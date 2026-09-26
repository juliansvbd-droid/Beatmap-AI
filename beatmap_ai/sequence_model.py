"""Mapping object by object: rhythm and placement from one sequence model.

The placement model (placement_model.py) decides where each object goes. This model
also decides *when* the next object comes, what it is and how long it lasts, one
object after the other, the way a mapper works through the timeline. Deciding in
order is what makes figures: a double tap followed by a pause, a triple into a jump,
a slider that holds over a vocal instead of two taps. A model that scores every beat
tick on its own (model.py) cannot see those.

For every object the transformer sees the earlier objects (timing, type, where they
went), the music at the object *and over the next four beats*, the star rating,
style measures and the community tags players voted for similar maps. It predicts:

* where this object goes (offset from the previous object, see placement_model),
  its slider shape and hitsounds;
* the gap to the next object in quarter beats (or "more than four beats"), whether
  the next object is a circle, slider or spinner, its length, repeats and whether it
  starts a new combo.

Long pauses are left to the per-frame rhythm model: after "more than four beats" the
next object starts where that model next hears a note.
"""

from __future__ import annotations

import copy
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, HitObject
from .placement_data import (BAR, BEAT, BEND, BMASK, CIRCLE, CLAP, CMASK, CU, CV, END, FINISH,
                             HITSOUND_BITS, KIND, LEN, N_COLUMNS, NC, OFFSET_SCALE, OMASK, OU, OV,
                             PHASE, SLIDER, SLIDES, T, VEL, WHISTLE, X)
from .placement_model import CausalBlock, mixture_nll
from .sequence_data import (DURATION_CLASSES, FEATURES, GAP_CLASSES, REPEAT_CLASSES,  # noqa: F401
                            SequenceSampler, batch_stream, build_sequence_maps, share_maps)


# Slider chord / length (1 = straight) of ranked maps: quantiles at CHORD_LEVELS per whole
# star rating, from ~2.2M sliders of 50 px or more. Most sliders are (nearly) straight;
# strongly curved ones get more common with difficulty.
CHORD_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
CHORD_QUANTILES = {
    1: (0.30, 0.566, 0.682, 0.826, 0.894, 0.951, 0.981, 1.0, 1.0),
    2: (0.35, 0.646, 0.764, 0.873, 0.928, 0.972, 0.995, 1.0, 1.0),
    3: (0.30, 0.567, 0.740, 0.865, 0.922, 0.972, 0.994, 1.0, 1.0),
    4: (0.15, 0.324, 0.488, 0.791, 0.896, 0.962, 0.991, 1.0, 1.0),
    5: (0.10, 0.222, 0.370, 0.664, 0.866, 0.953, 0.988, 1.0, 1.0),
    6: (0.07, 0.142, 0.272, 0.509, 0.804, 0.940, 0.984, 1.0, 1.0),
    7: (0.05, 0.095, 0.222, 0.510, 0.810, 0.941, 0.985, 1.0, 1.0),
}
STRAIGHT_BELOW_PX = 70.0  # shorter sliders are drawn straight
FULL_BEND_FROM_PX = 110.0  # from here on, curvature as drawn from CHORD_QUANTILES
# The table overstates bends as they come out of the renderer: at full strength 4-5 star
# maps got 12-17 % bent sliders (chord < 0.9 x length) and 3-7 % near-circles, ranked maps
# of those stars 4 % and 1 % (scripts/compare_maps.py). Half the bend matches them.
BEND_SCALE = 0.5


def human_chord(chord: tuple[float, float], stars: float, length: float,
                rng: np.random.Generator) -> tuple[float, float]:
    """Keep the model's slider direction but draw how curved it is from ranked maps.

    The chord head is a mixture of Gaussians over a point on (roughly) a ring -- a
    straight slider in any direction -- and fits that ring poorly: its draws land
    inside it, so most sliders came out curved. Short sliders stay straight: mappers
    practically never bend one under ~70 px (a bent short slider reads as a small circle),
    and bend them less up to ~110 px."""
    angle = math.atan2(chord[1], chord[0])
    ratio = 1.0
    if length >= STRAIGHT_BELOW_PX:
        levels = CHORD_QUANTILES[int(np.clip(round(stars), 1, 7))]
        ratio = float(np.interp(rng.random(), CHORD_LEVELS, levels))
        fade = min((length - STRAIGHT_BELOW_PX) / (FULL_BEND_FROM_PX - STRAIGHT_BELOW_PX), 1.0)
        ratio = 1.0 - (1.0 - ratio) * fade * BEND_SCALE
    return ratio * math.cos(angle), ratio * math.sin(angle)


class SequenceNet(nn.Module):
    def __init__(self, features: int = FEATURES, hidden: int = 320, layers: int = 8,
                 heads: int = 8, context: int = 128, offset_mixtures: int = 16,
                 chord_mixtures: int = 8):
        super().__init__()
        self.config = dict(features=features, hidden=hidden, layers=layers, heads=heads,
                           context=context, offset_mixtures=offset_mixtures,
                           chord_mixtures=chord_mixtures)
        self.inp = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.pos = nn.Parameter(torch.zeros(context, hidden))
        self.blocks = nn.ModuleList(CausalBlock(hidden, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.offset_head = nn.Linear(hidden, 5 * offset_mixtures)
        self.chord_head = nn.Linear(hidden, 5 * chord_mixtures)
        self.bend_head = nn.Linear(hidden, 1)
        self.hitsound_head = nn.Linear(hidden, 3)
        self.gap_head = nn.Linear(hidden, GAP_CLASSES)
        self.kind_head = nn.Linear(hidden, 3)
        self.duration_head = nn.Linear(hidden, DURATION_CLASSES)
        self.repeat_head = nn.Linear(hidden, REPEAT_CLASSES)
        self.combo_head = nn.Linear(hidden, 1)
        mask = torch.triu(torch.full((context, context), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        t = x.shape[1]
        h = self.inp(x) + self.pos[:t]
        for block in self.blocks:
            h = block(h, self.mask)
        h = self.norm(h)
        return {"offset": self.offset_head(h), "chord": self.chord_head(h),
                "bend": self.bend_head(h)[..., 0], "hitsound": self.hitsound_head(h),
                "gap": self.gap_head(h), "kind": self.kind_head(h),
                "duration": self.duration_head(h), "repeat": self.repeat_head(h),
                "combo": self.combo_head(h)[..., 0]}

    def step(self, x: torch.Tensor, pos: int,
             kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
            ) -> tuple[dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """A single autoregressive decode step for x of shape (B, 1, D).
        Uses KV caching in O(1) time without re-evaluating past tokens."""
        context = self.config["context"]
        pos_idx = min(pos, context - 1)
        h = self.inp(x) + self.pos[pos_idx:pos_idx + 1]
        new_cache = []
        for i, block in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None else None
            h, cache = block(h, kv_cache=past, max_context=context, use_cache=True)
            new_cache.append(cache)
        h = self.norm(h)
        out = {"offset": self.offset_head(h), "chord": self.chord_head(h),
               "bend": self.bend_head(h)[..., 0], "hitsound": self.hitsound_head(h),
               "gap": self.gap_head(h), "kind": self.kind_head(h),
               "duration": self.duration_head(h), "repeat": self.repeat_head(h),
               "combo": self.combo_head(h)[..., 0]}
        return out, new_cache


def sequence_losses(model: SequenceNet, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    out = model(batch["x"])
    y = batch["y"]
    present = ((y[..., T] > 0) | (y[..., X] > 0)).float()
    om, cm, bm = y[..., OMASK], y[..., CMASK], y[..., BMASK]

    def masked(values, mask):
        return (values * mask).sum() / mask.sum().clamp(min=1)

    offset = masked(mixture_nll(out["offset"], y[..., [OU, OV]]), om)
    chord = masked(mixture_nll(out["chord"], y[..., [CU, CV]]), cm)
    bend = masked(F.binary_cross_entropy_with_logits(out["bend"], y[..., BEND], reduction="none"), bm)
    hitsound = masked(F.binary_cross_entropy_with_logits(
        out["hitsound"], y[..., [WHISTLE, FINISH, CLAP]], reduction="none").mean(-1), present)
    rm = batch["rhythm_mask"] * present

    def ce(logits, target):
        return masked(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none")
                      .view_as(target), rm)

    gap = ce(out["gap"], batch["gap"])
    kind = ce(out["kind"], batch["kind"])
    slider_rm = rm * (batch["kind"] == SLIDER).float()
    duration = masked(F.cross_entropy(out["duration"].flatten(0, 1), batch["duration"].flatten(),
                                      reduction="none").view_as(rm), rm)
    repeat = masked(F.cross_entropy(out["repeat"].flatten(0, 1), batch["repeat"].flatten(),
                                    reduction="none").view_as(rm), slider_rm)
    combo = masked(F.binary_cross_entropy_with_logits(out["combo"], batch["combo"], reduction="none"), rm)
    total = (offset + 0.5 * chord + 0.2 * bend + 0.3 * hitsound
             + gap + 0.5 * kind + 0.5 * duration + 0.2 * repeat + 0.3 * combo)
    parts = {"offset": offset, "gap": gap, "kind": kind, "duration": duration, "combo": combo}
    return total, {k: v.item() for k, v in parts.items()}


def save_sequence(model: SequenceNet, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.config, "kind": "sequence", "state_dict": state}, path)


def load_sequence(path, device="cpu") -> SequenceNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = SequenceNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def is_sequence_checkpoint(path) -> bool:
    try:
        return torch.load(path, map_location="cpu", weights_only=True).get("kind") == "sequence"
    except Exception:
        return False


def train_sequence(data_dirs, out_path, tag_files=(), epochs: int = 30, steps_per_epoch: int = 1000,
                   batch_size: int = 48, lr: float = 4e-4, hidden: int = 320, layers: int = 8,
                   context: int = 128, device: str | None = None, seed: int = 0,
                   workers: int = 4, log=partial(print, flush=True)) -> Path:
    from .dataset import is_validation
    from .train import resolve_device
    device = resolve_device(device)
    torch.manual_seed(seed)
    maps = build_sequence_maps(data_dirs, tag_files, log=log)
    train_maps = [m for m in maps if not is_validation(m.song)]
    val_maps = [m for m in maps if is_validation(m.song)]
    log(f"{len(maps)} maps with {sum(len(m.objects) for m in maps):,} objects "
        f"({len(train_maps)} train, {len(val_maps)} validation); device={device}")
    val_sampler = SequenceSampler(val_maps, context, seed + 1, hide=False)
    val_batches = [{k: torch.from_numpy(v) for k, v in val_sampler.batch(batch_size).items()}
                   for _ in range(8)]
    shared = share_maps(train_maps, Path(out_path).with_suffix(".objects.npy"))
    del maps, train_maps, val_maps, val_sampler
    loader = batch_stream(shared, context, batch_size, seed, workers)
    model = SequenceNet(hidden=hidden, layers=layers, context=context).to(device)
    log(f"sequence model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    total_steps=epochs * steps_per_epoch)

    def validate() -> tuple[float, dict]:
        model.eval()
        totals, parts = [], {}
        with torch.no_grad():
            for b in val_batches:
                loss, p = sequence_losses(model, {k: v.to(device) for k, v in b.items()})
                totals.append(loss.item())
                for k, v in p.items():
                    parts.setdefault(k, []).append(v)
        model.train()
        return float(np.mean(totals)), {k: float(np.mean(v)) for k, v in parts.items()}

    out_path = Path(out_path)
    best, parts = validate()
    log(f"starting validation loss {best:.3f}")
    for epoch in range(1, epochs + 1):
        totals = []
        for _ in range(steps_per_epoch):
            batch = {k: torch.from_numpy(v).to(device) for k, v in next(loader).items()}
            batch["x"] = batch["x"].float()
            loss, _ = sequence_losses(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals.append(loss.item())
        val, parts = validate()
        detail = "  ".join(f"{k} {v:.3f}" for k, v in parts.items())
        log(f"epoch {epoch:3d}  loss {np.mean(totals):.3f}  validation {val:.3f}  ({detail})")
        if val < best:
            best = val
            save_sequence(model, out_path)
    if not out_path.exists():
        save_sequence(model, out_path)
    log(f"saved best sequence model (validation {best:.3f}) to {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Generation


class SequencePlacer:
    """Writes a map object by object with a SequenceNet: each step places the current
    object and draws when the next one comes, what it is and how long it lasts.

    ``sample`` returns a plan (PlannedObjects) and the placement choices; ``render``
    turns them into hit objects with every distance scaled (for star targeting), like
    placement_model.LearnedPlacer."""

    def __init__(self, model: SequenceNet, preset, features, timing, conditions: dict,
                 tags: dict[str, float] | None, note_scores: np.ndarray, grid,
                 threshold: float, rng: np.random.Generator, temperature: float = 0.8,
                 rhythm_temperature: float = 0.9, sv_at=None, guidance: float = 0.0):
        from .placement import circle_radius
        from .placement_data import section_energy
        from .sequence_data import CONDITIONS, TAGS
        from .style import encode_conditions
        self.model = model
        self.preset = preset
        self.features = features
        self.timing = timing
        tag_values = np.zeros(len(TAGS) + 1, dtype=np.float32)
        if tags:
            for name, value in tags.items():
                if name in TAGS:
                    tag_values[TAGS.index(name)] = value
        self.conditions = conditions or {}
        self.stars = float(conditions.get("stars", 4.0)) if conditions else 4.0
        self.cond = np.concatenate([encode_conditions(conditions, CONDITIONS), tag_values])
        self.energy = section_energy(features.mel)
        self.note_scores = note_scores  # frame model's note score per quarter-beat tick
        self.grid = grid  # quarter-beat TickGrid
        self.threshold = threshold
        self.rng = rng
        # Slider curvature is drawn per slider (by its time), not per candidate, so that
        # Best-of-N with a critic cannot pick bent sliders over straight ones.
        self.bend_seed = int(rng.integers(0, 2**31 - 1))
        self.temperature = temperature
        self.rhythm_temperature = rhythm_temperature
        self.stars = float(conditions.get("stars") or 4.0)
        # How much the frame model's note scores weigh in on the gap choice: the sequence
        # model knows figures (doubles, pauses), the frame model hears exactly where notes go.
        self.guidance = guidance
        self.radius = circle_radius(preset.cs)
        self.margin = self.radius * 0.6
        self.device = next(model.parameters()).device
        self.sv_at = sv_at

    def in_bounds(self, x: float, y: float) -> bool:
        m = self.margin
        return m <= x <= PLAYFIELD_WIDTH - m and m <= y <= PLAYFIELD_HEIGHT - m

    def _row(self, item) -> np.ndarray:
        row = np.zeros(N_COLUMNS, dtype=np.float32)
        beat_length = item.beat_length or self.timing.beat_length
        sv = self.sv_at(item.time) if self.sv_at is not None else 1.0
        velocity = self.preset.slider_multiplier * 100.0 * sv
        kind = SLIDER if item.kind == "slider" else CIRCLE
        slides = getattr(item, "slides", 1)
        row[[T, END, KIND, SLIDES, NC]] = item.time, item.end_time, kind, slides, float(item.new_combo)
        if kind == SLIDER:
            row[LEN] = (item.end_time - item.time) / slides / beat_length * velocity
        row[BEAT], row[VEL] = beat_length, velocity
        beats = (item.time - self.timing.offset_ms) / beat_length
        row[PHASE], row[BAR] = beats % 1.0, (beats % 4) / 4
        return row

    def _next_scored_tick(self, after: int) -> int | None:
        later = np.flatnonzero(self.note_scores[after:] >= self.threshold)
        return int(after + later[0]) if len(later) else None

    def _item(self, tick: int, kind: str, duration_ticks: int, slides: int, new_combo: bool):
        from .rhythm import PlannedObject
        times = self.grid.times
        end_tick = min(tick + duration_ticks * slides, len(times) - 1)
        beat_length = (float(self.grid.beat_length_ms[tick]) if self.grid.beat_length_ms is not None
                       else self.timing.beat_length)
        end_time = float(times[end_tick]) if kind == "slider" else float(times[tick])
        item = PlannedObject(float(times[tick]), kind, tick, 0.5, end_time,
                             int(self.grid.beat[tick] // 4), new_combo, beat_length)
        item.slides = slides
        return item

    @torch.no_grad()
    def sample(self, gap_bias: float = 0.0):
        """(plan, choices). ``gap_bias`` > 0 favours shorter gaps (more notes)."""
        from .placement_model import Choice, _Walker, sample_mixture, sample_mixture_guided
        from .sequence_data import LONG_GAP, sequence_features
        context = self.model.config["context"]
        first = self._next_scored_tick(0)
        if first is None:
            return [], []
        plan = [self._item(first, "circle", 0, 1, True)]
        choices = []
        rows = np.zeros((0, N_COLUMNS), dtype=np.float32)
        walker = _Walker(self)
        classes = np.arange(len(self.model.gap_head.bias))
        kv_cache = None
        while True:
            i = len(plan) - 1
            item = plan[i]
            rows = np.vstack([rows, self._row(item)])
            lo = max(0, i - context + 1)
            x = sequence_features(rows[lo:], self.features.mel, self.preset.cs, self.cond, self.energy)
            step_x = torch.from_numpy(x[-1:][None]).to(self.device)
            raw, kv_cache = self.model.step(step_x, i, kv_cache)
            out = {k: v[0, -1].float().cpu().numpy() for k, v in raw.items()}

            # Where this object goes (guided by boundaries and flow curvature).
            choice = None
            gap = (item.time - rows[i - 1, END]) / max(item.beat_length, 1.0) if i else 8.0
            for attempt in range(20):
                temp = self.temperature * (1 + 0.04 * attempt)
                u, v = sample_mixture_guided(out["offset"], walker, gap, self.rng, temp) * OFFSET_SCALE
                if i == 0 or walker.fits(u, v, item.time, gap):
                    choice = Choice((float(u), float(v)))
                    break
            choice = choice or Choice((float(u), float(v)))
            hit_p = 1.0 / (1.0 + np.exp(-out["hitsound"]))
            choice.hitsound = sum(bit for bit, p in zip(HITSOUND_BITS.values(), hit_p)
                                  if self.rng.random() < p)
            if item.kind == "slider":
                chord = sample_mixture(out["chord"], self.rng, self.temperature)
                choice.chord = human_chord((float(chord[0]), float(chord[1])), self.stars,
                                           float(rows[i, LEN]), self._bend_rng(item))
                bend_p = 1.0 / (1.0 + math.exp(-float(out["bend"])))
                choice.side = 1.0 if self.rng.random() < bend_p else -1.0
            choices.append(choice)
            walker.place(rows, i, item, choice, 1.0)

            # When the next object comes and what it is.
            logits = out["gap"].astype(np.float64) / self.rhythm_temperature
            busy = int(round((item.end_time - item.time) / max(item.beat_length, 1.0) * 4))
            min_gap = busy + (1 if item.kind == "slider" else 0)
            logits[0] = -np.inf
            logits[1:max(min_gap, 1)] = -np.inf
            logits[1:LONG_GAP] -= gap_bias * classes[1:LONG_GAP] / 4.0
            if self.guidance:
                ahead = np.clip(item.tick + classes[1:LONG_GAP], 0, len(self.note_scores) - 1)
                logits[1:LONG_GAP] += self.guidance * np.log(np.clip(self.note_scores[ahead], 1e-3, 1.0))
            p = np.exp(logits - logits.max())
            g = int(self.rng.choice(len(p), p=p / p.sum()))
            tick = self._next_scored_tick(item.tick + 17) if g == LONG_GAP else item.tick + g
            if tick is None or tick >= len(self.grid.times) - 1:
                break
            kind_p = np.exp(out["kind"] - out["kind"].max())
            kind_p[2] = 0.0  # no spinners: four beats of lookahead is too short for one
            kind = "slider" if self.rng.random() < kind_p[1] / kind_p.sum() else "circle"
            duration, slides = 0, 1
            if kind == "slider":
                d = np.exp(out["duration"] - out["duration"].max())
                d[0] = 0.0
                duration = int(self.rng.choice(len(d), p=d / d.sum()))
                r = np.exp(out["repeat"] - out["repeat"].max())
                slides = int(self.rng.choice(len(r), p=r / r.sum())) + 1
                if duration * slides > 16:
                    slides = 1
            combo = self.rng.random() < 1.0 / (1.0 + math.exp(-float(out["combo"])))
            plan.append(self._item(tick, kind, duration, slides, combo))
        return plan, choices

    def _quarter(self, item) -> int:
        """Nearest quarter-beat tick of an object (plans may use a 1/2 grid)."""
        times = self.grid.times
        j = int(np.searchsorted(times, item.time))
        lo = max(j - 1, 0)
        return lo + int(np.argmin(np.abs(times[lo:j + 1] - item.time)))

    def _bend_rng(self, item) -> np.random.Generator:
        return np.random.default_rng((self.bend_seed, int(round(item.time))))

    def _step_placement(self, item, out_plan, choices, rows, walker, scale, rng):
        from .placement_model import Choice, sample_mixture
        from .sequence_data import sequence_features
        context = self.model.config["context"]
        i = len(out_plan)
        out_plan.append(item)
        row = self._row(item)
        if i < len(rows):
            rows[i] = row
        else:
            rows = np.vstack([rows, row])
        x = sequence_features(rows[max(0, i - context + 1):i + 1], self.features.mel, self.preset.cs,
                              self.cond, self.energy)
        raw = self.model(torch.from_numpy(x)[None].to(self.device))
        o_len = raw["offset"].shape[-1]
        h_len = raw["hitsound"].shape[-1]
        c_len = raw["chord"].shape[-1]
        g_len = raw["gap"].shape[-1]
        packed = torch.cat([
            raw["offset"][0, -1].view(-1),
            raw["hitsound"][0, -1].view(-1),
            raw["chord"][0, -1].view(-1),
            raw["bend"][0, -1].view(-1),
            raw["gap"][0, -1].view(-1),
        ]).float().cpu().numpy()
        i1 = o_len
        i2 = i1 + h_len
        i3 = i2 + c_len
        i4 = i3 + 1
        out = {
            "offset": packed[:i1],
            "hitsound": packed[i1:i2],
            "chord": packed[i2:i3],
            "bend": packed[i3],
            "gap": packed[i4:i4 + g_len],
        }

        choice = fallback = None
        gap = (item.time - rows[i - 1, END]) / max(item.beat_length or 1.0, 1.0) if i else 8.0
        for attempt in range(30):
            temp = self.temperature * (1 + 0.05 * attempt)
            u, v = sample_mixture(out["offset"], rng, temp) * OFFSET_SCALE
            if i == 0 or walker.fits(u * scale, v * scale, item.time, gap):
                choice = Choice((float(u), float(v)))
                break
            if fallback is None and walker.fits(u * scale, v * scale):
                fallback = Choice((float(u), float(v)))
        choice = choice or fallback or Choice((float(u), float(v)))
        hit_p = 1.0 / (1.0 + np.exp(-out["hitsound"]))
        choice.hitsound = sum(bit for bit, p in zip(HITSOUND_BITS.values(), hit_p)
                              if rng.random() < p)
        if item.kind == "slider":
            chord = sample_mixture(out["chord"], rng, self.temperature)
            choice.chord = human_chord((float(chord[0]), float(chord[1])), self.stars,
                                       float(rows[i, LEN]), self._bend_rng(item))
            bend_p = 1.0 / (1.0 + math.exp(-float(out["bend"])))
            choice.side = 1.0 if rng.random() < bend_p else -1.0
        choices.append(choice)
        walker.place(rows, i, item, choice, scale)
        return out, rows

    def _edit_figures(self, item, out_plan, queue, k, gap_logits, drop_below, add_above, strong):
        gap_p = np.exp(gap_logits - gap_logits.max())
        gap_p /= gap_p.sum()
        here = self._quarter(item)
        nxt = queue[k] if k < len(queue) else None
        after = self._quarter(nxt) if nxt is not None else None
        before = self._quarter(out_plan[-2]) if len(out_plan) > 1 else None
        if after == here + 1:
            ends_run = k + 1 >= len(queue) or self._quarter(queue[k + 1]) != after + 1
            third = before == here - 1 and ends_run
            if third and gap_p[1] < drop_below and self.note_scores[after] < strong:
                k += 1  # make the triple a double
        elif (gap_p[1] > add_above and here + 1 < len(self.note_scores)
              and (after is None or after >= here + 3) and before != here - 1
              and self.note_scores[here + 1] >= 0.5 * self.threshold):
            queue.insert(k, self._item(here + 1, "circle", 0, 1, False))
        return k

    @torch.no_grad()
    def follow(self, plan, drop_below: float = 0.15, add_above: float = 0.6,
               strong: float = 0.8, scale: float = 1.0,
               critic=None, candidates: int = 4, chunk_size: int = 24):
        """Place a given rhythm (from the frame model) object by object, and let the
        sequence model edit its quick figures. When ``critic`` is provided with
        ``candidates`` > 1, section by section (16-32 objects) ``candidates`` variants are
        evaluated and the best one selected (Best-of-N). Returns (plan, choices)."""
        from .placement_model import _Walker
        queue = sorted((p for p in plan if p.kind != "spinner"), key=lambda p: p.time)
        out_plan, choices = [], []
        rows = np.zeros((len(queue) * 2 + 64, N_COLUMNS), dtype=np.float32)
        walker = _Walker(self)

        if critic is None or candidates <= 1:
            # Fast direct single-trajectory generation
            k = 0
            while k < len(queue):
                item = queue[k]
                if not hasattr(item, "slides"):
                    item.slides = 1
                out, rows = self._step_placement(item, out_plan, choices, rows, walker, scale, self.rng)
                k += 1
                if item.kind == "circle":
                    k = self._edit_figures(item, out_plan, queue, k, out["gap"], drop_below, add_above, strong)
            return out_plan, choices

        # Best-of-N chunk-by-chunk evaluation
        from .critic_data import CRITIC_WINDOW, critic_features
        chunk_size = max(16, min(chunk_size, 32))
        k = 0
        while k < len(queue):
            target_chunk = min(chunk_size, len(queue) - k)
            cand_results = []

            for c in range(candidates):
                cand_seed = int(self.rng.integers(0, 2**31 - 1))
                cand_rng = np.random.default_rng(cand_seed)

                cand_walker = copy.copy(walker)
                cand_walker.recent = list(walker.recent)
                cand_out_plan = list(out_plan)
                cand_choices = list(choices)
                cand_rows = rows.copy()
                cand_queue = [copy.copy(p) for p in queue]
                cand_k = k

                count = 0
                while cand_k < len(cand_queue) and count < target_chunk:
                    item = cand_queue[cand_k]
                    if not hasattr(item, "slides"):
                        item.slides = 1
                    out, cand_rows = self._step_placement(item, cand_out_plan, cand_choices, cand_rows,
                                                          cand_walker, scale, cand_rng)
                    count += 1
                    cand_k += 1
                    if item.kind == "circle":
                        cand_k = self._edit_figures(item, cand_out_plan, cand_queue, cand_k,
                                                    out["gap"], drop_below, add_above, strong)

                n_objs = len(cand_out_plan)
                ctx_len = min(n_objs, CRITIC_WINDOW)
                eval_rows = cand_rows[n_objs - ctx_len:n_objs]
                feats = critic_features(eval_rows, self.features.mel, self.stars, self.energy)
                pad = CRITIC_WINDOW - len(feats)
                if pad > 0:
                    feats_pad = np.pad(feats, ((0, pad), (0, 0)), mode="constant")
                    mask = np.pad(np.ones(len(feats), dtype=np.float32), (0, pad), mode="constant")
                else:
                    feats_pad = feats
                    mask = np.ones(CRITIC_WINDOW, dtype=np.float32)

                cand_results.append((
                    cand_out_plan, cand_choices, cand_rows, cand_walker, cand_queue, cand_k,
                    feats_pad, mask
                ))

            # Score candidates with critic
            batch_x = torch.from_numpy(np.stack([r[6] for r in cand_results])).to(self.device).float()
            batch_mask = torch.from_numpy(np.stack([r[7] for r in cand_results])).to(self.device).float()

            if hasattr(critic, "score"):
                scores = critic.score(batch_x, batch_mask)
            elif callable(critic):
                scores = critic(batch_x, batch_mask)
            else:
                scores = np.zeros(candidates)

            if isinstance(scores, torch.Tensor):
                scores_arr = scores.detach().cpu().numpy().ravel()
            else:
                scores_arr = np.asarray(scores).ravel()

            best_idx = int(np.argmax(scores_arr))
            out_plan, choices, rows, walker, queue, k = cand_results[best_idx][:6]

        return out_plan, choices

    def render(self, plan, choices, spacing_scale: float = 1.0) -> list[HitObject]:
        from .placement_model import _Walker
        rows = np.stack([self._row(item) for item in plan]) if plan else np.zeros((0, N_COLUMNS))
        walker = _Walker(self)
        objects = []
        for i, (item, choice) in enumerate(zip(plan, choices)):
            obj = walker.place(rows, i, item, choice, spacing_scale)
            if obj.kind == "slider":
                obj.slides = getattr(item, "slides", 1)
            objects.append(obj)
        return objects
