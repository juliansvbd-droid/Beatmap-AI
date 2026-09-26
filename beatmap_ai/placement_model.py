"""Learned placement: where mappers put the next object, learned from ranked maps.

Objects are read in order, like a mapper working through the timeline. For every object
a causal transformer sees the previous objects (timing, type, where they went), the
music at that moment and the map's star rating and style, and predicts:

* the offset from the previous object's end to this object's start, in a frame turned
  so that +x is the direction of the previous movement -- so "back and forth",
  "keep turning by 120 degrees" (triangles) or "stack" are the same numbers anywhere
  on the playfield, and patterns can be learned;
* for sliders, the chord from the slider's start to its far end (relative to its
  length, same frame) and which side the slider curves to;
* the object's hitsounds (whistle, finish, clap).

Besides the music right at the object it sees how intense the surrounding section is
compared with the whole song, so it can map a chorus bigger than a verse.

Offsets and chords are mixtures of Gaussians, so the model can hold several good
options at once (a jump either way, a stack, a turn) instead of averaging them.
Maps are then drawn by sampling one object at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, HitObject
from .placement_data import (  # noqa: F401  (re-exported for older imports)
    BAR,
    BEAT,
    BEND,
    BMASK,
    CIRCLE,
    CLAP,
    CMASK,
    CONDITIONS,
    CU,
    CV,
    END,
    EX,
    EY,
    FEATURES,
    FINISH,
    HCOS,
    HITSOUND_BITS,
    HSIN,
    KIND,
    LEN,
    MIN_MOVE,
    NC,
    N_COLUMNS,
    OFFSET_SCALE,
    OMASK,
    OU,
    OV,
    PHASE,
    PLACEMENT_CACHE,
    PlacementMap,
    SECTION_SECONDS,
    SLIDER,
    SLIDES,
    SPINNER,
    T,
    VEL,
    WHISTLE,
    WindowSampler,
    X,
    Y,
    build_placement_maps,
    map_objects,
    rotate,
    section_energy,
    slider_far_point,
    slider_side,
    token_features,
)


# --------------------------------------------------------------------------------------
# Model

import torch
import torch.nn.functional as F
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                max_context: int | None = None,
                use_cache: bool = False
               ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            if max_context is not None and k.shape[-2] > max_context:
                k = k[:, :, -max_context:]
                v = v[:, :, -max_context:]
            scores = q @ k.transpose(-1, -2) * (d // self.heads) ** -0.5
            if mask is not None:
                scores = scores + mask[:, :k.shape[-2]]
        else:
            scores = q @ k.transpose(-1, -2) * (d // self.heads) ** -0.5
            if mask is not None:
                scores = scores + mask[:t, :t]
        attn = (self.drop(torch.softmax(scores, dim=-1)) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.out(attn))
        out = x + self.drop(self.ff(self.norm2(x)))
        if use_cache or kv_cache is not None:
            return out, (k, v)
        return out


class PlacementNet(nn.Module):
    def __init__(self, features: int = FEATURES, hidden: int = 256, layers: int = 6,
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
        self.hitsound_head = nn.Linear(hidden, 3)  # whistle, finish, clap
        mask = torch.triu(torch.full((context, context), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        t = x.shape[1]
        h = self.inp(x) + self.pos[:t]
        for block in self.blocks:
            h = block(h, self.mask)
        h = self.norm(h)
        return (self.offset_head(h), self.chord_head(h), self.bend_head(h)[..., 0],
                self.hitsound_head(h))


def mixture_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of 2-D targets under a diagonal Gaussian mixture."""
    k = params.shape[-1] // 5
    logits, mu, log_sigma = params[..., :k], params[..., k:3 * k], params[..., 3 * k:]
    mu = mu.unflatten(-1, (k, 2))
    log_sigma = log_sigma.unflatten(-1, (k, 2)).clamp(-5.0, 3.0)
    z = (target.unsqueeze(-2) - mu) / log_sigma.exp()
    log_prob = -0.5 * (z ** 2).sum(-1) - log_sigma.sum(-1) - math.log(2 * math.pi)
    return -torch.logsumexp(F.log_softmax(logits, dim=-1) + log_prob, dim=-1)


def sample_mixture(params: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
    """One draw from the mixture. ``temperature`` narrows each component but leaves
    the choice between components alone: sharpening that choice would keep picking the
    most common option (often a small move or a stack) and shrink the whole map."""
    k = len(params) // 5
    logits = params[:k]
    p = np.exp(logits - logits.max())
    j = int(rng.choice(k, p=p / p.sum()))
    mu = params[k + 2 * j:k + 2 * j + 2]
    sigma = np.exp(np.clip(params[3 * k + 2 * j:3 * k + 2 * j + 2], -5, 3))
    return mu + sigma * temperature * rng.standard_normal(2)


def sample_mixture_guided(params: np.ndarray, walker, gap_beats: float,
                          rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
    """Sample mixture with boundary and flow awareness:
    penalizes components pointing outside the playfield and avoids harsh anti-flow reversals."""
    k = len(params) // 5
    logits = params[:k] / max(temperature, 1e-3)
    weights = np.exp(logits - logits.max())

    scores = np.zeros(k, dtype=np.float64)
    for j in range(k):
        mu = params[k + 2 * j:k + 2 * j + 2]
        u = float(mu[0]) * OFFSET_SCALE
        v = float(mu[1]) * OFFSET_SCALE
        dx, dy = rotate(u, v, walker.heading)
        x = walker.end[0] + dx
        y = walker.end[1] + dy

        m = walker.p.margin
        out_x = max(0.0, m - x) + max(0.0, x - (PLAYFIELD_WIDTH - m))
        out_y = max(0.0, m - y) + max(0.0, y - (PLAYFIELD_HEIGHT - m))
        pen = (out_x + out_y) / 25.0

        dist_to_edge = min(x, PLAYFIELD_WIDTH - x, y, PLAYFIELD_HEIGHT - y)
        if dist_to_edge < 60.0:
            center_dx = (PLAYFIELD_WIDTH / 2) - walker.end[0]
            center_dy = (PLAYFIELD_HEIGHT / 2) - walker.end[1]
            toward = (dx * center_dx + dy * center_dy) / (math.hypot(dx, dy) * math.hypot(center_dx, center_dy) + 1e-6)
            pen -= 0.6 * toward

        if gap_beats <= 0.35 and math.hypot(dx, dy) > MIN_MOVE:
            if u < -0.15 * OFFSET_SCALE:
                pen += 2.5

        scores[j] = -pen

    guided_weights = weights * np.exp(np.clip(scores, -8.0, 4.0))
    total_w = guided_weights.sum()
    if not math.isfinite(total_w) or total_w <= 0:
        guided_weights = weights
        total_w = guided_weights.sum()

    j = int(rng.choice(k, p=guided_weights / total_w))
    mu = params[k + 2 * j:k + 2 * j + 2]
    sigma = np.exp(np.clip(params[3 * k + 2 * j:3 * k + 2 * j + 2], -5, 3))
    return mu + sigma * temperature * rng.standard_normal(2)


def losses(model: PlacementNet, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    offset, chord, bend, hitsounds = model(batch["x"])
    y = batch["y"]
    present = ((y[..., T] > 0) | (y[..., X] > 0)).float()  # not padding
    om, cm, bm = y[..., OMASK], y[..., CMASK], y[..., BMASK]
    offset_nll = (mixture_nll(offset, y[..., [OU, OV]]) * om).sum() / om.sum().clamp(min=1)
    chord_nll = (mixture_nll(chord, y[..., [CU, CV]]) * cm).sum() / cm.sum().clamp(min=1)
    bend_loss = (F.binary_cross_entropy_with_logits(bend, y[..., BEND], reduction="none") * bm).sum() \
        / bm.sum().clamp(min=1)
    hitsound_loss = (F.binary_cross_entropy_with_logits(
        hitsounds, y[..., [WHISTLE, FINISH, CLAP]], reduction="none").mean(-1) * present).sum()
    hitsound_loss = hitsound_loss / present.sum().clamp(min=1)
    total = offset_nll + 0.5 * chord_nll + 0.2 * bend_loss + 0.3 * hitsound_loss
    return total, {"offset": offset_nll.item(), "chord": chord_nll.item(), "bend": bend_loss.item(),
                   "hitsound": hitsound_loss.item()}


def save_placement(model: PlacementNet, path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.config, "state_dict": state, **(extra or {})}, path)


def load_placement(path, device="cpu") -> PlacementNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PlacementNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def train_placement(data_dirs, out_path, epochs: int = 30, steps_per_epoch: int = 500,
                    batch_size: int = 64, lr: float = 5e-4, hidden: int = 256, layers: int = 6,
                    context: int = 128, device: str | None = None, seed: int = 0,
                    log=partial(print, flush=True)) -> Path:
    from .dataset import is_validation
    from .train import resolve_device
    device = resolve_device(device)
    torch.manual_seed(seed)
    maps = build_placement_maps(data_dirs, log=log)
    train_maps = [m for m in maps if not is_validation(m.song)]
    val_maps = [m for m in maps if is_validation(m.song)]
    log(f"{len(maps)} maps with {sum(len(m.objects) for m in maps):,} objects "
        f"({len(train_maps)} train, {len(val_maps)} validation); device={device}")
    sampler = WindowSampler(train_maps, context, seed)
    # A fixed validation set: the same windows, all conditions given, every epoch.
    val_sampler = WindowSampler(val_maps, context, seed + 1, hide=False)
    val_batches = [{k: torch.from_numpy(v) for k, v in val_sampler.batch(batch_size).items()}
                   for _ in range(8)]
    model = PlacementNet(hidden=hidden, layers=layers, context=context).to(device)
    log(f"placement model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    total_steps=epochs * steps_per_epoch)

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            values = [losses(model, {k: v.to(device) for k, v in b.items()})[0].item()
                      for b in val_batches]
        model.train()
        return float(np.mean(values))

    out_path = Path(out_path)
    best = validate()
    log(f"starting validation loss {best:.3f}")
    for epoch in range(1, epochs + 1):
        totals = []
        for _ in range(steps_per_epoch):
            batch = {k: torch.from_numpy(v).to(device) for k, v in sampler.batch(batch_size).items()}
            loss, _ = losses(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals.append(loss.item())
        val = validate()
        log(f"epoch {epoch:3d}  loss {np.mean(totals):.3f}  validation {val:.3f}")
        if val < best:
            best = val
            save_placement(model, out_path)
    if not out_path.exists():
        save_placement(model, out_path)
    log(f"saved best placement model (validation {best:.3f}) to {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Generation


def arc_bend(ratio: float) -> float:
    """Total turning angle of a circular arc whose chord is ``ratio`` times its length."""
    ratio = float(np.clip(ratio, 0.05, 1.0))
    if ratio > 0.999:
        return 0.0
    lo, hi = 1e-4, 2 * math.pi - 1e-3
    for _ in range(40):  # sin(b/2)/(b/2) falls from 1 to 0 on (0, 2*pi)
        mid = (lo + hi) / 2
        if math.sin(mid / 2) / (mid / 2) > ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Choice:
    offset: tuple[float, float]  # local frame, px
    chord: tuple[float, float] | None = None  # local frame, relative to slider length
    side: float = 1.0
    hitsound: int = 0  # osu! hitsound bits (2 whistle, 4 finish, 8 clap)


class LearnedPlacer:
    """Places a planned map with a PlacementNet; see ``sample`` and ``render``."""

    def __init__(self, model: PlacementNet, preset, mel: np.ndarray, conditions: dict,
                 rng: np.random.Generator, temperature: float = 0.75, offset_ms: float = 0.0,
                 sv_at=None):
        from .placement import circle_radius
        from .style import encode_conditions
        self.model = model
        self.preset = preset
        self.mel = mel
        self.energy = section_energy(mel)
        self.cond = encode_conditions(conditions, CONDITIONS)
        self.rng = rng
        self.temperature = temperature
        self.offset_ms = offset_ms  # time of a downbeat, for beat and bar positions
        self.radius = circle_radius(preset.cs)
        self.margin = self.radius * 0.6
        self.device = next(model.parameters()).device
        self.sv_at = sv_at

    def in_bounds(self, x: float, y: float) -> bool:
        m = self.margin
        return m <= x <= PLAYFIELD_WIDTH - m and m <= y <= PLAYFIELD_HEIGHT - m

    def _rows(self, plan) -> np.ndarray:
        """map_objects-style rows for the plan (timing only; positions are filled in)."""
        rows = np.zeros((len(plan), N_COLUMNS), dtype=np.float32)
        base_velocity = self.preset.slider_multiplier * 100.0
        for i, item in enumerate(plan):
            beat_length = item.beat_length or 500.0
            sv = self.sv_at(item.time) if self.sv_at is not None else 1.0
            velocity = base_velocity * sv
            kind = {"circle": CIRCLE, "slider": SLIDER, "spinner": SPINNER}[item.kind]
            rows[i, [T, END, KIND, SLIDES, NC]] = item.time, item.end_time, kind, 1, float(item.new_combo)
            rows[i, LEN] = (item.end_time - item.time) / beat_length * velocity if kind == SLIDER else 0
            rows[i, BEAT], rows[i, VEL] = beat_length, velocity
            beats = (item.time - self.offset_ms) / beat_length
            rows[i, PHASE], rows[i, BAR] = beats % 1.0, (beats % 4) / 4
        return rows

    @torch.no_grad()
    def sample(self, plan) -> list[Choice | None]:
        """Draw every object's offset (and slider chord) one after the other."""
        rows = self._rows(plan)
        context = self.model.config["context"]
        choices: list[Choice | None] = []
        state = _Walker(self)
        for i, item in enumerate(plan):
            if item.kind == "spinner":
                choices.append(None)
                state.spinner(rows, i)
                continue
            lo = max(0, i - context + 1)
            x = token_features(rows[lo:i + 1], self.mel, self.preset.cs, self.cond, self.energy)
            offset, chord, bend, hitsounds = self.model(torch.from_numpy(x)[None].to(self.device))
            offset, chord = offset[0, -1].cpu().numpy(), chord[0, -1].cpu().numpy()
            hit_p = torch.sigmoid(hitsounds[0, -1]).cpu().numpy()
            choice = None
            for attempt in range(20):
                temp = self.temperature * (1.0 + 0.05 * attempt)
                u, v = sample_mixture(offset, self.rng, temp) * OFFSET_SCALE
                gap = (item.time - rows[i - 1, END]) / max(rows[i, BEAT], 1.0) if i else 8.0
                if i == 0 or state.fits(u, v, item.time, gap):
                    choice = Choice((float(u), float(v)))
                    break
            if choice is None:
                choice = Choice((float(u), float(v)))
            choice.hitsound = sum(bit for bit, p in zip(HITSOUND_BITS.values(), hit_p)
                                  if self.rng.random() < p)
            if item.kind == "slider":
                choice.chord = tuple(float(c) for c in sample_mixture(chord, self.rng, self.temperature))
                p = 1.0 / (1.0 + math.exp(-float(bend[0, -1])))
                choice.side = 1.0 if self.rng.random() < p else -1.0
            choices.append(choice)
            state.place(rows, i, item, choice, 1.0)
        return choices

    def render(self, plan, choices: list[Choice | None], spacing_scale: float = 1.0) -> list[HitObject]:
        """Turn sampled choices into hit objects, scaling every offset (star targeting)."""
        rows = self._rows(plan)
        state = _Walker(self)
        objects = []
        for i, (item, choice) in enumerate(zip(plan, choices)):
            if choice is None:
                objects.append(HitObject(256, 192, item.time, "spinner", True, end_time=item.end_time))
                state.spinner(rows, i)
                continue
            objects.append(state.place(rows, i, item, choice, spacing_scale))
        return objects


class _Walker:
    """Keeps the running position and movement direction while placing objects."""

    def __init__(self, placer: LearnedPlacer):
        self.p = placer
        self.end = (PLAYFIELD_WIDTH / 2, PLAYFIELD_HEIGHT / 2)
        self.heading = 0.0
        self.recent: list[tuple[float, float, float]] = []  # (x, y, time) of earlier objects
        self.stacked = False  # the last object was stacked on the one before

    def fits(self, u: float, v: float, time: float | None = None, gap_beats: float = 1.0) -> bool:
        """On the playfield and readable: stacked pairs on 1/4 or 1/2 notes (no longer
        stack chains), no covering an object shown in the last second."""
        dx, dy = rotate(u, v, self.heading)
        x, y = self.end[0] + dx, self.end[1] + dy
        if not self.p.in_bounds(x, y):
            return False
        if time is None:
            return True
        if math.hypot(dx, dy) < MIN_MOVE:
            # At most two objects on one spot: stacked pairs on 1/4 or 1/2, no stack chains.
            return gap_beats <= 0.55 and not self.stacked
        for rx, ry, rt in self.recent[:-1]:  # the previous object is where we start from
            if time - rt < 1000 and math.hypot(x - rx, y - ry) < self.p.radius:
                return False
        return True

    def spinner(self, rows: np.ndarray, i: int) -> None:
        rows[i, [X, Y, EX, EY]] = 256, 192, 256, 192
        rows[i, HCOS], rows[i, HSIN] = math.cos(self.heading), math.sin(self.heading)
        self.end = (256.0, 192.0)

    def place(self, rows: np.ndarray, i: int, item, choice: Choice, scale: float) -> HitObject:
        from .placement import slider_path
        u, v = choice.offset
        u, v = u * scale, v * scale
        rows[i, HCOS], rows[i, HSIN] = math.cos(self.heading), math.sin(self.heading)
        # Out of the playfield (after scaling): turn the offset until it fits, then shrink.
        for turn in [0.0] + [s * k * math.radians(15) for k in range(1, 13) for s in (1, -1)]:
            dx, dy = rotate(*rotate(u, v, turn), self.heading)
            if self.p.in_bounds(self.end[0] + dx, self.end[1] + dy):
                break
        else:
            for shrink in (0.8, 0.6, 0.4, 0.2, 0.0):
                dx, dy = rotate(u * shrink, v * shrink, self.heading)
                if self.p.in_bounds(self.end[0] + dx, self.end[1] + dy):
                    break
        x = float(np.clip(self.end[0] + dx, self.p.margin, PLAYFIELD_WIDTH - self.p.margin))
        y = float(np.clip(self.end[1] + dy, self.p.margin, PLAYFIELD_HEIGHT - self.p.margin))
        self.stacked = math.hypot(x - self.end[0], y - self.end[1]) < MIN_MOVE
        lu, lv = rotate(x - self.end[0], y - self.end[1], -self.heading)
        rows[i, [OU, OV, OMASK]] = lu / OFFSET_SCALE, lv / OFFSET_SCALE, 1.0
        if math.hypot(x - self.end[0], y - self.end[1]) >= MIN_MOVE:
            self.heading = math.atan2(y - self.end[1], x - self.end[0])
        obj = HitObject(x, y, item.time, "circle", item.new_combo, choice.hitsound)
        for column, bit in HITSOUND_BITS.items():
            rows[i, column] = float(bool(choice.hitsound & bit))
        end = (x, y)
        length = float(rows[i, LEN])
        if item.kind == "slider" and choice.chord is not None and length >= 1.0:
            cu, cv = choice.chord
            ratio = math.hypot(cu, cv)
            chord_angle = math.atan2(cv, cu) + self.heading
            bend = choice.side * arc_bend(ratio)
            best_score = float("inf")
            path = None
            anchor = None
            if self.recent:
                rx, ry, _ = self.recent[-1]
                if 20.0 < math.hypot(rx - x, ry - y) < length + 120.0:
                    anchor = (rx, ry)

            for turn in [0.0] + [s * k * math.radians(15) for k in range(1, 9) for s in (1, -1)]:
                for b in (bend, -bend, bend * 0.5, 0.0):
                    candidate = slider_path((x, y), chord_angle + turn - b / 2, length, b)
                    points = [candidate[2](length * s / 8) for s in range(1, 9)]
                    if all(self.p.in_bounds(*pt) for pt in points):
                        score = abs(turn) + 0.3 * abs(b - bend)
                        if anchor is not None:
                            dists = [math.hypot(pt[0] - anchor[0], pt[1] - anchor[1]) for pt in points]
                            variance = float(np.std(dists))
                            score += 0.04 * variance
                        if score < best_score:
                            best_score = score
                            path = candidate
                if path is not None and best_score < 0.25:
                    break
            if path is None:
                item.end_time = item.time
            else:
                curve_type, points, _, _ = path
                obj.kind, obj.curve_type, obj.curve_points, obj.length = "slider", curve_type, points, length
                obj.slides = max(int(getattr(item, "slides", 1)), 1)
                end = points[-1]
                far = rotate(end[0] - x, end[1] - y, -self.heading)
                rows[i, [CU, CV, CMASK]] = far[0] / length, far[1] / length, 1.0
                if math.hypot(end[0] - x, end[1] - y) >= MIN_MOVE:
                    self.heading = math.atan2(end[1] - y, end[0] - x)
                if obj.slides % 2 == 0:  # back and forth an even number of times: ends at its head
                    end = (x, y)
                    self.heading += math.pi
        rows[i, [X, Y, EX, EY]] = x, y, end[0], end[1]
        self.end = (float(end[0]), float(end[1]))
        if obj.kind == "slider":
            self.recent.append(((x + end[0]) / 2, (y + end[1]) / 2, item.time))
        self.recent = (self.recent + [(x, y, item.time), (self.end[0], self.end[1], item.end_time)])[-12:]
        return obj
