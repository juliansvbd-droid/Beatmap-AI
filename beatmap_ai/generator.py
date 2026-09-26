"""End-to-end generation: audio file in, .osz beatmap set out."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import AudioFeatures, compute_features, load_audio, preview_time_ms
from .difficulty import (DEFAULT_STARS, DifficultyPreset, get_preset, preset_for_stars,
                         stars_for_density)
from .structure import copy_patterns, copy_rhythm, copy_sections, find_repeats, kiai_sections
from .style import map_style, star_rating, style_tags
from .osu import Beatmap, TimingPoint, write_osz
from .placement import Placer
from .rhythm import RHYTHM_VARIETY, PlannedObject, plan_objects
from .timing import (TimingEstimate, estimate_swing, estimate_timing, fit_offset,
                     track_variable_timing)

MAX_COMBO = 16
BUNDLED_MODEL = Path(__file__).parent / "models" / "rhythm.pt"
BUNDLED_SEQUENCE = Path(__file__).parent / "models" / "sequence.pt"
BUNDLED_PLACEMENT = Path(__file__).parent / "models" / "placement.pt"
BUNDLED_CRITIC = Path(__file__).parent / "models" / "critic.pt"
BREAK_MIN_GAP = 5000.0


def analyze(audio_path: str | Path, bpm: float | None = None,
            offset: float | None = None) -> tuple[AudioFeatures, TimingEstimate]:
    features = compute_features(load_audio(audio_path), chroma=True)
    if bpm is None:
        timing = estimate_timing(features)
        if offset is not None:
            timing.offset_ms = offset
            timing.swing_ratio, timing.swing_confidence = estimate_swing(
                features, timing.bpm, timing.offset_ms)
            timing.beat_times_ms, timing.beat_indices, timing.tempo_points = track_variable_timing(
                features, timing.bpm, timing.offset_ms)
    else:
        timing = TimingEstimate(bpm, offset if offset is not None else fit_offset(features, bpm))
        timing.swing_ratio, timing.swing_confidence = estimate_swing(
            features, timing.bpm, timing.offset_ms)
        timing.beat_times_ms, timing.beat_indices, timing.tempo_points = track_variable_timing(
            features, timing.bpm, timing.offset_ms)
    return features, timing


def assign_combos(plan: list[PlannedObject], preset: DifficultyPreset, beat_length: float) -> None:
    prev: PlannedObject | None = None
    length = 0
    for item in plan:
        new = (
            prev is None
            or item.kind == "spinner"
            or prev.kind == "spinner"
            or item.measure // preset.combo_measures != prev.measure // preset.combo_measures
            or item.time - prev.end_time > 4 * beat_length
            or length >= MAX_COMBO
        )
        item.new_combo = new
        length = 1 if new else length + 1
        prev = item


def approach_preempt(ar: float) -> float:
    if ar < 5:
        return 1200.0 + 600.0 * (5 - ar) / 5
    return 1200.0 - 750.0 * (ar - 5) / 5


def find_breaks(bm: Beatmap) -> list[tuple[float, float]]:
    breaks = []
    preempt = approach_preempt(bm.ar)
    objs = bm.hit_objects
    for a, b in zip(objs, objs[1:]):
        start, end = bm.end_time(a) + 200.0, b.time - preempt
        if b.time - bm.end_time(a) >= BREAK_MIN_GAP and end - start >= 650.0:
            breaks.append((start, end))
    return breaks


def model_conditions(model, preset: DifficultyPreset, stars: float | None,
                     style: dict[str, float] | None) -> float | dict[str, float]:
    """Condition values for the model: the density (and star rating) wanted, plus style
    strengths turned into values typical for maps of these stars (see ``style_values``)."""
    if not getattr(model, "conditions", None):
        return preset.density
    values = {"density": preset.density}
    if stars is not None:
        values["stars"] = stars
    values.update(style_values(model, stars, style))
    return values


def style_values(model, stars: float | None, style: dict[str, float] | None) -> dict[str, float]:
    """``style`` maps a style name from ``style.STYLES`` to a strength in 0..1 (or a
    condition that is not also a style name, e.g. "tech", to its raw value). Each strength moves the style's conditions from
    the median of the training maps at this star level towards their 95th (raise) or
    5th (lower) percentile. Conditions no style mentions stay "Auto"."""
    from .style import CONDITIONS, STYLES
    stats = getattr(model, "config", {}).get("cond_stats") or {}
    if not style or not stats:
        return {}
    wanted = stars if stars is not None else 4.0
    level = min(stats, key=lambda k: abs(float(k) + 0.5 - wanted))
    out = {}
    for name, strength in style.items():
        if name not in STYLES and name in CONDITIONS:
            out[name] = float(strength)
            continue
        for condition, direction in STYLES.get(name, {}).items():
            pct = stats[level].get(condition)
            if pct is None:
                continue
            p5, _, p50, _, p95 = pct
            target = p95 if direction >= 0.5 else p5
            out[condition] = p50 + float(np.clip(strength, 0.0, 1.0)) * (target - p50)
    return out


def generate_beatmap(
    features: AudioFeatures,
    timing: TimingEstimate,
    difficulty: str | float,
    seed: int = 0,
    model=None,
    threshold: float = 0.5,
    title: str = "Unknown",
    artist: str = "Unknown",
    audio_filename: str = "audio.mp3",
    selection: str = "threshold",
    style: dict[str, float] | None = None,
    log=None,
    placement=None,
    repeats: list | None = None,
    variety: float = RHYTHM_VARIETY,
    critic=None,
    candidates: int = 4,
) -> Beatmap:
    """One difficulty. ``difficulty`` is a preset name ("hard") or a star rating (4.5);
    for a star rating the map is adjusted until osu!'s star calculation agrees (needs
    ``rosu-pp-py``). ``style`` asks for a style, e.g. {"jump": 0.8} (see style_values).
    ``placement`` is a PlacementNet; without it objects are placed by rules.
    ``repeats`` (structure.find_repeats) are mapped like their first occurrence, and
    loud repeated sections get kiai time. ``variety`` (0..1) trades closeness to the beat
    for a more varied rhythm with longer sliders (see rhythm.variety_ratio)."""
    name = None
    if isinstance(difficulty, str) and difficulty.lower() in DEFAULT_STARS:
        # A named difficulty aims for the stars typical of that name (and keeps its name).
        name, difficulty = get_preset(difficulty).name, DEFAULT_STARS[difficulty.lower()]
    stars = None if isinstance(difficulty, str) else float(difficulty)
    preset = get_preset(difficulty) if stars is None else preset_for_stars(stars)
    if name is not None:
        preset = replace(preset, name=name)
    if model is not None:
        from .model import threshold_for
        threshold = threshold_for(model, stars if stars is not None
                                  else stars_for_density(preset.density), threshold)
    grid = (timing.tempo_points if timing.tempo_points
            else [(timing.offset_ms, timing.beat_length)])
    grid = [(time, beat_length, 4) for time, beat_length in grid]

    sequence = placement is not None and type(placement).__name__ == "SequenceNet"
    notes_for: dict[int, tuple] = {}  # id(plan) -> (plan, the frame model's note probabilities)
    # Star rating asked of the models, relative to the target (see the sequence branch of
    # the star targeting below): "make it harder" instead of "space it further apart".
    steer = {"k": 1.0, "critic": True}

    def build(density_scale: float, force_density: bool = False):
        rng = np.random.default_rng(seed)
        asked = stars * steer["k"] if stars is not None else None
        base_density = preset_for_stars(asked).density if steer["k"] != 1.0 else preset.density
        wanted = replace(preset, density=base_density * density_scale)
        # More notes wanted: also accept less certain ones (and the other way round).
        note_threshold = float(np.clip(threshold * density_scale ** -0.8, 0.08, 0.85))
        outputs: dict[str, np.ndarray] = {}
        if model is not None:
            from .model import predict_all
            outputs = predict_all(model, features, grid, model_conditions(model, wanted, asked, style))
        # Far more notes wanted than the model hears with confidence (a very high star
        # rating for a slow song): take its most likely ticks until the density fits.
        mode = "density" if force_density and outputs else selection
        plan = plan_objects(features, timing, wanted, rng, outputs.get("note"), outputs.get("slider"),
                            note_threshold, mode, outputs.get("sustain"), outputs.get("spacing"),
                            stars, variety if mode == selection else 0.0)
        if repeats and not sequence:
            plan = copy_rhythm(plan, repeats, timing.beat_length / preset.divisor,
                               4 * timing.beat_length)
        assign_combos(plan, wanted, timing.beat_length)
        notes_for[id(plan)] = (plan, outputs.get("note"))
        return plan

    kiai = kiai_sections(features, repeats) if repeats else []
    wanted_stars = stars if stars is not None else stars_for_density(preset.density)

    def kiai_sv_factor(st: float) -> float:
        if st >= 4.5:
            return 1.15
        elif st >= 3.0:
            return 1.10
        return 1.05

    sv_boost = kiai_sv_factor(wanted_stars)

    def sv_at(t: float) -> float:
        return sv_boost if any(start - 1 <= t <= end + 1 for start, end in kiai) else 1.0

    placement_conditions = {"density": preset.density, "stars": wanted_stars,
                            **style_values(model, wanted_stars, style)}
    sampled: dict[int, tuple] = {}  # id(plan) -> (plan, choices): sampled once per rhythm

    def follow_sequence(plan, base: float = 1.0):
        """Place the frame model's rhythm with the sequence model, which also turns some
        triples into doubles (see SequencePlacer.follow). Sampled once per rhythm."""
        from .audio import sample_peak
        from .rhythm import heuristic_scores, make_tick_grid
        from .sequence_model import SequencePlacer
        ranked = critic is not None and steer["critic"]
        key = (id(plan), round(base, 3), steer["k"], ranked)
        if key in sampled:
            return sampled[key]
        quarter = make_tick_grid(timing, features.duration * 1000.0, 4)
        notes = notes_for.get(id(plan), (None, None))[1]
        if notes is not None:
            scores = sample_peak(notes, quarter.score_times, radius=1)
        else:
            scores = heuristic_scores(features, quarter)
            scores = scores / (scores.max() + 1e-6)
        conditions = dict(placement_conditions, stars=wanted_stars * steer["k"])
        placer = SequencePlacer(placement, preset, features, timing, conditions,
                                style_tags(style), scores, quarter, threshold,
                                np.random.default_rng(seed + 1000), sv_at=sv_at)
        followed, choices = placer.follow(copy.deepcopy(plan), scale=base,
                                          critic=critic if ranked else None,
                                          candidates=candidates)
        sampled[key] = (plan, followed, choices, placer)
        return sampled[key]

    def place(plan, spacing_scale: float, base: float = 1.0) -> Beatmap:
        rng = np.random.default_rng(seed + 1000)
        bm = Beatmap(
            title=title, artist=artist, version=preset.name, audio_filename=audio_filename,
            preview_time=preview_time_ms(features), hp=preset.hp, cs=preset.cs, od=preset.od,
            ar=preset.ar, slider_multiplier=preset.slider_multiplier, beat_divisor=preset.divisor,
            timing_points=[TimingPoint(time, beat_length) for time, beat_length, _ in grid],
        )
        for start, end in kiai:
            bm.timing_points += [TimingPoint(start, -100.0 / sv_boost, uninherited=False, effects=1),
                                 TimingPoint(end, -100.0, uninherited=False, effects=0)]
        bm.timing_points.sort(key=lambda tp: (tp.time, not tp.uninherited))
        placed_plan = copy.deepcopy(plan)
        if sequence:
            _, followed, choices, placer = follow_sequence(plan, base)
            bm.hit_objects = placer.render(copy.deepcopy(followed), choices, spacing_scale)
            if repeats:
                bm.hit_objects = copy_sections(bm.hit_objects, repeats, np.random.default_rng(seed),
                                               bm.end_time)
        elif placement is not None:
            from .placement_model import LearnedPlacer
            placer = LearnedPlacer(placement, preset, features.mel, placement_conditions, rng,
                                   offset_ms=timing.offset_ms, sv_at=sv_at)
            if id(plan) not in sampled:
                sampled[id(plan)] = (plan, placer.sample(copy.deepcopy(plan)))
            bm.hit_objects = placer.render(placed_plan, sampled[id(plan)][1], spacing_scale)
        else:
            bm.hit_objects = Placer(preset, rng, timing.beat_length, spacing_scale, sv_at=sv_at).place(placed_plan)
        if repeats and not sequence:
            bm.hit_objects = copy_patterns(bm.hit_objects, placed_plan, np.random.default_rng(seed))
        bm.breaks = find_breaks(bm)
        if bm.hit_objects:
            # Give players time to see the first object before it must be hit.
            bm.audio_lead_in = max(0, round(approach_preempt(bm.ar) + 500 - bm.hit_objects[0].time))
        return bm

    plan = build(1.0)
    # With a star target the search places without the critic first (see below).
    steer["critic"] = stars is None
    bm = place(plan, 1.0)
    reached = star_rating(bm.to_osu_string()) if stars is not None else None
    if reached is None:
        steer["critic"] = True
        return place(plan, 1.0)

    # Reach the star rating the way a mapper would: spacing follows the style (large for
    # jump maps, the typical spacing of these stars otherwise); if that is far too easy
    # for the song, bigger jumps come first and more notes only after that; finally the
    # spacing is fine-tuned.
    styled_jump = style_values(model, stars, style).get("jump")
    jump_levels = [styled_jump] if styled_jump else jump_percentiles(model, stars)
    wanted_jump = jump_levels[0]

    def fit_spacing(plan) -> float:
        scale = 1.0
        for _ in range(2):
            jump = map_style(place(plan, scale))["jump"]
            if not np.isfinite(jump) or jump <= 0:
                break
            scale = float(np.clip(scale * wanted_jump / jump, 0.4, 2.5))
        return scale

    best = (abs(reached - stars), reached, bm)

    def consider(candidate: Beatmap) -> float:
        nonlocal best
        actual = star_rating(candidate.to_osu_string())
        if abs(actual - stars) < best[0]:
            best = (abs(actual - stars), actual, candidate)
        return actual

    if sequence:
        # Harder maps are not just wider ones: ask both models for a harder (or easier)
        # map -- they know how mappers do that (rhythm, streams, angles, jumps) -- and
        # leave only a small correction to the spacing.
        best_k, best_plan = 1.0, plan
        # Within a narrow band only: asking for far more stars brings in patterns (kick
        # sliders, streams) that belong to much harder maps.
        lo, hi = (1.0, 1.25) if reached < stars else (0.8, 1.0)
        for _ in range(6):
            if best[0] < 0.1:
                break
            steer["k"] = (lo * hi) ** 0.5
            candidate_plan = build(1.0)
            before = best[0]
            actual = consider(place(candidate_plan, 1.0))
            if best[0] < before:
                best_k, best_plan = steer["k"], candidate_plan
            lo, hi = (steer["k"], hi) if actual < stars else (lo, steer["k"])
        steer["k"], plan = best_k, best_plan
        if best[0] >= 0.15:
            # Still off: change the amount of notes (far too calm a song for the rating:
            # take the frame model's most likely ticks until the density fits).
            short = best[1] < stars
            force = model is not None and short and best[0] > 0.8
            lo, hi = (1.0, 1.8) if short else (0.5, 1.0)
            for _ in range(5):
                density_scale = (lo * hi) ** 0.5
                candidate_plan = build(density_scale, force)
                before = best[0]
                actual = consider(place(candidate_plan, 1.0))
                if best[0] < before:
                    plan = candidate_plan
                if best[0] < 0.15:
                    break
                lo, hi = (density_scale, hi) if actual < stars else (lo, density_scale)
        if critic is not None:
            # The search above placed without the critic (4x faster); rank the chosen
            # rhythm's placement with it now and fine-tune that one.
            steer["critic"] = True
            searched, best = best, (float("inf"), stars, None)
            consider(place(plan, 1.0))
        lo, hi = 0.85, 1.18
        for _ in range(6):
            if best[0] < 0.05:
                break
            mid = (lo * hi) ** 0.5
            actual = consider(place(plan, mid))
            lo, hi = (mid, hi) if actual < stars else (lo, mid)
        if critic is not None and best[0] > searched[0] + 0.2:
            best = searched  # the critic's pick moved the stars too far: keep the search's
        if log is not None:
            log(f"  [{preset.name}] target {stars:.2f}*, reached {best[1]:.2f}* "
                f"(asked the models for {stars * best_k:.2f}*)")
        return best[2]

    scale = fit_spacing(plan)
    actual = consider(place(plan, scale))
    for level in jump_levels[1:]:
        if actual >= stars - 0.8:
            break
        wanted_jump = level
        scale = fit_spacing(plan)
        actual = consider(place(plan, scale))
    if abs(actual - stars) >= 0.15:
        # Bisect the amount of notes (in log space). Fewer notes: raise the model's
        # threshold. More notes than it hears with confidence: take its most likely
        # ticks until the density fits.
        # Small shortfalls are left to the spacing (fine-tuned below); forcing notes is
        # only for songs far too calm for the rating.
        force = model is not None and actual < stars - 0.8
        lo, hi = (0.8, 2.5) if force else (1.0, 1.8) if actual < stars else (0.3, 1.0)
        for _ in range(6):
            density_scale = (lo * hi) ** 0.5
            candidate_plan = build(density_scale, force)
            candidate_scale = fit_spacing(candidate_plan)
            actual = consider(place(candidate_plan, candidate_scale))
            if abs(actual - stars) < abs(best[1] - stars) + 1e-9 or abs(actual - stars) < 0.15:
                plan, scale = candidate_plan, candidate_scale
            if abs(actual - stars) < 0.15:
                break
            lo, hi = (density_scale, hi) if actual < stars else (lo, density_scale)
    lo, hi = scale * 0.6, scale * 2.2
    for _ in range(10):
        if best[0] < 0.05:
            break
        mid = (lo * hi) ** 0.5
        actual = consider(place(plan, mid))
        lo, hi = (mid, hi) if actual < stars else (lo, mid)
    if log is not None:
        log(f"  [{preset.name}] target {stars:.2f}*, reached {best[1]:.2f}*")
    return best[2]


def jump_percentiles(model, stars: float) -> list[float]:
    """Median, 75th and 95th percentile distance snap of ranked maps with this rating."""
    stats = getattr(model, "config", {}).get("cond_stats") or {}
    if stats:
        level = min(stats, key=lambda k: abs(float(k) + 0.5 - stars))
        if "jump" in stats[level]:
            p5, p25, p50, p75, p95 = stats[level]["jump"]
            return [float(p50), float(p75), float(p95)]
    median = typical_jump(model, stars)
    return [median, median * 1.3, median * 1.7]


def typical_jump(model, stars: float) -> float:
    """Median distance snap of ranked maps with this star rating."""
    stats = getattr(model, "config", {}).get("cond_stats") or {}
    if stats:
        level = min(stats, key=lambda k: abs(float(k) + 0.5 - stars))
        if "jump" in stats[level]:
            return float(stats[level]["jump"][2])
    return float(np.interp(stars, [2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75],
                           [1.0, 1.21, 1.26, 1.62, 1.75, 1.91, 2.0]))


def inference_device() -> str:
    """The GPU when PyTorch sees one (CUDA or ROCm), else the CPU. Star targeting runs
    the model several times per difficulty, so this matters."""
    import torch
    if torch.cuda.is_available():
        from .train import resolve_device
        try:
            device = resolve_device("cuda")
            torch.zeros(1, device=device)
            return device
        except Exception:
            pass
    return "cpu"


def bundled_model() -> Path | None:
    """The rhythm model shipped with the package, if it and PyTorch are available."""
    if BUNDLED_MODEL.exists() and importlib.util.find_spec("torch") is not None:
        return BUNDLED_MODEL
    return None


def bundled_sequence() -> Path | None:
    """The sequence model shipped with the package, if it and PyTorch are available."""
    if BUNDLED_SEQUENCE.exists() and importlib.util.find_spec("torch") is not None:
        return BUNDLED_SEQUENCE
    return None


def bundled_placement() -> Path | None:
    """The placement or sequence model shipped with the package, if it and PyTorch are available."""
    if importlib.util.find_spec("torch") is None:
        return None
    if BUNDLED_SEQUENCE.exists():
        return BUNDLED_SEQUENCE
    if BUNDLED_PLACEMENT.exists():
        return BUNDLED_PLACEMENT
    return None


def bundled_critic() -> Path | None:
    """The critic model shipped with the package, if it and PyTorch are available."""
    if BUNDLED_CRITIC.exists() and importlib.util.find_spec("torch") is not None:
        return BUNDLED_CRITIC
    return None


def guess_metadata(audio_path: Path) -> tuple[str, str]:
    """"Artist - Title.mp3" -> (artist, title)."""
    artist, sep, title = audio_path.stem.partition(" - ")
    return (artist.strip(), title.strip()) if sep else ("Unknown", audio_path.stem)


def generate(
    audio_path: str | Path,
    out_path: str | Path | None = None,
    difficulties: list[str | float] = ("normal", "hard", "insane"),
    model_path: str | Path | None = "auto",
    bpm: float | None = None,
    offset: float | None = None,
    title: str | None = None,
    artist: str | None = None,
    seed: int = 0,
    style: dict[str, float] | None = None,
    placement_path: str | Path | None = "auto",
    repeat_sections: bool = True,
    variety: float = RHYTHM_VARIETY,
    critic_path: str | Path | None = "auto",
    candidates: int = 4,
    log=print,
) -> Path:
    """Write an .osz with one beatmap per difficulty (a preset name or a star rating).
    ``model_path`` is a rhythm model checkpoint, "auto" for the bundled model when
    available, or None for heuristics. ``style`` e.g. {"jump": 0.7}; None is "Auto".
    ``placement_path`` is a placement model, "auto" for the bundled one, or None for
    rule-based placement. ``critic_path`` is a critic model checkpoint for Best-of-N
    ranking, "auto" for the bundled critic if present, or None. ``repeat_sections``
    maps repeated sections (a returning chorus) with the same rhythm and patterns
    and adds kiai time."""
    audio_path = Path(audio_path)
    guessed_artist, guessed_title = guess_metadata(audio_path)
    artist, title = artist or guessed_artist, title or guessed_title
    out_path = Path(out_path) if out_path else audio_path.with_suffix(".osz")
    if audio_path.suffix.lower() not in (".mp3", ".ogg"):
        log(f"warning: osu!stable only plays .mp3/.ogg audio, got {audio_path.suffix}")

    features, timing = analyze(audio_path, bpm, offset)
    timing_notes = []
    if timing.swing_confidence >= 0.25 and timing.swing_ratio > 0.515:
        timing_notes.append(f"swing {timing.swing_ratio:.0%}")
    if timing.tempo_points and len(timing.tempo_points) > 1:
        timing_notes.append(f"{len(timing.tempo_points)} local tempo points")
    suffix = f" ({', '.join(timing_notes)})" if timing_notes else ""
    log(f"{artist} - {title}: {features.duration:.1f}s, "
        f"BPM {timing.bpm:g}, offset {timing.offset_ms:g} ms{suffix}")

    if model_path == "auto":
        model_path = bundled_model()
    model, threshold = None, 0.5
    if model_path is not None:
        from .model import load_checkpoint
        model, threshold = load_checkpoint(model_path, device=inference_device())
        device = next(model.parameters()).device
        log(f"rhythm: model {Path(model_path).name} on {device.type} (threshold {threshold:.2f})")
    else:
        log("rhythm: onset heuristics")
    if placement_path == "auto":
        placement_path = bundled_placement()
    placement = None
    if placement_path is not None:
        from .sequence_model import is_sequence_checkpoint
        if is_sequence_checkpoint(placement_path):
            from .sequence_model import load_sequence
            placement = load_sequence(placement_path, device=inference_device())
            log(f"placement: sequence model {Path(placement_path).name}")
        else:
            from .placement_model import load_placement
            placement = load_placement(placement_path, device=inference_device())
            log(f"placement: model {Path(placement_path).name}")
    else:
        log("placement: rules")

    if critic_path == "auto":
        critic_path = bundled_critic()
    critic = None
    if critic_path is not None and placement is not None and type(placement).__name__ == "SequenceNet":
        from .critic import load_critic
        critic = load_critic(critic_path, device=inference_device())
        log(f"critic: model {Path(critic_path).name} (Best-of-{candidates})")
    elif critic_path is not None:
        log("critic: off (requires sequence model)")

    repeats = []
    if repeat_sections and features.chroma is not None and not (
            timing.tempo_points and len(timing.tempo_points) > 1):
        repeats = find_repeats(features, timing.offset_ms, timing.beat_length)
        if repeats:
            log("repeated sections: " + ", ".join(
                f"{r.target_ms / 1000:.0f}s like {r.source_ms / 1000:.0f}s" for r in repeats))
    beatmaps = []
    for i, name in enumerate(difficulties):
        bm = generate_beatmap(
            features, timing, name, seed=seed + i, model=model, threshold=threshold,
            title=title, artist=artist, audio_filename="audio" + audio_path.suffix.lower(),
            style=style, log=log, placement=placement, repeats=repeats, variety=variety,
            critic=critic, candidates=candidates,
        )
        kinds = [o.kind for o in bm.hit_objects]
        log(f"  [{bm.version}] {len(kinds)} objects: {kinds.count('circle')} circles, "
            f"{kinds.count('slider')} sliders, {kinds.count('spinner')} spinners")
        beatmaps.append(bm)
    write_osz(out_path, audio_path, beatmaps)
    log(f"wrote {out_path}")
    return out_path
