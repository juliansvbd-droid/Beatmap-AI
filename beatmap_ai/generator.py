"""End-to-end generation: audio file in, .osz beatmap set out."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .audio import AudioFeatures, compute_features, load_audio, preview_time_ms
from .difficulty import DifficultyPreset, get_preset
from .osu import Beatmap, TimingPoint, write_osz
from .placement import Placer
from .rhythm import PlannedObject, plan_objects
from .timing import TimingEstimate, estimate_timing, fit_offset

MAX_COMBO = 16
BUNDLED_MODEL = Path(__file__).parent / "models" / "rhythm.pt"
BREAK_MIN_GAP = 5000.0


def analyze(audio_path: str | Path, bpm: float | None = None,
            offset: float | None = None) -> tuple[AudioFeatures, TimingEstimate]:
    features = compute_features(load_audio(audio_path))
    if bpm is None:
        timing = estimate_timing(features)
        if offset is not None:
            timing.offset_ms = offset
    else:
        timing = TimingEstimate(bpm, offset if offset is not None else fit_offset(features, bpm))
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


def generate_beatmap(
    features: AudioFeatures,
    timing: TimingEstimate,
    difficulty: str,
    seed: int = 0,
    model=None,
    threshold: float = 0.5,
    title: str = "Unknown",
    artist: str = "Unknown",
    audio_filename: str = "audio.mp3",
) -> Beatmap:
    preset = get_preset(difficulty)
    rng = np.random.default_rng(seed)
    note_probs = slider_probs = None
    if model is not None:
        from .model import predict
        grid = [(timing.offset_ms, timing.beat_length, 4)]
        note_probs, slider_probs = predict(model, features, grid, preset.density)

    plan = plan_objects(features, timing, preset, rng, note_probs, slider_probs, threshold)
    assign_combos(plan, preset, timing.beat_length)
    bm = Beatmap(
        title=title, artist=artist, version=preset.name, audio_filename=audio_filename,
        preview_time=preview_time_ms(features), hp=preset.hp, cs=preset.cs, od=preset.od,
        ar=preset.ar, slider_multiplier=preset.slider_multiplier, beat_divisor=preset.divisor,
        timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)],
    )
    bm.hit_objects = Placer(preset, rng, timing.beat_length).place(plan)
    bm.breaks = find_breaks(bm)
    if bm.hit_objects:
        # Give players time to see the first object before it must be hit.
        bm.audio_lead_in = max(0, round(approach_preempt(bm.ar) + 500 - bm.hit_objects[0].time))
    return bm


def bundled_model() -> Path | None:
    """The rhythm model shipped with the package, if it and PyTorch are available."""
    if BUNDLED_MODEL.exists() and importlib.util.find_spec("torch") is not None:
        return BUNDLED_MODEL
    return None


def guess_metadata(audio_path: Path) -> tuple[str, str]:
    """"Artist - Title.mp3" -> (artist, title)."""
    artist, sep, title = audio_path.stem.partition(" - ")
    return (artist.strip(), title.strip()) if sep else ("Unknown", audio_path.stem)


def generate(
    audio_path: str | Path,
    out_path: str | Path | None = None,
    difficulties: list[str] = ("normal", "hard", "insane"),
    model_path: str | Path | None = "auto",
    bpm: float | None = None,
    offset: float | None = None,
    title: str | None = None,
    artist: str | None = None,
    seed: int = 0,
    log=print,
) -> Path:
    """Write an .osz with one beatmap per difficulty. ``model_path`` is a rhythm model
    checkpoint, "auto" for the bundled model when available, or None for heuristics."""
    audio_path = Path(audio_path)
    guessed_artist, guessed_title = guess_metadata(audio_path)
    artist, title = artist or guessed_artist, title or guessed_title
    out_path = Path(out_path) if out_path else audio_path.with_suffix(".osz")
    if audio_path.suffix.lower() not in (".mp3", ".ogg"):
        log(f"warning: osu!stable only plays .mp3/.ogg audio, got {audio_path.suffix}")

    features, timing = analyze(audio_path, bpm, offset)
    log(f"{artist} - {title}: {features.duration:.1f}s, "
        f"BPM {timing.bpm:g}, offset {timing.offset_ms:g} ms")

    if model_path == "auto":
        model_path = bundled_model()
    model, threshold = None, 0.5
    if model_path is not None:
        from .model import load_checkpoint
        model, threshold = load_checkpoint(model_path)
        log(f"rhythm: model {Path(model_path).name} (threshold {threshold:.2f})")
    else:
        log("rhythm: onset heuristics")

    beatmaps = []
    for i, name in enumerate(difficulties):
        bm = generate_beatmap(
            features, timing, name, seed=seed + i, model=model, threshold=threshold,
            title=title, artist=artist, audio_filename="audio" + audio_path.suffix.lower(),
        )
        kinds = [o.kind for o in bm.hit_objects]
        log(f"  [{bm.version}] {len(kinds)} objects: {kinds.count('circle')} circles, "
            f"{kinds.count('slider')} sliders, {kinds.count('spinner')} spinners")
        beatmaps.append(bm)
    write_osz(out_path, audio_path, beatmaps)
    log(f"wrote {out_path}")
    return out_path
