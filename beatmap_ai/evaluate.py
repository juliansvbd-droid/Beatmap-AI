"""Measure how closely generated maps match human-made ones on held-out songs.

* Timing: is the detected BPM right, and how far is the detected offset from the
  mapper's (modulo one beat)?
* Rhythm: with the mapper's own timing, how well do generated note times match the
  mapper's (F1, ±30 ms)? Heuristic and model-based rhythm are scored the same way.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import AudioFeatures
from .dataset import (MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, note_density,
                      song_key)
from .osu import parse_osu
from .style import star_rating
from .difficulty import PRESETS, DifficultyPreset
from .osu import Beatmap
from .rhythm import plan_objects
from .timing import TimingEstimate, estimate_timing


def match_f1(pred: np.ndarray, truth: np.ndarray, tolerance: float = 30.0) -> float:
    """F1 of predicted vs true times, matching each true time at most once."""
    pred, truth = np.sort(pred), np.sort(truth)
    if len(pred) == 0 or len(truth) == 0:
        return 0.0
    used = np.zeros(len(truth), dtype=bool)
    hits = 0
    for p in pred:
        j = np.searchsorted(truth, p - tolerance)
        while j < len(truth) and truth[j] <= p + tolerance:
            if not used[j]:
                used[j] = True
                hits += 1
                break
            j += 1
    return 0.0 if hits == 0 else 2 * hits / (len(pred) + len(truth))


def constant_timing(bm: Beatmap) -> TimingEstimate | None:
    """The map's timing if it uses a single BPM on a single grid, else None."""
    red = [tp for tp in bm.timing_points if tp.uninherited and tp.beat_length > 0]
    if not red:
        return None
    first = red[0]
    for tp in red[1:]:
        beats = (tp.time - first.time) / first.beat_length
        if abs(tp.beat_length - first.beat_length) > 0.01 or abs(beats - round(beats)) * first.beat_length > 2:
            return None
    return TimingEstimate(bpm=first.bpm, offset_ms=first.time)


def closest_preset(density: float) -> DifficultyPreset:
    preset = min(PRESETS.values(), key=lambda p: abs(np.log(p.density / density)))
    return dataclasses.replace(preset, density=density)


def rhythm_f1(features: AudioFeatures, bm: Beatmap, timing: TimingEstimate,
              model=None, threshold: float = 0.5,
              selections: tuple[str, ...] = ("threshold",), stars: float | None = None) -> list[float]:
    """F1 of the generated rhythm against ``bm``, one value per model ``selections``
    mode (a single value without a model). The model is told the map's density and,
    if it knows them, its ``stars`` -- what the generator knows when asked for them."""
    density = note_density(bm)
    preset = closest_preset(density)
    outputs: dict[str, np.ndarray] = {}
    if model is not None:
        from .model import predict_all
        grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points if tp.uninherited]
        conditions = {"density": density}
        if stars is not None:
            conditions["stars"] = stars
        outputs = predict_all(model, features, grid, conditions)
    else:
        selections = ("threshold",)
    truth = np.array([o.time for o in bm.hit_objects])
    scores = []
    for selection in selections:
        plan = plan_objects(features, timing, preset, np.random.default_rng(0),
                            outputs.get("note"), outputs.get("slider"), threshold, selection,
                            outputs.get("sustain"), outputs.get("spacing"))
        scores.append(match_f1(np.array([p.time for p in plan]), truth))
    return scores


STAR_BUCKETS = ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 99))


def evaluate(data_dir: str | Path | list[str | Path], model_path: str | Path | None = None,
             val_fraction: float = 0.1, max_songs: int = 40, log=print) -> dict:
    """Timing and rhythm accuracy on held-out songs, overall and per star rating."""
    model, threshold = None, 0.5
    if model_path is not None:
        from .model import load_checkpoint
        model, threshold = load_checkpoint(model_path)

    songs: dict[str, list] = defaultdict(list)
    sources, seen = {}, set()
    for folder in [data_dir] if isinstance(data_dir, (str, Path)) else data_dir:
        for _, text, source, _ in iter_beatmap_texts(folder):
            if source.path.suffix == ".npy":  # Audio deleted, only the spectrogram is left.
                continue
            bm = parse_osu(text)
            key = song_key(bm)
            if (bm.mode == 0 and len(bm.hit_objects) >= MIN_OBJECTS
                    and is_validation(key, val_fraction) and map_key(bm) not in seen):
                seen.add(map_key(bm))
                # One audio file per song; its other versions may be cut differently.
                if sources.setdefault(key, source) == source:
                    songs[key].append((bm, star_rating(text)))
    # Spread the chosen songs over the whole (hash-ordered) list.
    ordered = sorted(songs)
    selected = [ordered[int(i * len(ordered) / max_songs)] for i in range(min(max_songs, len(ordered)))]

    bpm_ok, octave_ok, offset_errors = [], [], []
    f1_heuristic, f1_model, map_stars = [], [], []
    for sid in selected:
        try:
            features = sources[sid].load_features()
        except Exception as exc:
            log(f"skipping {sources[sid].key}: {exc}")
            continue
        maps = songs[sid]
        truth = next((t for t in (constant_timing(bm) for bm, _ in maps) if t is not None), None)
        if truth is not None:
            detected = estimate_timing(features)
            ratio = detected.bpm / truth.bpm
            bpm_ok.append(abs(detected.bpm - truth.bpm) < 0.5)
            octave_ok.append(min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.005)
            if bpm_ok[-1]:
                beat = truth.beat_length
                offset_errors.append((detected.offset_ms - truth.offset_ms + beat / 2) % beat - beat / 2)
        n = 0
        for bm, stars in maps:
            timing = constant_timing(bm)
            if timing is None:
                continue
            n += 1
            map_stars.append(stars if stars is not None else np.nan)
            f1_heuristic += rhythm_f1(features, bm, timing)
            if model is not None:
                f1_model += rhythm_f1(features, bm, timing, model, threshold, stars=stars)
        if n:
            log(f"  {maps[0][0].artist} - {maps[0][0].title}: "
                f"heuristic F1 {np.mean(f1_heuristic[-n:]):.3f}"
                + (f", model F1 {np.mean(f1_model[-n:]):.3f}" if f1_model else ""))

    offsets = np.array(offset_errors)
    results = {
        "songs": len(selected),
        "difficulties": len(f1_heuristic),
        "bpm_accuracy": float(np.mean(bpm_ok)) if bpm_ok else None,
        "bpm_accuracy_any_octave": float(np.mean(octave_ok)) if octave_ok else None,
        "offset_median_error_ms": float(np.median(offsets)) if len(offsets) else None,
        "offset_within_10ms": float(np.mean(np.abs(offsets) <= 10)) if len(offsets) else None,
        "rhythm_f1_heuristic": float(np.mean(f1_heuristic)) if f1_heuristic else None,
        "rhythm_f1_model": float(np.mean(f1_model)) if f1_model else None,
    }
    for k, v in results.items():
        log(f"{k:>26}: {v:.3f}" if isinstance(v, float) else f"{k:>26}: {v}")
    # The same F1 for each star range, so weak difficulties do not hide in the average.
    stars = np.array(map_stars, dtype=float)
    scores = np.array(f1_model if f1_model else f1_heuristic)
    per_stars = {}
    for lo, hi in STAR_BUCKETS:
        inside = (stars >= lo) & (stars < hi)
        if inside.any():
            label = f"{lo}-{hi}*" if hi < 99 else f"{lo}+*"
            per_stars[label] = (float(scores[inside].mean()), int(inside.sum()))
            log(f"{'rhythm F1 ' + label:>26}: {per_stars[label][0]:.3f}  ({per_stars[label][1]} maps)")
    results["rhythm_f1_by_stars"] = per_stars
    return results
