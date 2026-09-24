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
from .dataset import MIN_OBJECTS, is_validation, iter_beatmaps, note_density, song_id
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
              model=None, threshold: float = 0.5) -> float:
    density = note_density(bm)
    preset = closest_preset(density)
    note_probs = slider_probs = None
    if model is not None:
        from .model import predict
        grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points if tp.uninherited]
        note_probs, slider_probs = predict(model, features, grid, density)
    plan = plan_objects(features, timing, preset, np.random.default_rng(0),
                        note_probs, slider_probs, threshold)
    pred = np.array([p.time for p in plan])
    truth = np.array([o.time for o in bm.hit_objects])
    return match_f1(pred, truth)


def evaluate(data_dir: str | Path, model_path: str | Path | None = None,
             val_fraction: float = 0.1, max_songs: int = 40, log=print) -> dict:
    model, threshold = None, 0.5
    if model_path is not None:
        from .model import load_checkpoint
        model, threshold = load_checkpoint(model_path)

    songs: dict[str, list] = defaultdict(list)
    sources = {}
    for _, bm, source in iter_beatmaps(data_dir):
        sid = song_id(source.key)
        if bm.mode == 0 and len(bm.hit_objects) >= MIN_OBJECTS and is_validation(sid, val_fraction):
            songs[sid].append(bm)
            sources[sid] = source
    selected = sorted(songs)[:max_songs]

    bpm_ok, octave_ok, offset_errors = [], [], []
    f1_heuristic, f1_model = [], []
    for sid in selected:
        try:
            features = sources[sid].load_features()
        except Exception as exc:
            log(f"skipping {sources[sid].key}: {exc}")
            continue
        truth = next((t for t in map(constant_timing, songs[sid]) if t is not None), None)
        if truth is not None:
            detected = estimate_timing(features)
            ratio = detected.bpm / truth.bpm
            bpm_ok.append(abs(detected.bpm - truth.bpm) < 0.5)
            octave_ok.append(min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.005)
            if bpm_ok[-1]:
                beat = truth.beat_length
                offset_errors.append((detected.offset_ms - truth.offset_ms + beat / 2) % beat - beat / 2)
        for bm in songs[sid]:
            timing = constant_timing(bm)
            if timing is None:
                continue
            f1_heuristic.append(rhythm_f1(features, bm, timing))
            if model is not None:
                f1_model.append(rhythm_f1(features, bm, timing, model, threshold))
        log(f"  {songs[sid][0].artist} - {songs[sid][0].title}: "
            f"heuristic F1 {np.mean(f1_heuristic[-len(songs[sid]):]) if f1_heuristic else 0:.3f}"
            + (f", model F1 {np.mean(f1_model[-len(songs[sid]):]):.3f}" if f1_model else ""))

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
    return results
