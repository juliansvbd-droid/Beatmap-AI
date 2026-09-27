"""Torch-free song and section examples for the pre-mapping planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import FPS
from .dataset import is_validation
from .placement_data import BAR, BEAT, T
from .sequence_data import TAGS, build_sequence_maps, section_controls

PLANNER_FEATURES = 164  # mel mean/std, loudness, change, repetition similarity, time position
STAR_CLASSES = 14  # half-star bins from 0.5 to 7.0
SECTION_TYPES = ("intro", "verse", "build", "main", "bridge", "outro", "instrumental")


@dataclass
class PlannerExample:
    song: str
    features: np.ndarray
    controls: np.ndarray
    section_type: np.ndarray
    boundary: np.ndarray
    stars: np.ndarray
    styles: np.ndarray
    styles_known: bool
    max_stars: float
    # Stars of the difficulty whose section values these are (the planner's condition). The
    # song-level targets (star range, maximum, styles) are the same for all its difficulties.
    condition: float = 0.0


def audio_section_features(mel: np.ndarray, beat_length: float, bars: int = 4,
                           offset_ms: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize a whole song by musical chunks of ``bars`` bars, with a repetition
    similarity cue. Chunks start on bar lines (``offset_ms`` is a downbeat); the lead-in
    before the first full chunk is a chunk of its own. Also returns the chunk starts (ms)."""
    n_frames = mel.shape[1]
    chunk_frames = max(int(round(beat_length * 4 * bars * FPS / 1000.0)), 1)
    bar_ms = max(beat_length * 4, 1.0)
    first = int(round((offset_ms % bar_ms) * FPS / 1000.0))
    starts = list(range(first, n_frames, chunk_frames))
    if first > 0 or not starts:
        starts = [0] + starts
    vectors, intensity = [], []
    for start in starts:
        end = min(start + chunk_frames, n_frames)
        chunk = np.asarray(mel[:, start:end], dtype=np.float32)
        if not chunk.size:
            continue
        mean = chunk.mean(axis=1)
        std = chunk.std(axis=1)
        energy = float(np.mean(np.maximum(chunk, 0.0)))
        intensity.append(energy)
        vectors.append(np.concatenate([mean, std, [energy, 0.0, 0.0, start / max(n_frames, 1)] ]))
    features = np.asarray(vectors, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    starts_ms = np.asarray(starts[:len(features)], dtype=np.float32) * 1000.0 / FPS
    if len(features) == 0:
        return (np.zeros((1, PLANNER_FEATURES), dtype=np.float32), np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32))
    energy = features[:, 160]
    features[:, 161] = np.r_[0.0, np.diff(energy)]
    normed = features[:, :80] / (np.linalg.norm(features[:, :80], axis=1, keepdims=True) + 1e-6)
    for i in range(len(features)):
        if i >= 2:
            features[i, 162] = float(np.max(normed[i] @ normed[:i - 1].T))
    if len(intensity) > 1:
        scale = float(np.std(intensity)) + 1e-6
        intensity = (intensity - float(np.mean(intensity))) / scale
    return features, intensity, starts_ms


def _style_targets(maps) -> tuple[np.ndarray, bool]:
    tagged = [m for m in maps if m.tags is not None and m.tags[-1] > 0]
    if not tagged:
        return np.zeros(len(TAGS), dtype=np.float32), False
    chosen = max(tagged, key=lambda m: float(m.style.get("stars", 0.0)))
    tags = np.asarray(chosen.tags[:-1], dtype=np.float32)
    return np.pad(tags, (0, max(0, len(TAGS) - len(tags))))[:len(TAGS)], True


def _star_targets(maps) -> tuple[np.ndarray, float]:
    values = [float(m.style.get("stars", 0.0)) for m in maps if np.isfinite(m.style.get("stars", np.nan))]
    hist = np.zeros(STAR_CLASSES, dtype=np.float32)
    for value in values:
        hist[int(np.clip(round(value * 2) - 1, 0, STAR_CLASSES - 1))] += 1.0
    if hist.sum():
        hist /= hist.sum()
    return hist, max(values, default=0.0)


def _section_labels(maps, starts_ms: np.ndarray, ends_ms: np.ndarray, intensity: np.ndarray,
                    kiai_maps=None):
    """Section values of ``maps`` (one difficulty, or the mean over several); the kiai
    column comes from all ``kiai_maps`` of the song (kiai belongs to the song part)."""
    controls_per_map = []
    for m in maps:
        stars = float(m.style.get("stars", 4.5))
        values = np.zeros((len(starts_ms), 8), dtype=np.float32)
        for i, (start, end) in enumerate(zip(starts_ms, ends_ms)):
            ids = np.flatnonzero((m.objects[:, T] >= start) & (m.objects[:, T] < end))
            if len(ids):
                local = section_controls(m.objects, int(ids[0]), len(ids), stars, m.cs)
                if len(local):
                    values[i] = local.mean(axis=0)
        controls_per_map.append(values)
    controls = np.mean(controls_per_map, axis=0) if controls_per_map else np.zeros((len(starts_ms), 8), np.float32)
    for m in (kiai_maps or []):
        if any(m is other for other in maps):
            continue
        kiai = np.zeros_like(controls)
        for i, (start, end) in enumerate(zip(starts_ms, ends_ms)):
            ids = np.flatnonzero((m.objects[:, T] >= start) & (m.objects[:, T] < end))
            if len(ids):
                local = section_controls(m.objects, int(ids[0]), len(ids),
                                         float(m.style.get("stars", 4.5)), m.cs)
                if len(local):
                    kiai[i, 7] = local[:, 7].mean()
        controls_per_map.append(kiai)
    controls[:, 7] = np.max([v[:, 7] for v in controls_per_map], axis=0) if controls_per_map else 0.0

    # Coarse human-readable section labels from kiai, intensity, and where shifts occur.
    kind = np.ones(len(starts_ms), dtype=np.int64)  # verse by default
    if len(kind):
        kind[0] = 0
        kind[-1] = 5
    if len(intensity) > 1:
        slope = np.r_[np.diff(intensity), 0.0]
        rising = slope > max(float(np.std(slope)) * 0.5, 0.1)
        kind[rising] = 2
        kind[controls[:, 7] >= 0.5] = 3
        middle = (intensity < np.median(intensity)) & (np.arange(len(kind)) > 0) & (np.arange(len(kind)) < len(kind) - 1)
        kind[middle & (controls[:, 7] < 0.5)] = 4
    boundary = np.zeros(len(starts_ms), dtype=np.float32)
    if len(boundary) > 1:
        delta = np.abs(np.diff(controls, axis=0)).mean(axis=1)
        level_delta = np.abs(np.diff(intensity))
        boundary[1:] = ((delta > 0.15) | (level_delta > max(float(np.std(level_delta)), 0.1))).astype(np.float32)
    return controls, kind, boundary


def build_planner_examples(data_dirs, tag_files=(), tagger_path=None, log=print) -> list[PlannerExample]:
    """Group difficulties by song; one full-song example per difficulty, conditioned on
    that difficulty's stars (as it is asked when generating), with the song-level targets
    shared by all of the song's difficulties."""
    maps = build_sequence_maps(data_dirs, tag_files, log=log, tagger_path=tagger_path)
    groups: dict[str, list] = {}
    for m in maps:
        if len(m.objects) >= 20:
            groups.setdefault(m.song, []).append(m)
    examples = []
    for n, (song, song_maps) in enumerate(groups.items(), 1):
        primary = max(song_maps, key=lambda m: float(m.style.get("stars", 0.0)))
        try:
            mel = np.load(primary.mel_path, mmap_mode="r")
        except (OSError, ValueError):
            continue
        beat = float(np.median(primary.objects[:, BEAT]))
        first = primary.objects[0]
        downbeat = float(first[T] - first[BAR] * 4 * first[BEAT])  # start of its bar
        features, intensity, starts = audio_section_features(mel, beat, offset_ms=downbeat)
        ends = np.r_[starts[1:], np.inf].astype(np.float32)
        stars, maximum = _star_targets(song_maps)
        style_values, styles_known = _style_targets(song_maps)
        for m in song_maps:
            controls, kinds, boundaries = _section_labels([m], starts, ends, intensity, song_maps)
            examples.append(PlannerExample(song, features, controls, kinds, boundaries, stars,
                                           style_values, styles_known, maximum,
                                           float(m.style.get("stars", maximum))))
        if n % 1000 == 0:
            log(f"planner prepared {n}/{len(groups)} songs")
    log(f"planner examples: {len(examples)} difficulties of {len(groups)} songs; "
        f"{sum(len(e.features) for e in examples):,} sections")
    return examples


def split_examples(examples: list[PlannerExample]):
    train = [ex for ex in examples if not is_validation(ex.song)]
    validation = [ex for ex in examples if is_validation(ex.song)]
    return train, validation
