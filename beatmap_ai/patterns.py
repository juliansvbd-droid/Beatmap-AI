"""Torch-free beatmap pattern measurements for human and generated maps."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .osu import Beatmap, HitObject
from .placement_data import END, EX, EY, T, map_objects

STAR_BUCKETS = ("<3", "3-4.5", "4.5-6", "6+")
SUBDIVISIONS = {"1_4": 0.25, "1_6": 1.0 / 6.0, "1_8": 0.125}
RHYTHM_GROUPS = {"double": (2, 2), "triple": (3, 3), "burst_4_8": (4, 8), "stream_9p": (9, None)}
PATTERN_TYPES = ("zigzag", "triangle", "quadrilateral", "pentagon_star", "line", "arc_flow", "mixed")
DIFFICULTY_FEATURES = (
    "jump_radii_per_second",
    "sharp_turn_speed",
    "stream_run_length",
    "cross_screen_fraction",
)
PLAYFIELD_DIAGONAL = math.hypot(512.0, 384.0)


def star_bucket(stars: float | None) -> str | None:
    """Map an osu!standard star rating to the four reporting ranges."""
    if stars is None or not math.isfinite(float(stars)):
        return None
    if stars < 3.0:
        return "<3"
    if stars < 4.5:
        return "3-4.5"
    if stars < 6.0:
        return "4.5-6"
    return "6+"


def _notes_and_rows(bm: Beatmap):
    rows = map_objects(bm)
    return [(obj, rows[i]) for i, obj in enumerate(bm.hit_objects) if obj.kind != "spinner"]


def _beat_position(bm: Beatmap, time_ms: float) -> tuple[int, float, float]:
    red = max((tp for tp in bm.timing_points if tp.uninherited and tp.time <= time_ms + 1),
              key=lambda tp: tp.time,
              default=next(tp for tp in bm.timing_points if tp.uninherited))
    beat_length = max(float(red.beat_length), 1.0)
    beats = (time_ms - red.time) / beat_length
    return int(math.floor(beats / max(red.meter, 1))), beats % max(red.meter, 1), beat_length


def _rhythm_metrics(bm: Beatmap, notes: list[tuple[HitObject, np.ndarray]]) -> dict[str, float]:
    totals = Counter()
    n = len(notes)
    if n < 2:
        return {f"{group}_{subdivision}": 0.0
                for group in RHYTHM_GROUPS for subdivision in SUBDIVISIONS}
    gap_classes: list[str | None] = []
    for (previous, _), (current, _) in zip(notes, notes[1:]):
        beat_length = bm.timing_at(current.time)[0]
        gap = (current.time - previous.time) / max(beat_length, 1.0)
        match = next((name for name, value in SUBDIVISIONS.items()
                      if abs(gap - value) <= max(0.012, value * 0.10)), None)
        gap_classes.append(match)
    i = 0
    while i < len(gap_classes):
        subdivision = gap_classes[i]
        if subdivision is None:
            i += 1
            continue
        end = i + 1
        while end < len(gap_classes) and gap_classes[end] == subdivision:
            end += 1
        note_count = end - i + 1
        group = next((name for name, (minimum, maximum) in RHYTHM_GROUPS.items()
                      if note_count >= minimum and (maximum is None or note_count <= maximum)), None)
        if group:
            totals[(group, subdivision)] += 1
        i = end
    return {f"{group}_{subdivision}": totals[(group, subdivision)] * 100.0 / n
            for group in RHYTHM_GROUPS for subdivision in SUBDIVISIONS}


def _turn_angles(points: np.ndarray) -> np.ndarray:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    angles = []
    for first, second, a, b in zip(vectors, vectors[1:], lengths, lengths[1:]):
        if min(a, b) < 1e-6:
            angles.append(float("nan"))
        else:
            cross = first[0] * second[1] - first[1] * second[0]
            dot = float(np.dot(first, second))
            angles.append(math.degrees(math.atan2(cross, dot)))
    return np.asarray(angles, dtype=float)


def _classify_pattern(points: np.ndarray) -> tuple[str, float] | None:
    edges = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if len(edges) not in (3, 4, 5) or np.any(edges < 16.0):
        return None
    mean_edge = float(np.mean(edges))
    if mean_edge <= 0 or float(np.max(edges) / np.min(edges)) > 1.35:
        return None
    closed = float(np.linalg.norm(points[-1] - points[0])) <= max(18.0, mean_edge * 0.30)
    if closed:
        kind = {3: "triangle", 4: "quadrilateral", 5: "pentagon_star"}[len(edges)]
        return kind, mean_edge
    turns = _turn_angles(points)
    finite = turns[np.isfinite(turns)]
    if len(finite) == 0:
        return "mixed", mean_edge
    absolute = np.abs(finite)
    if float(np.mean(absolute >= 145.0)) >= 0.75:
        return "zigzag", mean_edge
    if float(np.mean(absolute <= 22.0)) >= 0.75:
        return "line", mean_edge
    signs = np.sign(finite[np.abs(finite) >= 8.0])
    if len(signs) >= 2 and np.all(signs == signs[0]) and float(np.mean((absolute >= 15) & (absolute <= 125))) >= 0.75:
        return "arc_flow", mean_edge
    return "mixed", mean_edge


def _shape_fingerprint(points: np.ndarray) -> tuple[int, ...]:
    """A translation, rotation, reflection and scale invariant ordered shape key."""
    edges = np.linalg.norm(np.diff(points, axis=0), axis=1)
    scale = max(float(np.mean(edges)), 1e-6)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2) / scale
    indices = np.triu_indices(len(points), 1)
    # Pairwise distances discard translation, rotation and reflection; coarse bins
    # allow small aim-coordinate variation between two renditions of one motif.
    quantized = np.rint(distances[indices] / 0.15).astype(int)
    return (len(points) - 1, *quantized.tolist())


def _bar_signatures(bm: Beatmap, notes: list[tuple[HitObject, np.ndarray]]) -> dict[int, tuple]:
    bars: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for obj, _ in notes:
        bar, phase, beat_length = _beat_position(bm, obj.time)
        _, meter, _ = _timing_at(bm, obj.time)
        phase_in_bar = ((obj.time - meter[0]) / beat_length) % max(meter[1], 1)
        bars[bar].append((int(round(phase_in_bar * 4)), 1 if obj.kind == "slider" else 0))
    return {bar: tuple(sorted(items)) for bar, items in bars.items()}


def _timing_at(bm: Beatmap, time_ms: float) -> tuple[float, tuple[float, int]]:
    red = max((tp for tp in bm.timing_points if tp.uninherited and tp.time <= time_ms + 1),
              key=lambda tp: tp.time,
              default=next(tp for tp in bm.timing_points if tp.uninherited))
    return float(red.beat_length), (float(red.time), int(red.meter))


def _jump_patterns(bm: Beatmap, notes: list[tuple[HitObject, np.ndarray]], radius: float):
    points = np.asarray([(obj.x, obj.y) for obj, _ in notes], dtype=float)
    found: list[dict[str, Any]] = []
    if len(points) >= 4:
        for start in range(len(points) - 3):
            for edge_count in (3, 4, 5):
                end = start + edge_count + 1
                if end > len(points):
                    continue
                classified = _classify_pattern(points[start:end])
                if classified is None:
                    continue
                kind, size = classified
                bar, phase, _ = _beat_position(bm, notes[start][0].time)
                found.append({
                    "type": kind,
                    "start_index": start,
                    "edge_count": edge_count,
                    "time_ms": float(notes[start][0].time),
                    "size_radii": size / max(radius, 1.0),
                    "fingerprint": _shape_fingerprint(points[start:end]),
                    "bar": bar,
                    "bar_phase": phase,
                })
    bars = _bar_signatures(bm, notes)
    return found, bars


def _movement_features(bm: Beatmap, notes: list[tuple[HitObject, np.ndarray]], radius: float):
    n = len(notes)
    values = {name: np.zeros(n, dtype=np.float32) for name in DIFFICULTY_FEATURES}
    vectors, speeds, distances = [], [], []
    for i in range(1, n):
        previous, previous_row = notes[i - 1]
        current, _ = notes[i]
        dx, dy = float(current.x - previous_row[EX]), float(current.y - previous_row[EY])
        distance = math.hypot(dx, dy)
        gap_ms = max(float(current.time - previous_row[END]), 1.0)
        beat_length = max(float(bm.timing_at(current.time)[0]), 1.0)
        values["jump_radii_per_second"][i] = distance / max(radius, 1.0) / (gap_ms / 1000.0)
        values["cross_screen_fraction"][i] = distance / PLAYFIELD_DIAGONAL
        vectors.append((dx, dy))
        speeds.append((i, beat_length))
        distances.append(distance)
    for j in range(1, len(vectors)):
        first, second = vectors[j - 1], vectors[j]
        a, b = math.hypot(*first), math.hypot(*second)
        if min(a, b) < 12.0:
            continue
        turn = abs(math.degrees(math.atan2(first[0] * second[1] - first[1] * second[0],
                                           first[0] * second[0] + first[1] * second[1])))
        object_index, beat_length = speeds[j]
        bpm = 60000.0 / beat_length
        values["sharp_turn_speed"][object_index] = turn / 180.0 * bpm / 180.0

    gaps = []
    for (previous, _), (current, _) in zip(notes, notes[1:]):
        beat_length = max(float(bm.timing_at(current.time)[0]), 1.0)
        gaps.append((current.time - previous.time) / beat_length)
    run_start = 0
    while run_start < len(gaps):
        if gaps[run_start] > 0.26:
            run_start += 1
            continue
        run_end = run_start + 1
        while run_end < len(gaps) and gaps[run_end] <= 0.26:
            run_end += 1
        run_length = run_end - run_start + 1
        if run_length >= 4:
            values["stream_run_length"][run_start:run_end + 1] = run_length
        run_start = run_end
    return values


def analyze_map(bm: Beatmap, stars: float | None = None, name: str = "") -> dict[str, Any]:
    """Return per-map rhythm, movement, repetition, slider and difficulty measurements."""
    if not any(tp.uninherited for tp in bm.timing_points):
        raise ValueError("beatmap has no uninherited timing points")
    notes = _notes_and_rows(bm)
    n = len(notes)
    radius = max(54.4 - 4.48 * float(bm.cs), 1.0)
    rhythm = _rhythm_metrics(bm, notes)
    patterns, bar_signatures = _jump_patterns(bm, notes, radius)
    type_counts = Counter(pattern["type"] for pattern in patterns)
    pattern_denominator = max(n, 1)

    by_shape: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for pattern in patterns:
        by_shape[pattern["fingerprint"]].append(pattern)
    repeated = {shape: items for shape, items in by_shape.items() if len(items) >= 2}
    repeated_indices = {id(item) for items in repeated.values() for item in items}
    music_indices: set[int] = set()
    stale_indices: set[int] = set()
    for items in repeated.values():
        phases: dict[int, list[dict[str, Any]]] = defaultdict(list)
        bar_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            phases[int(round(item["bar_phase"] * 4))].append(item)
            signature = bar_signatures.get(item["bar"])
            if signature:
                bar_groups[signature].append(item)
        aligned = set()
        for same_phase in phases.values():
            if len(same_phase) >= 2:
                aligned.update(id(item) for item in same_phase)
        for same_bar in bar_groups.values():
            if len(same_bar) >= 2:
                aligned.update(id(item) for item in same_bar)
        music_indices.update(aligned)
        starts = sorted(items, key=lambda item: item["start_index"])
        best_cluster: list[dict[str, Any]] = []
        for left in range(len(starts)):
            cluster = [item for item in starts[left:]
                       if item["start_index"] - starts[left]["start_index"] <= 20]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        if len(best_cluster) >= 5:
            stale_indices.update(id(item) for item in best_cluster)

    slider_items = [(obj, row) for obj, row in notes if obj.kind == "slider"]
    slider_lengths, slider_curvatures = [], []
    short_curved = repeated_sliders = 0
    for obj, row in slider_items:
        length = max(float(obj.length), 0.0)
        chord = math.hypot(float(row[EX] - obj.x), float(row[EY] - obj.y))
        ratio = min(chord / length, 1.0) if length > 0 else 1.0
        curvature = 1.0 - ratio
        slider_lengths.append(length / radius)
        slider_curvatures.append(curvature)
        repeated_sliders += int(obj.slides > 1)
        short_curved += int(length <= 100.0 and ratio <= 0.72)

    features = _movement_features(bm, notes, radius)
    metrics = dict(rhythm)
    metrics.update({
        "slider_share_pct": len(slider_items) * 100.0 / max(n, 1),
        "repeating_sliders_per_100": repeated_sliders * 100.0 / max(n, 1),
        "short_curved_sliders_per_100": short_curved * 100.0 / max(n, 1),
        "slider_length_radii_median": float(np.median(slider_lengths)) if slider_lengths else 0.0,
        "slider_curvature_median": float(np.median(slider_curvatures)) if slider_curvatures else 0.0,
        "patterns_per_100": len(patterns) * 100.0 / pattern_denominator,
        "pattern_size_radii_median": float(np.median([p["size_radii"] for p in patterns])) if patterns else 0.0,
        "pattern_length_median": float(np.median([p["edge_count"] for p in patterns])) if patterns else 0.0,
        "repeated_windows_pct": len(repeated_indices) * 100.0 / max(len(patterns), 1),
        "music_aligned_repeat_pct": len(music_indices) * 100.0 / max(len(patterns), 1),
        "stale_repeat_pct": len(stale_indices) * 100.0 / max(len(patterns), 1),
    })
    for pattern_type in PATTERN_TYPES:
        metrics[f"pattern_{pattern_type}_per_100"] = type_counts[pattern_type] * 100.0 / pattern_denominator

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pattern in patterns:
        group = examples[pattern["type"]]
        if len(group) < 3:
            group.append({"map": name, "time_ms": round(pattern["time_ms"]),
                          "edge_count": pattern["edge_count"],
                          "size_radii": round(pattern["size_radii"], 2)})
    return {
        "name": name,
        "stars": float(stars) if stars is not None else None,
        "star_bucket": star_bucket(stars),
        "object_count": n,
        "metrics": metrics,
        "examples": dict(examples),
        "_difficulty_values": {key: value.tolist() for key, value in features.items()},
    }


def build_reference(human_maps: list[dict[str, Any]]) -> dict[str, Any]:
    """Build compact 5th/50th/95th percentile limits from human maps by star range."""
    reference: dict[str, Any] = {"version": 1, "source": "human validation maps",
                                 "features": {bucket: {} for bucket in STAR_BUCKETS}}
    for bucket in STAR_BUCKETS:
        maps = [item for item in human_maps if item.get("star_bucket") == bucket]
        for feature in DIFFICULTY_FEATURES:
            arrays = [np.asarray(item["_difficulty_values"][feature], dtype=float)
                      for item in maps if item["_difficulty_values"].get(feature)]
            values = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
            if len(values):
                q05, q50, q95 = np.percentile(values, [5, 50, 95]).tolist()
                reference["features"][bucket][feature] = {
                    "p05": float(q05), "p50": float(q50), "p95": float(q95),
                    "objects": int(len(values)), "maps": len(maps),
                }
    return reference


def apply_reference(report: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    """Count unique objects above their human 95th-percentile limits."""
    bucket = report.get("star_bucket")
    feature_limits = (reference or {}).get("features", {}).get(bucket, {})
    values = report.get("_difficulty_values", {})
    n = int(report.get("object_count", 0))
    any_outlier = np.zeros(n, dtype=bool)
    per_feature = {}
    for feature in DIFFICULTY_FEATURES:
        limit = feature_limits.get(feature, {}).get("p95")
        feature_values = np.asarray(values.get(feature, []), dtype=float)
        if limit is None or len(feature_values) != n:
            continue
        mask = feature_values > float(limit)
        per_feature[feature] = int(mask.sum())
        any_outlier |= mask
        report["metrics"][f"difficulty_{feature}_over_p95_per_100"] = float(mask.sum() * 100.0 / max(n, 1))
    count = int(any_outlier.sum())
    report["difficulty"] = {
        "objects_over_p95": count,
        "objects_over_p95_per_100": float(count * 100.0 / max(n, 1)),
        "per_feature": per_feature,
    }
    report["metrics"]["difficulty_any_over_p95_per_100"] = report["difficulty"]["objects_over_p95_per_100"]
    return report


def clean_report(report: dict[str, Any]) -> dict[str, Any]:
    """Drop internal per-object arrays before displaying or serializing a report."""
    return {key: value for key, value in report.items() if not key.startswith("_")}

