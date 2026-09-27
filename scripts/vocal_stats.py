"""CPU-only evaluation of vocal alignment on held-out human osu! maps.

By default this reads the local ``data`` and ``best_maps`` folders. Pass an additional
dataset root explicitly when it is safe to read. The script is serial by design: it
does not import PyTorch, start worker processes, or write intermediate files into the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beatmap_ai.audio import SAMPLE_RATE, load_audio
from beatmap_ai.dataset import AudioSource, is_validation, iter_beatmap_texts, map_key, song_id, song_key
from beatmap_ai.osu import Beatmap, parse_osu
from beatmap_ai.style import star_rating
from beatmap_ai.vocals import (
    VAD_MIN_VOICED_RATIO, VocalAnalysis, analyze_mel, analyze_vocals,
)

STAR_BUCKETS = ("<3", "3-4.5", "4.5-6", "6+")
MATCH_TOLERANCE_MS = 30.0


@dataclass(slots=True)
class MapRecord:
    name: str
    beatmap: Beatmap
    source: AudioSource
    stars: float
    song: str


def _bucket(stars: float) -> str | None:
    if stars < 3.0:
        return "<3"
    if stars < 4.5:
        return "3-4.5"
    if stars < 6.0:
        return "4.5-6"
    if np.isfinite(stars):
        return "6+"
    return None


def _select_maps(roots: list[Path], per_bucket: int, seed: int) -> tuple[list[MapRecord], dict]:
    rng = random.Random(seed)
    maps_by_song: dict[str, dict[str, list[MapRecord]]] = {
        group: defaultdict(list) for group in STAR_BUCKETS
    }
    seen_keys: set[tuple] = set()
    counters = {"beatmaps_seen": 0, "non_validation": 0, "bad_maps": 0,
                "no_stars": 0, "validation_candidates": 0}
    for root in roots:
        if not root.exists():
            continue
        for name, text, source, _stamp in iter_beatmap_texts(root):
            counters["beatmaps_seen"] += 1
            try:
                beatmap = parse_osu(text)
            except Exception:
                counters["bad_maps"] += 1
                continue
            if beatmap.mode != 0 or len(beatmap.hit_objects) < 20:
                continue
            key = map_key(beatmap)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            song = song_key(beatmap)
            if not is_validation(song):
                counters["non_validation"] += 1
                continue
            counters["validation_candidates"] += 1
            stars = star_rating(text)
            if stars is None or not np.isfinite(stars):
                counters["no_stars"] += 1
                continue
            group = _bucket(float(stars))
            if group is None:
                continue
            maps_by_song[group][song].append(MapRecord(name, beatmap, source, float(stars), song))

    selected: list[MapRecord] = []
    for group in STAR_BUCKETS:
        song_maps = maps_by_song[group]
        for records in song_maps.values():
            rng.shuffle(records)
        songs = list(song_maps)
        rng.shuffle(songs)
        selected_group = []
        # Round-robin across songs first, then add their other difficulties if room
        # remains. This keeps one prolific song from filling a star bucket.
        depth = 0
        while len(selected_group) < per_bucket:
            added = False
            for song in songs:
                records = song_maps[song]
                if depth < len(records):
                    selected_group.append(records[depth])
                    added = True
                    if len(selected_group) == per_bucket:
                        break
            if not added:
                break
            depth += 1
        selected.extend(selected_group)
        counters[f"available_{group}"] = sum(len(items) for items in song_maps.values())
        counters[f"available_songs_{group}"] = len(song_maps)
        counters[f"selected_{group}"] = len(selected_group)
    return selected, counters


def _load_source(source: AudioSource) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (waveform, cached mel); exactly one value is non-None."""
    if source.path.name.lower() == "mel.npy":
        return None, np.load(source.path, mmap_mode="r")
    if source.member is None:
        return load_audio(source.path), None
    with zipfile.ZipFile(source.path) as archive:
        data = archive.read(source.member)
    # Decode archived audio from a short-lived file because miniaudio's robust decoder
    # is path-based. Nothing is left in the repository or dataset after this call.
    suffix = Path(source.member).suffix or ".audio"
    with tempfile.TemporaryDirectory(prefix="beatmap-vocal-audio-") as tmp:
        path = Path(tmp) / f"audio{suffix}"
        path.write_bytes(data)
        return load_audio(path), None


def _analyze_source(source: AudioSource, roots: list[Path]) -> tuple[VocalAnalysis, str]:
    if source.path.name.lower() == "mel.npy":
        return analyze_mel(np.load(source.path, mmap_mode="r")), "mel_file"
    adjacent = source.path.parent / "mel.npy"
    if source.member is None and adjacent.is_file():
        return analyze_mel(np.load(adjacent, mmap_mode="r")), "adjacent_mel_cache"
    cache_name = song_id(source.key) + ".npy"
    for root in roots:
        cached = root / ".beatmap_ai_cache" / cache_name
        if cached.is_file():
            return analyze_mel(np.load(cached, mmap_mode="r")), "feature_cache"
    waveform, mel = _load_source(source)
    if mel is not None:
        return analyze_mel(mel), "mel_file"
    assert waveform is not None
    return analyze_vocals(waveform, sr=SAMPLE_RATE), "raw_audio"


def _bar_sections(beatmap: Beatmap, duration_ms: float, bars: int = 4) -> list[tuple[float, float]]:
    timing = sorted((tp for tp in beatmap.timing_points if tp.uninherited and tp.beat_length > 0),
                    key=lambda tp: tp.time)
    if not timing or duration_ms <= 0:
        return []
    sections = []
    for i, point in enumerate(timing):
        next_change = timing[i + 1].time if i + 1 < len(timing) else duration_ms
        segment_start = max(0.0, point.time)
        segment_end = min(duration_ms, next_change)
        if segment_end <= segment_start:
            continue
        bar_ms = point.beat_length * max(1, point.meter)
        width = max(bar_ms * bars, 1.0)
        start = segment_start
        while start + width <= segment_end + 1e-6:
            sections.append((start, start + width))
            start += width
    return sections


def _beat_grid(beatmap: Beatmap, start_ms: float, end_ms: float) -> np.ndarray:
    points = sorted((tp for tp in beatmap.timing_points if tp.uninherited and tp.beat_length > 0),
                    key=lambda tp: tp.time)
    beats: list[float] = []
    for i, point in enumerate(points):
        limit = min(end_ms, points[i + 1].time if i + 1 < len(points) else end_ms)
        start = max(start_ms, point.time)
        if limit <= start:
            continue
        beat = float(point.beat_length)
        first = point.time + np.ceil((start - point.time) / beat) * beat
        beats.extend(np.arange(first, limit, beat, dtype=np.float64).tolist())
    return np.asarray(beats, dtype=np.float64)


def _hit_mask(notes: np.ndarray, events: np.ndarray, tolerance_ms: float = MATCH_TOLERANCE_MS) -> np.ndarray:
    if len(notes) == 0 or len(events) == 0:
        return np.zeros(len(notes), dtype=bool)
    events = np.sort(np.asarray(events, dtype=np.float64))
    idx = np.searchsorted(events, notes)
    left = np.clip(idx - 1, 0, len(events) - 1)
    right = np.clip(idx, 0, len(events) - 1)
    distance = np.minimum(np.abs(notes - events[left]), np.abs(notes - events[right]))
    return distance <= tolerance_ms


def _syllable_times(analysis: VocalAnalysis) -> np.ndarray:
    return np.asarray([row["time_ms"] for row in analysis.syllables], dtype=np.float64)


def _kiai_share(beatmap: Beatmap, start_ms: float, end_ms: float) -> float:
    points = sorted(beatmap.timing_points, key=lambda tp: tp.time)
    active = False
    cursor = start_ms
    covered = 0.0
    for point in points:
        if point.time <= start_ms:
            active = bool(point.effects & 1)
            continue
        if point.time >= end_ms:
            break
        if active:
            covered += point.time - cursor
        cursor = point.time
        active = bool(point.effects & 1)
    if active:
        covered += end_ms - cursor
    return float(np.clip(covered / max(end_ms - start_ms, 1e-9), 0.0, 1.0))


def _section_row(record: MapRecord, analysis: VocalAnalysis,
                 start_ms: float, end_ms: float, rng: random.Random,
                 random_shifts: int) -> dict | None:
    notes = np.asarray([
        obj.time for obj in record.beatmap.hit_objects
        if obj.kind in {"circle", "slider"} and start_ms <= obj.time < end_ms
    ], dtype=np.float64)
    if not len(notes):
        return None
    lo = max(0, int(start_ms * analysis.frame_rate / 1000.0))
    hi = min(analysis.n_frames, max(lo + 1, int(np.ceil(end_ms * analysis.frame_rate / 1000.0))))
    if lo >= analysis.n_frames:
        return None
    voice_energy = float(analysis.vocal_energy[lo:hi].sum())
    other_energy = float(analysis.other_energy[lo:hi].sum())
    vocal_share = voice_energy / max(voice_energy + other_energy, 1e-12)
    vir = voice_energy / max(other_energy, 1e-12)
    labels = analysis.labels[lo:hi]
    voiced_ratio = float(np.mean(labels == 1)) if len(labels) else 0.0
    syllables = _syllable_times(analysis)
    syllables = syllables[(syllables >= start_ms) & (syllables < end_ms)]
    percussion = analysis.percussive_onsets_ms
    percussion = percussion[(percussion >= start_ms) & (percussion < end_ms)]
    beats = _beat_grid(record.beatmap, start_ms, end_ms)
    beat_events = np.unique(np.concatenate((beats, percussion)))
    vocal_hits = _hit_mask(notes, syllables)
    beat_hits = _hit_mask(notes, beat_events)
    vocal_score = float(vocal_hits.mean())
    beat_score = float(beat_hits.mean())
    if vocal_score > beat_score + 0.05:
        label = "vocal"
    elif beat_score > vocal_score + 0.05:
        label = "beat"
    else:
        label = "mixed"

    # Circular random shifts preserve the syllable count and local note density while
    # breaking the actual timing relationship. Use a song-independent fixed seed.
    duration = end_ms - start_ms
    random_scores = []
    if len(syllables) and duration > 0:
        rel_syllables = syllables - start_ms
        for _ in range(random_shifts):
            shift = rng.random() * duration
            shifted = start_ms + np.mod(rel_syllables + shift, duration)
            random_scores.append(float(_hit_mask(notes, shifted).mean()))
    drum_density = len(percussion) / max(duration / 1000.0, 1e-9)
    return {
        "map": record.name,
        "song": record.song,
        "stars": record.stars,
        "star_bucket": _bucket(record.stars),
        "start_ms": float(start_ms),
        "end_ms": float(end_ms),
        "label": label,
        "follows_vocals": label == "vocal",
        "vocal_alignment": vocal_score,
        "beat_alignment": beat_score,
        "vocal_share": float(vocal_share),
        "vir": float(vir),
        "voiced_ratio": voiced_ratio,
        "is_vocal_section": voiced_ratio >= VAD_MIN_VOICED_RATIO,
        "drum_density": float(drum_density),
        "kiai": _kiai_share(record.beatmap, start_ms, end_ms) >= 0.5,
        "note_count": int(len(notes)),
        "syllable_count": int(len(syllables)),
        "note_syllable_hits": int(vocal_hits.sum()),
        "syllables_with_note": int(_hit_mask(syllables, notes).sum()),
        "random_shift_alignment": float(np.mean(random_scores)) if random_scores else 0.0,
        "example_note_ms": notes.tolist(),
        "example_syllable_ms": syllables.tolist(),
        "example_beat_ms": beat_events.tolist(),
    }


def _rule_thresholds(rows: list[dict]) -> tuple[float, float]:
    train = [row for row in rows if not _is_test_song(row["song"])]
    if not train:
        return 0.18, 1.5
    shares = np.asarray([r["vocal_share"] for r in train], dtype=np.float64)
    drums = np.asarray([r["drum_density"] for r in train], dtype=np.float64)
    share_grid = np.unique(np.quantile(shares, [0.25, 0.4, 0.5, 0.6, 0.75]))
    drum_grid = np.unique(np.quantile(drums, [0.25, 0.4, 0.5, 0.6, 0.75]))
    truth = np.asarray([r["follows_vocals"] for r in train], dtype=bool)
    best = (-1.0, 0.18, 1.5)
    for share_cut in share_grid:
        for drum_cut in drum_grid:
            pred = (shares >= share_cut) & (drums <= drum_cut)
            tpr = float(np.mean(pred[truth])) if np.any(truth) else 0.0
            tnr = float(np.mean(~pred[~truth])) if np.any(~truth) else 0.0
            score = (tpr + tnr) / 2.0
            if score > best[0]:
                best = (score, float(share_cut), float(drum_cut))
    return best[1], best[2]


def _is_test_song(song: str) -> bool:
    value = int(hashlib.sha1(song.encode("utf-8")).hexdigest()[:8], 16)
    return value % 5 == 0


def _rule_metrics(rows: list[dict]) -> dict:
    share_cut, drum_cut = _rule_thresholds(rows)
    test = [row for row in rows if _is_test_song(row["song"])]
    if not test:
        empty = {"test_sections": 0, "accuracy": None, "balanced_accuracy": None}
        return {"tuned_on_train": {**empty, "vocal_share_threshold": share_cut,
                                    "drum_density_threshold_per_s": drum_cut},
                "fixed_high_share_low_drum": empty, "majority_baseline_accuracy": None}
    truth = np.asarray([row["follows_vocals"] for row in test], dtype=bool)
    tuned = np.asarray([
        row["vocal_share"] >= share_cut and row["drum_density"] <= drum_cut for row in test
    ], dtype=bool)

    def score(prediction: np.ndarray) -> dict:
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        tn = int(np.count_nonzero(~prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        tpr = tp / (tp + fn) if tp + fn else float("nan")
        tnr = tn / (tn + fp) if tn + fp else float("nan")
        return {
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": float(np.nanmean([tpr, tnr])),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tpr if np.isfinite(tpr) else None,
            "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
        }

    fixed = np.asarray([
        row["vocal_share"] >= 0.20 and row["drum_density"] < 1.0 for row in test
    ], dtype=bool)
    return {
        "test_sections": len(test),
        "test_songs": len({row["song"] for row in test}),
        "test_positive_sections": int(truth.sum()),
        "majority_baseline_accuracy": float(max(truth.mean(), 1.0 - truth.mean())),
        "tuned_on_train": {
            **score(tuned),
            "vocal_share_threshold": share_cut,
            "drum_density_threshold_per_s": drum_cut,
        },
        "fixed_high_share_low_drum": {
            **score(fixed),
            "vocal_share_threshold": 0.20,
            "drum_density_threshold_per_s": 1.0,
        },
    }


def _group_metrics(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row[key]
        groups[str(value)].append(row)
    return {
        group: {
            "sections": len(items),
            "follow_vocals_percent": 100.0 * float(np.mean([r["follows_vocals"] for r in items])),
            "mean_vocal_alignment_percent": 100.0 * float(np.mean([r["vocal_alignment"] for r in items])),
            "mean_beat_alignment_percent": 100.0 * float(np.mean([r["beat_alignment"] for r in items])),
            "mean_vocal_share_percent": 100.0 * float(np.mean([r["vocal_share"] for r in items])),
            "mean_drum_density_per_s": float(np.mean([r["drum_density"] for r in items])),
        }
        for group, items in sorted(groups.items())
    }


def _summarize(rows: list[dict], map_counts: dict, timings: list[float], random_shifts: int) -> dict:
    vocal_sections = [row for row in rows if row["is_vocal_section"]]
    note_total = sum(r["note_count"] for r in vocal_sections)
    syllable_total = sum(r["syllable_count"] for r in vocal_sections)
    note_hits = sum(r["note_syllable_hits"] for r in vocal_sections)
    syllable_hits = sum(r["syllables_with_note"] for r in vocal_sections)
    held_songs = {r["song"] for r in rows}
    test_rows = [r for r in rows if _is_test_song(r["song"])]
    examples = []
    example_counts = {"vocal": 0, "beat": 0}
    seen_example_maps: set[str] = set()
    for row in sorted(rows, key=lambda r: (
        r["label"] != "vocal", -abs(r["vocal_alignment"] - r["beat_alignment"]), r["song"]
    )):
        label = row["label"]
        if label == "mixed" or example_counts[label] >= (5 if label == "vocal" else 3):
            continue
        if row["map"] in seen_example_maps:
            continue
        nearest = []
        target = row["example_syllable_ms"] if row["label"] == "vocal" else row["example_beat_ms"]
        for note in row["example_note_ms"]:
            if target:
                best = min(target, key=lambda time: abs(time - note))
                if abs(best - note) <= MATCH_TOLERANCE_MS:
                    nearest.append({"note_ms": round(note, 1), "reference_ms": round(best, 1)})
                    if len(nearest) == 3:
                        break
        if not nearest:
            continue
        examples.append({
            "map": row["map"], "stars": row["stars"], "label": row["label"],
            "section_start_ms": round(row["start_ms"]),
            "note_reference_examples": nearest,
            "vocal_alignment_percent": round(100.0 * row["vocal_alignment"], 1),
            "beat_alignment_percent": round(100.0 * row["beat_alignment"], 1),
        })
        example_counts[label] += 1
        seen_example_maps.add(row["map"])
    return {
        "maps": map_counts,
        "songs": len(held_songs),
        "sections": len(rows),
        "section_labels": {label: sum(row["label"] == label for row in rows)
                           for label in ("vocal", "beat", "mixed")},
        "follow_vocals_by_star_bucket": _group_metrics(rows, "star_bucket"),
        "follow_vocals_by_kiai": _group_metrics(rows, "kiai"),
        "follow_vocals_by_vocal_share_and_drum_density": {
            "vocal_share": _group_metrics([
                {**r, "vocal_share_band": "low (<20%)" if r["vocal_share"] < 0.2 else
                 "mid (20-40%)" if r["vocal_share"] < 0.4 else "high (>=40%)"} for r in rows
            ], "vocal_share_band"),
            "drum_density": _group_metrics([
                {**r, "drum_band": "low (<1/s)" if r["drum_density"] < 1.0 else
                 "mid (1-3/s)" if r["drum_density"] < 3.0 else "high (>=3/s)"} for r in rows
            ], "drum_band"),
        },
        "syllable_detection_in_voice_detected_sections": {
            "voice_detected_sections": len(vocal_sections),
            "human_notes": note_total,
            "note_syllable_hits": note_hits,
            "note_hit_percent": 100.0 * note_hits / note_total if note_total else None,
            "detected_syllables": syllable_total,
            "syllables_with_note": syllable_hits,
            "syllable_coverage_percent": 100.0 * syllable_hits / syllable_total if syllable_total else None,
            "tolerance_ms": MATCH_TOLERANCE_MS,
        },
        "random_shift_baseline": {
            "shifts_per_section": random_shifts,
            "mean_note_alignment_percent": 100.0 * float(np.mean([
                r["random_shift_alignment"] for r in rows if r["syllable_count"]
            ])) if any(r["syllable_count"] for r in rows) else None,
            "actual_mean_note_alignment_percent": 100.0 * float(np.mean([
                r["vocal_alignment"] for r in rows if r["syllable_count"]
            ])) if any(r["syllable_count"] for r in rows) else None,
        },
        "vocal_share_low_drum_rule": _rule_metrics(rows),
        "audio_runtime_seconds": {
            "unique_sources": len(timings),
            "total": float(sum(timings)),
            "median_per_source": float(np.median(timings)) if timings else None,
            "p95_per_source": float(np.percentile(timings, 95)) if timings else None,
        },
        "held_out_test_songs": len({r["song"] for r in test_rows}),
        "editor_examples": examples,
        "method": {
            "split": "dataset.is_validation(song_key) for source maps; deterministic 80/20 song split for the simple-rule test",
            "section": "4 bars between uninherited timing changes; note heads are circles and slider heads",
            "label": "vocal when note hits align with detected syllables > beat/percussion hits by >5 percentage points; inverse for beat; otherwise mixed",
            "beat_reference": "quarter-note beat grid plus whole-spectrum spectral-flux onset peaks",
            "drum_density": "transient proxy from full-spectrum spectral flux; it is not a drum stem",
            "tolerance_ms": MATCH_TOLERANCE_MS,
            "limitations": [
                "The 2-6 kHz energy band also contains cymbals and other instruments.",
                "Cached mel-only sources cannot provide a physical F0 estimate.",
                "This measures agreement with existing human maps; it is not a direct head-to-head generation test against Mapperatorinator.",
            ],
        },
    }


def evaluate(roots: list[Path], *, per_bucket: int = 50, seed: int = 2026,
             random_shifts: int = 100, dry_run: bool = False) -> dict:
    # Keep BLAS/FFT pools to one thread while the night run is using the machine.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    records, map_counts = _select_maps(roots, per_bucket, seed)
    map_counts["selected_total"] = len(records)
    map_counts["selected_songs"] = len({r.song for r in records})
    if dry_run:
        return {"maps": map_counts, "dry_run": True,
                "selected": [{"map": r.name, "stars": r.stars, "song": r.song,
                              "star_bucket": _bucket(r.stars)} for r in records]}

    grouped: dict[str, list[MapRecord]] = defaultdict(list)
    source_by_key: dict[str, AudioSource] = {}
    for record in records:
        grouped[record.source.key].append(record)
        source_by_key[record.source.key] = record.source

    rng = random.Random(seed + 1)
    rows: list[dict] = []
    timings: list[float] = []
    source_modes: dict[str, int] = defaultdict(int)
    done = 0
    for source_key, source_records in grouped.items():
        analysis, source_mode = _analyze_source(source_by_key[source_key], roots)
        source_modes[source_mode] += 1
        timings.append(analysis.runtime_seconds)
        duration_ms = analysis.n_frames * 1000.0 / analysis.frame_rate
        for record in source_records:
            sections = _bar_sections(record.beatmap, duration_ms, bars=4)
            for start_ms, end_ms in sections:
                row = _section_row(record, analysis, start_ms, end_ms, rng, random_shifts)
                if row is not None:
                    rows.append(row)
        done += 1
        if done % 10 == 0 or done == len(grouped):
            print(f"CPU vocal evaluation: {done}/{len(grouped)} distinct audio sources",
                  file=sys.stderr, flush=True)

    map_counts["evaluated_audio_sources"] = len(grouped)
    map_counts["audio_source_modes"] = dict(source_modes)
    map_counts["maps_with_sections"] = len({row["map"] for row in rows})
    result = _summarize(rows, map_counts, timings, random_shifts)
    result["seed"] = seed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path,
                        default=[REPO_ROOT / "data", REPO_ROOT / "best_maps"],
                        help="Read-only map roots; only local data and best_maps are used by default.")
    parser.add_argument("--per-bucket", type=int, default=50,
                        help="Maximum held-out songs/maps in each star bucket (default: 50).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-shifts", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only select and count validation maps; do not decode audio.")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON output path, preferably under D:\\BeatMap-AI-Dataset\\vocals.")
    args = parser.parse_args()
    if not 1 <= args.per_bucket <= 500:
        parser.error("--per-bucket must be between 1 and 500")
    if args.random_shifts < 1:
        parser.error("--random-shifts must be positive")
    roots = [path if path.is_absolute() else (REPO_ROOT / path) for path in args.roots]
    result = evaluate(roots, per_bucket=args.per_bucket, seed=args.seed,
                      random_shifts=args.random_shifts, dry_run=args.dry_run)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
        summary = {key: result.get(key) for key in ("maps", "songs", "sections", "section_labels")}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
