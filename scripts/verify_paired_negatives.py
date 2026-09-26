"""Check rhythm, slider-velocity and beat-phase parity in generated critic pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from beatmap_ai.critic_data import critic_features, resolve_human_osu_text
from beatmap_ai.osu import parse_osu
from beatmap_ai.placement_data import BEAT, END, KIND, LEN, PHASE, SLIDER, SLIDES, T, map_objects


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    array = np.asarray(values, dtype=np.float64)
    return (f"n={len(array)} mean={array.mean():.5f} std={array.std():.5f} "
            f"p10={np.percentile(array, 10):.5f} p50={np.percentile(array, 50):.5f} "
            f"p90={np.percentile(array, 90):.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=r"D:\BeatMap-AI-Dataset\critic_negatives_paired\manifest.jsonl")
    parser.add_argument("data", nargs="*", default=["data", "best_maps", r"D:\BeatMap-AI-Dataset"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--fix-slider-multipliers", action="store_true",
                        help="write the exact source multiplier into each paired negative map")
    args = parser.parse_args()

    folders = [Path(folder) for folder in args.data]
    rows = []
    missing = 0
    mismatches = 0
    fixed_multipliers = 0
    feature_deltas = {"slider": [], "position": []}
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= args.limit:
                break
            if not line.strip():
                continue
            entry = json.loads(line)
            negative_path = Path(entry["negative_path"])
            if not negative_path.is_file():
                missing += 1
                continue
            human_text = resolve_human_osu_text(entry, folders)
            if human_text is None:
                missing += 1
                continue

            human_map = parse_osu(human_text)
            negative_text = negative_path.read_text(encoding="utf-8")
            negative_map = parse_osu(negative_text)
            if args.fix_slider_multipliers and human_map.slider_multiplier != negative_map.slider_multiplier:
                old_line = next(
                    (line for line in negative_text.splitlines() if line.startswith("SliderMultiplier:")),
                    None,
                )
                if old_line is not None:
                    new_line = f"SliderMultiplier:{human_map.slider_multiplier!r}"
                    negative_text = negative_text.replace(old_line, new_line, 1)
                    negative_path.write_text(negative_text, encoding="utf-8")
                    negative_map = parse_osu(negative_text)
                    fixed_multipliers += 1
            start = int(entry.get("window_start", 0))
            count = int(entry.get("window_count", len(negative_map.hit_objects)))
            human_window = replace(human_map, hit_objects=human_map.hit_objects[start:start + count])
            human_objects = human_window.hit_objects
            negative_objects = negative_map.hit_objects
            reasons = []
            if len(human_objects) != len(negative_objects):
                reasons.append(f"object-count {len(human_objects)}!={len(negative_objects)}")
            for object_idx, (human_obj, negative_obj) in enumerate(zip(human_objects, negative_objects)):
                if human_obj.kind != negative_obj.kind:
                    reasons.append(f"kind[{object_idx}] {human_obj.kind}!={negative_obj.kind}")
                if abs(human_obj.time - negative_obj.time) > 0.01:
                    reasons.append(f"time[{object_idx}] {human_obj.time}!={negative_obj.time}")
                if human_obj.slides != negative_obj.slides:
                    reasons.append(f"slides[{object_idx}] {human_obj.slides}!={negative_obj.slides}")
                end_delta = abs(human_map.end_time(human_obj) - negative_map.end_time(negative_obj))
                if end_delta > 2.0:
                    reasons.append(f"end_time[{object_idx}] delta={end_delta:.3f}ms")
            if human_map.slider_multiplier != negative_map.slider_multiplier:
                reasons.append(f"slider_multiplier {human_map.slider_multiplier}!={negative_map.slider_multiplier}")
            human_timing = [tp.to_line() for tp in human_map.timing_points]
            negative_timing = [tp.to_line() for tp in negative_map.timing_points]
            if human_timing != negative_timing:
                reasons.append(f"timing_points lines {len(human_timing)}!={len(negative_timing)}")
            pair_bad = bool(reasons)
            mismatches += int(pair_bad)
            if pair_bad:
                print(f"Mismatch {len(rows) + 1}: {negative_path}: {'; '.join(reasons[:4])}")

            hrows = map_objects(human_window)
            nrows = map_objects(negative_map)
            if hrows.shape == nrows.shape:
                human_features = critic_features(hrows, None, float(entry["stars"]))
                negative_features = critic_features(nrows, None, float(entry["stars"]))
                feature_deltas["slider"].append(
                    np.abs(human_features[:, 9:12] - negative_features[:, 9:12]).ravel()
                )
                feature_deltas["position"].append(
                    np.abs(human_features[:, 12:23] - negative_features[:, 12:23]).ravel()
                )

            def pair_metrics(obj_rows: np.ndarray, beatmap):
                slider_mask = obj_rows[:, KIND] == SLIDER
                slider_speed = []
                for i in np.flatnonzero(slider_mask):
                    duration = obj_rows[i, END] - obj_rows[i, T]
                    slides = max(float(obj_rows[i, SLIDES]), 1.0)
                    if duration > 0:
                        slider_speed.append(float(obj_rows[i, LEN] * obj_rows[i, BEAT] * slides / duration))
                gaps = []
                for i in range(1, len(beatmap.hit_objects)):
                    obj, previous = beatmap.hit_objects[i], beatmap.hit_objects[i - 1]
                    beat_length, _ = beatmap.timing_at(obj.time)
                    gaps.append(float((obj.time - beatmap.end_time(previous)) / beat_length))
                return slider_speed, gaps, obj_rows[:, PHASE].astype(float).tolist()

            hmetrics = pair_metrics(hrows, human_window)
            nmetrics = pair_metrics(nrows, negative_map)
            rows.append({"human": hmetrics, "negative": nmetrics, "entry": entry, "pair_bad": pair_bad})

    labels = ("slider px per beat", "gap per beat", "beat phase")
    for metric_idx, label in enumerate(labels):
        human_values = [value for pair in rows for value in pair["human"][metric_idx]]
        negative_values = [value for pair in rows for value in pair["negative"][metric_idx]]
        deltas = []
        for pair in rows:
            h = pair["human"][metric_idx]
            n = pair["negative"][metric_idx]
            if len(h) == len(n):
                deltas.extend(abs(a - b) for a, b in zip(h, n))
        print(f"{label}: human {describe(human_values)}")
        print(f"{label}: negative {describe(negative_values)}")
        print(f"{label}: paired max |delta|={max(deltas, default=0.0):.8f}; "
              f"mean |delta|={np.mean(deltas) if deltas else 0.0:.8f}")

    for name, batches in feature_deltas.items():
        deltas = np.concatenate(batches) if batches else np.zeros(0, dtype=np.float32)
        nonzero = int(np.count_nonzero(deltas > 1e-7))
        print(f"Critic {name} feature parity: values={len(deltas)} differ>{1e-7:g}={nonzero}; "
              f"max |delta|={float(deltas.max()) if len(deltas) else 0.0:.8f}; "
              f"mean |delta|={float(deltas.mean()) if len(deltas) else 0.0:.8f}")

    starts = [int(pair["entry"].get("window_start", 0)) for pair in rows]
    kiai_count = sum(bool(pair["entry"].get("kiai_window")) for pair in rows)
    print(f"Pairs checked: {len(rows)}; missing files/sources: {missing}; "
          f"multiplier values corrected: {fixed_multipliers}; rhythm/timing mismatches: {mismatches}")
    print(f"Window starts: {describe(starts)}; Kiai windows: {kiai_count}/{len(rows)}")
    return int(missing > 0 or mismatches > 0 or not rows)


if __name__ == "__main__":
    raise SystemExit(main())
