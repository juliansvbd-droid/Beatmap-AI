"""Generate paired negative beatmaps (human rhythm, AI placement) for the Critic.

For each selected human beatmap:
- Extract the exact human rhythm (object times, circle/slider kinds, lengths, repeats, combos)
- Place objects with SequencePlacer.follow (~75%) or rule-based Placer (~25%)
- Resulting maps have identical rhythm, stars, density and audio, differing ONLY in spatial placement.
- Target: 3,000 maps balanced across 4 star buckets (<3, 3-4.5, 4.5-6, 6+).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import sys
import time
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beatmap_ai.audio import FPS, AudioFeatures
from beatmap_ai.dataset import AudioSource, MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, note_density, song_key
from beatmap_ai.difficulty import preset_for_stars
from beatmap_ai.evaluate import constant_timing
from beatmap_ai.generator import inference_device
from beatmap_ai.osu import Beatmap, TimingPoint, parse_osu
from beatmap_ai.placement import Placer
from beatmap_ai.rhythm import PlannedObject, make_tick_grid
from beatmap_ai.sequence_model import SequencePlacer, load_sequence
from beatmap_ai.style import star_rating
from beatmap_ai.timing import TimingEstimate
from beatmap_ai.train import resolve_device


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def star_bucket(stars: float) -> str:
    if stars < 3.0:
        return "<3"
    if stars < 4.5:
        return "3-4.5"
    if stars < 6.0:
        return "4.5-6"
    return "6+"


def get_audio_features(source: AudioSource) -> AudioFeatures:
    """Fast audio loader: loads pre-computed mel.npy in 0.5ms when available."""
    if source.path.is_file():
        mel_file = source.path.parent / "mel.npy"
        if mel_file.is_file():
            try:
                mel = np.load(mel_file)
                n = mel.shape[1]
                dur = float(n * 256.0 / 22050.0)
                return AudioFeatures(
                    mel=mel,
                    onset=np.zeros(n, dtype=np.float32),
                    rms=np.zeros(n, dtype=np.float32),
                    bass_onset=np.zeros(n, dtype=np.float32),
                    duration=dur,
                )
            except Exception:
                pass
    return source.load_features()


def kiai_window_starts(bm: Beatmap, window_count: int, min_fraction: float = 2.0 / 3.0) -> list[int]:
    """Return object-index windows that are mostly inside human-marked Kiai sections."""
    count = len(bm.hit_objects)
    if count == 0 or window_count <= 0:
        return []

    timing_points = sorted(bm.timing_points, key=lambda tp: tp.time)
    active_intervals = []
    for i, tp in enumerate(timing_points):
        next_time = timing_points[i + 1].time if i + 1 < len(timing_points) else float("inf")
        if tp.effects & 1:
            active_intervals.append((tp.time, next_time))
    if not active_intervals:
        return []

    interval_starts = np.asarray([start for start, _ in active_intervals], dtype=np.float64)
    interval_ends = np.asarray([end for _, end in active_intervals], dtype=np.float64)
    object_times = np.asarray([obj.time for obj in bm.hit_objects], dtype=np.float64)
    interval_idx = np.searchsorted(interval_starts, object_times, side="right") - 1
    valid = interval_idx >= 0
    is_kiai = np.zeros(count, dtype=np.int32)
    is_kiai[valid] = (object_times[valid] < interval_ends[interval_idx[valid]]).astype(np.int32)
    width = min(window_count, count)
    prefix = np.concatenate(([0], np.cumsum(is_kiai)))
    starts = []
    for start in range(count - width + 1):
        if (prefix[start + width] - prefix[start]) / width >= min_fraction:
            starts.append(start)
    return starts


def human_to_plan(
    bm: Beatmap,
    timing: TimingEstimate,
    quarter,
    max_objects: int = 96,
    start_index: int = 0,
) -> tuple[list[PlannedObject], list]:
    """Convert human hit objects into a list of PlannedObjects preserving exact rhythm."""
    plan = []
    spinners = []
    start_index = max(0, min(int(start_index), max(0, len(bm.hit_objects) - 1)))
    end_index = start_index + max_objects if max_objects else len(bm.hit_objects)
    objs = bm.hit_objects[start_index:end_index]

    for o in objs:
        if o.kind == "spinner":
            spinners.append(o)
            continue
        if o.kind not in ("circle", "slider"):
            continue
        b_len, sv = bm.timing_at(o.time)
        end_time = bm.end_time(o)
        tick = int(np.searchsorted(quarter.times, o.time))
        if tick > 0 and tick < len(quarter.times):
            if abs(quarter.times[tick - 1] - o.time) < abs(quarter.times[tick] - o.time):
                tick -= 1
        p = PlannedObject(
            time=float(o.time),
            kind=o.kind,
            tick=tick,
            intensity=0.5,
            end_time=float(end_time),
            measure=int((o.time - timing.offset_ms) / max(b_len * 4.0, 1.0)),
            new_combo=bool(o.new_combo),
            beat_length=float(b_len),
            slides=int(getattr(o, "slides", 1)),
        )
        plan.append(p)

    return plan, spinners


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="*", default=["data", "best_maps", "D:/BeatMap-AI-Dataset"],
                        help="directories containing source maps")
    parser.add_argument("--out-dir", default=r"D:\BeatMap-AI-Dataset\critic_negatives_paired",
                        help="directory where paired negative maps and manifest are saved")
    parser.add_argument("--target-per-bucket", type=int, default=750,
                        help="target number of maps per star bucket (total = 4 * target)")
    parser.add_argument("--sequence", default="beatmap_ai/models/sequence.pt",
                        help="sequence model checkpoint")
    parser.add_argument("--rule-ratio", type=float, default=0.25,
                        help="fraction placed with rule-based Placer (~0.25)")
    parser.add_argument("--max-objects", type=int, default=96,
                        help="max objects per map to place (speeds up generation, preserves distribution)")
    parser.add_argument("--device", default=None, help="device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log-every", type=int, default=50, help="log progress interval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    existing_keys = set()
    bucket_counts = Counter()
    kiai_bucket_counts = Counter()
    total_existing = 0

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    neg_path = Path(entry.get("negative_path", ""))
                    if neg_path.exists() and neg_path.stat().st_size > 50:
                        mkey_str = entry.get("map_key_str", "")
                        existing_keys.add(mkey_str)
                        b = entry.get("bucket", star_bucket(entry.get("stars", 4.0)))
                        bucket_counts[b] += 1
                        if entry.get("kiai_window"):
                            kiai_bucket_counts[b] += 1
                        total_existing += 1
                except Exception:
                    pass

    print(f"Loaded {total_existing} existing paired maps from {manifest_path}: {dict(bucket_counts)}", flush=True)
    total_target = args.target_per_bucket * 4
    if total_existing >= total_target and all(bucket_counts[b] >= args.target_per_bucket for b in ("<3", "3-4.5", "4.5-6", "6+")):
        print(f"Already reached full target of {total_target} maps across all buckets. Exiting.", flush=True)
        return

    # Load and optimize sequence model
    device = resolve_device(args.device or "cuda")
    print(f"Loading sequence model on {device}...", flush=True)
    seq = load_sequence(args.sequence, device)
    dummy_x = torch.zeros((1, 64, 830), device=device)
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(seq, dummy_x, strict=False)
            traced.config = seq.config
            seq_model = traced
            print("Successfully JIT-traced sequence model for ROCm.", flush=True)
    except Exception as e:
        print(f"JIT-trace fallback to standard model: {e}", flush=True)
        seq_model = seq

    rng = np.random.default_rng(args.seed)

    print("Scanning dataset directories for balanced human source maps...", flush=True)
    candidates_by_bucket = defaultdict(list)
    needed = {b: max(0, args.target_per_bucket - bucket_counts[b]) for b in ("<3", "3-4.5", "4.5-6", "6+")}
    kiai_needed = {
        b: max(0, int(np.ceil(args.target_per_bucket / 3)) - kiai_bucket_counts[b])
        for b in ("<3", "3-4.5", "4.5-6", "6+")
    }
    scan_extra = {b: max(40, int(np.ceil(needed[b] * 0.2))) if needed[b] else 0 for b in needed}
    scan_limits = {b: needed[b] + scan_extra[b] for b in needed}
    kiai_scan_limits = {
        b: kiai_needed[b] + max(20, int(np.ceil(kiai_needed[b] * 0.2))) if kiai_needed[b] else 0
        for b in kiai_needed
    }
    kiai_candidates_by_bucket = defaultdict(list)

    def enough_candidates(bucket: str) -> bool:
        return (len(candidates_by_bucket[bucket]) >= scan_limits[bucket]
                and len(kiai_candidates_by_bucket[bucket]) >= kiai_scan_limits[bucket])

    seen_scan_keys = set(existing_keys)

    for folder_str in args.data:
        folder = Path(folder_str)
        if not folder.exists():
            continue
        print(f"Scanning {folder}...", flush=True)

        is_large_flat = (folder / "tags.json").exists() or (folder / "catalog.json").exists()
        iterator = _scan_flat(folder) if is_large_flat else iter_beatmap_texts(folder)

        for name, text, source, _ in iterator:
            if all(enough_candidates(b) for b in ("<3", "3-4.5", "4.5-6", "6+")):
                break
            stars = star_rating(text)
            if stars is None:
                continue
            b = star_bucket(stars)
            if enough_candidates(b):
                continue
            try:
                bm = parse_osu(text)
                if bm.mode != 0 or len(bm.hit_objects) < MIN_OBJECTS:
                    continue
                mkey = map_key(bm)
                mkey_str = f"{mkey[0]}::{mkey[1]}::{mkey[2]}::{mkey[3]}"
                if mkey_str in seen_scan_keys:
                    continue
                seen_scan_keys.add(mkey_str)

                window_count = min(args.max_objects, len(bm.hit_objects)) if args.max_objects else len(bm.hit_objects)
                kiai_starts = kiai_window_starts(bm, window_count)
                candidate = {
                    "name": name,
                    "text": text,
                    "bm": bm,
                    "source": source,
                    "stars": float(stars),
                    "bucket": b,
                    "mkey_str": mkey_str,
                    "song_key": song_key(bm),
                    "is_val": is_validation(song_key(bm)),
                    "density": note_density(bm),
                    "window_count": window_count,
                    "kiai_starts": kiai_starts,
                }
                if len(candidates_by_bucket[b]) < scan_limits[b]:
                    candidates_by_bucket[b].append(candidate)
                if kiai_starts and len(kiai_candidates_by_bucket[b]) < kiai_scan_limits[b]:
                    kiai_candidates_by_bucket[b].append(candidate)
            except Exception:
                continue

    total_candidates = sum(len(v) for v in candidates_by_bucket.values())
    print(f"Collected {total_candidates} candidates to generate: "
          f"{ {b: len(candidates_by_bucket[b]) for b in ('<3', '3-4.5', '4.5-6', '6+')} }", flush=True)

    # Put Kiai-eligible maps first and keep extra candidates available if placement fails.
    all_candidates = []
    for b in ("<3", "3-4.5", "4.5-6", "6+"):
        pool = candidates_by_bucket[b]
        kiai_pool = list(kiai_candidates_by_bucket[b])
        rng.shuffle(pool)
        rng.shuffle(kiai_pool)
        kiai_keys = {candidate["mkey_str"] for candidate in kiai_pool}
        ordered = kiai_pool + [candidate for candidate in pool if candidate["mkey_str"] not in kiai_keys]
        if len(kiai_pool) < kiai_needed[b]:
            print(f"WARNING: {b} has only {len(kiai_pool)}/{kiai_needed[b]} eligible Kiai windows.", flush=True)
        if len(ordered) < needed[b]:
            print(f"WARNING: {b} has only {len(ordered)}/{needed[b]} source maps.", flush=True)
        all_candidates.extend(ordered)

    by_song = defaultdict(list)
    for c in all_candidates:
        by_song[c["song_key"]].append(c)

    song_keys = list(by_song.keys())
    kiai_songs = [key for key in song_keys if any(diff["kiai_starts"] for diff in by_song[key])]
    other_songs = [key for key in song_keys if key not in set(kiai_songs)]
    rng.shuffle(kiai_songs)
    rng.shuffle(other_songs)
    song_keys = kiai_songs + other_songs

    manifest_file = open(manifest_path, "a", encoding="utf-8")
    generated_count = total_existing
    start_time = time.time()

    print(f"Beginning paired generation across {len(song_keys)} songs (max {args.max_objects} objs/map)...", flush=True)

    try:
        for song_idx, s_key in enumerate(song_keys):
            diffs = by_song[s_key]
            # Load audio features fast
            try:
                features = get_audio_features(diffs[0]["source"])
            except Exception as e:
                # Corrupt audio, skip
                continue

            song_folder_name = sanitize_filename(f"{diffs[0]['bm'].artist} - {diffs[0]['bm'].title}")[:60]
            song_out_dir = out_dir / song_folder_name
            song_out_dir.mkdir(parents=True, exist_ok=True)

            for diff in diffs:
                if diff["mkey_str"] in existing_keys:
                    continue

                b = diff["bucket"]
                if bucket_counts[b] >= args.target_per_bucket:
                    continue

                bm = diff["bm"]
                stars = diff["stars"]
                density = diff["density"]
                preset = replace(preset_for_stars(stars), slider_multiplier=bm.slider_multiplier)

                timing = constant_timing(bm)
                if timing is None:
                    uninherited = [tp for tp in bm.timing_points if tp.uninherited and tp.beat_length > 0]
                    if not uninherited:
                        continue
                    timing = TimingEstimate(bpm=60000.0 / uninherited[0].beat_length, offset_ms=uninherited[0].time)

                quarter = make_tick_grid(timing, features.duration * 1000.0, 4)
                window_count = min(args.max_objects, len(bm.hit_objects)) if args.max_objects else len(bm.hit_objects)
                required_kiai_total = int(np.ceil(args.target_per_bucket / 3))
                if kiai_bucket_counts[b] < required_kiai_total and diff["kiai_starts"]:
                    window_start = int(rng.choice(diff["kiai_starts"]))
                else:
                    window_start = int(rng.integers(max(len(bm.hit_objects) - window_count + 1, 1)))
                window_end = window_start + window_count
                source_window = bm.hit_objects[window_start:window_end]
                plan, spinners = human_to_plan(
                    bm, timing, quarter, max_objects=args.max_objects, start_index=window_start
                )
                if len(plan) < 10:
                    continue

                use_rules = (rng.random() < args.rule_ratio)
                placer_name = "rules" if use_rules else "sequence"
                sv_at = lambda at: bm.timing_at(at)[1]

                try:
                    if use_rules:
                        hit_objects = []
                        for _ in range(5):
                            try_plan = deepcopy(plan)
                            placer = Placer(preset, rng, timing.beat_length, 1.0, sv_at=sv_at)
                            hit_objects = placer.place(try_plan)
                            if len(hit_objects) == len(plan) and all(
                                src.kind == out.kind for src, out in zip(
                                    [obj for obj in source_window if obj.kind != "spinner"], hit_objects
                                )
                            ):
                                break
                    else:
                        scores = np.zeros(len(quarter.times), dtype=np.float32)
                        seq_placer = SequencePlacer(seq_model, preset, features, timing,
                                                    {"density": density, "stars": stars},
                                                    None, scores, quarter, 0.5, rng, 0.8, 0.9,
                                                    sv_at=sv_at)
                        out_plan, choices = seq_placer.follow(deepcopy(plan), drop_below=0.0, add_above=1.0)
                        hit_objects = seq_placer.render(out_plan, choices)
                except Exception as e:
                    continue

                # Add spinners
                all_objects = sorted(hit_objects + spinners, key=lambda o: o.time)
                if len(all_objects) != len(source_window) or any(
                    source.kind != output.kind
                    or abs(source.time - output.time) > 0.01
                    or source.slides != output.slides
                    or abs(bm.end_time(source) - (source.time if output.kind == "circle" else bm.end_time(output))) > 2.0
                    for source, output in zip(source_window, all_objects)
                ):
                    continue

                gen_bm = Beatmap(
                    title=bm.title,
                    artist=bm.artist,
                    creator=bm.creator,
                    version=f"{bm.version} [AI-{placer_name}]",
                    audio_filename="audio.mp3",
                    hp=preset.hp,
                    cs=preset.cs,
                    od=preset.od,
                    ar=preset.ar,
                    slider_multiplier=bm.slider_multiplier,
                    beat_divisor=preset.divisor,
                    timing_points=[replace(tp) for tp in bm.timing_points],
                )
                gen_bm.hit_objects = all_objects
                osu_content = gen_bm.to_osu_string()

                safe_ver = sanitize_filename(diff["bm"].version)[:40]
                file_name = f"{safe_ver}_{placer_name}_{generated_count:05d}.osu"
                neg_file_path = song_out_dir / file_name
                neg_file_path.write_text(osu_content, encoding="utf-8")

                human_path_str = str(diff["source"].path.resolve()) if diff["source"].path.is_file() else ""
                manifest_entry = {
                    "negative_path": str(neg_file_path.resolve()),
                    "human_path": human_path_str,
                    "human_name": diff["name"],
                    "audio_path": human_path_str,
                    "mel_path": str(diff["source"].path.parent / "mel.npy") if (diff["source"].path.parent / "mel.npy").exists() else None,
                    "song_key": diff["song_key"],
                    "map_key_str": diff["mkey_str"],
                    "is_val": diff["is_val"],
                    "stars": float(stars),
                    "bucket": b,
                    "density": float(density),
                    "placer": placer_name,
                    "rhythm_source": "human",
                    "window_start": window_start,
                    "window_count": window_count,
                    "kiai_window": bool(window_start in diff["kiai_starts"]),
                    "slider_multiplier": bm.slider_multiplier,
                    "timing_points": len(bm.timing_points),
                    "objects": len(all_objects),
                }

                manifest_file.write(json.dumps(manifest_entry) + "\n")
                manifest_file.flush()

                existing_keys.add(diff["mkey_str"])
                bucket_counts[b] += 1
                if manifest_entry["kiai_window"]:
                    kiai_bucket_counts[b] += 1
                generated_count += 1

                if generated_count % args.log_every == 0:
                    elapsed = time.time() - start_time
                    maps_done = generated_count - total_existing
                    speed = maps_done / max(elapsed, 0.1)
                    rem = (total_target - generated_count) / max(speed, 0.01)
                    print(f"[{generated_count:4d}/{total_target}] ({speed:.2f} maps/s, ~{rem/60:.1f}m rem) "
                          f"Buckets: {dict(bucket_counts)}", flush=True)

            if generated_count >= total_target and all(bucket_counts[b] >= args.target_per_bucket for b in ("<3", "3-4.5", "4.5-6", "6+")):
                break

    finally:
        manifest_file.close()

    elapsed = time.time() - start_time
    print(f"\nFinished paired generation! Total generated: {generated_count} in {elapsed:.1f}s. Final buckets: {dict(bucket_counts)}", flush=True)
    print(f"Kiai windows: {dict(kiai_bucket_counts)}", flush=True)


def _scan_flat(root: Path):
    for entry in os.scandir(root):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        audio = None
        osus = []
        for sub in os.scandir(entry.path):
            sname = sub.name.lower()
            if sname in ("audio.mp3", "audio.ogg") or sname.endswith(".mp3"):
                audio = Path(sub.path)
            elif sname.endswith(".osu"):
                osus.append(Path(sub.path))
        if audio is not None:
            src = AudioSource(audio.resolve())
            for osu_path in osus:
                try:
                    text = osu_path.read_text(encoding="utf-8", errors="replace")
                    yield str(osu_path.relative_to(root)), text, src, (0, 0)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
