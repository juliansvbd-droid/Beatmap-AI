"""Generate negative beatmap examples (AI-generated maps) for training the critic.

For human beatmaps of training (and held-out validation) songs, generate corresponding
AI maps with matching timing, density and star rating:
- Rhythm planned with the frame rhythm model (plan_objects)
- Placement with SequencePlacer.follow (~75%) and rule-based Placer (~25%)
- Saved as .osu text under D:\\BeatMap-AI-Dataset\\critic_negatives\\...
- Resumable: skips maps already present in manifest or on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beatmap_ai.audio import sample_peak
from beatmap_ai.dataset import (AudioSource, MIN_OBJECTS, is_validation,
                                iter_beatmap_texts, map_key, note_density, song_key)
from beatmap_ai.difficulty import preset_for_stars
from beatmap_ai.evaluate import closest_preset, constant_timing
from beatmap_ai.generator import assign_combos, inference_device
from beatmap_ai.model import load_checkpoint, predict_all, threshold_for
from beatmap_ai.osu import Beatmap, TimingPoint, parse_osu
from beatmap_ai.placement import Placer
from beatmap_ai.rhythm import make_tick_grid, plan_objects
from beatmap_ai.sequence_model import SequencePlacer, load_sequence
from beatmap_ai.style import star_rating
from beatmap_ai.train import resolve_device


def sanitize_filename(name: str) -> str:
    """Replace characters invalid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def fast_iter_beatmap_texts(root: str | Path):
    root = Path(root)
    # Check if folder is the large flat dataset
    if not (root / "tags.json").exists() and not (root / "catalog.json").exists():
        for item in iter_beatmap_texts(root):
            yield item
        return

    # Fast scan for large folder structures like D:/BeatMap-AI-Dataset/<set_id>/
    for entry in os.scandir(root):
        if entry.is_dir() and not entry.name.startswith("."):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="*", default=["data", "best_maps", "D:/BeatMap-AI-Dataset"],
                        help="directories containing source .osz / .osu files")
    parser.add_argument("--out-dir", default=r"D:\BeatMap-AI-Dataset\critic_negatives",
                        help="directory where negative .osu files and manifest are saved")
    parser.add_argument("--target", type=int, default=5200,
                        help="target number of negative maps to generate")
    parser.add_argument("--rhythm", default="beatmap_ai/models/rhythm.pt",
                        help="rhythm model checkpoint")
    parser.add_argument("--sequence", default="beatmap_ai/models/sequence.pt",
                        help="sequence placement model checkpoint")
    parser.add_argument("--rule-ratio", type=float, default=0.25,
                        help="fraction of maps placed with rule-based Placer (~0.25)")
    parser.add_argument("--device", default=None,
                        help="device to use (cuda/cpu), default: resolve_device('cuda')")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log-every", type=int, default=20, help="progress log interval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    existing_manifest = set()
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
                    if neg_path.exists() and neg_path.stat().st_size > 100:
                        existing_manifest.add(entry.get("map_key_str", ""))
                        total_existing += 1
                except Exception:
                    pass

    print(f"Loaded {total_existing} existing negative maps from {manifest_path}", flush=True)
    if total_existing >= args.target:
        print(f"Already reached target of {args.target} maps. Exiting.", flush=True)
        return

    device = resolve_device(args.device) if args.device else inference_device()
    print(f"Using device: {device}", flush=True)
    print(f"Loading sequence model from {args.sequence}...", flush=True)
    seq = load_sequence(args.sequence, device)
    dummy_x = torch.zeros((1, 64, 830), device=device)
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(seq, dummy_x, strict=False)
            traced.config = seq.config
            seq = traced
            print("Successfully JIT-traced sequence model for fast inference.", flush=True)
    except Exception as e:
        print(f"JIT trace fallback: {e}", flush=True)

    print(f"Loading rhythm model from {args.rhythm}...", flush=True)
    rhythm, threshold = load_checkpoint(args.rhythm, device=device)

    rng = np.random.default_rng(args.seed)

    print("Scanning human beatmaps from source folders...", flush=True)
    seen_map_keys = set()
    eligible_maps = []

    for folder in args.data:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        print(f"Scanning {folder_path}...", flush=True)
        for name, text, source, _ in fast_iter_beatmap_texts(folder_path):
            if source.path.suffix == ".npy":
                continue
            try:
                bm = parse_osu(text)
            except Exception:
                continue

            if bm.mode != 0 or len(bm.hit_objects) < MIN_OBJECTS or len(bm.hit_objects) > 320:
                continue

            mkey = map_key(bm)
            mkey_str = f"{mkey[0]}::{mkey[1]}::{mkey[2]}::{mkey[3]}"
            if mkey_str in seen_map_keys:
                continue
            seen_map_keys.add(mkey_str)

            timing = constant_timing(bm)
            if timing is None:
                continue

            stars = star_rating(text)
            if stars is None:
                continue

            key = song_key(bm)
            mel_path = source.path.parent / "mel.npy" if source.path.is_file() else None

            eligible_maps.append({
                "mkey_str": mkey_str,
                "name": name,
                "text": text,
                "source": source,
                "mel_path": str(mel_path) if mel_path and mel_path.exists() else None,
                "song_key": key,
                "is_val": is_validation(key),
                "stars": stars,
                "timing": timing,
                "density": note_density(bm),
                "artist": bm.artist,
                "title": bm.title,
                "version": bm.version,
                "cs": bm.cs,
            })

            # Check if we have collected plenty of candidates
            if len(eligible_maps) >= int(args.target * 1.5):
                break
        if len(eligible_maps) >= int(args.target * 1.5):
            break

    # Group by song to reuse features and avoid redundant audio loads
    by_song = defaultdict(list)
    for m in eligible_maps:
        by_song[m["song_key"]].append(m)

    n_val_diffs = sum(1 for m in eligible_maps if m["is_val"])
    n_train_diffs = len(eligible_maps) - n_val_diffs
    print(f"Found {len(eligible_maps)} eligible human difficulties "
          f"({n_train_diffs} train, {n_val_diffs} val across {len(by_song)} songs).", flush=True)

    songs_ordered = list(by_song.keys())
    rng.shuffle(songs_ordered)

    generated_count = total_existing
    manifest_file = open(manifest_path, "a", encoding="utf-8")

    start_time = time.time()
    try:
        for song_idx, s_key in enumerate(songs_ordered):
            if generated_count >= args.target:
                break

            diffs = by_song[s_key]
            # Check if all diffs for this song are already generated
            remaining_diffs = [d for d in diffs if d["mkey_str"] not in existing_manifest]
            if not remaining_diffs:
                continue

            # Load audio features once per song
            try:
                first_src = remaining_diffs[0]["source"]
                features = first_src.load_features()
            except Exception as e:
                # Corrupt audio or read error, skip song
                continue

            song_folder_name = sanitize_filename(f"{remaining_diffs[0]['artist']} - {remaining_diffs[0]['title']}")[:60]
            song_out_dir = out_dir / song_folder_name
            song_out_dir.mkdir(parents=True, exist_ok=True)

            for diff in remaining_diffs:
                if generated_count >= args.target:
                    break

                diff_key_str = diff["mkey_str"]
                if diff_key_str in existing_manifest:
                    continue

                timing = diff["timing"]
                stars = diff["stars"]
                density = diff["density"]
                preset = preset_for_stars(stars)
                preset_rules = closest_preset(density)

                grid = [(timing.offset_ms, timing.beat_length, 4)]
                try:
                    with torch.inference_mode():
                        outputs = predict_all(rhythm, features, grid, {"density": density, "stars": stars})
                        quarter = make_tick_grid(timing, features.duration * 1000.0, 4)
                        scores = sample_peak(outputs["note"], quarter.score_times, radius=1)

                        frame_plan = plan_objects(features, timing, preset_rules, rng,
                                                  outputs["note"], outputs["slider"],
                                                  threshold_for(rhythm, stars, threshold), "threshold",
                                                  outputs["sustain"], outputs["spacing"], stars)
                        assign_combos(frame_plan, preset_rules, timing.beat_length)

                        if len(frame_plan) < 10:
                            continue

                        use_rules = (rng.random() < args.rule_ratio)
                        placer_name = "rules" if use_rules else "sequence"

                        if use_rules:
                            placer = Placer(preset, rng, timing.beat_length, 1.0)
                            hit_objects = placer.place(frame_plan)
                        else:
                            seq_placer = SequencePlacer(seq, preset, features, timing,
                                                        {"density": density, "stars": stars},
                                                        None, scores, quarter, threshold, rng, 0.8, 0.9)
                            plan, choices = seq_placer.follow(frame_plan)
                            hit_objects = seq_placer.render(plan, choices)

                    if len(hit_objects) < 10:
                        continue

                    gen_bm = Beatmap(
                        title=diff["artist"],
                        artist=diff["title"],
                        version=f"{diff['version']} [AI-{placer_name}]",
                        audio_filename="audio.mp3",
                        hp=preset.hp,
                        cs=preset.cs,
                        od=preset.od,
                        ar=preset.ar,
                        slider_multiplier=preset.slider_multiplier,
                        beat_divisor=preset.divisor,
                        timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)],
                    )
                    gen_bm.hit_objects = hit_objects
                    osu_content = gen_bm.to_osu_string()

                    safe_ver = sanitize_filename(diff["version"])[:40]
                    file_name = f"{safe_ver}_{placer_name}_{generated_count:05d}.osu"
                    neg_file_path = song_out_dir / file_name
                    neg_file_path.write_text(osu_content, encoding="utf-8")

                    manifest_entry = {
                        "negative_path": str(neg_file_path.resolve()),
                        "human_path": str(diff["source"].path.resolve()) if diff["source"].path.is_file() else "",
                        "human_name": diff["name"],
                        "audio_path": str(diff["source"].path.resolve()),
                        "mel_path": diff["mel_path"],
                        "song_key": diff["song_key"],
                        "map_key_str": diff_key_str,
                        "is_val": diff["is_val"],
                        "stars": float(stars),
                        "density": float(density),
                        "placer": placer_name,
                        "objects": len(hit_objects),
                    }
                    manifest_file.write(json.dumps(manifest_entry) + "\n")
                    manifest_file.flush()

                    existing_manifest.add(diff_key_str)
                    generated_count += 1

                    if generated_count % args.log_every == 0 or generated_count == args.target:
                        elapsed = time.time() - start_time
                        per_sec = (generated_count - total_existing) / max(elapsed, 0.001)
                        print(f"[{generated_count}/{args.target}] {diff['artist']} - {diff['title']} "
                              f"({placer_name}, {len(hit_objects)} objs) - {per_sec:.2f} maps/s", flush=True)

                except Exception as e:
                    # Skip failure and keep moving
                    continue

    finally:
        manifest_file.close()

    total_time = time.time() - start_time
    print(f"\nFinished! Total negative maps: {generated_count} (generated in {total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
