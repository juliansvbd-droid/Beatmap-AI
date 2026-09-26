"""Side-by-side A/B comparison of SequencePlacer: Baseline vs Old Critic vs New Critic.

Evaluates on 40 validation songs across all star levels (<3, 3-4.5, 4.5-6, 6+):
- Rhythm F1
- Critic score
- Movement statistics (turns, same turn as before, spacing CV, overlaps, doubles/triples, sliders)
- Generation time per difficulty
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_stats import describe

from beatmap_ai.audio import sample_peak
from beatmap_ai.critic import load_critic
from beatmap_ai.critic_data import CRITIC_WINDOW, critic_features
from beatmap_ai.dataset import MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, note_density, song_key
from beatmap_ai.difficulty import preset_for_stars
from beatmap_ai.evaluate import closest_preset, constant_timing, match_f1
from beatmap_ai.generator import assign_combos, inference_device
from beatmap_ai.model import load_checkpoint, predict_all, threshold_for
from beatmap_ai.osu import Beatmap, TimingPoint, parse_osu
from beatmap_ai.placement_data import map_objects
from beatmap_ai.rhythm import make_tick_grid, plan_objects
from beatmap_ai.sequence_model import SequencePlacer, load_sequence
from beatmap_ai.style import star_rating
from beatmap_ai.timing import TimingEstimate


def star_bucket(stars: float) -> str:
    if stars < 3.0:
        return "<3"
    if stars < 4.5:
        return "3-4.5"
    if stars < 6.0:
        return "4.5-6"
    return "6+"


def score_beatmap(critic_model, bm: Beatmap, features, stars: float, device) -> float:
    """Evaluate average critic probability for a generated beatmap."""
    if critic_model is None:
        return 0.0
    objs = map_objects(bm)
    if len(objs) < 16:
        return 0.5
    feats = critic_features(objs, features.mel, stars)
    n = len(feats)
    windows = []
    masks = []
    step = 32
    for start in range(0, max(1, n - CRITIC_WINDOW + 1), step):
        w = feats[start:start + CRITIC_WINDOW]
        pad = CRITIC_WINDOW - len(w)
        if pad > 0:
            w_pad = np.pad(w, ((0, pad), (0, 0)), mode="constant")
            m = np.pad(np.ones(len(w), dtype=np.float32), (0, pad), mode="constant")
        else:
            w_pad = w
            m = np.ones(CRITIC_WINDOW, dtype=np.float32)
        windows.append(w_pad)
        masks.append(m)
    if not windows:
        return 0.5
    with torch.inference_mode():
        x_t = torch.from_numpy(np.array(windows, dtype=np.float32)).to(device)
        m_t = torch.from_numpy(np.array(masks, dtype=np.float32)).to(device)
        logits = critic_model(x_t, m_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        return float(np.mean(probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="*", default=["data", "best_maps", "D:/BeatMap-AI-Dataset"])
    parser.add_argument("--sequence", default="beatmap_ai/models/sequence.pt")
    parser.add_argument("--rhythm", default="beatmap_ai/models/rhythm.pt")
    parser.add_argument("--old-critic", default="checkpoints/critic/critic_v1.pt")
    parser.add_argument("--new-critic", default="D:/BeatMap-AI-Dataset/critic_paired.pt")
    parser.add_argument("--max-songs", type=int, default=40, help="max validation songs to evaluate")
    parser.add_argument("--max-diffs-per-song", type=int, default=0,
                        help="max diffs per song; 0 evaluates every difficulty")
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()

    device = inference_device()
    print(f"Loading models on {device}...")
    seq = load_sequence(args.sequence, device)
    dummy_x = torch.zeros((1, 64, 830), device=device)
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(seq, dummy_x, strict=False)
            traced.config = seq.config
            seq = traced
    except Exception:
        pass

    rhythm, threshold = load_checkpoint(args.rhythm, device=device)

    if not Path(args.old_critic).exists():
        parser.error(f"old critic checkpoint not found: {args.old_critic}")
    if not Path(args.new_critic).exists():
        parser.error(f"new critic checkpoint not found: {args.new_critic}")
    old_critic = load_critic(args.old_critic, device=device)
    print(f"Loaded old critic from {args.old_critic}")
    new_critic = load_critic(args.new_critic, device=device)
    print(f"Loaded new critic from {args.new_critic}")

    print(f"Collecting validation beatmaps across up to {args.max_songs} validation songs...")
    by_song = defaultdict(list)
    seen_maps = set()

    for folder in args.data:
        p = Path(folder)
        if not p.exists():
            continue
        for name, text, source, _ in iter_beatmap_texts(p):
            if source.path.suffix == ".npy":
                continue
            try:
                bm = parse_osu(text)
                key = song_key(bm)
                if (bm.mode == 0 and len(bm.hit_objects) >= MIN_OBJECTS
                        and is_validation(key) and map_key(bm) not in seen_maps):
                    seen_maps.add(map_key(bm))
                    stars = star_rating(text)
                    if stars is not None:
                        by_song[key].append((bm, text, source, float(stars)))
            except Exception:
                continue

    val_songs = list(by_song.keys())[:args.max_songs]
    total_diffs = sum(
        len(by_song[k]) if args.max_diffs_per_song <= 0
        else min(len(by_song[k]), args.max_diffs_per_song)
        for k in val_songs
    )
    print(f"Evaluating {len(val_songs)} validation songs ({total_diffs} difficulties) on {device}...")

    # Data structures for collecting metrics
    # Levels: overall and by bucket
    def make_bucket_dict():
        return {
            "human": defaultdict(list),
            "baseline": defaultdict(list),
            "old_critic": defaultdict(list),
            "new_critic": defaultdict(list),
            "f1_base": [],
            "f1_old": [],
            "f1_new": [],
            "score_base_new": [],
            "score_old_new": [],
            "score_new_new": [],
            "time_base": [],
            "time_old": [],
            "time_new": [],
        }

    overall = make_bucket_dict()
    buckets = {b: make_bucket_dict() for b in ("<3", "3-4.5", "4.5-6", "6+")}

    eval_idx = 0
    start_eval_time = time.time()

    for s_idx, skey in enumerate(val_songs):
        diffs = (by_song[skey] if args.max_diffs_per_song <= 0
                 else by_song[skey][:args.max_diffs_per_song])
        try:
            features = diffs[0][2].load_features()
        except Exception:
            continue

        for bm, text, source, stars in diffs:
            eval_idx += 1
            b = star_bucket(stars)
            timing = constant_timing(bm)
            if timing is None:
                uninherited = [tp for tp in bm.timing_points if tp.uninherited and tp.beat_length > 0]
                if not uninherited:
                    continue
                timing = TimingEstimate(bpm=60000.0 / uninherited[0].beat_length, offset_ms=uninherited[0].time)

            density = note_density(bm)
            preset = preset_for_stars(stars)
            preset_rules = closest_preset(density)
            grid = [(timing.offset_ms, timing.beat_length, 4)]

            with torch.inference_mode():
                outputs = predict_all(rhythm, features, grid, {"density": density, "stars": stars})
            quarter = make_tick_grid(timing, features.duration * 1000.0, 4)
            scores = sample_peak(outputs["note"], quarter.score_times, radius=1)

            # Common frame plan
            frame_plan = plan_objects(features, timing, preset_rules, np.random.default_rng(eval_idx),
                                      outputs["note"], outputs["slider"],
                                      threshold_for(rhythm, stars, threshold), "threshold",
                                      outputs["sustain"], outputs["spacing"], stars)
            assign_combos(frame_plan, preset_rules, timing.beat_length)
            if len(frame_plan) < 10:
                continue

            # 1. Baseline generation (no critic)
            placer_base = SequencePlacer(seq, preset, features, timing, {"density": density, "stars": stars},
                                         None, scores, quarter, threshold, np.random.default_rng(eval_idx))
            t0 = time.time()
            with torch.inference_mode():
                plan_b, choices_b = placer_base.follow(list(frame_plan))
            dur_b = time.time() - t0
            gen_b = Beatmap(slider_multiplier=preset.slider_multiplier, cs=preset.cs,
                            timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)])
            gen_b.hit_objects = placer_base.render(plan_b, choices_b)

            # 2. Old Critic generation (if available)
            dur_old = dur_b
            gen_old = gen_b
            if old_critic is not None:
                placer_old = SequencePlacer(seq, preset, features, timing, {"density": density, "stars": stars},
                                            None, scores, quarter, threshold, np.random.default_rng(eval_idx))
                t0 = time.time()
                with torch.inference_mode():
                    plan_o, choices_o = placer_old.follow(list(frame_plan), critic=old_critic, candidates=args.candidates)
                dur_old = time.time() - t0
                gen_old = Beatmap(slider_multiplier=preset.slider_multiplier, cs=preset.cs,
                                  timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)])
                gen_old.hit_objects = placer_old.render(plan_o, choices_o)

            # 3. New Critic generation (if available)
            dur_new = dur_b
            gen_new = gen_b
            if new_critic is not None:
                placer_new = SequencePlacer(seq, preset, features, timing, {"density": density, "stars": stars},
                                            None, scores, quarter, threshold, np.random.default_rng(eval_idx))
                t0 = time.time()
                with torch.inference_mode():
                    plan_n, choices_n = placer_new.follow(list(frame_plan), critic=new_critic, candidates=args.candidates)
                dur_new = time.time() - t0
                gen_new = Beatmap(slider_multiplier=preset.slider_multiplier, cs=preset.cs,
                                  timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)])
                gen_new.hit_objects = placer_new.render(plan_n, choices_n)

            # Compute Rhythm F1 vs Human
            human_times = np.array([o.time for o in bm.hit_objects])
            f1_b = match_f1(np.array([o.time for o in gen_b.hit_objects]), human_times)
            f1_o = match_f1(np.array([o.time for o in gen_old.hit_objects]), human_times)
            f1_n = match_f1(np.array([o.time for o in gen_new.hit_objects]), human_times)

            # Compute New Critic Scores
            eval_critic_model = new_critic or old_critic
            sc_b = score_beatmap(eval_critic_model, gen_b, features, stars, device)
            sc_o = score_beatmap(eval_critic_model, gen_old, features, stars, device)
            sc_n = score_beatmap(eval_critic_model, gen_new, features, stars, device)

            # Compute Movement Stats
            desc_h = describe(bm, timing.beat_length, timing.offset_ms)
            desc_b = describe(gen_b, timing.beat_length, timing.offset_ms)
            desc_o = describe(gen_old, timing.beat_length, timing.offset_ms)
            desc_n = describe(gen_new, timing.beat_length, timing.offset_ms)

            # Record in overall and bucket dicts
            for target_dict in (overall, buckets[b]):
                target_dict["f1_base"].append(f1_b)
                target_dict["f1_old"].append(f1_o)
                target_dict["f1_new"].append(f1_n)
                target_dict["score_base_new"].append(sc_b)
                target_dict["score_old_new"].append(sc_o)
                target_dict["score_new_new"].append(sc_n)
                target_dict["time_base"].append(dur_b)
                target_dict["time_old"].append(dur_old)
                target_dict["time_new"].append(dur_new)
                for k, v in desc_h.items():
                    target_dict["human"][k].append(v)
                for k, v in desc_b.items():
                    target_dict["baseline"][k].append(v)
                for k, v in desc_o.items():
                    target_dict["old_critic"][k].append(v)
                for k, v in desc_n.items():
                    target_dict["new_critic"][k].append(v)

            print(f"[{eval_idx:2d}/{total_diffs}] {bm.artist} - {bm.title} [{bm.version} ({stars:.1f}* | {b})]: "
                  f"Score {sc_b:.3f} -> {sc_n:.3f} | Base {dur_b:.2f}s -> New {dur_new:.2f}s", flush=True)

    print("\n" + "=" * 90)
    print("A/B EVALUATION RESULTS (Step 5 Verification - 40 Validation Songs)")
    print("=" * 90)

    movement_keys = (
        "gap 1/4", "gap 1/2", "gap 3/4", "gap 1", "gap more", "same gap as before",
        "3-gap patterns /100 obj", "notes off the beat", "spacing px/beat (median)",
        "spacing variation (cv)", "slider share", "slider length kinds", "doubles /100 obj",
        "triples /100 obj", "bursts 4-5 /100 obj", "streams 6+ /100 obj",
        "turns: straight <30deg", "turns: sharp >120deg", "turns: back+forth >160",
        "same turn as before", "stacks", "covers a recent object",
    )

    def print_table(title: str, d: dict):
        if not d["f1_base"]:
            print(f"\nNo data for {title}")
            return
        print(f"\n--- {title} (N = {len(d['f1_base'])} maps) ---")
        print(f"{'Metric':<30}{'Human':>14}{'Baseline':>14}{'Old Critic':>14}{'New Critic':>14}")
        print("-" * 86)
        print(f"{'Rhythm F1':<30}{'--':>14}{np.mean(d['f1_base']):14.3f}{np.mean(d['f1_old']):14.3f}{np.mean(d['f1_new']):14.3f}")
        print(f"{'Critic Probability':<30}{'--':>14}{np.mean(d['score_base_new']):14.3%}{np.mean(d['score_old_new']):14.3%}{np.mean(d['score_new_new']):14.3%}")
        print(f"{'Generation Time / Map':<30}{'--':>14}{np.mean(d['time_base']):13.2f}s{np.mean(d['time_old']):13.2f}s{np.mean(d['time_new']):13.2f}s")
        overhead = np.mean(d['time_new']) / max(np.mean(d['time_base']), 0.001)
        print(f"{'Runtime Overhead':<30}{'--':>14}{'1.0x':>14}{np.mean(d['time_old'])/max(np.mean(d['time_base']), 0.001):13.1f}x{overhead:13.1f}x")

        for k in movement_keys:
            if k in d["human"] and d["human"][k]:
                h = np.mean(d["human"][k])
                b_val = np.mean(d["baseline"][k])
                o_val = np.mean(d["old_critic"][k])
                n_val = np.mean(d["new_critic"][k])
                print(f"{k:<30}{h:14.3f}{b_val:14.3f}{o_val:14.3f}{n_val:14.3f}")

    def normalized_movement_error(d: dict, model_key: str) -> float:
        errors = []
        for key in movement_keys:
            if d["human"][key] and d[model_key][key]:
                human_mean = float(np.mean(d["human"][key]))
                model_mean = float(np.mean(d[model_key][key]))
                errors.append(abs(model_mean - human_mean) / max(abs(human_mean), 0.05))
        return float(np.mean(errors)) if errors else float("nan")

    print_table("OVERALL COMPARISON", overall)
    for b in ("<3", "3-4.5", "4.5-6", "6+"):
        print_table(f"STAR BUCKET: {b}", buckets[b])

    print("\nMean normalized absolute movement-stat error vs human (lower is closer):")
    for model_key, label in (("baseline", "Baseline"), ("old_critic", "Old critic"),
                             ("new_critic", "New critic")):
        print(f"  {label}: {normalized_movement_error(overall, model_key):.4f}")

    print("=" * 90)


if __name__ == "__main__":
    main()
