"""Compare maps written by the sequence model with human maps of held-out songs.

For each human difficulty of a validation song the sequence model writes a map with
the human's timing, density and star rating (styles left on Auto). Reported: rhythm
F1 (generated vs. human note times, +-30 ms) per star range, and the same rhythm and
movement statistics as map_stats.py.

    python scripts/eval_sequence.py data best_maps D:/maps -m sequence.pt --rhythm rhythm.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from map_stats import describe  # noqa: E402

from beatmap_ai.audio import sample_peak  # noqa: E402
from beatmap_ai.dataset import MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, note_density, song_key  # noqa: E402
from beatmap_ai.difficulty import preset_for_stars  # noqa: E402
from beatmap_ai.evaluate import STAR_BUCKETS, closest_preset, constant_timing, match_f1, rhythm_f1  # noqa: E402
from beatmap_ai.generator import assign_combos  # noqa: E402
from beatmap_ai.generator import inference_device  # noqa: E402
from beatmap_ai.model import load_checkpoint, predict_all, threshold_for  # noqa: E402
from beatmap_ai.osu import Beatmap, TimingPoint, parse_osu  # noqa: E402
from beatmap_ai.rhythm import make_tick_grid, plan_objects  # noqa: E402
from beatmap_ai.sequence_model import SequencePlacer, load_sequence  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="+")
    parser.add_argument("-m", "--model", required=True, help="sequence model")
    parser.add_argument("--rhythm", default="beatmap_ai/models/rhythm.pt",
                        help="frame rhythm model (restarts after long pauses)")
    parser.add_argument("--max-songs", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--rhythm-temperature", type=float, default=0.9)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--follow", action="store_true",
                        help="rhythm from the frame model; the sequence model places it and edits doubles")
    parser.add_argument("--drop-below", type=float, default=0.15)
    parser.add_argument("--add-above", type=float, default=0.6)
    parser.add_argument("--critic", default=None, help="critic model checkpoint for Best-of-N")
    parser.add_argument("--candidates", type=int, default=4, help="candidates to evaluate per chunk with critic")
    args = parser.parse_args()

    device = inference_device()
    seq = load_sequence(args.model, device)
    import torch
    dummy_x = torch.zeros((1, 64, 830), device=device)
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(seq, dummy_x, strict=False)
            traced.config = seq.config
            seq = traced
    except Exception:
        pass

    rhythm, threshold = load_checkpoint(args.rhythm, device=device)
    critic = None
    if args.critic:
        from beatmap_ai.critic import load_critic
        critic = load_critic(args.critic, device=device)

    songs, sources, seen = defaultdict(list), {}, set()
    for folder in args.data:
        for _, text, source, _ in iter_beatmap_texts(folder):
            if source.path.suffix == ".npy":
                continue
            bm = parse_osu(text)
            key = song_key(bm)
            if (bm.mode == 0 and len(bm.hit_objects) >= MIN_OBJECTS and is_validation(key)
                    and map_key(bm) not in seen and constant_timing(bm) is not None):
                seen.add(map_key(bm))
                if sources.setdefault(key, source) == source:
                    songs[key].append((bm, star_rating(text)))
    ordered = sorted(songs)
    chosen = [ordered[int(i * len(ordered) / args.max_songs)] for i in range(min(args.max_songs, len(ordered)))]

    f1s, baseline, stars_list = [], [], []
    human, generated = defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(0)
    for n, key in enumerate(chosen):
        features = sources[key].load_features()
        for bm, stars in songs[key]:
            if stars is None:
                continue
            timing = constant_timing(bm)
            preset = preset_for_stars(stars)
            density = note_density(bm)
            grid = [(timing.offset_ms, timing.beat_length, 4)]
            notes = predict_all(rhythm, features, grid, {"density": density, "stars": stars})["note"]
            quarter = make_tick_grid(timing, features.duration * 1000.0, 4)
            scores = sample_peak(notes, quarter.score_times, radius=1)
            placer = SequencePlacer(seq, preset, features, timing, {"density": density, "stars": stars},
                                    None, scores, quarter, threshold, rng, args.temperature,
                                    args.rhythm_temperature, guidance=args.guidance)
            if args.follow:
                preset_rules = closest_preset(density)
                outputs = predict_all(rhythm, features, grid, {"density": density, "stars": stars})
                frame_plan = plan_objects(features, timing, preset_rules, np.random.default_rng(0),
                                          outputs["note"], outputs["slider"],
                                          threshold_for(rhythm, stars, threshold), "threshold",
                                          outputs["sustain"], outputs["spacing"], stars)
                assign_combos(frame_plan, preset_rules, timing.beat_length)
                plan, choices = placer.follow(frame_plan, args.drop_below, args.add_above,
                                              critic=critic, candidates=args.candidates)
            else:
                plan, choices = placer.sample()
            if len(plan) < 10:
                continue
            gen = Beatmap(slider_multiplier=preset.slider_multiplier, cs=preset.cs,
                          timing_points=[TimingPoint(timing.offset_ms, timing.beat_length)])
            gen.hit_objects = placer.render(plan, choices)
            f1s.append(match_f1(np.array([o.time for o in gen.hit_objects]),
                                np.array([o.time for o in bm.hit_objects])))
            stars_list.append(stars)
            # The per-frame model on the same map, for comparison.
            baseline += rhythm_f1(features, bm, timing, rhythm, threshold_for(rhythm, stars, threshold),
                                  stars=stars)
            for name, value in describe(bm, timing.beat_length, timing.offset_ms).items():
                human[name].append(value)
            for name, value in describe(gen, timing.beat_length, timing.offset_ms).items():
                generated[name].append(value)
        print(f"[{n + 1}/{len(chosen)}] {songs[key][0][0].artist} - {songs[key][0][0].title}", flush=True)

    f1s, base, stars_arr = np.array(f1s), np.array(baseline), np.array(stars_list)
    print("\nrhythm F1          sequence  per-frame model")
    print(f"  overall          {f1s.mean():.3f}     {base.mean():.3f}   ({len(f1s)} maps)")
    for lo, hi in STAR_BUCKETS:
        inside = (stars_arr >= lo) & (stars_arr < hi)
        if inside.any():
            label = f"{lo}-{hi}*" if hi < 99 else f"{lo}+*"
            print(f"  {label:15}  {f1s[inside].mean():.3f}     {base[inside].mean():.3f}   ({inside.sum()} maps)")
    print(f"\n{'':28}{'human':>10}{'generated':>12}")
    for name in human:
        print(f"{name:28}{np.mean(human[name]):10.3f}{np.mean(generated[name]):12.3f}")


if __name__ == "__main__":
    main()
