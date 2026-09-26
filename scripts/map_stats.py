"""Compare generated maps with human maps of the same (held-out) songs.

For every validation song, each human osu!standard difficulty is regenerated with the
human's own timing and a similar difficulty preset, then both are described by:

* rhythm mix: share of gaps of 1/4, 1/2, 1, 2 and more beats between objects
* monotony: how often a gap equals the previous one (a "metronome" map is near 100%)
* rhythm variety: number of distinct 4-object gap patterns per 100 objects
* notes off the main beat (on 1/2 or 1/4 positions)
* spacing: distance per beat of gap, and its variation
* sliders: share of objects, and variety of slider lengths
* movement: how sharply the cursor turns between objects (straight on, sharp turns,
  back and forth), how often the same turn repeats (patterns), stacks and objects
  that cover a recent one

    python scripts/map_stats.py data best_maps -m beatmap_ai/models/rhythm.pt
"""

from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict

import numpy as np

from beatmap_ai.dataset import MIN_OBJECTS, is_validation, iter_beatmaps, map_key, note_density, song_key
from beatmap_ai.evaluate import closest_preset, constant_timing
from beatmap_ai.difficulty import stars_for_density
from beatmap_ai.generator import assign_combos, inference_device
from beatmap_ai.placement import Placer
from beatmap_ai.rhythm import plan_objects


def describe(bm, beat_length: float, offset: float) -> dict[str, float]:
    objs = [o for o in bm.hit_objects if o.kind != "spinner"]
    starts = np.array([o.time for o in objs])
    ends = np.array([bm.end_time(o) for o in objs])
    gaps = (starts[1:] - starts[:-1]) / beat_length
    snapped = np.round(gaps * 4) / 4
    mix = {f"gap {k}": float(np.mean(sel)) for k, sel in (
        ("1/4", snapped <= 0.25), ("1/2", snapped == 0.5), ("3/4", snapped == 0.75),
        ("1", snapped == 1.0), ("more", snapped > 1.0))}
    same = float(np.mean(snapped[1:] == snapped[:-1]))
    patterns = {tuple(snapped[i:i + 3]) for i in range(len(snapped) - 2)}
    phase = ((starts - offset) / beat_length) % 1.0
    off_beat = float(np.mean(np.minimum(phase, 1 - phase) > 0.1))
    # Distance from the previous object's end to the next object's start, per beat.
    dist, per_beat = [], []
    for a, b, oa in zip(objs, objs[1:], objs):
        end = (oa.curve_points[-1] if oa.kind == "slider" and oa.curve_points else (oa.x, oa.y))
        d = math.dist(end, (b.x, b.y))
        gap = (b.time - bm.end_time(oa)) / beat_length
        if 0.2 <= gap <= 1.05:
            dist.append(d)
            per_beat.append(d / max(gap, 0.25))
    # Turning angle between consecutive movements (within a beat, longer than 20 px).
    turns, repeat_turn = [], []
    moves = []
    for a, b in zip(objs, objs[1:]):
        end = (a.curve_points[-1] if a.kind == "slider" and a.curve_points and a.slides % 2 else (a.x, a.y))
        gap = (b.time - bm.end_time(a)) / beat_length
        moves.append((b.x - end[0], b.y - end[1], gap))
    for (ax, ay, ga), (bx, by, gb) in zip(moves, moves[1:]):
        if ga <= 1.05 and gb <= 1.05 and math.hypot(ax, ay) > 20 and math.hypot(bx, by) > 20:
            turn = math.degrees(abs(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))
            turns.append(turn)
    turns = np.array(turns)
    if len(turns) > 2:
        repeat_turn = np.abs(np.diff(turns)) < 15
    stacks = sum(1 for dx, dy, g in moves if math.hypot(dx, dy) < 5 and g <= 1.05)
    radius = 54.4 - 4.48 * bm.cs
    overlaps = 0
    for i, b in enumerate(objs):
        for a in objs[max(0, i - 8):max(0, i - 1)]:
            if b.time - a.time < 1000 and 5 < math.dist((a.x, a.y), (b.x, b.y)) < radius:
                overlaps += 1
                break
    # Runs of objects 1/4 beat (or less) apart: doubles, triples, bursts, streams.
    runs, run = [], 1
    for g in snapped:
        if g <= 0.25:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    runs = np.array(runs)
    per100 = 100.0 / max(len(objs), 1)
    sliders = [o for o in objs if o.kind == "slider"]
    lengths = np.round((ends - starts)[[o.kind == "slider" for o in objs]] / beat_length * 4) / 4
    return {
        **mix,
        "same gap as before": same,
        "3-gap patterns /100 obj": 100 * len(patterns) / max(len(objs), 1),
        "notes off the beat": off_beat,
        "spacing px/beat (median)": float(np.median(per_beat)) if per_beat else 0.0,
        "spacing variation (cv)": float(np.std(dist) / (np.mean(dist) + 1e-6)) if dist else 0.0,
        "slider share": len(sliders) / max(len(objs), 1),
        "slider length kinds": float(len(set(lengths.tolist()))),
        "doubles /100 obj": float((runs == 2).sum() * per100),
        "triples /100 obj": float((runs == 3).sum() * per100),
        "bursts 4-5 /100 obj": float(((runs >= 4) & (runs <= 5)).sum() * per100),
        "streams 6+ /100 obj": float((runs >= 6).sum() * per100),
        "turns: straight <30deg": float(np.mean(turns < 30)) if len(turns) else 0.0,
        "turns: sharp >120deg": float(np.mean(turns > 120)) if len(turns) else 0.0,
        "turns: back+forth >160": float(np.mean(turns > 160)) if len(turns) else 0.0,
        "same turn as before": float(np.mean(repeat_turn)) if len(repeat_turn) else 0.0,
        "stacks": stacks / max(len(objs), 1),
        "covers a recent object": overlaps / max(len(objs), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="+")
    parser.add_argument("-m", "--model")
    parser.add_argument("--selection", default="threshold", choices=("threshold", "density"))
    parser.add_argument("--max-maps", type=int, default=60)
    parser.add_argument("--threshold", type=float, help="override the checkpoint's threshold")
    parser.add_argument("--placement", help="placement model (default: rule-based placement)")
    parser.add_argument("--temperature", type=float, default=0.9, help="placement sampling temperature")
    args = parser.parse_args()

    model, threshold = None, 0.5
    if args.model:
        from beatmap_ai.model import load_checkpoint, predict_all
        model, threshold = load_checkpoint(args.model, device=inference_device())
        threshold = args.threshold or threshold

    placement = None
    if args.placement:
        from beatmap_ai.placement_model import LearnedPlacer, load_placement
        placement = load_placement(args.placement, device=inference_device())

    maps, sources, seen = defaultdict(list), {}, set()
    for folder in args.data:
        for _, bm, source in iter_beatmaps(folder):
            key = song_key(bm)
            if (bm.mode == 0 and len(bm.hit_objects) >= MIN_OBJECTS and is_validation(key)
                    and map_key(bm) not in seen and constant_timing(bm) is not None):
                seen.add(map_key(bm))
                if sources.setdefault(key, source) == source:
                    maps[key].append(bm)
    human, generated = defaultdict(list), defaultdict(list)
    done = 0
    for key in sorted(maps):
        features = sources[key].load_features()
        for bm in maps[key]:
            if done >= args.max_maps:
                break
            timing = constant_timing(bm)
            preset = closest_preset(note_density(bm))
            out = {}
            if model is not None:
                grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points if tp.uninherited]
                out = predict_all(model, features, grid, note_density(bm))
            rng = np.random.default_rng(0)
            plan = plan_objects(features, timing, preset, rng, out.get("note"), out.get("slider"),
                                threshold, args.selection, out.get("sustain"), out.get("spacing"))
            if len(plan) < 10:
                continue
            assign_combos(plan, preset, timing.beat_length)
            gen = type(bm)(slider_multiplier=preset.slider_multiplier, timing_points=bm.timing_points[:1])
            gen.timing_points = [tp for tp in bm.timing_points if tp.uninherited][:1]
            if placement is not None:
                placer = LearnedPlacer(placement, preset, features.mel,
                                       {"density": note_density(bm), "stars": stars_for_density(note_density(bm))},
                                       rng, temperature=args.temperature, offset_ms=timing.offset_ms)
                gen.hit_objects = placer.render(plan, placer.sample(copy.deepcopy(plan)))
            else:
                gen.hit_objects = Placer(preset, rng, timing.beat_length).place(plan)
            gen.cs = preset.cs
            for name, value in describe(bm, timing.beat_length, timing.offset_ms).items():
                human[name].append(value)
            for name, value in describe(gen, timing.beat_length, timing.offset_ms).items():
                generated[name].append(value)
            done += 1
    print(f"{done} difficulties from {len(maps)} held-out songs"
          f" ({'model ' + args.model + ', ' + args.selection if model else 'onset heuristics'})")
    print(f"{'':28}{'human':>10}{'generated':>12}")
    for name in human:
        print(f"{name:28}{np.mean(human[name]):10.3f}{np.mean(generated[name]):12.3f}")


if __name__ == "__main__":
    main()
