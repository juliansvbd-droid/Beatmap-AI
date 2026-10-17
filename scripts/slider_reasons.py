"""Why the rhythm planner makes a circle where the human mapper starts a slider.

For holdout maps (``scripts/rhythm_holdout.py``) the rhythm model builds the generator's
first rhythm (as ``compare_rhythm_models.py``, without the pre-planner's style). The slider decisions of
``rhythm.plan_objects`` are then replayed with the same random numbers, recording for
every circle which rule kept it from becoming a slider:

- ``no_room``: the next note is too close even after a quarter-beat release,
- ``half_beat_rest``: a slider would fit, but the planner leaves half a beat after a
  slider unless ``quick_release`` comes up,
- ``kick``: nothing is held (sustain head) and the slider head is below
  ``kick_confidence``,
- ``below_cut``: the slider head is below ``slider_cut`` (share cap) or 0.5,
- ``spinner``/``last``.

Every human slider head is matched (within 30 ms) to the generated object: slider,
circle with one of the reasons, or no object at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_ai import generator, rhythm  # noqa: E402
from beatmap_ai.audio import sample_peak  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402

REASONS = ("slider", "no_room", "half_beat_rest", "kick", "below_cut", "spinner", "last")


def replay(features, timing, preset, seed, outputs, threshold, stars, plan) -> tuple[list, bool]:
    """(reason per planned non-spinner object, replay matches the real plan)."""
    divisor = preset.divisor
    grid = rhythm.make_tick_grid(timing, features.duration * 1000.0, divisor)
    model_times = grid.score_times if grid.score_times is not None else grid.times
    scores = sample_peak(outputs["note"], model_times, radius=1)
    rms = sample_peak(features.rms, grid.times, radius=2)
    active = rms > rhythm.ACTIVE_RMS
    slider_probs, sustain = outputs["slider"], outputs["sustain"]
    objects = [o for o in plan if o.kind != "spinner"]
    ticks = [o.tick for o in objects]
    intensity = np.clip(sample_peak(features.onset, grid.times + rhythm.ONSET_LAG_MS, radius=2),
                        0.0, 1.0)
    recovery = max(int(round(preset.slider_recovery_beats * divisor)), 1)
    min_slider = max(int(round(preset.min_slider_beats * divisor)), 1)
    held = sample_peak(sustain, model_times, radius=0) > 0.5
    max_slider = max(max(int(round(preset.max_slider_beats * divisor)), min_slider), 4 * divisor)
    level = stars
    kick_confidence = float(np.interp(level, [4.0, 6.0, 7.0], [0.92, 0.8, 0.65]))
    repeat_chance = float(np.interp(level, [2.0, 3.5, 5.0], [0.35, 0.9, 0.5]))
    half_beat = max(int(round(0.5 * divisor)), recovery)
    quick_release = float(np.interp(level, [4.0, 5.5, 6.5], [0.08, 0.15, 0.25]))
    at_notes = sample_peak(slider_probs, grid.times[ticks], radius=1)
    most = float(np.interp(level, [2.0, 3.5, 5.0, 6.0], [0.68, 0.66, 0.55, 0.48]))
    slider_cut = max(0.5, float(np.quantile(at_notes, 1.0 - most)))
    rng = np.random.default_rng(seed)
    reasons, kinds = [], []
    for n, tick in enumerate(ticks):
        if n + 1 >= len(ticks):
            reasons.append("last")
            kinds.append("circle")
            continue
        next_tick = ticks[n + 1]
        gap_beats = (next_tick - tick) / divisor
        rest = recovery if rng.random() < quick_release else half_beat
        room = min(next_tick - tick - rest, max_slider)
        quick_room = min(next_tick - tick - recovery, max_slider)
        run = 1
        while run < room and held[tick + run]:
            run += 1
        length = max(run, min_slider) if room >= min_slider else 0
        head = sample_peak(slider_probs, grid.times[tick:tick + 1], radius=1)[0]
        kick = run == 1 and head < kick_confidence
        if kick:
            length = 0
        local_beat = (float(grid.beat_length_ms[tick]) if grid.beat_length_ms is not None
                      else timing.beat_length)
        if (6 <= gap_beats <= 16 and active[tick:next_tick].mean() > 0.8
                and (intensity[tick + 1:next_tick] > 0.5).mean() < 0.15):
            lead = max(preset.min_gap_beats, 1.0) * local_beat
            start, end = float(grid.times[tick]) + lead, float(grid.times[next_tick]) - lead
            if end - start >= 2 * local_beat:
                reasons.append("spinner")
                kinds.append("circle")
                continue
        if length >= min_slider and head > slider_cut:
            reasons.append("slider")
            kinds.append("slider")
            slides = rhythm.repeat_slides(intensity, tick, length, divisor)
            if slides > 1:
                rng.random()
            continue
        kinds.append("circle")
        if room < min_slider:
            reasons.append("no_room" if quick_room < min_slider else "half_beat_rest")
        elif kick:
            reasons.append("kick")
        else:
            reasons.append("below_cut")
    return reasons, kinds == [o.kind for o in objects]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=generator.BUNDLED_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-stars", type=float, default=3.0)
    parser.add_argument("--max-stars", type=float, default=6.5)
    parser.add_argument("--max-songs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from beatmap_ai.audio import compute_features, load_audio
    from beatmap_ai.difficulty import preset_for_stars
    from beatmap_ai.evaluate import constant_timing
    from beatmap_ai.model import load_checkpoint, predict_all, threshold_for
    device = generator.inference_device()
    model, base_threshold = load_checkpoint(args.model, device=device)
    data = json.loads(args.holdout.read_text(encoding="utf-8"))
    by_song: dict[str, list[dict]] = {}
    for m in data["maps"]:
        if (m["used"] is None and m["constant_timing"] and m["stars"] is not None
                and args.min_stars <= m["stars"] <= args.max_stars):
            by_song.setdefault(m["song"], []).append(m)
    songs = [key for key in data["holdout_order"] if key in by_song][:args.max_songs]
    results = []
    for key in songs:
        features = compute_features(load_audio(by_song[key][0]["audio"]), chroma=True)
        for m in by_song[key]:
            human = parse_osu(Path(m["audio"]).with_name(Path(m["file"]).name)
                              .read_text(encoding="utf-8", errors="replace"))
            timing = constant_timing(human)
            stars = float(m["stars"])
            preset = preset_for_stars(stars)
            threshold = float(np.clip(threshold_for(model, stars, base_threshold), 0.08, 0.85))
            outputs = predict_all(model, features, [(timing.offset_ms, timing.beat_length, 4)],
                                  generator.model_conditions(model, preset, stars, None))
            plan = rhythm.plan_objects(features, timing, preset, np.random.default_rng(args.seed),
                                       outputs["note"], outputs["slider"], threshold,
                                       "threshold", outputs["sustain"], outputs["spacing"],
                                       stars, generator.RHYTHM_VARIETY)
            reasons, exact = replay(features, timing, preset, args.seed, outputs, threshold,
                                    stars, plan)
            ours = [o for o in plan if o.kind != "spinner"]
            times = np.array([o.time for o in ours])
            notes = [o for o in human.hit_objects if o.kind != "spinner"]
            heads = [o.time for o in notes if o.kind == "slider"]
            outcome, detail = Counter(), Counter()
            beat = timing.beat_length
            for i, obj in enumerate(notes):
                if obj.kind != "slider":
                    continue
                t = obj.time
                j = int(np.argmin(np.abs(times - t))) if len(times) else -1
                reason = "no_object" if j < 0 or abs(times[j] - t) > 30 else reasons[j]
                outcome[reason] += 1
                if reason in ("half_beat_rest", "no_room", "kick") and j + 1 < len(times) \
                        and i + 1 < len(notes):
                    # Quarter beats: our next note, the human slider's length and the human's
                    # next note, e.g. "2|2|4" = we tap again after 1/2 beat where the human
                    # holds a 1/2-beat slider and waits a whole beat.
                    ours_gap = round((times[j + 1] - t) / beat * 4)
                    length = round((human.end_time(obj) - t) / beat * 4)
                    human_gap = round((notes[i + 1].time - t) / beat * 4)
                    detail[f"{reason}:{ours_gap}|{length}|{human_gap}"] += 1
            results.append({"song": key, "version": m["version"], "stars": stars,
                            "replay_exact": exact, "ours": dict(Counter(reasons)),
                            "human_slider_heads": len(heads), "human_heads_got": dict(outcome),
                            "missed_gaps": dict(detail),
                            "human_objects": sum(o.kind != "spinner" for o in human.hit_objects),
                            "our_objects": len(ours)})
            print(f"{m['artist']} - {m['title']} [{m['version']}] {stars:.2f}*: exact {exact}; "
                  f"human slider heads {len(heads)} -> {dict(outcome)}", flush=True)
        args.output.write_text(json.dumps({"model": str(args.model), "results": results},
                                          indent=1), encoding="utf-8")
    total, heads, ours_total, gaps = Counter(), 0, Counter(), Counter()
    for r in results:
        total.update(r["human_heads_got"])
        heads += r["human_slider_heads"]
        ours_total.update(r["ours"])
        gaps.update(r["missed_gaps"])
    print("missed heads (reason:our gap|human slider length|human gap, in 1/4 beats):",
          gaps.most_common(12))
    print("human slider heads:", heads, {k: f"{100 * v / max(heads, 1):.1f}%" for k, v in total.items()})
    n = sum(ours_total.values())
    print("our objects:", n, {k: f"{100 * v / max(n, 1):.1f}%" for k, v in ours_total.items()})
    print("replay exact:", sum(r["replay_exact"] for r in results), "/", len(results))


if __name__ == "__main__":
    main()
