"""Compare two rhythm checkpoints on untouched validation maps, rhythm only.

For every holdout map (``scripts/rhythm_holdout.py``) both models build the rhythm the
generator builds first for that star rating (``generate_beatmap``'s ``build(1.0)``:
preset and note threshold for the stars, the pre-planner's style, threshold selection,
same seed), on the human map's timing. No placement and no star search, so both sides
always use the same selection. Reports figures, sliders, pauses and rhythm F1 per
rhythm stage (``audit_rhythm_pipeline.Observer``) against the human map, and checks
whether the shared outputs (note, slider, sustain, spacing) of the two models agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_rhythm_pipeline import STAGES, Observer, _sha  # noqa: E402
from beatmap_ai import generator  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.rhythm_audit import events, summary  # noqa: E402

SHARED = ("note", "slider", "sustain", "spacing")


def rhythm_stages(model, threshold, features, timing, stars, style, seed, truth):
    from beatmap_ai.difficulty import preset_for_stars
    from beatmap_ai.model import predict_all, threshold_for
    preset = preset_for_stars(stars)
    note_threshold = float(np.clip(threshold_for(model, stars, threshold), 0.08, 0.85))
    grid = [(timing.offset_ms, timing.beat_length, 4)]
    outputs = predict_all(model, features, grid,
                          generator.model_conditions(model, preset, stars, style))
    with Observer() as observer:
        generator.plan_objects(features, timing, preset, np.random.default_rng(seed),
                               outputs.get("note"), outputs.get("slider"), note_threshold,
                               "threshold", outputs.get("sustain"), outputs.get("spacing"),
                               stars, generator.RHYTHM_VARIETY, human_constraints=True,
                               figure_probs=outputs.get("figure_link"),
                               figure_threshold=outputs.get("figure_threshold", 0.5))
    plan = observer.plans[observer.plan_order[0]]
    stages = {name: summary(plan[name], timing, truth) for name in STAGES[:6] if name in plan}
    return stages, plan.get("reinforce"), outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=generator.BUNDLED_MODEL)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-stars", type=float, default=3.0)
    parser.add_argument("--max-stars", type=float, default=6.5)
    parser.add_argument("--max-songs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from beatmap_ai.audio import compute_features, load_audio
    from beatmap_ai.evaluate import constant_timing
    from beatmap_ai.model import load_checkpoint
    from beatmap_ai.planner import load_planner, plan_song
    device = generator.inference_device()
    models = {name: load_checkpoint(path, device=device)
              for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    planner = load_planner(Path(generator.__file__).parent / "models" / "planner.pt",
                           device=device)
    data = json.loads(args.holdout.read_text(encoding="utf-8"))
    by_song: dict[str, list[dict]] = {}
    for m in data["maps"]:
        if (m["used"] is None and m["constant_timing"] and m["stars"] is not None
                and args.min_stars <= m["stars"] <= args.max_stars):
            by_song.setdefault(m["song"], []).append(m)
    songs = [key for key in data["holdout_order"] if key in by_song][:args.max_songs]
    meta = {"baseline": str(args.baseline), "baseline_sha256": _sha(args.baseline),
            "candidate": str(args.candidate), "candidate_sha256": _sha(args.candidate),
            "seed": args.seed, "device": str(device), "songs": len(songs),
            "thresholds": {name: [t, getattr(m, "thresholds_by_stars", {}),
                                  getattr(m, "figure_threshold", None)]
                           for name, (m, t) in models.items()}}
    results = []
    for n, key in enumerate(songs):
        maps = by_song[key]
        features = compute_features(load_audio(maps[0]["audio"]), chroma=True)
        for m in sorted(maps, key=lambda m: m["stars"]):
            human = parse_osu(Path(m["audio"]).with_name(Path(m["file"]).name)
                              .read_text(encoding="utf-8", errors="replace"))
            timing = constant_timing(human)
            stars = float(m["stars"])
            advice = plan_song(planner, features.mel, timing.beat_length, stars, timing.offset_ms)
            suggested = {"jump": max(advice["styles"].get("skillset/jumps", 0.0),
                                     advice["styles"].get("jumps/wide", 0.0)),
                         "stream": advice["styles"].get("skillset/streams", 0.0),
                         "tech": advice["styles"].get("skillset/tech", 0.0),
                         "flow": advice["styles"].get("streams/flow aim", 0.0)}
            style = {k: min(0.4, float(v) * 0.4)
                     for k, v in sorted(suggested.items(), key=lambda item: -item[1])[:2]
                     if v >= 0.35} or None
            truth = np.array([o.time for o in human.hit_objects if o.kind != "spinner"])
            row = {"song": key, "set": m["set"], "version": m["version"], "stars": stars,
                   "bpm": m["bpm"], "human": summary(events(human.hit_objects, human.end_time),
                                                     human, truth)}
            outputs = {}
            for name, (model, threshold) in models.items():
                stages, reinforce, outputs[name] = rhythm_stages(
                    model, threshold, features, timing, stars, style, args.seed, truth)
                row[name] = {"stages": stages, "reinforce": reinforce}
            row["shared_outputs_max_diff"] = max(
                float(np.max(np.abs(outputs["baseline"][k] - outputs["candidate"][k])))
                for k in SHARED if k in outputs["baseline"] and k in outputs["candidate"])
            results.append(row)
            print(f"[{n + 1}/{len(songs)}] {m['artist']} - {m['title']} [{m['version']}] "
                  f"{stars:.2f}*: objects human {row['human']['objects']}, "
                  f"baseline {row['baseline']['stages']['planned']['objects']}, "
                  f"candidate {row['candidate']['stages']['planned']['objects']}; "
                  f"F1 {row['baseline']['stages']['planned']['rhythm_f1']:.3f} / "
                  f"{row['candidate']['stages']['planned']['rhythm_f1']:.3f}", flush=True)
        args.output.write_text(json.dumps({"metadata": meta, "results": results}, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
