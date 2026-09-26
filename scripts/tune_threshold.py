"""Choose the model's note threshold by the rhythm F1 of generated maps.

The threshold saved during training is tuned on raw per-frame predictions. The
generator snaps notes to the beat grid and applies the difficulty's rules, so its best
threshold can differ. Held-out songs are split in two halves: thresholds are tuned on
one half (per star range) and the result is reported on the other, so the reported F1
is not flattered by the tuning.

    python scripts/tune_threshold.py data best_maps -m model.pt --write
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict

import numpy as np
import torch

from beatmap_ai.dataset import MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, note_density, song_key
from beatmap_ai.evaluate import closest_preset, constant_timing, match_f1
from beatmap_ai.model import load_checkpoint, predict_all
from beatmap_ai.osu import parse_osu
from beatmap_ai.rhythm import plan_objects
from beatmap_ai.style import star_rating

THRESHOLDS = np.round(np.arange(0.20, 0.71, 0.05), 2)
STAR_EDGES = (0, 3, 4.5, 6, 99)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="+")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--max-songs", type=int, default=80)
    parser.add_argument("--device", default="cpu", help='"cuda" to run the model on the GPU')
    parser.add_argument("--write", action="store_true",
                        help="store the tuned thresholds in the checkpoint")
    args = parser.parse_args()
    from beatmap_ai.train import resolve_device
    model, _ = load_checkpoint(args.model, device=resolve_device(args.device))

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
    chosen = [ordered[int(i * len(ordered) / args.max_songs)]
              for i in range(min(args.max_songs, len(ordered)))]

    rows = []  # (half, star bucket, f1 per threshold)
    for n, key in enumerate(chosen):
        features = sources[key].load_features()
        half = int(hashlib.sha1(key.encode()).hexdigest()[-1], 16) % 2
        for bm, stars in songs[key]:
            timing = constant_timing(bm)
            density = note_density(bm)
            grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points if tp.uninherited]
            cond = {"density": density}
            if stars is not None:
                cond["stars"] = stars
            out = predict_all(model, features, grid, cond)
            truth = np.array([o.time for o in bm.hit_objects])
            preset = closest_preset(density)
            f1 = []
            for th in THRESHOLDS:
                plan = plan_objects(features, timing, preset, np.random.default_rng(0),
                                    out["note"], out["slider"], float(th), "threshold",
                                    out.get("sustain"), out.get("spacing"))
                f1.append(match_f1(np.array([p.time for p in plan]), truth))
            bucket = int(np.searchsorted(STAR_EDGES, stars if stars is not None else 4.0, "right") - 1)
            rows.append((half, bucket, np.array(f1)))
        print(f"[{n + 1}/{len(chosen)}] {songs[key][0][0].artist} - {songs[key][0][0].title}", flush=True)

    def mean_f1(half, bucket=None):
        sel = [f for h, b, f in rows if h == half and (bucket is None or b == bucket)]
        return np.mean(sel, axis=0) if sel else None

    print("\nthreshold  " + " ".join(f"{t:5.2f}" for t in THRESHOLDS))
    for half in (0, 1):
        print(f"half {half}    " + " ".join(f"{v:5.3f}" for v in mean_f1(half)))
    tuned, report = {}, []
    for b in range(len(STAR_EDGES) - 1):
        tune = mean_f1(0, b)
        test = mean_f1(1, b)
        if tune is None:
            continue
        best = int(np.argmax(tune))
        tuned[f"{STAR_EDGES[b]}"] = float(THRESHOLDS[best])
        label = f"{STAR_EDGES[b]}-{STAR_EDGES[b + 1]}*" if STAR_EDGES[b + 1] < 99 else f"{STAR_EDGES[b]}+*"
        if test is not None:
            report.append((label, THRESHOLDS[best], test[best], test.max()))
    print("\nstars      tuned threshold   F1 on the other half (best possible there)")
    for label, th, f1, best in report:
        print(f"{label:10} {th:15.2f}   {f1:.3f} ({best:.3f})")
    # Overall: tuned per bucket, measured on the other half.
    chosen_f1 = []
    for h, b, f in rows:
        if h == 1 and str(STAR_EDGES[b]) in tuned:
            chosen_f1.append(f[list(THRESHOLDS).index(tuned[str(STAR_EDGES[b])])])
    print(f"\noverall rhythm F1 on the held-out half with tuned thresholds: {np.mean(chosen_f1):.3f}"
          f" ({len(chosen_f1)} maps)")
    if args.write:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        ckpt["thresholds_by_stars"] = tuned
        torch.save(ckpt, args.model)
        print(f"saved thresholds {tuned} to {args.model}")


if __name__ == "__main__":
    main()
