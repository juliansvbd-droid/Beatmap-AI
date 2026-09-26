"""How closely do two human mappers agree on the rhythm of the same song?

Pairs of difficulties of the same song from different mappers, with the same timing
(BPM and offset) and a similar star rating, are compared with the same rhythm F1 the
evaluation uses. This is the ceiling a model can sensibly reach: there is no single
right map.

    python scripts/human_agreement.py data best_maps D:/maps
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np

from beatmap_ai.dataset import MIN_OBJECTS, iter_beatmap_texts, map_key, song_key
from beatmap_ai.evaluate import STAR_BUCKETS, constant_timing, match_f1
from beatmap_ai.osu import parse_osu
from beatmap_ai.style import star_rating


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="+")
    parser.add_argument("--max-star-gap", type=float, default=0.4)
    args = parser.parse_args()

    songs, seen = defaultdict(list), set()
    for folder in args.data:
        for _, text, source, _ in iter_beatmap_texts(folder):
            bm = parse_osu(text)
            if bm.mode != 0 or len(bm.hit_objects) < MIN_OBJECTS or map_key(bm) in seen:
                continue
            seen.add(map_key(bm))
            timing = constant_timing(bm)
            if timing is not None:
                songs[song_key(bm)].append((bm, timing, source.path.parent))
    # Keep songs mapped in more than one beatmap set.
    songs = {k: v for k, v in songs.items() if len({m[2] for m in v}) > 1}

    scores = defaultdict(list)
    for maps in songs.values():
        stars = {}
        for (a, ta, pa), (b, tb, pb) in itertools.combinations(maps, 2):
            if pa == pb or abs(ta.bpm - tb.bpm) > 0.05:
                continue
            beat = ta.beat_length
            shift = (ta.offset_ms - tb.offset_ms) % beat
            if min(shift, beat - shift) > 3:  # Different audio cut or offset.
                continue
            for bm in (a, b):
                if id(bm) not in stars:
                    stars[id(bm)] = star_rating(bm.to_osu_string()) or np.nan
            sa, sb = stars[id(a)], stars[id(b)]
            if not (np.isfinite(sa) and np.isfinite(sb)) or abs(sa - sb) > args.max_star_gap:
                continue
            times_a = np.array([o.time for o in a.hit_objects])
            times_b = np.array([o.time for o in b.hit_objects])
            lo, hi = max(times_a.min(), times_b.min()), min(times_a.max(), times_b.max())
            times_a = times_a[(times_a >= lo) & (times_a <= hi)]
            times_b = times_b[(times_b >= lo) & (times_b <= hi)]
            if min(len(times_a), len(times_b)) < 50:
                continue
            scores[(sa + sb) / 2].append(match_f1(times_a, times_b))
    values = np.array([(s, f) for s, fs in scores.items() for f in fs])
    print(f"{len(values)} pairs from {len(songs)} songs mapped in several sets")
    for lo, hi in STAR_BUCKETS:
        inside = (values[:, 0] >= lo) & (values[:, 0] < hi)
        if inside.any():
            f = values[inside, 1]
            label = f"{lo}-{hi}*" if hi < 99 else f"{lo}+*"
            print(f"{label:8} median {np.median(f):.3f}  (25%-75%: {np.percentile(f, 25):.3f}-"
                  f"{np.percentile(f, 75):.3f}, {inside.sum()} pairs)")
    print(f"all      median {np.median(values[:, 1]):.3f}")


if __name__ == "__main__":
    main()
