"""Slider playability stats (share, chains, quick notes after a slider end, off-beat
sliders) of the .osu files in a folder next to ranked maps per star range. Torch-free.

    python scripts/slider_stats.py Vergleich/7-neueste
"""
import sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
from beatmap_ai.osu import parse_osu
from beatmap_ai.style import star_rating
from beatmap_ai.dataset import iter_beatmap_texts

def stats(bm):
    objs = [o for o in bm.hit_objects if o.kind != 'spinner']
    n = len(objs)
    sl = [o for o in objs if o.kind == 'slider']
    beat = lambda o: bm.timing_at(o.time)[0]
    # after-slider gap (slider end -> next note), in beats
    after = []
    for a, b in zip(objs, objs[1:]):
        if a.kind == 'slider':
            after.append((b.time - bm.end_time(a)) / beat(a))
    after = np.array(after) if after else np.zeros(1)
    # runs of consecutive sliders
    runs, r = [], 0
    for o in objs:
        if o.kind == 'slider': r += 1
        else:
            if r: runs.append(r)
            r = 0
    if r: runs.append(r)
    runs = np.array(runs) if runs else np.zeros(1)
    in_chain = runs[runs >= 3].sum() / max(len(sl), 1)
    # sliders that start on a 1/2 off-beat (not on a beat)
    red = [tp for tp in bm.timing_points if tp.uninherited]
    def phase(t):
        tp = max((p for p in red if p.time <= t + 1), key=lambda p: p.time, default=red[0])
        return ((t - tp.time) / tp.beat_length) % 1.0
    off = np.mean([min(phase(o.time), 1 - phase(o.time)) > 0.1 for o in sl]) if sl else 0
    return dict(share=len(sl) / n, quick_after=np.mean(after < 0.3) if len(sl) else 0,
                chain=in_chain, offbeat=off)

def line(name, s):
    return f"{name:28} Slider {s['share']:4.0%} | Ketten>=3 {s['chain']:4.0%} | Note <1/4 Beat nach Slider-Ende {s['quick_after']:4.0%} | Slider auf Offbeat {s['offbeat']:4.0%}"

buckets = [(0, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 99)]
hum = defaultdict(list)
for folder in ['data', 'best_maps']:
    for _, text, _, _ in iter_beatmap_texts(folder):
        if 'Mode: 0' not in text[:2000]: continue
        s = star_rating(text)
        if s is None: continue
        b = next(i for i, (lo, hi) in enumerate(buckets) if lo <= s < hi)
        if len(hum[b]) >= 80: continue
        try: hum[b].append(stats(parse_osu(text)))
        except Exception: pass
for i, (lo, hi) in enumerate(buckets):
    m = {k: np.median([h[k] for h in hum[i]]) for k in hum[i][0]}
    print(line(f"Mensch {lo}-{hi}* (Median)", m))
for f in sorted(Path(sys.argv[1]).glob('*.osu')):
    t = f.read_text(encoding='utf-8')
    print(line(f"{f.stem} {star_rating(t):.1f}*", stats(parse_osu(t))))
