"""Compare generated .osu files (e.g. the same song across model versions) with each other
and with human ranked maps of the same star range. Torch-free, so it can run while a GPU
job is busy.

    python scripts/compare_maps.py Vergleich/1-vor-01b Vergleich/2-critic-v1 --human data
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402

BUCKETS = ((0.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 99.0))
METRICS = (
    ("stars", "Sterne", "{:.2f}"),
    ("ar", "AR", "{:.1f}"),
    ("cs", "CS", "{:.1f}"),
    ("nps", "Noten/s", "{:.2f}"),
    ("sliders", "Slider-Anteil", "{:.0%}"),
    ("repeat", "davon wiederholend", "{:.0%}"),
    ("curved", "davon gebogen", "{:.0%}"),
    ("short_slider", "Slider < 1 Beat", "{:.0%}"),
    ("tiny_slider", "Slider kürzer als 2 Kreise", "{:.0%}"),
    ("round_slider", "Kreisslider (Bogen > 1,6× Sehne)", "{:.0%}"),
    ("slider_px", "Slider-Länge (px, median)", "{:.0f}"),
    ("g_quarter", "Abstand 1/4 Beat", "{:.0%}"),
    ("g_half", "Abstand 1/2 Beat", "{:.0%}"),
    ("g_one", "Abstand 1 Beat", "{:.0%}"),
    ("g_long", "Abstand > 1 Beat", "{:.0%}"),
    ("doubles", "Doubles /100", "{:.1f}"),
    ("triples", "Triples /100", "{:.1f}"),
    ("bursts", "Bursts 4-8 /100", "{:.1f}"),
    ("streams", "Streams 9+ /100", "{:.1f}"),
    ("ds_1", "Distanz/Beat bei 1/1 (px)", "{:.0f}"),
    ("ds_cv", "Streuung der Abstände (CV)", "{:.2f}"),
    ("jump_px", "Sprung 1/2 Beat (px, median)", "{:.0f}"),
    ("stack", "Stacks", "{:.1%}"),
    ("sharp", "scharfe Wendungen >120°", "{:.0%}"),
    ("straight", "gerade <30°", "{:.0%}"),
    ("same_turn", "gleiche Wendung wie davor", "{:.0%}"),
    ("nc_len", "Noten pro Combo", "{:.1f}"),
)


def describe(text: str) -> dict | None:
    bm = parse_osu(text)
    objs = [o for o in bm.hit_objects if o.kind != "spinner"]
    if len(objs) < 30:
        return None
    s = {"stars": star_rating(text) or float("nan"), "ar": bm.ar, "cs": bm.cs}
    dur = (objs[-1].time - objs[0].time) / 1000.0
    s["nps"] = len(objs) / max(dur, 1.0)
    sliders = [o for o in objs if o.kind == "slider"]
    s["sliders"] = len(sliders) / len(objs)
    beat_of = {id(o): bm.timing_at(o.time)[0] for o in objs}
    if sliders:
        s["repeat"] = np.mean([o.slides > 1 for o in sliders])
        s["curved"] = np.mean([o.curve_type in ("B", "P") and len(o.curve_points) >= 2
                               and _chord(o) < 0.9 * o.length for o in sliders])
        s["short_slider"] = np.mean([(bm.end_time(o) - o.time) / o.slides < 0.9 * beat_of[id(o)]
                                     for o in sliders])
        radius = 54.4 - 4.48 * bm.cs
        s["tiny_slider"] = np.mean([o.length < 4 * radius for o in sliders])
        s["round_slider"] = np.mean([o.curve_type in ("B", "P") and len(o.curve_points) >= 2
                                     and o.length > 1.6 * max(_chord(o), 1.0) for o in sliders])
        s["slider_px"] = float(np.median([o.length for o in sliders]))
    else:
        s["repeat"] = s["curved"] = s["short_slider"] = 0.0
        s["tiny_slider"] = s["round_slider"] = 0.0
        s["slider_px"] = float("nan")

    gaps, dists, ds1, jumps = [], [], [], []
    for a, b in zip(objs, objs[1:]):
        beat = beat_of[id(b)]
        g = (b.time - bm.end_time(a)) / beat
        start_gap = (b.time - a.time) / beat
        ex, ey = (a.curve_points[-1] if a.kind == "slider" and a.slides % 2 == 1 and a.curve_points
                  else (a.x, a.y))
        d = math.hypot(b.x - ex, b.y - ey)
        gaps.append(start_gap)
        dists.append((g, d))
        if 0.9 <= g <= 1.1:
            ds1.append(d)
        if 0.45 <= start_gap <= 0.55 and a.kind == "circle":
            jumps.append(d)
    gaps = np.array(gaps)
    s["g_quarter"] = np.mean(gaps < 0.3)
    s["g_half"] = np.mean((gaps >= 0.3) & (gaps < 0.6))
    s["g_one"] = np.mean((gaps >= 0.6) & (gaps < 1.2))
    s["g_long"] = np.mean(gaps >= 1.2)

    # Runs of circles 1/4 beat (or faster) apart: 2 = double, 3 = triple, ...
    runs, run = [], 1
    for a, b, g in zip(objs, objs[1:], gaps):
        if a.kind == "circle" and g < 0.3:
            run += 1
        else:
            if run > 1:
                runs.append(run)
            run = 1
    if run > 1:
        runs.append(run)
    runs = np.array(runs)
    per = 100.0 / len(objs)
    s["doubles"] = np.sum(runs == 2) * per
    s["triples"] = np.sum(runs == 3) * per
    s["bursts"] = np.sum((runs >= 4) & (runs <= 8)) * per
    s["streams"] = np.sum(runs >= 9) * per

    s["ds_1"] = float(np.median(ds1)) if ds1 else float("nan")
    # Spacing consistency: distance per beat for gaps of 1/2 and 1 beat, coefficient of variation.
    per_beat = [d / g for g, d in dists if 0.45 <= g <= 1.1 and d > 5]
    s["ds_cv"] = float(np.std(per_beat) / np.mean(per_beat)) if len(per_beat) > 5 else float("nan")
    s["jump_px"] = float(np.median(jumps)) if jumps else float("nan")
    s["stack"] = np.mean([d < 5 for _, d in dists])

    turns = []
    pts = [(o.x, o.y) for o in objs]
    for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        if math.hypot(*v1) < 20 or math.hypot(*v2) < 20:
            turns.append(None)
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        turns.append(math.degrees(math.atan2(cross, dot)))
    valid = [t for t in turns if t is not None]
    s["sharp"] = np.mean([abs(t) > 120 for t in valid]) if valid else float("nan")
    s["straight"] = np.mean([abs(t) < 30 for t in valid]) if valid else float("nan")
    same = [abs(t1 - t0) < 25 for t0, t1 in zip(turns, turns[1:])
            if t0 is not None and t1 is not None and abs(t0) >= 30]
    s["same_turn"] = np.mean(same) if same else float("nan")
    combos = sum(o.new_combo for o in objs) or 1
    s["nc_len"] = len(objs) / combos
    return s


def _chord(o) -> float:
    fx, fy = o.curve_points[-1]
    return math.hypot(fx - o.x, fy - o.y)


def human_reference(folders: list[str], limit: int) -> dict:
    from beatmap_ai.dataset import iter_beatmap_texts

    by_bucket = defaultdict(list)
    for folder in folders:
        for _, text, _, _ in iter_beatmap_texts(folder):
            if "Mode: 0" not in text[:2000] and "Mode:0" not in text[:2000]:
                continue
            try:
                s = describe(text)
            except Exception:
                continue
            if s is None or math.isnan(s["stars"]):
                continue
            b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= s["stars"] < hi)
            if len(by_bucket[b]) < limit:
                by_bucket[b].append(s)
            if all(len(by_bucket[i]) >= limit for i in range(len(BUCKETS))):
                return by_bucket
    return by_bucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", help="folders with generated .osu files, one per version")
    parser.add_argument("--human", nargs="*", default=[], help="human map folders for the reference")
    parser.add_argument("--per-bucket", type=int, default=80)
    args = parser.parse_args()

    maps = {}
    for folder in args.folders:
        for path in sorted(Path(folder).glob("*.osu")):
            s = describe(path.read_text(encoding="utf-8", errors="replace"))
            if s is not None:
                maps[(path.stem, Path(folder).name)] = s
    ref = human_reference(args.human, args.per_bucket) if args.human else {}

    diffs = sorted({d for d, _ in maps}, key=lambda d: np.nanmean([maps[k]["stars"] for k in maps if k[0] == d]))
    versions = [Path(f).name for f in args.folders]
    for diff in diffs:
        cols = [(v, maps[(diff, v)]) for v in versions if (diff, v) in maps]
        stars = np.nanmean([s["stars"] for _, s in cols])
        b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= stars < hi)
        human = ref.get(b, [])
        lo, hi = BUCKETS[b]
        header = [name for name, _ in cols] + ([f"Mensch {lo:g}-{hi:g}* (n={len(human)})"] if human else [])
        print(f"\n### {diff}\n")
        print("| Merkmal | " + " | ".join(header) + " |")
        print("|---|" + "---|" * len(header))
        for key, label, fmt in METRICS:
            row = [fmt.format(s[key]) if not (isinstance(s[key], float) and math.isnan(s[key])) else "–"
                   for _, s in cols]
            if human:
                vals = np.array([h[key] for h in human], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    row.append(f"{fmt.format(med)} ({fmt.format(q1)}–{fmt.format(q3)})")
                else:
                    row.append("–")
            print(f"| {label} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
