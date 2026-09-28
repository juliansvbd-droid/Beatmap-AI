"""How mappers build jump sections: geometric shapes, and the regularities used instead.

A jump run is a stretch of >= 4 hit circles at the same rhythm (gaps within 12 %) whose
moves are all jumps (> 1.5 circle radii). Per run: side lengths, signed turn angles,
regular shapes (triangle/square/pentagon/hexagon, star, zigzag, line), how even the
spacing is, how "round" the angles are (near 45/60/90 ...), whether the rotation keeps
its direction, whether the path crosses itself, and whether an earlier run's shape comes
back (moved, turned or mirrored). Torch-free.

    python scripts/jump_geometry.py --human data best_maps --files Vergleich/7-neueste
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402

BUCKETS = ((0, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 99))
ROUND = (30.0, 45.0, 60.0, 72.0, 90.0, 108.0, 120.0, 135.0, 144.0, 150.0, 180.0)


def jump_runs(bm):
    objs = [o for o in bm.hit_objects if o.kind != "spinner"]
    radius = 54.4 - 4.48 * bm.cs
    runs, current = [], []
    for a, b in zip(objs, objs[1:]):
        gap = b.time - bm.end_time(a)
        start = (a.curve_points[-1] if a.kind == "slider" and a.slides % 2 == 1 and a.curve_points
                 else (a.x, a.y))
        jump = math.hypot(b.x - start[0], b.y - start[1]) > 1.5 * radius
        same = current and abs(gap - current[-1][2]) <= 0.12 * max(current[-1][2], 1.0)
        if jump and a.kind == "circle" and b.kind == "circle" and (not current or same):
            if not current:
                current = [((a.x, a.y), a.time, gap)]
            current.append(((b.x, b.y), b.time, gap))
        else:
            if len(current) >= 4:
                runs.append([p for p, _, _ in current])
            current = [((a.x, a.y), a.time, gap), ((b.x, b.y), b.time, gap)] if jump and a.kind == b.kind == "circle" else []
    if len(current) >= 4:
        runs.append([p for p, _, _ in current])
    return runs


def turns_of(points):
    out = []
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        out.append(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])))
    return out


def crosses(points):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    segs = list(zip(points, points[1:]))
    n = 0
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            a, b = segs[i]
            c, d = segs[j]
            if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
                n += 1
    return n


def shape(points, sides, turns):
    cv = float(np.std(sides) / max(np.mean(sides), 1e-6))
    t = np.asarray(turns)
    if len(t) == 0:
        return "other"
    if np.all(np.abs(t) < 20):
        return "line"
    signs = np.sign(t[np.abs(t) >= 20])
    alternating = len(signs) >= 2 and np.all(signs[1:] != signs[:-1])
    if alternating and np.std(np.abs(t)) < 20:
        return "zigzag / back and forth"
    same_sign = len(signs) and (np.all(signs > 0) or np.all(signs < 0))
    if same_sign and cv < 0.2 and np.std(np.abs(t)) < 15:
        exterior = float(np.mean(np.abs(t)))
        for name, angle in (("triangle", 120), ("square", 90), ("pentagon", 72), ("hexagon", 60),
                            ("star", 144)):
            if abs(exterior - angle) <= 15:
                return name
        return "regular arc"
    if same_sign:
        return "curve, uneven"
    return "other"


def signature(points):
    """Shape up to position, size, rotation and mirroring: relative sides and |turns|."""
    sides = [math.dist(a, b) for a, b in zip(points, points[1:])]
    scale = max(np.mean(sides), 1e-6)
    return tuple(np.round(np.asarray(sides) / scale, 1)) + tuple(np.round(np.abs(turns_of(points)) / 15))


def describe(bm):
    runs = jump_runs(bm)
    res = {"runs": len(runs), "shapes": Counter(), "side_cv": [], "round_angle": [],
           "rotation_keep": [], "crossing_runs": 0, "repeat_runs": 0, "objects": len(bm.hit_objects)}
    seen = []
    for points in runs:
        sides = [math.dist(a, b) for a, b in zip(points, points[1:])]
        t = turns_of(points)
        res["shapes"][shape(points, sides, t)] += 1
        res["side_cv"].append(float(np.std(sides) / max(np.mean(sides), 1e-6)))
        big = [abs(x) for x in t if abs(x) >= 20]
        res["round_angle"].extend(min(abs(abs(x) - r) for r in ROUND) <= 5 for x in big)
        signs = [np.sign(x) for x in t if abs(x) >= 20]
        if len(signs) >= 2:
            res["rotation_keep"].append(max(signs.count(1), signs.count(-1)) / len(signs))
        res["crossing_runs"] += crosses(points) > 0
        sig = signature(points[:4])
        res["repeat_runs"] += sig in seen
        seen.append(sig)
    return res


def summarise(results):
    runs = sum(r["runs"] for r in results)
    shapes = Counter()
    for r in results:
        shapes.update(r["shapes"])
    cv = [x for r in results for x in r["side_cv"]]
    rnd = [x for r in results for x in r["round_angle"]]
    rot = [x for r in results for x in r["rotation_keep"]]
    return {
        "maps": len(results), "runs": runs,
        "runs_per_1000_objects": 1000 * runs / max(sum(r["objects"] for r in results), 1),
        "shapes": {k: v / max(runs, 1) for k, v in shapes.most_common()},
        "even_spacing": float(np.mean(np.asarray(cv) < 0.15)) if cv else float("nan"),
        "side_cv_median": float(np.median(cv)) if cv else float("nan"),
        "round_angles": float(np.mean(rnd)) if rnd else float("nan"),
        "rotation_kept": float(np.mean(rot)) if rot else float("nan"),
        "crossing": sum(r["crossing_runs"] for r in results) / max(runs, 1),
        "repeated_shape": sum(r["repeat_runs"] for r in results) / max(runs, 1),
    }


def load(paths, limit_per_bucket=None):
    from beatmap_ai.dataset import iter_beatmap_texts
    buckets = defaultdict(list)
    for folder in paths:
        folder = Path(folder)
        texts = ([(f.name, f.read_text(encoding="utf-8", errors="replace")) for f in folder.rglob("*.osu")]
                 if limit_per_bucket is None else
                 ((name, text) for name, text, _, _ in iter_beatmap_texts(folder)))
        for _, text in texts:
            if "Mode: 0" not in text[:3000] and "Mode:0" not in text[:3000]:
                continue
            stars = star_rating(text)
            if stars is None:
                continue
            b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= stars < hi)
            if limit_per_bucket and len(buckets[b]) >= limit_per_bucket:
                continue
            try:
                buckets[b].append(describe(parse_osu(text)))
            except Exception:
                continue
    return buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", nargs="*", default=[])
    parser.add_argument("--files", nargs="*", default=[], help="folders of generated maps, one label each")
    parser.add_argument("--per-bucket", type=int, default=400)
    args = parser.parse_args()
    sets = {"Mensch": load(args.human, args.per_bucket)} if args.human else {}
    for folder in args.files:
        sets[Path(folder).name] = load([folder])
    for b, (lo, hi) in enumerate(BUCKETS):
        print(f"\n### {lo:g}-{hi:g} Sterne")
        for name, buckets in sets.items():
            if not buckets.get(b):
                continue
            s = summarise(buckets[b])
            top = ", ".join(f"{k} {v:.0%}" for k, v in list(s["shapes"].items())[:6])
            print(f"{name:14} Maps {s['maps']:4} | Sprungfolgen/1000 Obj {s['runs_per_1000_objects']:5.1f} | "
                  f"gleiche Abstände {s['even_spacing']:.0%} | runde Winkel {s['round_angles']:.0%} | "
                  f"Drehrichtung gehalten {s['rotation_kept']:.0%} | kreuzt sich {s['crossing']:.0%} | "
                  f"Form wiederholt {s['repeated_shape']:.0%}")
            print(f"{'':14} Formen: {top}")


if __name__ == "__main__":
    main()
