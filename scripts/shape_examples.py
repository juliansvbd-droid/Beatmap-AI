"""Share of jumps per shape (beatmap_ai.jump_shapes) by star level, human vs generated maps,
and example places (map, time) to check the shape detection in the osu! editor.

    python scripts/shape_examples.py --human data best_maps --files Vergleich/7-neueste
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_ai.jump_shapes import SHAPES, object_shapes  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.placement_data import T, map_objects  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402

BUCKETS = ((0, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 99))


def texts(folder: Path, generated: bool):
    if generated:
        for f in folder.rglob("*.osu"):
            yield f.name, f.read_text(encoding="utf-8", errors="replace")
    else:
        from beatmap_ai.dataset import iter_beatmap_texts
        for name, text, _, _ in iter_beatmap_texts(folder):
            yield name, text


def gather(folders, generated: bool, per_bucket: int):
    counts = defaultdict(Counter)
    maps = Counter()
    examples = defaultdict(list)
    for folder in folders:
        for name, text in texts(Path(folder), generated):
            if "Mode: 0" not in text[:3000]:
                continue
            stars = star_rating(text)
            if stars is None:
                continue
            b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= stars < hi)
            if maps[b] >= per_bucket:
                continue
            try:
                bm = parse_osu(text)
                objects = map_objects(bm)
            except Exception:
                continue
            labels = object_shapes(objects, bm.cs)
            maps[b] += 1
            counts[b]["objects"] += len(objects)
            for idx in np.flatnonzero(labels):
                counts[b][SHAPES[labels[idx]]] += 1
            for shape_id in set(labels[labels > 0].tolist()):
                shape = SHAPES[shape_id]
                if len(examples[shape]) < 4 and not generated:
                    first = int(np.flatnonzero(labels == shape_id)[0])
                    examples[shape].append(f"{name[:70]} @ {int(objects[first, T])} ms ({stars:.1f}*)")
    return counts, maps, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", nargs="*", default=[])
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--per-bucket", type=int, default=400)
    args = parser.parse_args()
    sets = {}
    examples = {}
    if args.human:
        c, m, examples = gather(args.human, False, args.per_bucket)
        sets["Mensch"] = (c, m)
    for folder in args.files:
        c, m, _ = gather([folder], True, 10 ** 9)
        sets[Path(folder).name] = (c, m)
    shapes = [s for s in SHAPES if s != "none"]
    for b, (lo, hi) in enumerate(BUCKETS):
        print(f"\n### {lo:g}-{hi:g} Sterne   (Objekte in Sprungfolgen, dann Anteil je Form daran)")
        print(f"{'':16}{'Maps':>5}{'in Folgen':>10}" + "".join(f"{s[:9]:>10}" for s in shapes))
        for name, (counts, maps) in sets.items():
            if not maps.get(b):
                continue
            in_runs = max(sum(counts[b][s] for s in shapes), 1)
            share = in_runs / max(counts[b]["objects"], 1)
            print(f"{name[:16]:16}{maps[b]:>5}{share:>10.1%}"
                  + "".join(f"{counts[b][s] / in_runs:>10.0%}" for s in shapes))
    if examples:
        print("\nBeispiele (menschliche Maps) zum Nachsehen im Editor:")
        for shape in shapes:
            for line in examples.get(shape, [])[:3]:
                print(f"  {shape:15} {line}")


if __name__ == "__main__":
    main()
