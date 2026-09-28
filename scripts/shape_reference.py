"""How often human mappers turn a same-rhythm run of circles into a jump shape, and which
shape, per star level and rhythm. Written to beatmap_ai/shape_reference.json, which the
generator's shape plan (v4) draws from. Torch-free.

    python scripts/shape_reference.py --per-bucket 400
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beatmap_ai.dataset import is_validation, iter_beatmap_texts, song_key  # noqa: E402
from beatmap_ai.jump_shapes import SHAPES, gap_class, object_shapes, rhythm_runs  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.placement_data import map_objects  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402
from night_phase import data_dirs  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "beatmap_ai" / "shape_reference.json"
COVERED = 0.6  # a rhythm run counts as a jump run when this share of it is in jump shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bucket", type=int, default=400, help="maps per whole star")
    args = parser.parse_args()
    maps = Counter()
    runs = defaultdict(Counter)  # (stars, gap) -> {"runs", "jump"}
    shapes = defaultdict(Counter)  # (stars, gap) -> shape -> runs
    for folder in data_dirs():
        for name, text, _, _ in iter_beatmap_texts(folder):
            if "Mode: 0" not in text[:3000]:
                continue
            stars = star_rating(text)
            if stars is None:
                continue
            level = int(min(stars, 9))
            if maps[level] >= args.per_bucket:
                continue
            try:
                bm = parse_osu(text)
                if is_validation(song_key(bm)):
                    continue
                objects = map_objects(bm)
            except Exception:
                continue
            maps[level] += 1
            labels = object_shapes(objects, bm.cs)
            for run in rhythm_runs(objects):
                key = f"{level}/{gap_class(objects, run)}"
                runs[key]["runs"] += 1
                inside = labels[run]
                if np.mean(inside > 0) < COVERED:
                    continue
                runs[key]["jump"] += 1
                named = Counter(SHAPES[i] for i in inside if i > 0)
                shapes[key][named.most_common(1)[0][0]] += 1
        if all(maps[level] >= args.per_bucket for level in range(1, 8)):
            break
    table = {}
    for key in sorted(runs):
        total = sum(shapes[key].values())
        table[key] = {"runs": runs[key]["runs"],
                      "jump_share": runs[key]["jump"] / max(runs[key]["runs"], 1),
                      "shapes": {s: shapes[key][s] / max(total, 1) for s in SHAPES[1:] if shapes[key][s]}}
    OUT.write_text(json.dumps({"maps_per_star": dict(sorted(maps.items())), "covered": COVERED,
                               "table": table}, indent=1), encoding="utf-8")
    print(f"maps per star: {dict(sorted(maps.items()))}")
    for key, row in table.items():
        top = ", ".join(f"{s} {v:.0%}" for s, v in sorted(row["shapes"].items(), key=lambda kv: -kv[1])[:5])
        print(f"{key:5} runs {row['runs']:6}  jump {row['jump_share']:5.0%}  {top}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
