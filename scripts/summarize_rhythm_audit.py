"""Markdown tables from the rhythm audit (no PyTorch).

- ``--stages pipeline.json``: mean per map at every stage of the chosen rhythm, plus the
  first rhythm the star search built, and a human reference (the holdout maps
  themselves, or validation maps of similar stars from ``--holdout``).
- ``--compare compare.json``: rhythm-only comparison of two checkpoints per star range.
- ``--pair a/pipeline.json b/pipeline.json``: full pipeline, same maps and seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

COLUMNS = (("objects", "Obj.", 0), ("double_1_4_per_100", "Doubles", 2),
           ("triple_1_4_per_100", "Triples", 2), ("burst_4_8_1_4_per_100", "Bursts 4–8", 2),
           ("stream_9p_1_4_per_100", "Streams 9+", 2), ("slider_pct", "Slider %", 1),
           ("repeat_slider_per_100", "Wdh.-Slider", 2), ("gap_1_4_pct", "1/4-Abst. %", 1),
           ("gap_1_2_pct", "1/2-Abst. %", 1), ("pause_1_per_100", "Pausen ≥1 Beat", 2),
           ("pause_2_per_100", "Pausen ≥2 Beats", 2), ("rhythm_f1", "F1", 3))
STAGE_NAMES = (("initial_planned", "0 Sternsuche: erster Rhythmus (fertig geplant)"),
               ("selected", "1 Tick-Auswahl"), ("after_shape", "2 nach shape_triples"),
               ("after_reinforce", "3 nach reinforce_figures (v5)"),
               ("after_absorb", "4 nach absorb_into_sliders"),
               ("planned", "5 plan_objects fertig (Slider gesetzt)"),
               ("after_follow", "6 nach SequencePlacer.follow"), ("rendered", "7 gerendert"),
               ("final", "8 final (nach Refrain-Kopien)"))
BANDS = ((3.0, 4.5), (4.5, 6.0), (6.0, 6.5))


def _mean(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r is not None and key in r and r[key] is not None]
    return float(np.mean(values)) if values else float("nan")


def _table(rows: list[tuple[str, list[dict]]], columns=COLUMNS) -> str:
    present = [c for c in columns if any(any(c[0] in r for r in group if r) for _, group in rows)]
    lines = ["| | " + " | ".join(c[1] for c in present) + " |",
             "|---|" + "---|" * len(present)]
    for name, group in rows:
        cells = []
        for key, _, digits in present:
            value = _mean(group, key)
            cells.append("–" if not np.isfinite(value) else f"{value:.{digits}f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stages(path: Path, holdout: Path | None, difficulty: str | None) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = [r for r in data["results"] if difficulty is None or r["difficulty"] == difficulty]
    rows = []
    human = [r["source_summary"] for r in results if r["source"] == "human"]
    if human:
        rows.append(("Mensch (dieselben Maps)", human))
    elif holdout is not None:
        maps = json.loads(holdout.read_text(encoding="utf-8"))["maps"]
        lo, hi = {"insane": (4.2, 4.95), "expert": (5.2, 6.2)}.get(difficulty, (4.2, 6.2))
        ref = [m["human"] for m in maps if m["stars"] is not None and lo <= m["stars"] <= hi]
        rows.append((f"Mensch ({len(ref)} Val.-Maps {lo}–{hi}★)", ref))
    order = [key for key, _ in STAGE_NAMES[1:]]
    for key, name in STAGE_NAMES:
        if key == "initial_planned":
            group = [r["star_search"]["initial_planned"] for r in results]
        else:
            # A stage that did not run (shape_triples and reinforce_figures only follow the
            # threshold selection) left the rhythm as the stage before it had it.
            group = []
            for r in results:
                earlier = order[:order.index(key) + 1]
                group.append(next((r["stages"][k] for k in reversed(earlier) if k in r["stages"]),
                                  None))
            if not any(key in r["stages"] for r in results):
                continue
        rows.append((name, group))
    modes = [r["selection_mode"] for r in results]
    built = [r["star_search"]["rhythms_built"] for r in results]
    changed = sum(r["star_search"]["chosen_index"] != 0 for r in results)
    star_error = _mean(results, "star_error")
    title = (f"{path.parent.name} ({difficulty or 'alle'}): {len(results)} Maps; Auswahl "
             f"{modes.count('threshold')}× Schwelle, {modes.count('density')}× Dichte; "
             f"Sternsuche baute im Mittel {np.mean(built):.1f} Rhythmen, {changed}× nicht den "
             f"ersten gewählt; Sternfehler {star_error:.2f}")
    return title + "\n\n" + _table(rows)


def compare(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    results = data["results"]
    for lo, hi in BANDS:
        group = [r for r in results if lo <= r["stars"] < hi or (hi == 6.5 and r["stars"] == hi)]
        if not group:
            continue
        rows = [("Mensch", [r["human"] for r in group]),
                ("v4 (bundled)", [r["baseline"]["stages"]["planned"] for r in group]),
                ("v5-frozen-strict", [r["candidate"]["stages"]["planned"] for r in group])]
        f1 = np.array([[r["baseline"]["stages"]["planned"]["rhythm_f1"],
                        r["candidate"]["stages"]["planned"]["rhythm_f1"]] for r in group])
        delta = f1[:, 1] - f1[:, 0]
        rng = np.random.default_rng(0)
        boot = [rng.choice(delta, len(delta)).mean() for _ in range(2000)]
        added = [r["candidate"]["reinforce"]["added"] for r in group if r["candidate"]["reinforce"]]
        budget = [r["candidate"]["reinforce"]["budget"] for r in group if r["candidate"]["reinforce"]]
        out.append(f"{lo}–{hi}★: {len(group)} Maps aus {len({r['song'] for r in group})} Songs; "
                   f"ΔF1 v5−v4 {delta.mean():+.4f} (95 %-Bootstrap {np.quantile(boot, 0.025):+.4f}"
                   f" … {np.quantile(boot, 0.975):+.4f}), v5 besser/gleich/schlechter "
                   f"{(delta > 0).sum()}/{(delta == 0).sum()}/{(delta < 0).sum()}; "
                   f"reinforce ergänzt {np.sum(added)} Noten (Deckel {np.sum(budget)})\n\n"
                   + _table(rows))
    diff = max(r["shared_outputs_max_diff"] for r in results)
    out.append(f"Größte Abweichung der gemeinsamen Ausgaben (Note/Slider/Sustain/Abstand) "
               f"v4 vs. v5: {diff}")
    return "\n\n".join(out)


def pair(first: Path, second: Path) -> str:
    a = json.loads(first.read_text(encoding="utf-8"))["results"]
    b = json.loads(second.read_text(encoding="utf-8"))["results"]
    lines = ["| Map | Auswahl v4/v5 | Obj. v4/v5/Mensch | Slider % | D/T/B je 100 v4 | v5 | "
             "Mensch | Sterne Ziel / v4 / v5 | F1 v4/v5 |", "|---|---|---|---|---|---|---|---|---|"]
    for x, y in zip(a, b):
        assert (x["song"], x["difficulty"]) == (y["song"], y["difficulty"])
        fx, fy, h = x["stages"]["final"], y["stages"]["final"], x["source_summary"]

        def dtb(s):
            return (f"{s['double_1_4_per_100']:.1f}/{s['triple_1_4_per_100']:.1f}/"
                    f"{s['burst_4_8_1_4_per_100']:.1f}")
        f1 = (f"{fx.get('rhythm_f1', float('nan')):.3f}/{fy.get('rhythm_f1', float('nan')):.3f}")
        lines.append(
            f"| {x['song'][:28]} {x['difficulty']} | {x['selection_mode'][:3]}/"
            f"{y['selection_mode'][:3]} | {fx['objects']}/{fy['objects']}/{h['objects']} | "
            f"{fx['slider_pct']:.0f}/{fy['slider_pct']:.0f}/{h['slider_pct']:.0f} | {dtb(fx)} | "
            f"{dtb(fy)} | {dtb(h)} | {x['target_stars']:.2f} / {x['generated_stars']:.2f} / "
            f"{y['generated_stars']:.2f} | {f1} |")
    rows = [("Mensch", [x["source_summary"] for x in a]),
            ("v4 final", [x["stages"]["final"] for x in a]),
            ("v5 final", [y["stages"]["final"] for y in b])]
    errors = (f"Sternfehler im Mittel v4 {_mean(a, 'star_error'):.3f}, v5 {_mean(b, 'star_error'):.3f}; "
              f"gleiche Auswahlart in {sum(x['selection_mode'] == y['selection_mode'] for x, y in zip(a, b))}"
              f"/{len(a)} Paaren")
    return "\n".join(lines) + "\n\n" + errors + "\n\n" + _table(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path)
    parser.add_argument("--difficulty")
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--pair", type=Path, nargs=2)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.stages:
        print(stages(args.stages, args.holdout, args.difficulty))
    if args.compare:
        print(compare(args.compare))
    if args.pair:
        print(pair(*args.pair))


if __name__ == "__main__":
    main()
