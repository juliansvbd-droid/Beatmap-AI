"""Compare osu!standard rhythm and movement patterns without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from beatmap_ai.dataset import is_validation, iter_beatmap_texts, map_key, song_key
from beatmap_ai.osu import Beatmap, parse_osu
from beatmap_ai.patterns import (
    DIFFICULTY_FEATURES,
    PATTERN_TYPES,
    RHYTHM_GROUPS,
    STAR_BUCKETS,
    SUBDIVISIONS,
    analyze_map,
    apply_reference,
    build_reference,
    clean_report,
    star_bucket,
)
from beatmap_ai.style import star_rating

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "beatmap_ai" / "pattern_reference.json"
GROUP_LABELS = {
    "double": "Doubles",
    "triple": "Triples",
    "burst_4_8": "Bursts (4–8)",
    "stream_9p": "Streams (9+)",
}
SUBDIVISION_LABELS = {"1_4": "1/4", "1_6": "1/6", "1_8": "1/8"}


def _worker(payload: tuple[Beatmap, float | None, str]) -> dict[str, Any]:
    bm, stars, name = payload
    return analyze_map(bm, stars=stars, name=name)


def _analyze(payloads: list[tuple[Beatmap, float | None, str]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(payloads) < 2:
        return [_worker(item) for item in payloads]
    # The CLI counts the main process as one; total analysis processes never exceed four.
    with ProcessPoolExecutor(max_workers=workers - 1) as pool:
        return list(pool.map(_worker, payloads, chunksize=max(1, len(payloads) // (workers * 8))))


def _excluded_human_path(name: str) -> bool:
    parts = name.replace("/", "\\").lower().split("\\")
    return any(part.startswith("critic_negatives") for part in parts)


def _normal_name(value: str) -> str:
    return value.replace("/", "\\").strip("\\.").casefold()


def _name_matches(name: str, relative_name: str) -> bool:
    name, relative_name = _normal_name(name), _normal_name(relative_name)
    return name == relative_name or name.endswith("\\" + relative_name)


def _collect_human(roots: list[str], max_maps: int | None = None,
                   paired_names: set[str] | None = None) -> tuple[list[tuple[Beatmap, float, str]], dict[str, int]]:
    candidates: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    seen: set[tuple] = set()
    counts = {"seen": 0, "validation": 0, "missing_stars": 0, "excluded_generated": 0}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            print(f"Übersprungen, nicht gefunden: {root}", file=sys.stderr)
            continue
        print(f"Lese menschliche Maps aus {root} …", file=sys.stderr)
        for name, text, _, _ in iter_beatmap_texts(root):
            counts["seen"] += 1
            if _excluded_human_path(name):
                counts["excluded_generated"] += 1
                continue
            try:
                bm = parse_osu(text)
                if bm.mode != 0 or len(bm.hit_objects) < 20 or not any(tp.uninherited for tp in bm.timing_points):
                    continue
                key = song_key(bm)
                if not is_validation(key):
                    continue
                counts["validation"] += 1
                difficulty_key = map_key(bm)
                if difficulty_key in seen:
                    continue
                stars = star_rating(text)
                if stars is None or not math.isfinite(stars):
                    counts["missing_stars"] += 1
                    continue
                bucket = star_bucket(stars)
                if bucket is None:
                    continue
                seen.add(difficulty_key)
                display_name = f"{root.resolve()}\\{name}"
                candidates[bucket].append((text, float(stars), display_name))
            except (ValueError, IndexError, StopIteration):
                continue
    selected = _balanced_select(candidates, max_maps, paired_names or set())
    return [(parse_osu(text), stars, name) for text, stars, name in selected], counts


def _balanced_select(candidates: dict[str, list], max_maps: int | None,
                     required_names: set[str] | None = None) -> list:
    for bucket in STAR_BUCKETS:
        candidates[bucket].sort(key=lambda item: hashlib.sha1(item[2].encode("utf-8")).hexdigest())
    total = sum(len(candidates[bucket]) for bucket in STAR_BUCKETS)
    limit = min(total, max_maps) if max_maps is not None else total
    base, remainder = divmod(limit, len(STAR_BUCKETS))
    target_quotas = {bucket: base + int(i < remainder) for i, bucket in enumerate(STAR_BUCKETS)}
    required_names = required_names or set()
    selected = []
    quotas = {}
    for bucket in STAR_BUCKETS:
        pinned = [item for item in candidates[bucket]
                  if any(_name_matches(item[2], name) for name in required_names)]
        pinned_ids = {item[2] for item in pinned}
        ordinary = [item for item in candidates[bucket] if item[2] not in pinned_ids]
        take = max(target_quotas[bucket], len(pinned))
        chosen = pinned + ordinary[:max(take - len(pinned), 0)]
        selected.extend(chosen)
        quotas[bucket] = len(chosen)
    remaining = limit - len(selected)
    while remaining > 0:
        available = [(len(candidates[bucket]) - quotas[bucket], bucket) for bucket in STAR_BUCKETS]
        count, bucket = max(available)
        if count <= 0:
            break
        quotas[bucket] += 1
        selected.append(candidates[bucket][quotas[bucket] - 1])
        remaining -= 1
    selected.sort(key=lambda item: (STAR_BUCKETS.index(star_bucket(item[1])), item[2]))
    return selected


def _collect_manifest(path: str) -> tuple[list[tuple[Beatmap, float, str]], dict[str, int], dict[str, dict[str, Any]]]:
    payloads, counts = [], {"entries": 0, "missing_files": 0, "invalid": 0}
    pair_info: dict[str, dict[str, Any]] = {}
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            counts["entries"] += 1
            try:
                entry = json.loads(line)
                map_path = Path(entry.get("negative_path") or entry.get("path") or entry.get("osu_path"))
                stars = float(entry["stars"])
                if not map_path.is_file():
                    counts["missing_files"] += 1
                    continue
                text = map_path.read_text(encoding="utf-8", errors="replace")
                bm = parse_osu(text)
                if bm.mode != 0 or len(bm.hit_objects) < 20 or not any(tp.uninherited for tp in bm.timing_points):
                    counts["invalid"] += 1
                    continue
                resolved = str(map_path.resolve())
                payloads.append((bm, stars, resolved))
                pair_info[_normal_name(resolved)] = {
                    "human_name": entry.get("human_name", ""),
                    "is_validation": bool(entry.get("is_val", False)),
                }
            except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
                counts["invalid"] += 1
                print(f"Ungültiger Manifest-Eintrag in Zeile {line_number}", file=sys.stderr)
    return payloads, counts, pair_info


def _collect_files(paths: list[str]) -> tuple[list[tuple[Beatmap, float | None, str]], dict[str, int]]:
    found = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".osu":
            found.add(path.resolve())
        elif path.is_dir():
            found.update(p.resolve() for p in path.rglob("*.osu") if p.is_file())
    payloads, counts = [], {"files": len(found), "missing_stars": 0, "invalid": 0}
    for path in sorted(found, key=lambda item: str(item).lower()):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            bm = parse_osu(text)
            if bm.mode != 0 or len(bm.hit_objects) < 20 or not any(tp.uninherited for tp in bm.timing_points):
                counts["invalid"] += 1
                continue
            stars = star_rating(text)
            if stars is None:
                counts["missing_stars"] += 1
            payloads.append((bm, stars, str(path)))
        except (OSError, ValueError, IndexError, StopIteration):
            counts["invalid"] += 1
    return payloads, counts


def _summary(reports: list[dict[str, Any]], buckets: tuple[str, ...] = STAR_BUCKETS) -> dict[str, Any]:
    result = {}
    for bucket in buckets:
        items = [report for report in reports if report.get("star_bucket") == bucket]
        keys = sorted({key for item in items for key in item["metrics"]})
        result[bucket] = {"maps": len(items), "mean": {}, "std": {}}
        for key in keys:
            values = [item["metrics"][key] for item in items if key in item["metrics"]]
            if values:
                import numpy as np
                result[bucket]["mean"][key] = float(np.mean(values))
                result[bucket]["std"][key] = float(np.std(values))
    return result


def _top_differences(human: list[dict[str, Any]], ai: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = []
    for bucket in STAR_BUCKETS:
        hs = [item for item in human if item.get("star_bucket") == bucket]
        ks = [item for item in ai if item.get("star_bucket") == bucket]
        if not hs or not ks:
            continue
        keys = set.intersection(*(set(item["metrics"]) for item in hs))
        keys &= set.intersection(*(set(item["metrics"]) for item in ks))
        for key in keys:
            h_values = [item["metrics"][key] for item in hs]
            k_values = [item["metrics"][key] for item in ks]
            import numpy as np
            h_mean, k_mean = float(np.mean(h_values)), float(np.mean(k_values))
            delta = k_mean - h_mean
            scale = max(float(np.std(h_values)), abs(h_mean) * 0.05, 0.05)
            ranked.append({"bucket": bucket, "metric": key, "human": h_mean,
                           "ai": k_mean, "delta": delta, "effect": abs(delta) / scale})
    return sorted(ranked, key=lambda item: item["effect"], reverse=True)[:limit]


def _human_relative_name(name: str) -> str:
    normalized = _normal_name(name)
    marker = "\\data\\"
    return normalized.rsplit(marker, 1)[-1] if marker in normalized else normalized


def _pair_validation_maps(human: list[dict[str, Any]], ai: list[dict[str, Any]],
                          pair_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for generated in ai:
        info = pair_info.get(_normal_name(generated["name"]), {})
        if not info.get("is_validation") or not info.get("human_name"):
            continue
        source = next((item for item in human
                       if _name_matches(item["name"], info["human_name"])), None)
        if source is None:
            continue
        shared = set(source["metrics"]) & set(generated["metrics"])
        pairs.append({
            "star_bucket": generated.get("star_bucket"),
            "human_map": source["name"],
            "ai_map": generated["name"],
            "metrics_delta_ai_minus_human": {
                key: generated["metrics"][key] - source["metrics"][key] for key in sorted(shared)
            },
        })
    return pairs


def _paired_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for bucket in STAR_BUCKETS:
        items = [item for item in pairs if item.get("star_bucket") == bucket]
        keys = sorted({key for item in items for key in item["metrics_delta_ai_minus_human"]})
        result[bucket] = {"pairs": len(items), "mean_delta": {}}
        for key in keys:
            values = [item["metrics_delta_ai_minus_human"][key]
                      for item in items if key in item["metrics_delta_ai_minus_human"]]
            result[bucket]["mean_delta"][key] = float(sum(values) / len(values)) if values else 0.0
    return result


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _cohort_mean(summary: dict[str, Any], bucket: str, cohort: str, metric: str) -> float | None:
    return summary.get(bucket, {}).get(cohort, {}).get("mean", {}).get(metric)


def _render_markdown(human: list[dict[str, Any]], ai: list[dict[str, Any]],
                     files: list[dict[str, Any]], reference: dict[str, Any] | None,
                     paired: list[dict[str, Any]]) -> str:
    cohorts = {"Mensch": human, "KI (Manifest)": ai}
    summary = {bucket: {name: _summary(items, (bucket,))[bucket]
                        for name, items in cohorts.items()} for bucket in STAR_BUCKETS}
    lines = ["# Mustervergleich Mensch vs. KI", "",
             f"- Menschliche Maps: **{len(human)}** (Validierungssongs, sternbalanciert)",
             f"- KI-Maps: **{len(ai)}**", f"- Eigene Dateien/Ordner: **{len(files)}**",
             f"- Schwierigkeitsschwellen: {'vorhanden' if reference else 'nicht vorhanden'}", ""]

    lines += ["## Rhythmusgruppen je 100 Objekte", ""]
    for group, _ in RHYTHM_GROUPS.items():
        lines += [f"### {GROUP_LABELS[group]}", "",
                  "| Sterne | Mensch/KI Maps | 1/4 M/KI | 1/6 M/KI | 1/8 M/KI |",
                  "|---|---:|---:|---:|---:|"]
        for bucket in STAR_BUCKETS:
            h = summary[bucket]["Mensch"]["mean"]
            k = summary[bucket]["KI (Manifest)"]["mean"]
            counts = f"{summary[bucket]['Mensch']['maps']}/{summary[bucket]['KI (Manifest)']['maps']}"
            values = [f"{_fmt(h.get(f'{group}_{sub}'))}/{_fmt(k.get(f'{group}_{sub}'))}"
                      for sub in SUBDIVISIONS]
            lines.append(f"| {bucket} | {counts} | " + " | ".join(values) + " |")
        lines.append("")

    lines += ["## Sprungmuster je 100 Objekte", "",
              "| Sterne | Mensch/KI Maps | Alle Muster | Zickzack | Dreieck | Viereck | Fünfeck/Stern | Linie | Bogen/Flow | Gemischt | Median: Größe (Radien) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for bucket in STAR_BUCKETS:
        h, k = summary[bucket]["Mensch"], summary[bucket]["KI (Manifest)"]
        hmean, kmean = h["mean"], k["mean"]
        counts = f"{h['maps']}/{k['maps']}"
        vals = [("patterns_per_100", "patterns_per_100")]
        values = [f"{_fmt(hmean.get('patterns_per_100'))}/{_fmt(kmean.get('patterns_per_100'))}"]
        for kind in PATTERN_TYPES:
            key = f"pattern_{kind}_per_100"
            values.append(f"{_fmt(hmean.get(key))}/{_fmt(kmean.get(key))}")
        values.append(f"{_fmt(hmean.get('pattern_size_radii_median'))}/{_fmt(kmean.get('pattern_size_radii_median'))}")
        lines.append(f"| {bucket} | {counts} | " + " | ".join(values) + " |")
    lines += ["", "## Wiederholung, Slider und Schwierigkeit", "",
              "| Sterne | Wiederholte Muster % M/KI | Songteil-/Taktpassung % M/KI | Stumpfe Wiederholung % M/KI | Slider % M/KI | Wiederhol-Slider /100 M/KI | Kurz & stark gebogen /100 M/KI | Objekte über P95 /100 M/KI |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for bucket in STAR_BUCKETS:
        h, k = summary[bucket]["Mensch"], summary[bucket]["KI (Manifest)"]
        hmean, kmean = h["mean"], k["mean"]
        pairs = [
            ("repeated_windows_pct", "repeated_windows_pct"),
            ("music_aligned_repeat_pct", "music_aligned_repeat_pct"),
            ("stale_repeat_pct", "stale_repeat_pct"),
            ("slider_share_pct", "slider_share_pct"),
            ("repeating_sliders_per_100", "repeating_sliders_per_100"),
            ("short_curved_sliders_per_100", "short_curved_sliders_per_100"),
            ("difficulty_any_over_p95_per_100", "difficulty_any_over_p95_per_100"),
        ]
        values = [f"{_fmt(hmean.get(a))}/{_fmt(kmean.get(b))}" for a, b in pairs]
        lines.append(f"| {bucket} | " + " | ".join(values) + " |")

    lines += ["", "## Muster-Abweichung vom menschlichen Referenzprofil", "",
              "Mittelwert des Scores pro Map; 0 entspricht exakt dem menschlichen Medianprofil, kleinere Werte sind menschlicher.", "",
              "| Sterne | Mensch | KI |", "|---|---:|---:|"]
    for bucket in STAR_BUCKETS:
        hmean = summary[bucket]["Mensch"]["mean"].get("pattern_deviation")
        kmean = summary[bucket]["KI (Manifest)"]["mean"].get("pattern_deviation")
        lines.append(f"| {bucket} | {_fmt(hmean, 3)} | {_fmt(kmean, 3)} |")

    paired_stats = _paired_summary(paired)
    lines += ["", "## Direkt gepaarte Validierungssongs", "",
              "Nur Manifest-Einträge mit is_val=true, deren menschliche Quell-Map im Datensatz gefunden wurde. Werte zeigen KI minus Mensch.", "",
              "| Sterne | Paare | Δ Doubles 1/4 /100 | Δ Muster /100 | Δ Slider-Anteil (Pp) | Δ Objekte über P95 /100 |",
              "|---|---:|---:|---:|---:|---:|"]
    for bucket in STAR_BUCKETS:
        item = paired_stats[bucket]
        deltas = item["mean_delta"]
        values = [deltas.get("double_1_4"), deltas.get("patterns_per_100"),
                  deltas.get("slider_share_pct"), deltas.get("difficulty_any_over_p95_per_100")]
        lines.append(f"| {bucket} | {item['pairs']} | " + " | ".join(_fmt(value, 3) for value in values) + " |")

    top = _top_differences(human, ai)
    lines += ["", "## Fünf größte Abweichungen", ""]
    if top:
        lines.append("; ".join(f"{item['bucket']}: {item['metric']} Mensch {_fmt(item['human'])}, "
                              f"KI {_fmt(item['ai'])} (Δ {_fmt(item['delta'], 3)})"
                              for item in top) + ".")
    else:
        lines.append("Für gemeinsame Sternbereiche liegen nicht genug Maps in beiden Gruppen vor.")

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cohort, reports in (("Mensch", human), ("KI", ai), ("Datei", files)):
        for report in reports:
            for kind, items in report.get("examples", {}).items():
                for item in items:
                    if len(examples[kind]) < 3 and all(old["map"] != item["map"] for old in examples[kind]):
                        examples[kind].append({**item, "cohort": cohort, "stars": report.get("stars")})
    lines += ["", "## Beispiele zum Nachschauen", ""]
    for kind in PATTERN_TYPES:
        items = examples.get(kind, [])
        if not items:
            continue
        rendered = "; ".join(f"{item['cohort']} {item['map']} @ {item['time_ms']} ms "
                              f"({item['edge_count']} Sprünge, {item['size_radii']} Radien)"
                              for item in items)
        lines.append(f"- {kind}: {rendered}")
    if not any(examples.values()):
        lines.append("Keine geometrischen Muster in den ausgewerteten Maps erkannt.")
    lines += ["", "## Definitionen", "",
              "- Rhythmusgruppen zählen maximale Folgen gleichmäßiger Startabstände nahe 1/4, 1/6 oder 1/8 Beat; angegeben sind Ereignisse je 100 Nicht-Spinner-Objekte.",
              "- Sprungmuster verwenden Fenster mit 3–5 gleich großen Sprüngen. Die Form wird über relative paarweise Punktabstände verglichen, daher bleiben Verschiebung, Drehung und Spiegelung erhalten.",
              "- Takt-/Songteilpassung ist ein Näherungswert: gleiche Form plus gleiche Beatposition oder wiederkehrendes Objekt-/Slider-Muster im Takt.",
              "- Kurz und stark gebogen bedeutet Sliderlänge höchstens 100 px und Sehne höchstens 72 % der Pfadlänge.",
              "- P95 zählt je Objekt jede Überschreitung eines menschlichen 95. Perzentils der jeweiligen Sternklasse; die Gesamtrate zählt Objekte nur einmal.", ""]
    lines.insert(-1, "- Muster-Abweichung pro Map ist der mittlere robuste Z-Abstand ihrer Kennzahlen zum menschlichen Medianprofil der Sternklasse; 0 bedeutet gleiche Kennzahlen wie der Median.")
    if files:
        lines += [f"Zusätzliche Einzeldateien/Ordner: {len(files)} Maps; Messwerte sind im JSON-Ergebnis enthalten.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", nargs="+", default=[], metavar="ORDNER",
                        help="Menschliche Map-Ordner; nur Validierungssongs werden verwendet")
    parser.add_argument("--manifest", help="JSONL-Manifest erzeugter Maps, etwa critic_negatives/manifest.jsonl")
    parser.add_argument("--files", nargs="+", default=[], metavar="PFAD",
                        help="Einzelne .osu-Dateien oder Ordner zum Messen")
    parser.add_argument("--max-maps", type=int,
                        help="Maximale Zahl menschlicher/Datei-Maps; Auswahl wird nach Sternen balanciert")
    parser.add_argument("--workers", type=int, default=1,
                        help="Gesamtzahl der Analyseprozesse (1–4, einschließlich Hauptprozess; Standard 1)")
    parser.add_argument("--write-reference", action="store_true",
                        help="Speichert menschliche P05/P50/P95-Schwellen nach beatmap_ai/pattern_reference.json")
    parser.add_argument("--json", action="store_true", help="Gibt statt Markdown ein JSON-Ergebnis aus")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers muss zwischen 1 und 4 liegen")
    if args.max_maps is not None and args.max_maps < 1:
        parser.error("--max-maps muss mindestens 1 sein")
    if not args.human and not args.manifest and not args.files:
        parser.error("Mindestens --human, --manifest oder --files angeben")

    manifest_payloads, manifest_scan, pair_info = (
        _collect_manifest(args.manifest) if args.manifest else ([], {}, {})
    )
    paired_names = {item["human_name"] for item in pair_info.values()
                    if item["is_validation"] and item["human_name"]}
    human_payloads, human_scan = (
        _collect_human(args.human, args.max_maps, paired_names) if args.human else ([], {})
    )
    if args.human:
        print(f"Validierungskandidaten: {human_scan.get('validation', 0)}; "
              f"Maps ohne rosu-pp-Sterne: {human_scan.get('missing_stars', 0)}; "
              f"ausgewählt: {len(human_payloads)}", file=sys.stderr)

    file_payloads, file_scan = _collect_files(args.files) if args.files else ([], {})
    if args.max_maps is not None and file_payloads:
        grouped: dict[str, list] = defaultdict(list)
        for item in file_payloads:
            bucket = star_bucket(item[1])
            if bucket:
                grouped[bucket].append(item)
        file_payloads = _balanced_select(grouped, args.max_maps)

    print(f"Analysiere {len(human_payloads)} menschliche, {len(manifest_payloads)} KI- und "
          f"{len(file_payloads)} eigene Maps mit {args.workers} Prozess(en) …", file=sys.stderr)
    human_reports = _analyze(human_payloads, args.workers)
    reference = build_reference(human_reports) if human_reports else None
    if reference and args.write_reference:
        REFERENCE_PATH.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    elif not reference and REFERENCE_PATH.is_file():
        try:
            reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reference = None
    for report in human_reports:
        apply_reference(report, reference)
    ai_reports = _analyze(manifest_payloads, args.workers)
    file_reports = _analyze(file_payloads, args.workers)
    for report in ai_reports + file_reports:
        apply_reference(report, reference)
    paired = _pair_validation_maps(human_reports, ai_reports, pair_info)

    result = {
        "counts": {"human": len(human_reports), "manifest": len(ai_reports), "files": len(file_reports),
                   "human_scan": human_scan, "manifest_scan": manifest_scan, "file_scan": file_scan},
        "summary": {
            "human": _summary(human_reports),
            "manifest": _summary(ai_reports),
            "files": _summary(file_reports),
        },
        "largest_differences": _top_differences(human_reports, ai_reports),
        "paired_validation_summary": _paired_summary(paired),
        "paired_validation_maps": paired,
        "reference": reference,
        "definitions": {
            "rhythm_groups_per_100": "maximal runs of equal 1/4, 1/6 or 1/8-beat start gaps",
            "pattern_windows": "similar-length 3–5-jump motifs; repeated by pairwise distance fingerprint",
            "music_alignment": "repeat at same beat-in-bar or in a bar with a repeated object/slider signature",
            "short_curved_slider": "path length <= 100 px and chord/path <= 0.72",
            "difficulty_outliers": "unique objects exceeding one or more same-star human P95 limits",
            "pattern_deviation": "mean robust-z distance from the human median map-metric profile; 0 is the human median profile",
        },
        "human_maps": [clean_report(item) for item in human_reports],
        "manifest_maps": [clean_report(item) for item in ai_reports],
        "files": [clean_report(item) for item in file_reports],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_render_markdown(human_reports, ai_reports, file_reports, reference, paired))


if __name__ == "__main__":
    main()
