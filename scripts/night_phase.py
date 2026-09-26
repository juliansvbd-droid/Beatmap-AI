"""Run one resumable, time-boxed phase of the BeatMap-AI night workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("D:/BeatMap-AI-Dataset")
PYTHON = REPO / ".venv-rocm" / "Scripts" / "python.exe"
RUN_DIR: Path
DEADLINE: float


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def data_dirs() -> list[Path]:
    return [path for path in (REPO / "data", REPO / "best_maps", DATA_ROOT) if path.is_dir()]


def tag_files() -> list[Path]:
    candidates = [DATA_ROOT / "tags.json", REPO / "data" / "tags.json"]
    return [path for path in candidates if path.is_file()]


def _remaining() -> float:
    return max(0.0, DEADLINE - time.time())


def _run_child(command: list[str], name: str, timeout: float) -> int:
    log(f"{name}: start, budget {timeout / 60:.1f} min")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with (RUN_DIR / f"{name}.log").open("a", encoding="utf-8", buffering=1) as output:
        try:
            process = subprocess.Popen(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT,
                                       creationflags=flags)
            try:
                code = process.wait(timeout=max(timeout, 1.0))
            except subprocess.TimeoutExpired:
                log(f"{name}: Zeitlimit erreicht; beende Prozessbaum")
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    process.kill()
                process.wait(timeout=30)
                code = 124
        except Exception:
            traceback.print_exc(file=output)
            code = 1
    log(f"{name}: exit {code}")
    return code


def phase_a() -> int:
    from beatmap_ai.placement_data import build_placement_maps

    total = 0
    for root in data_dirs():
        log(f"phase A: build/reuse placement-v3 cache for {root}")
        maps = build_placement_maps([root], log=log)
        total += len(maps)
        log(f"phase A: {len(maps)} maps cached in {root}")
        del maps
        if time.time() >= DEADLINE:
            break
    (RUN_DIR / "preparation.json").write_text(
        json.dumps({"cached_maps": total, "data_dirs": [str(path) for path in data_dirs()]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if total else 1


def phase_b() -> int:
    from beatmap_ai.tagger import train_tagger

    train_tagger(data_dirs(), tag_files(), RUN_DIR / "tagger.pt", epochs=24,
                 device="cuda", deadline=DEADLINE - 45, log=log)
    return 0 if (RUN_DIR / "tagger.pt").is_file() else 1


def phase_c() -> int:
    import gc
    import torch

    from beatmap_ai.sequence_model import train_sequence_v3

    variants = [
        ("small", 384, 6, 192),
        ("large", 512, 8, 256),
    ]
    results = {}
    tagger = RUN_DIR / "tagger.pt"
    tagger_path = tagger if tagger.is_file() else None
    for index, (name, hidden, layers, context) in enumerate(variants):
        if _remaining() < 90:
            log(f"phase C: not enough time to start {name} probe")
            break
        out = RUN_DIR / f"probe_{name}.pt"
        soft_deadline = min(DEADLINE - 45, time.time() + max(60, _remaining() / (2 - index)))
        try:
            train_sequence_v3(data_dirs(), out, tag_files(), epochs=2, steps_per_epoch=300,
                              batch_size=4, hidden=hidden, layers=layers, context=context,
                              device="cuda", workers=2, tagger_path=tagger_path,
                              deadline=soft_deadline, log=log)
            state = torch.load(out, map_location="cpu", weights_only=True)
            results[name] = {"hidden": hidden, "layers": layers, "context": context,
                             "parameters": sum(value.numel() for value in state["state_dict"].values()),
                             "val_loss": float(state.get("metrics", {}).get("val_loss", float("inf")))}
            log(f"phase C {name}: {results[name]}")
        except Exception as exc:
            log(f"phase C {name} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[name] = {"hidden": hidden, "layers": layers, "context": context,
                             "error": f"{type(exc).__name__}: {exc}"}
        del out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = "small"
    small, large = results.get("small", {}), results.get("large", {})
    if ("val_loss" in large and "val_loss" in small
            and large["val_loss"] < small["val_loss"] * 0.98):
        selected = "large"
    payload = {"variants": results, "selected": selected,
               "selection_reason": "large is selected only with at least 2% lower validation loss; otherwise small"}
    (RUN_DIR / "probe_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"phase C selected {selected} probe configuration")
    return 0 if results else 1


def phase_d() -> int:
    import json as json_module

    from beatmap_ai.sequence_model import train_sequence_v3

    selection_path = RUN_DIR / "probe_selection.json"
    selected = "small"
    if selection_path.is_file():
        selected = json_module.loads(selection_path.read_text(encoding="utf-8")).get("selected", "small")
    spec = {"small": (384, 6, 192), "large": (512, 8, 256)}.get(selected, (384, 6, 192))
    hidden, layers, context = spec
    tagger = RUN_DIR / "tagger.pt"
    tagger_path = tagger if tagger.is_file() else None
    log(f"phase D: selected={selected}, hidden={hidden}, layers={layers}, context={context}")
    train_sequence_v3(data_dirs(), RUN_DIR / "sequence-v3.pt", tag_files(), epochs=1000,
                      steps_per_epoch=200, batch_size=8, hidden=hidden, layers=layers,
                      context=context, device="cuda", workers=2, tagger_path=tagger_path,
                      deadline=DEADLINE - 60, log=log)
    return 0 if (RUN_DIR / "sequence-v3.pt").is_file() else 1


def phase_e() -> int:
    import gc
    import torch

    from beatmap_ai.planner import train_planner
    from beatmap_ai.planner_data import build_planner_examples

    tagger = RUN_DIR / "tagger.pt"
    tagger_path = tagger if tagger.is_file() else None
    examples = build_planner_examples(data_dirs(), tag_files(), tagger_path=tagger_path, log=log)
    results = {}
    sizes = (128, 256)
    for index, hidden in enumerate(sizes):
        if _remaining() < 120:
            log(f"phase E: not enough time to start hidden={hidden}")
            break
        out = RUN_DIR / f"planner-{hidden}.pt"
        soft_deadline = min(DEADLINE - 45, time.time() + max(60, _remaining() / (2 - index)))
        try:
            train_planner(data_dirs(), tag_files(), out, tagger_path=tagger_path, epochs=12,
                          batch_size=16, hidden=hidden, device="cuda", log=log,
                          deadline=soft_deadline, examples=examples)
            checkpoint = torch.load(out.with_name(out.stem + ".best.pt"), map_location="cpu",
                                    weights_only=True)
            metrics = checkpoint.get("metrics", {})
            results[str(hidden)] = {"path": str(out.with_name(out.stem + ".best.pt")),
                                    "parameters": sum(value.numel() for value in checkpoint["state_dict"].values()),
                                    **metrics}
        except Exception as exc:
            log(f"phase E hidden={hidden} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[str(hidden)] = {"error": f"{type(exc).__name__}: {exc}"}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results or not any("selection_score" in item for item in results.values()):
        (RUN_DIR / "planner-selection.json").write_text(
            json.dumps({"variants": results, "selected": None}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return 1
    valid = [(int(hidden), result) for hidden, result in results.items()
             if "selection_score" in result]
    best_score = max(result["selection_score"] for _, result in valid)
    selected_hidden, selected = min(
        ((hidden, result) for hidden, result in valid
         if best_score - result["selection_score"] <= 0.001), key=lambda pair: pair[0])
    source = Path(selected["path"])
    shutil.copy2(source, RUN_DIR / "planner.pt")
    shutil.copy2(source, RUN_DIR / "planner.best.pt")
    latest = RUN_DIR / f"planner-{selected_hidden}.last.pt"
    if latest.is_file():
        shutil.copy2(latest, RUN_DIR / "planner.last.pt")
    payload = {"variants": results, "selected": selected_hidden,
               "tie_break": "hidden=128 is selected if selection scores differ by at most 0.001"}
    (RUN_DIR / "planner-selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"phase E selected hidden={selected_hidden}: {selected}")
    del examples
    return 0


def phase_f() -> int:
    from beatmap_ai.songfit import train_songfit

    train_songfit(data_dirs(), RUN_DIR / "songfit.pt", epochs=1000, steps_per_epoch=200,
                  batch_size=32, device="cuda", deadline=DEADLINE - 45, log=log)
    return 0 if (RUN_DIR / "songfit.pt").is_file() else 1


def _find_blindtest_songs() -> dict[str, Path]:
    music = REPO / "Music"
    files = [path for path in music.rglob("*") if path.is_file()
             and path.suffix.lower() in {".mp3", ".ogg"}] if music.is_dir() else []
    result = {}
    for label, terms in (("GUERREIRO", ("montagem", "guerreiro")),
                         ("ALQUIMIA", ("montagem", "alquimia"))):
        candidates = [path for path in files
                      if all(term in str(path).lower() for term in terms)]
        if not candidates:
            candidates = [path for path in files if all(term in path.stem.lower() for term in terms[1:])]
        if candidates:
            result[label] = sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))[0]
    return result


def _generate_blind_maps(sequence_path: Path) -> list[Path]:
    output_dir = REPO / "Vergleich" / "v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = []
    for label, audio in _find_blindtest_songs().items():
        if _remaining() < 180:
            log(f"phase G: skipping {label}; too little time remains")
            break
        archive = output_dir / f"{label}_Insane_v3.osz"
        command = [str(PYTHON), "-u", "-m", "beatmap_ai", "generate", str(audio),
                   "-o", str(archive), "-d", "4.5", "--sequence", str(sequence_path),
                   "--critic", "auto", "--passes", "1"]
        code = _run_child(command, f"blindtest-{label}", min(900, _remaining() - 90))
        if code:
            continue
        extraction = RUN_DIR / "blindtest-osu" / label
        extraction.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as bundle:
                osu_name = next(name for name in bundle.namelist() if name.lower().endswith(".osu"))
                destination = extraction / f"{label}_Insane_v3.osu"
                destination.write_bytes(bundle.read(osu_name))
                maps.append(destination)
        except (OSError, zipfile.BadZipFile, StopIteration) as exc:
            log(f"phase G: unable to inspect {archive}: {exc}")
    return maps


def _pattern_report(v3_maps: list[Path]) -> None:
    output = RUN_DIR / "pattern_stats.json"
    paths = [Path("D:/Mapperatorinator/compare")]
    if v3_maps:
        paths.append(RUN_DIR / "blindtest-osu")
    command = [str(PYTHON), "-u", "scripts/pattern_stats.py", "--files",
               *(str(path) for path in paths), "--workers", "1", "--json"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with output.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=sys.stderr,
                                   creationflags=flags)
        try:
            code = process.wait(timeout=max(min(_remaining() - 120, 480), 1))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                process.kill()
            process.wait(timeout=30)
            code = 124
    log(f"pattern_stats exit {code}: {output}")
    if code:
        return
    payload = json.loads(output.read_text(encoding="utf-8"))
    lines = ["# Nachtlauf: Phase G – Muster- und P95-Messung", "",
             f"Analysierte Benchmark-Maps: {len(payload.get('files', []))}; "
             f"Blindtest-Maps: {len(v3_maps)}; menschliche Referenz aus `beatmap_ai/pattern_reference.json`.", "",
             "## Durchschnitt nach Sternbereich", "",
             "| Sternbereich | Maps | Ausreißer gesamt/100 | Sprungtempo/100 | scharfe Wendung schnell/100 | Stream-Länge/100 | Cross-Screen/100 | Musterabweichung |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metric_keys = ["difficulty_any_over_p95_per_100",
                   "difficulty_jump_radii_per_second_over_p95_per_100",
                   "difficulty_sharp_turn_speed_over_p95_per_100",
                   "difficulty_stream_run_length_over_p95_per_100",
                   "difficulty_cross_screen_fraction_over_p95_per_100"]
    summary = payload.get("summary", {}).get("files", {})
    for bucket, values in summary.items():
        mean = values.get("mean", {})
        cells = [mean.get(key) for key in metric_keys] + [mean.get("pattern_deviation")]
        rendered = ["—" if value is None else f"{float(value):.3f}" for value in cells]
        lines.append(f"| {bucket} | {values.get('maps', 0)} | " + " | ".join(rendered) + " |")
    lines += ["", "## Einzelkarten", "", "| Map | Sterne | Gesamt/100 | Sprungtempo | scharfe Wendung schnell | Stream-Länge | Cross-Screen | Musterabweichung |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for report in payload.get("files", []):
        metrics = report.get("metrics", {})
        values = [metrics.get(key) for key in metric_keys] + [report.get("pattern_deviation", {}).get("score")]
        rendered = ["—" if value is None else f"{float(value):.3f}" for value in values]
        lines.append(f"| {Path(report.get('name', 'Map')).name} | {report.get('stars', '—')} | "
                     + " | ".join(rendered) + " |")
    (RUN_DIR / "Phase-G-Muster-und-P95.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_g() -> int:
    result_file = RUN_DIR / "evaluation-results.json"
    results = []
    sequence_path = RUN_DIR / "sequence-v3.pt"
    if sequence_path.is_file():
        maps = _generate_blind_maps(sequence_path)
        _pattern_report(maps)
        human_roots = [REPO / "data", REPO / "best_maps", DATA_ROOT]
        base = [str(PYTHON), "-u", "scripts/eval_sequence.py",
                *(str(path) for path in human_roots if path.is_dir())]
        variants = [
            ("v2-with-crutches", REPO / "beatmap_ai/models/sequence.pt", (), 40),
            ("v3-with-crutches", sequence_path, (), 40),
            ("v3-no-crutches", sequence_path, ("--no-crutches",), 40),
            ("v3-with-planner", sequence_path, ("--planner", str(RUN_DIR / "planner.best.pt")), 10),
            ("v3-with-songfit", sequence_path,
             ("--songfit", str(RUN_DIR / "songfit.pt"), "--passes", "2"), 10),
        ]
        available = max(_remaining() - 90, 0)
        for index, (name, model, extra, max_songs) in enumerate(variants):
            if _remaining() < 120:
                results.append({"name": name, "status": "skipped: reserved time for final report"})
                continue
            if name == "v3-with-planner" and not (RUN_DIR / "planner.best.pt").is_file():
                results.append({"name": name, "status": "skipped: no planner checkpoint"})
                continue
            if name == "v3-with-songfit" and not (RUN_DIR / "songfit.pt").is_file():
                results.append({"name": name, "status": "skipped: no song-fit checkpoint"})
                continue
            runs_left = sum(1 for item in variants[index:]
                            if item[0] not in {row["name"] for row in results})
            budget = min(available / max(runs_left, 1), max(_remaining() - 90, 0))
            command = [*base, "--model", str(model), "--rhythm", str(REPO / "beatmap_ai/models/rhythm.pt"),
                       "--critic", str(REPO / "beatmap_ai/models/critic.pt"), "--follow",
                       "--max-songs", str(max_songs), "--passes", "1", *extra]
            code = _run_child(command, f"eval-{name}", budget)
            results.append({"name": name, "status": "completed" if code == 0 else f"exit {code}",
                            "max_songs": max_songs, "model": str(model)})
            available = max(_remaining() - 90, 0)
    else:
        log("phase G: no v3 checkpoint; evals and blind maps skipped")
        _pattern_report([])
    result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Nachtlauf: Messphase G", "", "## Muster und P95", "",
             "Siehe `Phase-G-Muster-und-P95.md`; der Bericht enthält `difficulty_any_over_p95_per_100`, "
             "alle vier Merkmale und `pattern_deviation` für `D:\\Mapperatorinator\\compare` sowie erzeugte Insane-Maps.", "",
             "## Folgeauswertungen", "", "| Variante | Status | Songs |", "|---|---|---:|"]
    for row in results:
        lines.append(f"| {row['name']} | {row['status']} | {row.get('max_songs', '—')} |")
    lines += ["", "Jede Auswertung schreibt Fortschritt und Messwerte in die zugehörige `eval-*.log`.",
              "Zeitlimits stoppen den jeweiligen Prozessbaum; danach bleiben die übrigen Messschritte möglich.", ""]
    (RUN_DIR / "Phase-G-Messung.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


PHASES = {"A": phase_a, "B": phase_b, "C": phase_c, "D": phase_d,
          "E": phase_e, "F": phase_f, "G": phase_g}


def main() -> None:
    global RUN_DIR, DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    RUN_DIR = Path(args.run_dir)
    DEADLINE = args.deadline
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Training environment not found: {PYTHON}")
    log(f"phase {args.phase} started; {len(data_dirs())} data roots; remaining={_remaining() / 60:.1f} min")
    try:
        code = PHASES[args.phase]()
    except Exception as exc:
        log(f"phase {args.phase} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 1
    log(f"phase {args.phase} finished with exit {code}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
