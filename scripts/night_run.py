"""Detached, deadline-bounded supervisor for the sequential night workflow."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("D:/BeatMap-AI-Dataset")
PYTHON = REPO / ".venv-rocm" / "Scripts" / "python.exe"
TOTAL_SECONDS = 10 * 60 * 60
PHASE_BUDGETS = {"A": 45 * 60, "B": 45 * 60, "C": 45 * 60,
                 "D": 5 * 60 * 60, "E": 75 * 60, "F": 75 * 60, "G": 60 * 60}
PHASE_NAMES = {"A": "Datenvorbereitung", "B": "Tagger", "C": "v3-Probelauf",
               "D": "v3-Haupttraining", "E": "Vorplanung 128/256",
               "F": "Song-Passung", "G": "Messung und Blindtest"}
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
STOP_REQUESTED = False


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _append_master(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as log:
        log.write(f"[{_timestamp()}] {line}\n")
        log.flush()


def _read_status_section(text: str) -> tuple[str, int, int]:
    heading = "## Currently running / in progress"
    start = text.find(heading)
    if start < 0:
        raise ValueError("docs/STATUS.md has no Currently running section")
    body_start = start + len(heading)
    end = text.find("\n## ", body_start)
    return heading, body_start, len(text) if end < 0 else end


def _update_status(run_dir: Path, phase: str | None, summary: str | None = None) -> None:
    path = REPO / "docs" / "STATUS.md"
    text = path.read_text(encoding="utf-8")
    heading, body_start, body_end = _read_status_section(text)
    body = text[body_start:body_end]
    if phase is not None:
        details = (f"- **Nachtlauf läuft: Phase {phase} – {PHASE_NAMES[phase]}.** PID "
                   f"{os.getpid()}, gestartet {_timestamp()}; Protokoll: `{run_dir / 'night_run.log'}`.")
        old_clause = re.compile(r"- Keine laufenden BeatMap-AI-Mess- oder Trainingsjobs\.")
        if old_clause.search(body):
            body = old_clause.sub(lambda _: details, body, count=1)
        else:
            body = re.sub(r"(?m)^- \*\*Nachtlauf läuft:.*$\n?", "", body)
            body = "\n" + details + body
    else:
        details = f"- **Kein BeatMap-AI-Job läuft.** {summary or 'Nachtlauf beendet.'}"
        old_clause = re.compile(r"(?m)^- \*\*Nachtlauf läuft:.*$")
        if old_clause.search(body):
            body = old_clause.sub(lambda _: details, body, count=1)
        else:
            body = "\n" + details + body
    text = text[:body_start] + body + text[body_end:]

    phase_results = _load_phase_results(run_dir)
    completed = {key: value.get("exit_code") == 0 for key, value in phase_results.items()}
    statuses = {
        "03": "läuft" if phase is not None else (
            "fertig" if completed.get("D") and (run_dir / "sequence-v3.pt").is_file()
            else "teilweise"),
        "05": "läuft" if phase is not None else "beendet; siehe Nachtlaufbericht",
        "04": "wartet auf Phase E" if phase is not None else (
            "Checkpoint trainiert" if (run_dir / "planner.best.pt").is_file() else "teilweise"),
        "02": "wartet auf Phase F" if phase is not None else (
            "Checkpoint trainiert" if (run_dir / "songfit.pt").is_file() else "teilweise"),
    }
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for prompt, state in statuses.items():
            if line.startswith(f"| {prompt} |"):
                cells = line.split("|")
                cells[-2] = f" **{state}** "
                lines[index] = "|".join(cells)
    if lines:
        lines[0] = "# Projektstand BeatMap-AI"
    for index, line in enumerate(lines):
        if line.startswith("_Zuletzt aktualisiert:"):
            lines[index] = f"_Zuletzt aktualisiert: {_timestamp()}, von Codex._"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _insert_worklog(run_dir: Path, heading: str, lines: list[str]) -> None:
    path = REPO / "docs" / "WORKLOG.md"
    text = path.read_text(encoding="utf-8")
    position = text.find("\n## ")
    if position < 0:
        position = len(text)
    entry = "\n\n" + heading + "\n" + "\n".join(f"- {line}" for line in lines) + "\n"
    path.write_text(text[:position] + entry + text[position:], encoding="utf-8")


def _load_phase_results(run_dir: Path) -> dict:
    path = run_dir / "phase-results.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_phase_results(run_dir: Path, results: dict) -> None:
    path = run_dir / "phase-results.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _available_budget(phase: str, started: float, overall_deadline: float) -> float:
    remaining = max(0.0, overall_deadline - time.time())
    if phase in {"A", "B", "C"}:
        return min(PHASE_BUDGETS[phase], remaining)
    if phase == "D":
        # Preserve the full 60-minute measurement phase. D may use up to five hours,
        # including time saved by earlier phases.
        return min(PHASE_BUDGETS[phase], max(0.0, remaining - PHASE_BUDGETS["G"]))
    if phase == "E":
        # D has already had priority. Share the time left after reserving G between E/F.
        return min(PHASE_BUDGETS[phase], max(0.0, remaining - PHASE_BUDGETS["G"]) / 2)
    if phase == "F":
        return min(PHASE_BUDGETS[phase], max(0.0, remaining - PHASE_BUDGETS["G"]))
    return min(PHASE_BUDGETS[phase], remaining)


def _keep_awake(enabled: bool) -> None:
    if os.name != "nt":
        return
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enabled else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def _stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.kill()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _handle_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _new_run_dir() -> Path:
    base = DATA_ROOT / "night" / datetime.now().strftime("%Y-%m-%d")
    if not base.exists():
        return base
    return base.with_name(base.name + "_" + datetime.now().strftime("%H%M%S"))


def main() -> None:
    global STOP_REQUESTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"ROCm environment not found: {PYTHON}")
    run_dir = Path(args.run_dir) if args.run_dir else _new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    master = run_dir / "night_run.log"
    results = _load_phase_results(run_dir)
    started = time.time()
    overall_deadline = started + TOTAL_SECONDS
    _keep_awake(True)
    signal.signal(signal.SIGTERM, _handle_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_stop)
    _append_master(master, f"RUN START pid={os.getpid()} run_dir={run_dir} total_budget=600 min")
    _update_status(run_dir, "A")
    _insert_worklog(run_dir, f"## {_timestamp()} – Codex (Prompt 05, Nachtlauf gestartet)", [
        f"Code und Tests wurden vor dem Start abgeschlossen; der detached Lauf läuft nacheinander durch Phasen A–G.",
        f"Aktueller Laufordner: `{run_dir}`; Supervisor-PID {os.getpid()}; Hauptlog `{master}`.",
        "Kein Modell wird automatisch nach `beatmap_ai/models/` übernommen. Phasen D erhält Vorrang; die Messphase G bleibt bis zu 60 Minuten reserviert.",
    ])
    try:
        for phase in "ABCDEFG":
            if STOP_REQUESTED:
                _append_master(master, "STOP requested; remaining phases skipped")
                break
            phase_seconds = _available_budget(phase, started, overall_deadline)
            if phase_seconds < 30:
                result = {"exit_code": 125, "status": "skipped: no budget", "budget_seconds": phase_seconds}
                results[phase] = result
                _append_master(master, f"PHASE {phase} SKIP budget={phase_seconds:.0f}s; no time remaining")
                _save_phase_results(run_dir, results)
                continue
            phase_started = time.time()
            phase_deadline = phase_started + phase_seconds
            _update_status(run_dir, phase)
            _append_master(master, f"PHASE {phase} START name={PHASE_NAMES[phase]} budget={phase_seconds / 60:.1f} min")
            command = [str(PYTHON), "-u", str(REPO / "scripts" / "night_phase.py"),
                       "--phase", phase, "--run-dir", str(run_dir), "--deadline", str(phase_deadline)]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                with (run_dir / f"phase-{phase}.log").open("a", encoding="utf-8", buffering=1) as output:
                    process = subprocess.Popen(command, cwd=REPO, stdout=output,
                                               stderr=subprocess.STDOUT, creationflags=flags)
                    try:
                        code = process.wait(timeout=phase_seconds)
                        status = "completed" if code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        _append_master(master, f"PHASE {phase} HARD TIMEOUT; ending its process tree")
                        _stop_process_tree(process)
                        code, status = 124, "timed out; last checkpoint retained"
            except Exception as exc:
                code, status = 1, f"launch error: {type(exc).__name__}: {exc}"
                _append_master(master, f"PHASE {phase} EXCEPTION {status}")
                with (run_dir / f"phase-{phase}.log").open("a", encoding="utf-8") as output:
                    traceback.print_exc(file=output)
            elapsed = time.time() - phase_started
            results[phase] = {"exit_code": code, "status": status,
                              "duration_seconds": round(elapsed, 1),
                              "budget_seconds": round(phase_seconds, 1)}
            _save_phase_results(run_dir, results)
            _append_master(master, f"PHASE {phase} END exit={code} status={status} duration={elapsed / 60:.1f} min")
    finally:
        elapsed = time.time() - started
        _append_master(master, f"RUN END duration={elapsed / 3600:.2f} h phases={json.dumps(results, ensure_ascii=False)}")
        failures = [f"{key}:{value.get('status')}" for key, value in results.items()
                    if value.get("exit_code") != 0]
        phase_summary = ", ".join("{}={}".format(key, value.get("status"))
                                   for key, value in results.items()) or "keine"
        summary = (f"Nachtlauf beendet nach {elapsed / 3600:.2f} h. "
                   f"Phasenstatus: {phase_summary}. "
                   f"Berichte/Checkpoints: `{run_dir}`. "
                   + (f"Offen/abgebrochen: {', '.join(failures)}." if failures else "Alle Phasen meldeten Erfolg."))
        _update_status(run_dir, None, summary)
        _insert_worklog(run_dir, f"## {_timestamp()} – Codex (Prompt 05, Nachtlauf beendet)", [
            f"Gesamtlaufzeit: {elapsed / 3600:.2f} Stunden.",
            f"Phasenstatus: {json.dumps(results, ensure_ascii=False)}.",
            f"Checkpoint- und Messordner: `{run_dir}`; Hauptlog `{master}`.",
            "Kein Checkpoint wurde automatisch in der App aktiviert; blindtest maps liegen – falls erzeugt – unter `Vergleich/v3/`.",
        ])
        _keep_awake(False)


if __name__ == "__main__":
    main()
