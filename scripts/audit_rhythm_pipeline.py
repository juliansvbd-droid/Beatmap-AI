"""Trace rhythm changes along the real v4 Forms generation path.

This diagnostic observes existing functions without changing their output. Run one
process at a time: generation loads the ROCm rhythm, planner and sequence models.

Sources are existing .osz files (named difficulties Insane/Expert = 4.5*/5.6*, as the
archived v4 candidates) or ``--holdout`` (``scripts/rhythm_holdout.py``): human maps of
validation songs nobody tuned on. A holdout map is generated at the human map's star
rating on the human map's timing, and every stage is compared with it (rhythm F1 within
30 ms, figures, sliders, pauses; see ``beatmap_ai.rhythm_audit``).

Stages of the chosen rhythm: tick selection (threshold or density), ``shape_triples``,
``reinforce_figures`` (quarter-link models only), ``absorb_into_sliders``, the finished
``plan_objects``, ``SequencePlacer.follow``, render, refrain copies (final). The star
search is the difference between the first rhythm it builds and the chosen one.

``--force-selection threshold`` replaces the star search's density selection by the
threshold selection (at the same note threshold): for comparing two rhythm models
with the same selection on both sides.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_ai import generator, rhythm, rhythm_figures, sequence_model  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.rhythm_audit import events as _events, summary as _summary  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402

STAGES = ("selected", "after_shape", "after_reinforce", "before_absorb", "after_absorb",
          "planned", "after_follow", "rendered", "final")


def _tick_events(grid, ticks) -> list[dict]:
    return [{"time": round(float(grid.times[int(t)]), 3), "end": round(float(grid.times[int(t)]), 3),
             "kind": "circle", "slides": 1} for t in ticks]


def _signature(items: list[dict]) -> tuple:
    return tuple((item["time"], item["kind"]) for item in items)


def _overlap(first, second, *, positions: bool = False) -> int:
    def identity(obj):
        if isinstance(obj, dict):
            return round(obj["time"]), obj["kind"]
        base = (round(obj.time), obj.kind)
        return base + (round(obj.x), round(obj.y)) if positions else base
    return len({identity(e) for e in first} & {identity(e) for e in second})


def _reinforce_diagnostics(ticks, note_scores, link_scores, *, divisor, threshold,
                           link_threshold=0.5, min_note_ratio=0.45, max_extra_ratio=0.05,
                           **_) -> dict:
    """Why reinforce_figures adds (or does not add) quarter notes: how many picked ticks
    pass each of its conditions in turn."""
    chosen = set(map(int, ticks))
    step = max(divisor // 4, 1)
    anchors = sorted(chosen)
    link = np.asarray([float(link_scores[t]) for t in anchors]) if anchors else np.zeros(0)
    free = [t for t in anchors if t + step < len(note_scores) and t + step not in chosen]
    linked = [t for t in free if link_scores[t] >= link_threshold]
    heard = [t for t in linked if note_scores[t + step] >= min_note_ratio * threshold]
    return {"ticks": len(anchors), "next_quarter_free": len(free),
            "link_ok": len(linked), "note_ok": len(heard),
            "budget": int(round(len(anchors) * max_extra_ratio)),
            "link_threshold": float(link_threshold), "note_threshold": float(threshold),
            "link_max": float(link.max()) if len(link) else 0.0,
            "link_p95": float(np.quantile(link, 0.95)) if len(link) else 0.0,
            "link_ge_threshold_pct": 100.0 * float((link >= link_threshold).mean())
            if len(link) else 0.0}


class Observer:
    def __init__(self, force_selection: str | None = None):
        self.force_selection = force_selection
        self.active = None
        self.plans = {}
        self.plan_order = []
        self.followed = {}
        self.renders = []
        self.originals = {}

    def __enter__(self):
        self.originals = {
            "grid": rhythm.make_tick_grid,
            "shape": rhythm.shape_triples,
            "density": rhythm.select_ticks_by_density,
            "absorb": rhythm.absorb_into_sliders,
            "reinforce": rhythm_figures.reinforce_figures,
            "plan": generator.plan_objects,
            "follow": sequence_model.SequencePlacer.follow,
            "render": sequence_model.SequencePlacer.render,
        }
        observer = self

        def grid(*args, **kwargs):
            result = observer.originals["grid"](*args, **kwargs)
            if observer.active is not None:
                observer.active["grid"] = result
            return result

        def shape(ticks, *args, **kwargs):
            result = observer.originals["shape"](ticks, *args, **kwargs)
            if observer.active is not None:
                current = observer.active
                current["selected"] = _tick_events(current["grid"], ticks)
                current["after_shape"] = _tick_events(current["grid"], result)
            return result

        def density(*args, **kwargs):
            result = observer.originals["density"](*args, **kwargs)
            if observer.active is not None:
                observer.active["selected"] = _tick_events(observer.active["grid"], result)
            return result

        def reinforce(ticks, note_scores, link_scores, **kwargs):
            result = observer.originals["reinforce"](ticks, note_scores, link_scores, **kwargs)
            if observer.active is not None:
                current = observer.active
                current["after_reinforce"] = _tick_events(current["grid"], result)
                current["reinforce"] = dict(
                    _reinforce_diagnostics(ticks, note_scores, link_scores, **kwargs),
                    added=int(len(result) - len(ticks)))
            return result

        def absorb(ticks, *args, **kwargs):
            result = observer.originals["absorb"](ticks, *args, **kwargs)
            if observer.active is not None:
                current = observer.active
                current["before_absorb"] = _tick_events(current["grid"], ticks)
                current["after_absorb"] = _tick_events(current["grid"], result)
            return result

        def plan(*args, **kwargs):
            args = list(args)
            mode = args[7] if len(args) > 7 else kwargs.get("selection", "threshold")
            if observer.force_selection and mode != observer.force_selection:
                if len(args) > 7:
                    args[7] = observer.force_selection
                else:
                    kwargs["selection"] = observer.force_selection
                mode = observer.force_selection
            preset = args[2]
            current = {"selection_mode": mode, "index": len(observer.plan_order),
                       "density": round(float(preset.density), 4),
                       "note_threshold": round(float(args[6] if len(args) > 6 else
                                                     kwargs.get("threshold", 0.5)), 4)}
            observer.active = current
            try:
                result = observer.originals["plan"](*args, **kwargs)
            finally:
                observer.active = None
            current.pop("grid", None)
            current["planned"] = _events(result)
            key = _signature(current["planned"])
            observer.plans.setdefault(key, current)
            observer.plan_order.append(key)
            return result

        def follow(placer, plan_items, *args, **kwargs):
            key = _signature(_events(plan_items))
            result = observer.originals["follow"](placer, plan_items, *args, **kwargs)
            observer.followed[id(placer)] = {"plan_key": key, "after_follow": _events(result[0])}
            return result

        def render(placer, plan_items, *args, **kwargs):
            result = observer.originals["render"](placer, plan_items, *args, **kwargs)
            observer.renders.append({"placer": id(placer), "objects": copy.deepcopy(result)})
            return result

        rhythm.make_tick_grid = grid
        rhythm.shape_triples = shape
        rhythm.select_ticks_by_density = density
        rhythm.absorb_into_sliders = absorb
        rhythm_figures.reinforce_figures = reinforce
        generator.plan_objects = plan
        sequence_model.SequencePlacer.follow = follow
        sequence_model.SequencePlacer.render = render
        return self

    def __exit__(self, *_):
        rhythm.make_tick_grid = self.originals["grid"]
        rhythm.shape_triples = self.originals["shape"]
        rhythm.select_ticks_by_density = self.originals["density"]
        rhythm.absorb_into_sliders = self.originals["absorb"]
        rhythm_figures.reinforce_figures = self.originals["reinforce"]
        generator.plan_objects = self.originals["plan"]
        sequence_model.SequencePlacer.follow = self.originals["follow"]
        sequence_model.SequencePlacer.render = self.originals["render"]

    def report(self, final, timing, truth=None):
        final_events = _events(final.hit_objects, final.end_time)
        if not self.renders:
            raise RuntimeError("the selected sequence model produced no render trace")
        best = max(self.renders, key=lambda r: _overlap(r["objects"], final.hit_objects,
                                                        positions=True))
        followed = self.followed[best["placer"]]
        plan = self.plans[followed["plan_key"]]
        values = {name: plan[name] for name in STAGES[:6] if name in plan}
        values["after_follow"] = followed["after_follow"]
        values["rendered"] = _events(best["objects"], final.end_time)
        values["final"] = final_events
        order = [name for name in STAGES if name in values]
        stages = {name: _summary(values[name], timing, truth) for name in order}
        changes = {}
        for a, b in zip(order, order[1:]):
            shared = _overlap(values[a], values[b])
            changes[f"{a}_to_{b}"] = {"shared": shared, "removed": len(values[a]) - shared,
                                      "added": len(values[b]) - shared}
        initial = self.plans[self.plan_order[0]]
        search = {
            "rhythms_built": len(self.plan_order),
            "chosen_index": plan["index"], "chosen_density": plan["density"],
            "chosen_note_threshold": plan["note_threshold"],
            "initial_selection_mode": initial["selection_mode"],
            "initial_density": initial["density"],
            "initial_note_threshold": initial["note_threshold"],
            "initial_planned": _summary(initial["planned"], timing, truth),
            "initial_to_chosen_shared": _overlap(initial["planned"], plan["planned"]),
        }
        return {"selection_mode": plan["selection_mode"], "stages": stages,
                "changes": changes, "star_search": search,
                "reinforce": plan.get("reinforce"),
                "initial_reinforce": initial.get("reinforce"),
                "matched_render_positions": _overlap(best["objects"], final.hit_objects,
                                                     positions=True),
                "render_candidates": len(self.renders), "plan_candidates": len(self.plans)}


def _source_audio(archive: zipfile.ZipFile) -> str:
    candidates = [name for name in archive.namelist()
                  if name.lower().endswith((".mp3", ".ogg"))]
    if len(candidates) != 1:
        raise ValueError(f"expected one audio file in .osz, found {len(candidates)}")
    return candidates[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def holdout_jobs(path: Path, songs: int, insane=(4.2, 4.95), expert=(5.2, 6.2),
                 max_duration: float = 240.0) -> list[dict]:
    """The first ``songs`` unused validation songs (rhythm_holdout order) with an
    Insane-like and an Expert-like human map on constant timing; per song the map
    closest to 4.5* and the one closest to 5.6*."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_song: dict[str, list[dict]] = {}
    for m in data["maps"]:
        if (m["used"] is None and m["constant_timing"] and m["stars"] is not None
                and 30.0 <= m["duration_s"] <= max_duration):
            by_song.setdefault(m["song"], []).append(m)
    jobs = []
    for key in data["holdout_order"]:
        maps = by_song.get(key, [])
        pick = []
        for (lo, hi), centre, name in ((insane, 4.5, "insane"), (expert, 5.6, "expert")):
            inside = [m for m in maps if lo <= m["stars"] <= hi]
            if inside:
                pick.append((name, min(inside, key=lambda m: abs(m["stars"] - centre))))
        if len(pick) == 2 and len({m["audio"] for _, m in pick}) == 1:
            jobs.append({"song": key, "maps": pick})
        if len(jobs) >= songs:
            break
    return jobs


def run(paths: list[Path], output: Path, seed: int, rhythm_path: Path | None = None,
        holdout: Path | None = None, holdout_songs: int = 4,
        force_selection: str | None = None, dataset: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from beatmap_ai.audio import compute_features, load_audio
    from beatmap_ai.critic import load_critic
    from beatmap_ai.evaluate import constant_timing
    from beatmap_ai.model import load_checkpoint
    from beatmap_ai.planner import load_planner, plan_song
    from beatmap_ai.sequence_model import load_sequence
    from beatmap_ai.structure import find_repeats

    device = generator.inference_device()
    models = Path(generator.__file__).parent / "models"
    rhythm_path = rhythm_path or models / "rhythm.pt"
    rhythm_model, threshold = load_checkpoint(rhythm_path, device=device)
    placement = load_sequence(models / "sequence-v4.pt", device=device)
    planner = load_planner(models / "planner.pt", device=device)
    critic = load_critic(models / "critic.pt", device=device)
    generator.SHAPE_GUIDE = True
    generator.HUMAN_DENSITY = True
    generator.JUMP_BY_CONTROL = True
    repo = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(["git", *a], text=True, cwd=repo)  # noqa: E731
    metadata = {"commit": git("rev-parse", "HEAD").strip(),
                "dirty_files": [line[3:] for line in git("status", "--porcelain").splitlines()],
                "seed": seed, "model": str(rhythm_path), "model_sha256": _sha(rhythm_path),
                "figure_head": int(getattr(rhythm_model, "outputs", 4)) >= 5,
                "sequence": "sequence-v4.pt", "sequence_sha256": _sha(models / "sequence-v4.pt"),
                "planner": "planner.pt", "planner_sha256": _sha(models / "planner.pt"),
                "critic": "critic.pt", "critic_sha256": _sha(models / "critic.pt"),
                "device": str(device), "torch": torch.__version__,
                "python": platform.python_version(), "force_selection": force_selection,
                "options": ["shape-guide", "human-density", "jump-control", "variety=0.5",
                            "passes=1", "candidates=4"]}
    jobs = []
    for path in paths:
        jobs.append({"kind": "osz", "path": path})
    if holdout is not None:
        for job in holdout_jobs(holdout, holdout_songs):
            jobs.append({"kind": "holdout", **job})
    results = []
    for job in jobs:
        if job["kind"] == "osz":
            path = job["path"]
            song_dir = output / path.stem
            song_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                audio_name = _source_audio(archive)
                audio_path = song_dir / Path(audio_name).name
                if not audio_path.exists():
                    audio_path.write_bytes(archive.read(audio_name))
                source_maps = [parse_osu(archive.read(name).decode("utf-8-sig"))
                               for name in archive.namelist() if name.lower().endswith(".osu")]
            features, timing = generator.analyze(audio_path)
            items = []
            for name in ("insane", "expert"):
                source = next((bm for bm in source_maps if bm.version.lower() == name), None)
                if source is not None:
                    items.append((name, name, 4.5 if name == "insane" else 5.6, source, None,
                                  timing))
            label = path.name
        else:
            first = job["maps"][0][1]
            audio_path = Path(first["audio"])
            song_dir = output / f"{first['set']}-{job['song'][:40]}"
            song_dir.mkdir(parents=True, exist_ok=True)
            features = compute_features(load_audio(audio_path), chroma=True)
            items = []
            for name, m in job["maps"]:
                human = parse_osu(Path(m["audio"]).with_name(Path(m["file"]).name)
                                  .read_text(encoding="utf-8", errors="replace"))
                # The human map's timing (straight grid), so rhythm F1 measures the rhythm.
                items.append((name, float(m["stars"]), float(m["stars"]), human, m,
                              constant_timing(human)))
            label = f"{first['artist']} - {first['title']}"
        for i, (name, difficulty, target, source, info, timing) in enumerate(items):
            repeats = (find_repeats(features, timing.offset_ms, timing.beat_length)
                       if features.chroma is not None and not (
                           timing.tempo_points and len(timing.tempo_points) > 1) else [])
            advice = plan_song(planner, features.mel, timing.beat_length, target,
                               timing.offset_ms)
            suggested = {"jump": max(advice["styles"].get("skillset/jumps", 0.0),
                                     advice["styles"].get("jumps/wide", 0.0)),
                         "stream": advice["styles"].get("skillset/streams", 0.0),
                         "tech": advice["styles"].get("skillset/tech", 0.0),
                         "flow": advice["styles"].get("streams/flow aim", 0.0)}
            style = {key: min(0.4, float(value) * 0.4)
                     for key, value in sorted(suggested.items(), key=lambda item: -item[1])[:2]
                     if value >= 0.35} or None
            truth = None
            if info is not None:
                truth = np.array([o.time for o in source.hit_objects if o.kind != "spinner"])
            with Observer(force_selection) as observer:
                bm = generator.generate_beatmap(
                    features, timing, difficulty, seed=seed + i, model=rhythm_model,
                    threshold=threshold, title=source.title, artist=source.artist,
                    audio_filename=audio_path.name, style=style, placement=placement,
                    repeats=repeats, critic=critic, candidates=4, passes=1,
                    section_profile=advice["sections"], crutches=True,
                )
            report = observer.report(bm, timing, truth)
            generated = star_rating(bm.to_osu_string())
            source_events = _events(source.hit_objects, source.end_time)
            report.update({"song": label, "difficulty": name, "target_stars": target,
                           "generated_stars": generated,
                           "star_error": None if generated is None else abs(generated - target),
                           "source": "human" if info is not None else "archived candidate",
                           "source_stars": star_rating(source.to_osu_string()),
                           "source_summary": _summary(source_events, source, truth),
                           "source_to_current_same_times": _overlap(source_events, _events(
                               bm.hit_objects, bm.end_time))})
            if info is not None:
                report.update({"set": info["set"], "file": info["file"],
                               "version": info["version"], "bpm": info["bpm"]})
            results.append(report)
            (song_dir / f"{name}.osu").write_text(bm.to_osu_string(), encoding="utf-8")
            (output / "pipeline.json").write_text(
                json.dumps({"metadata": metadata, "results": results}, indent=2),
                encoding="utf-8")
            print(f"{label}: {name} {generated:.2f}* (target {target:.2f}); "
                  f"{report['stages']['final']['objects']} objects, "
                  f"selection {report['selection_mode']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, help="existing .osz files")
    parser.add_argument("--output", required=True, type=Path, help="directory on D: for results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rhythm-model", type=Path,
                        help="checkpoint to compare against the bundled rhythm model")
    parser.add_argument("--holdout", type=Path, help="JSON from scripts/rhythm_holdout.py")
    parser.add_argument("--holdout-songs", type=int, default=4)
    parser.add_argument("--force-selection", choices=("threshold", "density"))
    args = parser.parse_args()
    if not args.sources and args.holdout is None:
        parser.error("give .osz files and/or --holdout")
    run(args.sources, args.output, args.seed, args.rhythm_model, args.holdout,
        args.holdout_songs, args.force_selection)


if __name__ == "__main__":
    main()
