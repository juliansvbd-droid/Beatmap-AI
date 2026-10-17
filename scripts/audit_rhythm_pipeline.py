"""Trace rhythm changes along the real v4 Forms generation path.

This diagnostic observes existing functions without changing their output. Run one
process at a time: generation loads the ROCm rhythm, planner and sequence models.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beatmap_ai import generator, rhythm, sequence_model
from beatmap_ai.patterns import _rhythm_metrics
from beatmap_ai.osu import parse_osu
from beatmap_ai.style import star_rating


def _events(objects) -> list[dict]:
    return [
        {"time": round(float(obj.time), 3), "kind": obj.kind,
         "slides": int(getattr(obj, "slides", 1)),
         "x": round(float(obj.x), 3) if hasattr(obj, "x") else None,
         "y": round(float(obj.y), 3) if hasattr(obj, "y") else None}
        for obj in objects if obj.kind != "spinner"
    ]


def _tick_events(grid, ticks) -> list[dict]:
    return [{"time": round(float(grid.times[int(t)]), 3), "kind": "circle", "slides": 1,
             "x": None, "y": None} for t in ticks]


def _signature(events: list[dict]) -> tuple:
    return tuple((item["time"], item["kind"]) for item in events)


def _summary(events: list[dict], timing) -> dict:
    if hasattr(timing, "timing_at"):
        view = timing
    else:
        points = timing.tempo_points or [(timing.offset_ms, timing.beat_length)]
        starts = [float(p[0]) for p in points]
        beats = [float(p[1]) for p in points]

        class TimingView:
            def timing_at(self, time):
                return beats[max(0, bisect_right(starts, time) - 1)], 1.0

        view = TimingView()

    notes = [(SimpleNamespace(**item), None) for item in events]
    metrics = _rhythm_metrics(view, notes)
    n = len(events)
    return {
        "objects": n,
        "double_1_4_per_100": metrics["double_1_4"],
        "triple_1_4_per_100": metrics["triple_1_4"],
        "burst_4_8_1_4_per_100": metrics["burst_4_8_1_4"],
        "slider_pct": 100.0 * sum(e["kind"] == "slider" for e in events) / max(n, 1),
        "repeat_slider_per_100": 100.0 * sum(e["kind"] == "slider" and e["slides"] > 1
                                         for e in events) / max(n, 1),
    }


def _overlap(first: list[dict], second: list[dict], *, positions: bool = False) -> int:
    def identity(item):
        base = (round(item["time"]), item["kind"])
        return base + (round(item["x"]), round(item["y"])) if positions else base
    return len({identity(e) for e in first} & {identity(e) for e in second})


class Observer:
    def __init__(self):
        self.active = None
        self.plans = {}
        self.followed = {}
        self.renders = []
        self.originals = {}

    def __enter__(self):
        self.originals = {
            "grid": rhythm.make_tick_grid,
            "shape": rhythm.shape_triples,
            "density": rhythm.select_ticks_by_density,
            "absorb": rhythm.absorb_into_sliders,
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

        def absorb(ticks, *args, **kwargs):
            result = observer.originals["absorb"](ticks, *args, **kwargs)
            if observer.active is not None:
                current = observer.active
                current["before_absorb"] = _tick_events(current["grid"], ticks)
                current["after_absorb"] = _tick_events(current["grid"], result)
            return result

        def plan(*args, **kwargs):
            current = {"selection_mode": args[7] if len(args) > 7 else
                       kwargs.get("selection", "threshold")}
            observer.active = current
            try:
                result = observer.originals["plan"](*args, **kwargs)
            finally:
                observer.active = None
            current.pop("grid", None)
            current["planned"] = _events(result)
            observer.plans[_signature(current["planned"])] = current
            return result

        def follow(placer, plan_items, *args, **kwargs):
            key = _signature(_events(plan_items))
            result = observer.originals["follow"](placer, plan_items, *args, **kwargs)
            observer.followed[id(placer)] = {"plan_key": key, "after_follow": _events(result[0])}
            return result

        def render(placer, plan_items, *args, **kwargs):
            result = observer.originals["render"](placer, plan_items, *args, **kwargs)
            observer.renders.append({"placer": id(placer), "rendered": _events(result)})
            return result

        rhythm.make_tick_grid = grid
        rhythm.shape_triples = shape
        rhythm.select_ticks_by_density = density
        rhythm.absorb_into_sliders = absorb
        generator.plan_objects = plan
        sequence_model.SequencePlacer.follow = follow
        sequence_model.SequencePlacer.render = render
        return self

    def __exit__(self, *_):
        rhythm.make_tick_grid = self.originals["grid"]
        rhythm.shape_triples = self.originals["shape"]
        rhythm.select_ticks_by_density = self.originals["density"]
        rhythm.absorb_into_sliders = self.originals["absorb"]
        generator.plan_objects = self.originals["plan"]
        sequence_model.SequencePlacer.follow = self.originals["follow"]
        sequence_model.SequencePlacer.render = self.originals["render"]

    def report(self, final, timing):
        final_events = _events(final.hit_objects)
        if not self.renders:
            raise RuntimeError("the selected sequence model produced no render trace")
        best = max(self.renders, key=lambda r: _overlap(r["rendered"], final_events,
                                                        positions=True))
        followed = self.followed[best["placer"]]
        plan = self.plans[followed["plan_key"]]
        stages = {name: _summary(events, timing) for name, events in (
            (name, plan[name]) for name in ("selected", "after_shape", "before_absorb",
                                          "after_absorb", "planned") if name in plan
        )}
        stages["after_follow"] = _summary(followed["after_follow"], timing)
        stages["rendered"] = _summary(best["rendered"], timing)
        stages["final"] = _summary(final_events, timing)
        changes = {}
        order = [name for name in ("selected", "after_shape", "before_absorb", "after_absorb",
                                   "planned", "after_follow", "rendered", "final") if name in stages]
        values = {**plan, "after_follow": followed["after_follow"],
                  "rendered": best["rendered"], "final": final_events}
        for a, b in zip(order, order[1:]):
            overlap = _overlap(values[a], values[b])
            times_a = {round(e["time"]) for e in values[a]}
            times_b = {round(e["time"]) for e in values[b]}
            changes[f"{a}_to_{b}"] = {"shared": overlap,
                                        "removed": len(values[a]) - overlap,
                                        "added": len(values[b]) - overlap,
                                        "shared_times": len(times_a & times_b)}
        return {"selection_mode": plan["selection_mode"], "stages": stages,
                "changes": changes,
                "matched_render_positions": _overlap(best["rendered"], final_events,
                                                       positions=True),
                "render_candidates": len(self.renders), "plan_candidates": len(self.plans)}


def _source_audio(archive: zipfile.ZipFile) -> str:
    candidates = [name for name in archive.namelist()
                  if name.lower().endswith((".mp3", ".ogg"))]
    if len(candidates) != 1:
        raise ValueError(f"expected one audio file in .osz, found {len(candidates)}")
    return candidates[0]


def run(paths: list[Path], output: Path, seed: int, rhythm_path: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    from beatmap_ai.model import load_checkpoint
    from beatmap_ai.planner import load_planner, plan_song
    from beatmap_ai.sequence_model import load_sequence
    from beatmap_ai.critic import load_critic
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
    metadata = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "seed": seed, "model": str(rhythm_path), "sequence": "sequence-v4.pt",
                "planner": "planner.pt", "critic": "critic.pt",
                "options": ["shape-guide", "human-density", "jump-control", "variety=0.5",
                            "passes=1", "candidates=4"]}
    results = []
    for path in paths:
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
        repeats = (find_repeats(features, timing.offset_ms, timing.beat_length)
                   if features.chroma is not None and not (timing.tempo_points and
                                                          len(timing.tempo_points) > 1) else [])
        for i, difficulty in enumerate(("insane", "expert")):
            source = next((bm for bm in source_maps if bm.version.lower() == difficulty), None)
            if source is None:
                continue
            target = 4.5 if difficulty == "insane" else 5.6
            advice = plan_song(planner, features.mel, timing.beat_length, target,
                               timing.offset_ms)
            suggested = {"jump": max(advice["styles"].get("skillset/jumps", 0.0),
                                     advice["styles"].get("jumps/wide", 0.0)),
                         "stream": advice["styles"].get("skillset/streams", 0.0),
                         "tech": advice["styles"].get("skillset/tech", 0.0),
                         "flow": advice["styles"].get("streams/flow aim", 0.0)}
            style = {name: min(0.4, float(value) * 0.4)
                     for name, value in sorted(suggested.items(), key=lambda item: -item[1])[:2]
                     if value >= 0.35} or None
            with Observer() as observer:
                bm = generator.generate_beatmap(
                    features, timing, difficulty, seed=seed + i, model=rhythm_model,
                    threshold=threshold, title=source.title, artist=source.artist,
                    audio_filename=audio_path.name, style=style, placement=placement,
                    repeats=repeats, critic=critic, candidates=4, passes=1,
                    section_profile=advice["sections"], crutches=True,
                )
            report = observer.report(bm, timing)
            report.update({"song": path.name, "difficulty": difficulty,
                           "target_stars": target,
                           "archived_candidate_stars": star_rating(source.to_osu_string()),
                           "archived_candidate": _summary(_events(source.hit_objects), source),
                           "generated_stars": star_rating(bm.to_osu_string()),
                           "archived_to_current_same_times": _overlap(
                               _events(source.hit_objects), _events(bm.hit_objects))})
            results.append(report)
            (song_dir / f"{difficulty}.osu").write_text(bm.to_osu_string(), encoding="utf-8")
            (output / "pipeline.json").write_text(
                json.dumps({"metadata": metadata, "results": results}, indent=2),
                encoding="utf-8")
            print(f"{path.name}: {difficulty} {report['generated_stars']:.2f}*; "
                  f"{report['stages']['final']['objects']} objects", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="existing .osz files")
    parser.add_argument("--output", required=True, type=Path, help="directory on D: for results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rhythm-model", type=Path,
                        help="checkpoint to compare against the bundled rhythm model")
    args = parser.parse_args()
    run(args.sources, args.output, args.seed, args.rhythm_model)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
