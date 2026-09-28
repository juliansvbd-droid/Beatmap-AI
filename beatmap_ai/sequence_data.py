"""Training data for the sequence model (no PyTorch needed, so batches can be built in
worker processes cheaply). See sequence_model.py for the model itself."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import FPS
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .placement_data import (BAR, BEAT, CIRCLE, CLAP, CMASK, CU, CV, END, EX, EY, FINISH, HCOS,
                             HSIN, KIAI, KIND, N_COLUMNS, NC, OMASK, OU, OV, PHASE, SLIDER,
                             SLIDES, T, WHISTLE,
                             VEL, X, Y, PlacementMap, WindowSampler, augment_objects,
                             build_placement_maps)
from .jump_shapes import SHAPE_FEATURES, shape_features
from .style import encode_conditions

CONDITIONS = ("density", "stars", "jump", "stream", "sliders")
# Community tags the model is conditioned on (share of the difficulty's top votes).


TAGS = (
    "skillset/jumps", "jumps/sharp", "jumps/wide", "jumps/cross-screen", "jumps/linear",
    "jumps/triangles", "jumps/back and forth", "jumps/squares",
    "streams/bursts", "skillset/streams", "streams/flow aim", "streams/stamina",
    "streams/spaced streams", "streams/cutstreams", "streams/doubles",
    "skillset/alt", "skillset/tech", "tech/aim control", "tech/finger control",
    "tech/slider tech", "sliders/complex slidershapes",
    "expression/simple", "style/clean", "style/geometric", "style/symmetrical",
    "style/freeform", "expression/chaotic", "reading/overlaps", "reading/perfect stacks",
    "expression/repetition",
)


LOOKAHEAD = 17  # quarter-beat slots from this object to four beats ahead
SECTION_CONTROLS = 8  # density, jump size, stream, sliders, turns, fast turns, cross-screen, kiai

try:
    _PATTERN_REFERENCE = json.loads((Path(__file__).with_name("pattern_reference.json"))
                                    .read_text(encoding="utf-8"))
except (OSError, ValueError):
    _PATTERN_REFERENCE = {}


LONG_GAP = 17  # gap class for "more than four beats"


GAP_CLASSES = 18  # 0 is unused (no two objects at the same time), 1..16 quarter beats, 17 long


DURATION_CLASSES = 17  # 0 (circle) .. 16 quarter beats


REPEAT_CLASSES = 3  # 1, 2, 3+ slides


def tag_vector(votes: dict[str, int] | None) -> np.ndarray:
    """Tag values (votes relative to the difficulty's most voted tag) and a "known" flag."""
    out = np.zeros(len(TAGS) + 1, dtype=np.float32)
    if votes:
        top = max(votes.values())
        for i, name in enumerate(TAGS):
            out[i] = votes.get(name, 0) / top
        out[-1] = 1.0
    return out


def load_tags(paths) -> dict[tuple[str, str], dict[str, int]]:
    """(set id, difficulty name) -> {tag name: votes} from fetch_tags.py files."""
    out = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        store = json.loads(path.read_text(encoding="utf-8"))
        names = store.get("names", {})
        for set_id, diffs in store.get("sets", {}).items():
            for version, tags in diffs.items():
                if tags:
                    out[(str(set_id), version)] = {names.get(t, t): n for t, n in tags.items()}
    return out


def map_ids(name: str) -> tuple[str, str] | None:
    """(set id, difficulty name) from a dataset name such as "123/A - B (C) [Hard].osu"
    or "001 123 A - B.osz/A - B (C) [Hard].osu"."""
    version = re.search(r"\[([^\[\]]*)\]\.osu$", name)
    first = re.split(r"[\\/]", name)[0].split()
    if not version or not first:
        return None
    set_id = first[1] if len(first) > 1 and first[0].isdigit() and first[1].isdigit() else first[0]
    return (set_id, version.group(1)) if set_id.isdigit() else None


def rhythm_targets(objects: np.ndarray) -> dict[str, np.ndarray]:
    """For each object: the next object's gap (class), kind, length, slides and combo."""
    n = len(objects)
    beat = np.maximum(objects[:, BEAT], 1.0)
    gap = np.zeros(n, dtype=np.int64)
    kind = np.zeros(n, dtype=np.int64)
    duration = np.zeros(n, dtype=np.int64)
    repeat = np.zeros(n, dtype=np.int64)
    combo = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    if n > 1:
        nxt = objects[1:]
        q = np.rint((nxt[:, T] - objects[:-1, T]) / beat[:-1] * 4).astype(int)
        gap[:-1] = np.where(q > 16, LONG_GAP, np.clip(q, 1, 16))
        kind[:-1] = nxt[:, KIND].astype(int)
        d = np.rint((nxt[:, END] - nxt[:, T]) / np.maximum(nxt[:, BEAT], 1.0) * 4).astype(int)
        duration[:-1] = np.where(nxt[:, KIND] == CIRCLE, 0, np.clip(d, 1, 16))
        repeat[:-1] = np.clip(nxt[:, SLIDES].astype(int), 1, 3) - 1
        combo[:-1] = nxt[:, NC]
        mask[:-1] = 1.0
    return {"gap": gap, "kind": kind, "duration": duration, "repeat": repeat, "combo": combo,
            "rhythm_mask": mask}


def sequence_features(objects: np.ndarray, mel: np.ndarray, cs: float, cond: np.ndarray,
                      energy: np.ndarray) -> np.ndarray:
    """Model input per object: placement_model.token_features' inputs plus the music of
    the next four beats (mel bands halved to 40, at every quarter beat)."""
    n = len(objects)
    o = objects
    prev = np.vstack([np.zeros((1, N_COLUMNS), dtype=np.float32), o[:-1]])
    first = np.zeros(n, dtype=bool)
    first[0] = True
    beat = np.maximum(o[:, BEAT], 1.0)
    gap_start = np.where(first, 8.0, np.clip((o[:, T] - prev[:, T]) / beat, 0, 8))
    gap_end = np.where(first, 8.0, np.clip((o[:, T] - prev[:, END]) / beat, 0, 8))
    kind = np.eye(3, dtype=np.float32)[o[:, KIND].astype(int)]
    prev_kind = np.eye(3, dtype=np.float32)[prev[:, KIND].astype(int)] * (~first)[:, None]
    duration = np.clip((o[:, END] - o[:, T]) / beat, 0, 8) / 4.0
    phase = 2 * np.pi * o[:, PHASE]
    bar = 2 * np.pi * o[:, BAR]
    prev_offset = prev[:, [OU, OV]] * prev[:, [OMASK]]
    prev_chord = prev[:, [CU, CV]] * prev[:, [CMASK]]

    # Read the audio once for the whole window, then gather from memory.
    slots = o[:, T][:, None] + np.arange(LOOKAHEAD)[None, :] * beat[:, None] / 4.0
    frames = np.rint(slots * FPS / 1000.0).astype(int)
    total = mel.shape[1]
    lo = int(np.clip(frames.min() - 2, 0, total - 1))
    hi = int(np.clip(frames.max() + 3, lo + 1, total))
    local = np.asarray(mel[:, lo:hi], dtype=np.float32)
    idx = np.clip(frames - lo, 0, local.shape[1] - 1)
    here = np.clip(idx[:, :1] + np.arange(-2, 3)[None, :], 0, local.shape[1] - 1)
    audio = local[:, here].mean(axis=2).T  # (n, 80) around the object
    ahead_idx = np.clip(idx[:, :, None] + np.arange(-1, 2)[None, None, :], 0, local.shape[1] - 1)
    ahead = local[:, ahead_idx].mean(axis=3)  # (80, n, 17)
    ahead = ahead.reshape(40, 2, n, LOOKAHEAD).mean(axis=1).transpose(1, 2, 0).reshape(n, -1)
    section = energy[np.clip(frames[:, 0], 0, len(energy) - 1)]

    columns = [
        np.log1p(gap_start)[:, None], np.log1p(gap_end)[:, None], kind, prev_kind,
        duration[:, None], (o[:, SLIDES] / 4.0)[:, None], o[:, [NC]],
        np.stack([np.sin(phase), np.cos(phase), np.sin(bar), np.cos(bar)], axis=1),
        (o[:, VEL] / 200.0)[:, None], np.full((n, 1), cs / 5.0, dtype=np.float32),
        prev_offset, prev_chord,
        np.stack([prev[:, EX] / PLAYFIELD_WIDTH, prev[:, EY] / PLAYFIELD_HEIGHT], axis=1) * (~first)[:, None],
        o[:, [HCOS, HSIN]], audio, (section * 4.0)[:, None], prev[:, [WHISTLE, FINISH, CLAP]],
        np.broadcast_to(cond, (n, len(cond))), ahead,
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


FEATURES = (2 + 3 + 3 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2 + 2 + 2 + 80 + 1 + 3
            + 2 * len(CONDITIONS) + len(TAGS) + 1 + 40 * LOOKAHEAD)
V3_FEATURES = FEATURES + SECTION_CONTROLS
V4_FEATURES = V3_FEATURES + SHAPE_FEATURES  # v4: jump shape of every object (jump_shapes)
SHAPE_MARGIN = 12  # objects read around a window so that runs at its edges keep their shape
SHAPE_HIDE = 0.3  # share of training windows without shapes (generation without a plan)


def window_shapes(objects: np.ndarray, start: int, count: int, cs: float) -> np.ndarray:
    """shape_features for objects[start:start + count], judged with their neighbours."""
    lo = max(start - SHAPE_MARGIN, 0)
    features = shape_features(objects[lo:start + count + SHAPE_MARGIN], cs)
    return features[start - lo:start - lo + count]


def section_controls(objects: np.ndarray, start: int, count: int, stars: float,
                     cs: float = 4.0) -> np.ndarray:
    """Human local section targets over a centered eight-bar window per object.

    Vectorised: per-move quantities are computed once for the whole map and summed over
    each window with prefix sums (same values as _section_controls_reference, which looped
    in Python per object and window and took hours over the data set)."""
    output = np.zeros((count, SECTION_CONTROLS), dtype=np.float32)
    n = len(objects)
    if n < 2 or count <= 0:
        return output
    times = objects[:, T].astype(np.float64)
    mean_density = n / max((times[-1] - times[0]) / 1000.0, 1.0)
    star_group = "<3" if stars < 3 else "3-4.5" if stars < 4.5 else "4.5-6" if stars < 6 else "6+"
    limit = _PATTERN_REFERENCE.get("features", {}).get(star_group, {})
    sharp_limit = float(limit.get("sharp_turn_speed", {}).get("p95", 0.85))
    cross_limit = float(limit.get("cross_screen_fraction", {}).get("p95", 0.33))
    radius = max(64.0 - 4.48 * (float(cs) - 4.0), 20.0)
    diagonal = math.hypot(512.0, 384.0)

    # Move j (into object j, j >= 1): distance from the previous end, stream/cross flags.
    dx = np.zeros(n)
    dy = np.zeros(n)
    dx[1:] = objects[1:, X].astype(np.float64) - objects[:-1, EX]
    dy[1:] = objects[1:, Y].astype(np.float64) - objects[:-1, EY]
    dist = np.hypot(dx, dy)
    beats = objects[:, BEAT].astype(np.float64)
    stream = (dist / np.maximum(beats, 1.0) <= 0.5).astype(np.float64)
    cross = (dist / diagonal > cross_limit).astype(np.float64)
    stream[0] = cross[0] = dist[0] = 0.0
    # Turn at object j (j >= 2): between move j-1 and move j, both longer than 5 px.
    valid = np.zeros(n, dtype=bool)
    sharp = np.zeros(n)
    fast = np.zeros(n)
    if n >= 3:
        n1, n2 = dist[1:-1], dist[2:]
        ok = (n1 > 5) & (n2 > 5)
        cos = (dx[1:-1] * dx[2:] + dy[1:-1] * dy[2:]) / np.maximum(n1 * n2, 1e-12)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        is_sharp = turn > 120.0
        bpm = 60000.0 / np.maximum(beats[2:], 1.0)
        valid[2:] = ok
        sharp[2:] = np.where(ok, is_sharp, 0.0)
        fast[2:] = np.where(ok, is_sharp & (turn / 180.0 * bpm / 180.0 > sharp_limit), 0.0)

    def prefix(values):
        return np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])

    c_dist, c_stream, c_cross = prefix(dist), prefix(stream), prefix(cross)
    c_valid, c_sharp, c_fast = prefix(valid), prefix(sharp), prefix(fast)
    c_slider = prefix(objects[:, KIND] == SLIDER)
    c_kiai = prefix(objects[:, KIAI] > 0.5)

    index = np.arange(start, min(start + count, n))
    beat = np.maximum(beats[index], 1.0)
    lo = np.searchsorted(times, times[index] - 16.0 * beat, side="left")
    hi = np.searchsorted(times, times[index] + 16.0 * beat, side="right")
    local = hi - lo
    span = np.maximum((times[hi - 1] - times[lo]) / 1000.0, beat / 1000.0)
    density = local / span
    a = np.maximum(lo + 1, 1)
    steps = np.maximum(hi - a, 0)
    turns = c_valid[hi] - c_valid[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_distance = np.where(steps > 0, (c_dist[hi] - c_dist[a]) / steps, 0.0)
        values = np.stack([
            np.clip(np.log1p(density) / max(math.log1p(max(mean_density, 1.0)), 1e-4), 0, 2) / 2,
            np.clip(mean_distance / max(radius * 5.0, 1.0), 0, 2) / 2,
            np.where(steps > 0, (c_stream[hi] - c_stream[a]) / steps, 0.0),
            (c_slider[hi] - c_slider[lo]) / local,
            np.where(turns > 0, (c_sharp[hi] - c_sharp[a]) / turns, 0.0),
            np.where(turns > 0, (c_fast[hi] - c_fast[a]) / turns, 0.0),
            np.where(steps > 0, (c_cross[hi] - c_cross[a]) / steps, 0.0),
            (c_kiai[hi] - c_kiai[lo]) / local,
        ], axis=1)
    output[:len(index)] = values
    return output


def _section_controls_reference(objects: np.ndarray, start: int, count: int, stars: float,
                     cs: float = 4.0) -> np.ndarray:
    """Loop version of section_controls, kept as the reference its tests compare to."""
    output = np.zeros((count, SECTION_CONTROLS), dtype=np.float32)
    if len(objects) < 2 or count <= 0:
        return output
    times = objects[:, T]
    valid_gaps = np.diff(times) / np.maximum(objects[:-1, BEAT], 1.0)
    valid_gaps = valid_gaps[np.isfinite(valid_gaps) & (valid_gaps > 0)]
    mean_density = len(objects) / max((times[-1] - times[0]) / 1000.0, 1.0)
    star_group = "<3" if stars < 3 else "3-4.5" if stars < 4.5 else "4.5-6" if stars < 6 else "6+"
    limit = _PATTERN_REFERENCE.get("features", {}).get(star_group, {})
    sharp_limit = float(limit.get("sharp_turn_speed", {}).get("p95", 0.85))
    cross_limit = float(limit.get("cross_screen_fraction", {}).get("p95", 0.33))
    radius = max(64.0 - 4.48 * (float(cs) - 4.0), 20.0)
    diagonal = math.hypot(512.0, 384.0)
    for out_i, index in enumerate(range(start, min(start + count, len(objects)))):
        beat = max(float(objects[index, BEAT]), 1.0)
        lo = int(np.searchsorted(times, times[index] - 16.0 * beat, side="left"))
        hi = int(np.searchsorted(times, times[index] + 16.0 * beat, side="right"))
        local = objects[lo:hi]
        if not len(local):
            continue
        span = max((local[-1, T] - local[0, T]) / 1000.0, beat / 1000.0)
        density = len(local) / span
        steps = []
        turns, fast_turns, cross = [], [], []
        for j in range(max(lo + 1, 1), hi):
            prev, cur = objects[j - 1], objects[j]
            gap_ms = max(float(cur[T] - prev[END]), 1.0)
            dx, dy = float(cur[X] - prev[EX]), float(cur[Y] - prev[EY])
            distance = math.hypot(dx, dy)
            steps.append((distance, gap_ms, float(cur[BEAT])))
            if j >= 2:
                before = objects[j - 1]
                prior = objects[j - 2]
                v1 = np.array([before[X] - prior[EX], before[Y] - prior[EY]], dtype=float)
                v2 = np.array([cur[X] - before[EX], cur[Y] - before[EY]], dtype=float)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 5 and n2 > 5:
                    turn = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))
                    sharp = turn > 120.0
                    turns.append(float(sharp))
                    bpm = 60000.0 / max(float(cur[BEAT]), 1.0)
                    fast_turns.append(float(sharp and turn / 180.0 * bpm / 180.0 > sharp_limit))
            cross.append(float(distance / diagonal > cross_limit))
        mean_distance = float(np.mean([v[0] for v in steps])) if steps else 0.0
        stream = float(np.mean([v[0] / max(v[2], 1.0) <= 0.5 for v in steps])) if steps else 0.0
        output[out_i] = (
            np.clip(math.log1p(density) / max(math.log1p(max(mean_density, 1.0)), 1e-4), 0, 2) / 2,
            np.clip(mean_distance / max(radius * 5.0, 1.0), 0, 2) / 2,
            stream,
            float(np.mean(local[:, KIND] == SLIDER)),
            float(np.mean(turns)) if turns else 0.0,
            float(np.mean(fast_turns)) if fast_turns else 0.0,
            float(np.mean(cross)) if cross else 0.0,
            float(np.mean(local[:, KIAI] > 0.5)),
        )
    return output


@dataclass
class SequenceMap(PlacementMap):
    tags: np.ndarray | None = None


def _v3_style_bucket(m) -> str:
    """Coarse main style used to balance v3 windows within each half-star level."""
    tags = np.asarray(getattr(m, "tags", []), dtype=np.float32)
    groups = {
        "jump": ("skillset/jumps", "jumps/sharp", "jumps/wide", "jumps/cross-screen",
                 "jumps/linear", "jumps/triangles", "jumps/back and forth", "jumps/squares"),
        "stream": ("skillset/streams", "streams/bursts", "streams/flow aim", "streams/stamina",
                   "streams/spaced streams", "streams/cutstreams", "streams/doubles"),
        "tech": ("skillset/tech", "tech/aim control", "tech/finger control", "tech/slider tech"),
        "flow/simple": ("expression/simple", "style/clean", "style/freeform", "expression/repetition"),
    }
    scores = {}
    for name, members in groups.items():
        indices = [TAGS.index(tag) for tag in members if tag in TAGS]
        scores[name] = float(np.max(tags[indices])) if indices and len(tags) > max(indices) else 0.0
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] >= 0.35 and ordered[1][1] >= 0.35:
        if ordered[0][1] - ordered[1][1] <= 0.15:
            return "mixed"
    if ordered and ordered[0][1] >= 0.3:
        return ordered[0][0]

    style = getattr(m, "style", {}) or {}
    fallback = {
        "jump": float(np.clip((float(style.get("jump", 0.0)) - 0.7) / 0.8, 0, 1)),
        "stream": float(np.clip(float(style.get("stream", 0.0)) / 0.5, 0, 1)),
        "tech": float(np.clip(float(style.get("tech", 0.0)), 0, 1)),
        "flow/simple": float(np.clip(float(style.get("sliders", 0.0)) * 0.7
                                      + (1.0 - float(style.get("stream", 0.0))) * 0.3, 0, 1)),
    }
    ordered = sorted(fallback.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] >= 0.35 and ordered[1][1] >= 0.35:
        if ordered[0][1] - ordered[1][1] <= 0.15:
            return "mixed"
    return ordered[0][0] if ordered and ordered[0][1] >= 0.35 else "flow/simple"


def _v3_sampling_weights(maps) -> np.ndarray:
    """Give each available (half-star, main-style) bucket equal sampling weight."""
    stars = np.asarray([m.style.get("stars", np.nan) for m in maps], dtype=float)
    levels = np.where(np.isfinite(stars), np.floor(np.nan_to_num(stars) * 2), -1)
    buckets = [(int(level), _v3_style_bucket(m)) for level, m in zip(levels, maps)]
    counts = Counter(buckets)
    weights = np.asarray([1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    return weights / weights.sum()


def build_sequence_maps(roots, tag_files=(), cache_dir=None, log=print,
                       tagger_path=None) -> list[SequenceMap]:
    tags = load_tags(tag_files)
    maps = []
    for m in build_placement_maps(roots, cache_dir, log=log):
        ids = map_ids(m.name)
        votes = tags.get(ids) if ids else None
        maps.append(SequenceMap(**vars(m), tags=tag_vector(votes)))
    if tagger_path and Path(tagger_path).is_file():
        from .tagger import load_tagger, tagger_features, tag_probabilities
        tagger = load_tagger(tagger_path)
        metrics = torch_load_tagger_metrics(tagger_path)
        auc_by_tag = metrics.get("auc_by_tag", {})
        predicted = 0
        for m in maps:
            if m.tags[-1] > 0:
                continue
            values = tag_probabilities(
                tagger, tagger_features(m.objects, float(m.style.get("stars", 4.0)), m.cs)[None]
            )[0]
            keep = np.array([float(auc_by_tag.get(name, 0.5)) >= 0.58 for name in TAGS])
            values = np.where(keep & (values >= 0.35), values, 0.0)
            if values.max() > 0:
                m.tags[:-1] = values / values.max()
                m.tags[-1] = 0.5  # Distinguish calibrated predictions from community votes.
                predicted += 1
        log(f"tagger supplied calibrated tags for {predicted} additional maps")
    tagged = sum(1 for m in maps if m.tags[-1] > 0)
    log(f"{tagged} of {len(maps)} maps have community tags")
    return maps


def torch_load_tagger_metrics(path) -> dict:
    """Read scalar tagger metrics without initializing an additional device context."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("metrics", {})


class SequenceSampler(WindowSampler):
    """Windows of objects with style, tag and rhythm targets. One extra object past the
    window supplies the last object's "next" targets."""

    def __init__(self, maps, context: int, seed: int = 0, hide: bool = True, v3: bool = False,
                 shapes: bool = False):
        super().__init__(maps, context, seed, hide)
        self.v3 = v3 or shapes
        self.shapes = shapes
        if v3:
            self.weights = _v3_sampling_weights(self.maps)

    def conditions(self, m: SequenceMap) -> np.ndarray:
        values = dict(m.style)
        tags = m.tags.copy()
        if self.hide:
            if self.rng.random() < 0.1:
                values = {}
            for name in CONDITIONS:
                if self.rng.random() < (0.5 if name == "density" else 0.3):
                    values.pop(name, None)
            if self.rng.random() < 0.25:
                tags[:] = 0.0  # "tags unknown", as for untagged maps and Auto
        return np.concatenate([encode_conditions(values, CONDITIONS), tags])

    def batch(self, size: int) -> dict[str, np.ndarray]:
        out: dict[str, list] = {}
        for _ in range(size):
            m = self.maps[int(self.rng.choice(len(self.maps), p=self.weights))]
            start = int(self.rng.integers(max(len(m.objects) - self.context, 0) + 1))
            objects = m.objects[start:start + self.context + 1]
            mirrored = False
            if self.hide:
                flip_x = bool(self.rng.integers(2))
                flip_y = bool(self.rng.integers(2))
                objects = augment_objects(objects, flip_x, flip_y)
                mirrored = flip_x != flip_y
            window = objects[:self.context]
            x = sequence_features(window, self.mel(m.mel_path), m.cs, self.conditions(m),
                                  self.energy(m.mel_path))
            if self.v3:
                stars = float(m.style.get("stars", 4.5))
                local = section_controls(m.objects, start, len(window), stars, m.cs)
                if self.hide and self.rng.random() < 0.4:
                    local[:] = 0.0
                x = np.concatenate([x, local], axis=1)
            if self.shapes:
                shape = window_shapes(m.objects, start, len(window), m.cs)
                if mirrored:
                    shape[:, -1] *= -1.0  # a mirror image turns the other way
                if self.hide and self.rng.random() < SHAPE_HIDE:
                    shape[:] = 0.0
                x = np.concatenate([x, shape], axis=1)
            targets = {k: v[:len(window)] for k, v in rhythm_targets(objects).items()}
            pad = self.context - len(window)
            items = {"x": np.pad(x, ((0, pad), (0, 0))), "y": np.pad(window, ((0, pad), (0, 0))),
                     **{k: np.pad(v, (0, pad)) for k, v in targets.items()}}
            for k, v in items.items():
                out.setdefault(k, []).append(v)
        return {k: np.stack(v) for k, v in out.items()}


@dataclass
class MapRef:
    """A map whose objects live in a shared memory-mapped array."""
    start: int
    end: int
    song: str
    mel_path: Path
    cs: float
    style: dict
    tags: np.ndarray
    name: str = ""


class SharedMaps:
    def __init__(self, path: Path, refs: list[MapRef]):
        self.path = path
        self.refs = refs
        self._objects = None

    def objects(self, ref: MapRef) -> np.ndarray:
        if self._objects is None:
            self._objects = np.load(self.path, mmap_mode="r")
        return np.asarray(self._objects[ref.start:ref.end])

    def __getstate__(self):
        return {"path": self.path, "refs": self.refs, "_objects": None}


def share_maps(maps: list[SequenceMap], path: Path) -> SharedMaps:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(m.objects) for m in maps)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, N_COLUMNS))
    refs, at = [], 0
    for m in maps:
        out[at:at + len(m.objects)] = m.objects
        refs.append(MapRef(at, at + len(m.objects), m.song, m.mel_path, m.cs, m.style, m.tags, m.name))
        at += len(m.objects)
    out.flush()
    del out
    return SharedMaps(path, refs)


class SharedSampler(SequenceSampler):
    def __init__(self, shared: SharedMaps, context: int, seed: int = 0, hide: bool = True,
                 v3: bool = False, shapes: bool = False):
        self.shared = shared
        self.shapes = shapes
        v3 = v3 or shapes
        refs = [r for r in shared.refs if r.end - r.start >= 16]
        self.maps = refs
        self.context = context
        self.rng = np.random.default_rng(seed)
        self.hide = hide
        self.v3 = v3
        stars = np.array([r.style.get("stars", np.nan) for r in refs], dtype=float)
        if v3:
            self.weights = _v3_sampling_weights(refs)
        else:
            buckets = np.where(np.isfinite(stars), np.floor(np.nan_to_num(stars) * 2), -1)
            _, inverse, counts = np.unique(buckets, return_inverse=True, return_counts=True)
            weights = 1.0 / np.sqrt(counts[inverse])
            self.weights = weights / weights.sum()
        self._mels, self._energy = {}, {}

    def batch(self, size: int) -> dict[str, np.ndarray]:
        out: dict[str, list] = {}
        for _ in range(size):
            ref = self.maps[int(self.rng.choice(len(self.maps), p=self.weights))]
            n = ref.end - ref.start
            start = int(self.rng.integers(max(n - self.context, 0) + 1))
            objects = self.shared.objects(ref)[start:start + self.context + 1]
            window = objects[:self.context]
            x = sequence_features(window, self.mel(ref.mel_path), ref.cs, self.conditions(ref),
                                  self.energy(ref.mel_path))
            if self.v3:
                all_objects = self.shared.objects(ref)
                stars = float(ref.style.get("stars", 4.5))
                local = section_controls(all_objects, start, len(window), stars, ref.cs)
                if self.hide and self.rng.random() < 0.4:
                    local[:] = 0.0
                x = np.concatenate([x, local], axis=1)
            if self.shapes:
                shape = window_shapes(self.shared.objects(ref), start, len(window), ref.cs)
                if self.hide and self.rng.random() < SHAPE_HIDE:
                    shape[:] = 0.0
                x = np.concatenate([x, shape], axis=1)
            targets = {k: v[:len(window)] for k, v in rhythm_targets(objects).items()}
            pad = self.context - len(window)
            items = {"x": np.pad(x, ((0, pad), (0, 0))), "y": np.pad(window, ((0, pad), (0, 0))),
                     **{k: np.pad(v, (0, pad)) for k, v in targets.items()}}
            for k, v in items.items():
                out.setdefault(k, []).append(v)
        return {k: np.stack(v) for k, v in out.items()}


# Batches are built by worker processes that only load NumPy (not PyTorch, which
# reserves several GB per process with ROCm on Windows).
_worker = None


def _init_worker(shared: SharedMaps, context: int, batch_size: int, seed: int,
                 v3: bool = False, shapes: bool = False) -> None:
    global _worker
    _worker = (SharedSampler(shared, context, seed + os.getpid(), v3=v3, shapes=shapes), batch_size)


def _make_batch(_=None) -> dict[str, np.ndarray]:
    sampler, size = _worker
    batch = sampler.batch(size)
    batch["x"] = batch["x"].astype(np.float16)  # half the bytes through the pipe
    return batch


def batch_stream(shared: SharedMaps, context: int, batch_size: int, seed: int, workers: int = 4,
                 prefetch: int = 8, v3: bool = False, shapes: bool = False):
    """Endless training batches from ``workers`` processes (or this one with 0)."""
    if workers <= 0:
        _init_worker(shared, context, batch_size, seed, v3=v3, shapes=shapes)
        while True:
            yield _make_batch()
    pool = multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(shared, context, batch_size, seed, v3, shapes))
    pending = deque(pool.apply_async(_make_batch) for _ in range(prefetch))
    try:
        while True:
            result = pending.popleft()
            pending.append(pool.apply_async(_make_batch))
            yield result.get()
    finally:
        pool.terminate()
