"""Training data for the sequence model (no PyTorch needed, so batches can be built in
worker processes cheaply). See sequence_model.py for the model itself."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import FPS
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .placement_data import (BAR, BEAT, CIRCLE, CLAP, CMASK, CU, CV, END, EX, EY, FINISH, HCOS,
                             HSIN, KIND, N_COLUMNS, NC, OMASK, OU, OV, PHASE, SLIDES, T, VEL,
                             WHISTLE, PlacementMap, WindowSampler, augment_objects,
                             build_placement_maps)
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


@dataclass
class SequenceMap(PlacementMap):
    tags: np.ndarray | None = None


def build_sequence_maps(roots, tag_files=(), cache_dir=None, log=print) -> list[SequenceMap]:
    tags = load_tags(tag_files)
    maps = []
    for m in build_placement_maps(roots, cache_dir, log=log):
        ids = map_ids(m.name)
        votes = tags.get(ids) if ids else None
        maps.append(SequenceMap(**vars(m), tags=tag_vector(votes)))
    tagged = sum(1 for m in maps if m.tags[-1] > 0)
    log(f"{tagged} of {len(maps)} maps have community tags")
    return maps


class SequenceSampler(WindowSampler):
    """Windows of objects with style, tag and rhythm targets. One extra object past the
    window supplies the last object's "next" targets."""

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
            if self.hide:
                flip_x = bool(self.rng.integers(2))
                flip_y = bool(self.rng.integers(2))
                objects = augment_objects(objects, flip_x, flip_y)
            window = objects[:self.context]
            x = sequence_features(window, self.mel(m.mel_path), m.cs, self.conditions(m),
                                  self.energy(m.mel_path))
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
    def __init__(self, shared: SharedMaps, context: int, seed: int = 0, hide: bool = True):
        self.shared = shared
        refs = [r for r in shared.refs if r.end - r.start >= 16]
        self.maps = refs
        self.context = context
        self.rng = np.random.default_rng(seed)
        self.hide = hide
        stars = np.array([r.style.get("stars", np.nan) for r in refs], dtype=float)
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


def _init_worker(shared: SharedMaps, context: int, batch_size: int, seed: int) -> None:
    global _worker
    _worker = (SharedSampler(shared, context, seed + os.getpid()), batch_size)


def _make_batch(_=None) -> dict[str, np.ndarray]:
    sampler, size = _worker
    batch = sampler.batch(size)
    batch["x"] = batch["x"].astype(np.float16)  # half the bytes through the pipe
    return batch


def batch_stream(shared: SharedMaps, context: int, batch_size: int, seed: int, workers: int = 4,
                 prefetch: int = 8):
    """Endless training batches from ``workers`` processes (or this one with 0)."""
    if workers <= 0:
        _init_worker(shared, context, batch_size, seed)
        while True:
            yield _make_batch()
    pool = multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(shared, context, batch_size, seed))
    pending = deque(pool.apply_async(_make_batch) for _ in range(prefetch))
    try:
        while True:
            result = pending.popleft()
            pending.append(pool.apply_async(_make_batch))
            yield result.get()
    finally:
        pool.terminate()
