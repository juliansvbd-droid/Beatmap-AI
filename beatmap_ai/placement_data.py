"""Object sequences for the learned placement and sequence models (no PyTorch needed).

Per-object rows (``map_objects``), model inputs (``token_features``), the maps with
their cached objects (``build_placement_maps``) and random training windows
(``WindowSampler``). The models live in placement_model.py and sequence_model.py.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .audio import FPS
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, Beatmap, HitObject

# Columns of the per-object array (see map_objects).
T, END, X, Y, EX, EY, KIND, SLIDES, NC, LEN, BEAT, VEL, PHASE, BAR = range(14)
OU, OV, OMASK, CU, CV, CMASK, BEND, BMASK, HCOS, HSIN = range(14, 24)
WHISTLE, FINISH, CLAP = range(24, 27)
N_COLUMNS = 27
HITSOUND_BITS = {WHISTLE: 2, FINISH: 4, CLAP: 8}
SECTION_SECONDS = 4.0  # half-width of the window for section intensity
CIRCLE, SLIDER, SPINNER = 0, 1, 2
MIN_MOVE = 5.0  # px; shorter movements (stacks) keep the previous direction
OFFSET_SCALE = 100.0  # offsets are predicted in units of 100 px
CONDITIONS = ("density", "stars", "jump", "stream", "sliders")
PLACEMENT_CACHE = "placement.pkl"


def rotate(vx: float, vy: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * vx - s * vy, s * vx + c * vy


def slider_far_point(obj: HitObject) -> tuple[float, float]:
    return obj.curve_points[-1] if obj.curve_points else (obj.x, obj.y)


def slider_side(obj: HitObject) -> float | None:
    """+1 if the slider bends to the left of its chord, -1 to the right, None if straight."""
    if obj.curve_type == "L" or len(obj.curve_points) < 2:
        return None
    fx, fy = slider_far_point(obj)
    cx, cy = fx - obj.x, fy - obj.y
    mx, my = obj.curve_points[0][0] - obj.x, obj.curve_points[0][1] - obj.y
    cross = cx * my - cy * mx
    if abs(cross) < 1e-6 * (cx * cx + cy * cy + 1.0):
        return None
    return 1.0 if cross > 0 else -1.0


def augment_objects(objects: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Apply horizontal and/or vertical playfield symmetry to an object sequence.
    Preserves heading and relative angle consistency across all columns."""
    if not (flip_x or flip_y) or len(objects) == 0:
        return objects
    o = objects.copy()
    if flip_x:
        o[:, X] = PLAYFIELD_WIDTH - o[:, X]
        o[:, EX] = PLAYFIELD_WIDTH - o[:, EX]
        o[:, HCOS] = -o[:, HCOS]
    if flip_y:
        o[:, Y] = PLAYFIELD_HEIGHT - o[:, Y]
        o[:, EY] = PLAYFIELD_HEIGHT - o[:, EY]
        o[:, HSIN] = -o[:, HSIN]
    if flip_x ^ flip_y:
        o[:, OV] = -o[:, OV]
        o[:, CV] = -o[:, CV]
        mask = o[:, BMASK] > 0
        o[mask, BEND] = 1.0 - o[mask, BEND]
    return o


def map_objects(bm: Beatmap) -> np.ndarray:
    """Per-object inputs and targets (N, N_COLUMNS) for one osu!standard map."""
    objs = bm.hit_objects
    out = np.zeros((len(objs), N_COLUMNS), dtype=np.float32)
    first = next(tp for tp in bm.timing_points if tp.uninherited)
    heading = 0.0
    prev_end = None
    for i, o in enumerate(objs):
        beat_length, sv = bm.timing_at(o.time)
        end = bm.end_time(o)
        kind = SLIDER if o.kind == "slider" else SPINNER if o.kind == "spinner" else CIRCLE
        if kind == SPINNER:
            ex, ey = 256.0, 192.0
        elif kind == SLIDER and o.slides % 2 == 1:
            ex, ey = slider_far_point(o)
        else:
            ex, ey = o.x, o.y
        # Beat and bar position from the uninherited timing point in effect.
        red = max((tp for tp in bm.timing_points if tp.uninherited and tp.time <= o.time + 1),
                  key=lambda tp: tp.time, default=first)
        beats = (o.time - red.time) / beat_length
        row = out[i]
        row[[T, END, X, Y, EX, EY]] = o.time, end, o.x, o.y, ex, ey
        row[KIND], row[SLIDES], row[NC] = kind, o.slides, float(o.new_combo)
        row[LEN], row[BEAT] = o.length, beat_length
        row[VEL] = bm.slider_multiplier * 100.0 * sv
        row[PHASE], row[BAR] = beats % 1.0, (beats % red.meter) / red.meter
        row[HCOS], row[HSIN] = math.cos(heading), math.sin(heading)
        for column, bit in HITSOUND_BITS.items():
            row[column] = float(bool(o.hit_sound & bit))
        if prev_end is not None and kind != SPINNER and objs[i - 1].kind != "spinner":
            dx, dy = o.x - prev_end[0], o.y - prev_end[1]
            u, v = rotate(dx, dy, -heading)
            gap = (o.time - out[i - 1, END]) / beat_length
            row[OU], row[OV] = u / OFFSET_SCALE, v / OFFSET_SCALE
            row[OMASK] = float(gap <= 4.0)
            if math.hypot(dx, dy) >= MIN_MOVE:
                heading = math.atan2(dy, dx)
        if kind == SLIDER and o.length > 10:
            fx, fy = slider_far_point(o)
            cu, cv = rotate(fx - o.x, fy - o.y, -heading)
            row[CU], row[CV], row[CMASK] = cu / o.length, cv / o.length, 1.0
            side = slider_side(o)
            if side is not None and math.hypot(cu, cv) < 0.97 * o.length:
                # Side relative to the frame: mirrored frames flip it.
                row[BEND], row[BMASK] = float(side > 0), 1.0
            if o.slides % 2 == 1 and math.hypot(fx - o.x, fy - o.y) >= MIN_MOVE:
                heading = math.atan2(fy - o.y, fx - o.x)
        prev_end = (ex, ey)
    return out


def section_energy(mel: np.ndarray) -> np.ndarray:
    """Per-frame loudness of the surrounding section minus the song's average, (T,)."""
    loudness = np.asarray(mel, dtype=np.float32).mean(axis=0)
    width = max(int(2 * SECTION_SECONDS * FPS), 1)
    kernel = np.ones(width, dtype=np.float32) / width
    padded = np.pad(loudness, width // 2, mode="edge")
    smooth = np.convolve(padded, kernel, mode="same")[width // 2:width // 2 + len(loudness)]
    return (smooth - loudness.mean()).astype(np.float32)


def token_features(objects: np.ndarray, mel: np.ndarray, cs: float, cond: np.ndarray,
                   energy: np.ndarray) -> np.ndarray:
    """Model input for each object of ``objects`` (a window of map_objects rows);
    ``energy`` is section_energy(mel)."""
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
    frames = np.clip(np.rint(o[:, T] * FPS / 1000.0).astype(int), 0, mel.shape[1] - 1)
    window = np.clip(frames[:, None] + np.arange(-2, 3)[None, :], 0, mel.shape[1] - 1)
    audio = np.asarray(mel[:, window.ravel()], dtype=np.float32).reshape(mel.shape[0], n, 5).mean(axis=2).T
    columns = [
        np.log1p(gap_start)[:, None], np.log1p(gap_end)[:, None], kind, prev_kind,
        duration[:, None], (o[:, SLIDES] / 4.0)[:, None], o[:, [NC]],
        np.stack([np.sin(phase), np.cos(phase), np.sin(bar), np.cos(bar)], axis=1),
        (o[:, VEL] / 200.0)[:, None], np.full((n, 1), cs / 5.0, dtype=np.float32),
        prev_offset, prev_chord,
        np.stack([prev[:, EX] / PLAYFIELD_WIDTH, prev[:, EY] / PLAYFIELD_HEIGHT], axis=1) * (~first)[:, None],
        o[:, [HCOS, HSIN]], audio, (energy[frames] * 4.0)[:, None],
        prev[:, [WHISTLE, FINISH, CLAP]],
        np.broadcast_to(cond, (n, len(cond))),
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


FEATURES = 2 + 3 + 3 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2 + 2 + 2 + 80 + 1 + 3 + 2 * len(CONDITIONS)


# --------------------------------------------------------------------------------------
# Data


@dataclass
class PlacementMap:
    name: str
    song: str
    mel_path: Path
    cs: float
    style: dict
    objects: np.ndarray


def build_placement_maps(roots, cache_dir=None, log=print) -> list[PlacementMap]:
    """Object sequences of every usable map, sharing build_examples' deduplication,
    spectrogram caches and train/validation split."""
    from .dataset import build_examples, iter_beatmap_texts
    from .osu import parse_osu
    roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(r) for r in roots]
    examples = {ex.name: ex for ex in build_examples(roots, cache_dir, log=log)}
    maps = []
    for folder in roots:
        folder_cache = Path(cache_dir) if cache_dir else folder / ".beatmap_ai_cache"
        cache_file = folder_cache / PLACEMENT_CACHE
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError):
            cached = {}
        fresh, parsed = {}, 0
        for name, text, _, stamp in iter_beatmap_texts(folder):
            ex = examples.get(name)
            if ex is None:
                continue
            hit = cached.get(name)
            if hit is not None and hit[0] == stamp:
                objects, cs = hit[1], hit[2]
            else:
                bm = parse_osu(text)
                try:
                    objects, cs = map_objects(bm), bm.cs
                except (ValueError, StopIteration):
                    continue
                parsed += 1
                if parsed % 5000 == 0:
                    log(f"  read objects of {parsed} beatmaps in {folder}")
            fresh[name] = (stamp, objects, cs)
            maps.append(PlacementMap(name, ex.song, ex.mel_path, cs, ex.style or {}, objects))
        if parsed:
            tmp = cache_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(fresh, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_file)
    return maps


class WindowSampler:
    """Random windows of consecutive objects, star levels balanced like ChunkSampler."""

    def __init__(self, maps: list[PlacementMap], context: int, seed: int = 0, hide: bool = True):
        self.maps = [m for m in maps if len(m.objects) >= 16]
        self.context = context
        self.rng = np.random.default_rng(seed)
        self.hide = hide
        stars = np.array([m.style.get("stars", np.nan) for m in self.maps], dtype=float)
        buckets = np.where(np.isfinite(stars), np.floor(np.nan_to_num(stars) * 2), -1)
        _, inverse, counts = np.unique(buckets, return_inverse=True, return_counts=True)
        weights = 1.0 / np.sqrt(counts[inverse])
        self.weights = weights / weights.sum()
        self._mels: dict[Path, np.ndarray] = {}
        self._energy: dict[Path, np.ndarray] = {}

    def mel(self, path: Path) -> np.ndarray:
        if path not in self._mels:
            if len(self._mels) > 4000:
                self._mels.clear()
            self._mels[path] = np.load(path, mmap_mode="r")
        return self._mels[path]

    def energy(self, path: Path) -> np.ndarray:
        if path not in self._energy:
            self._energy[path] = section_energy(self.mel(path)).astype(np.float16)
        return self._energy[path].astype(np.float32)

    def conditions(self, m: PlacementMap) -> np.ndarray:
        from .style import encode_conditions
        values = dict(m.style)
        if self.hide:
            if self.rng.random() < 0.1:
                values = {}
            for name in CONDITIONS:
                if self.rng.random() < (0.5 if name == "density" else 0.3):
                    values.pop(name, None)
        return encode_conditions(values, CONDITIONS)

    def window(self, m: PlacementMap, start: int) -> tuple[np.ndarray, np.ndarray]:
        objects = m.objects[start:start + self.context]
        if self.hide:
            flip_x = bool(self.rng.integers(2))
            flip_y = bool(self.rng.integers(2))
            objects = augment_objects(objects, flip_x, flip_y)
        x = token_features(objects, self.mel(m.mel_path), m.cs, self.conditions(m),
                           self.energy(m.mel_path))
        return x, objects

    def batch(self, size: int) -> dict[str, np.ndarray]:
        xs, ys = [], []
        for _ in range(size):
            m = self.maps[int(self.rng.choice(len(self.maps), p=self.weights))]
            start = int(self.rng.integers(max(len(m.objects) - self.context, 0) + 1))
            x, objects = self.window(m, start)
            pad = self.context - len(objects)
            xs.append(np.pad(x, ((0, pad), (0, 0))))
            ys.append(np.pad(objects, ((0, pad), (0, 0))))
        return {"x": np.stack(xs), "y": np.stack(ys)}
