"""Song structure: repeated sections (a chorus coming back) and where kiai time goes.

Mappers usually map a repeated chorus with the same rhythm and patterns as the first
time, sometimes mirrored, and turn on kiai time in the loud, repeated parts. This
module finds those parts from the audio and applies both to a generated map.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .audio import FPS, AudioFeatures
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, HitObject

BEATS_PER_BAR = 4


@dataclass
class Repeat:
    source_ms: float  # start of the first occurrence
    target_ms: float  # start of the repetition
    length_ms: float
    similarity: float


def bar_starts(features: AudioFeatures, offset_ms: float, beat_length: float) -> np.ndarray:
    bar = BEATS_PER_BAR * beat_length
    first = offset_ms - math.floor(offset_ms / bar) * bar
    return np.arange(first, features.duration * 1000.0 - bar, bar)


def bar_vectors(features: AudioFeatures, starts: np.ndarray, bar_ms: float) -> np.ndarray:
    """One vector per bar: its harmony (chroma) and timbre (mel), each normalised."""
    parts = []
    sources = [features.mel] + ([features.chroma] if features.chroma is not None else [])
    for source in sources:
        rows = []
        for start in starts:
            a = int(start * FPS / 1000.0)
            b = max(int((start + bar_ms) * FPS / 1000.0), a + 1)
            rows.append(np.asarray(source[:, a:b], dtype=np.float32).mean(axis=1))
        rows = np.array(rows)
        rows -= rows.mean(axis=0)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-6
        parts.append(rows)
    return np.concatenate(parts, axis=1) / math.sqrt(len(parts))


def find_repeats(features: AudioFeatures, offset_ms: float, beat_length: float,
                 bars: int = 8, min_similarity: float = 0.8) -> list[Repeat]:
    """Later sections that repeat an earlier one closely enough to be mapped the same."""
    bar_ms = BEATS_PER_BAR * beat_length
    starts = bar_starts(features, offset_ms, beat_length)
    if len(starts) < 2 * bars:
        return []
    v = bar_vectors(features, starts, bar_ms)
    sim = v @ v.T
    n = len(starts)
    # Mean similarity along each diagonal segment of ``bars`` bars.
    candidates = []
    for lag in range(bars, n - bars + 1):
        diag = np.diagonal(sim, offset=lag)  # sim[i, i + lag]
        window = np.convolve(diag, np.ones(bars) / bars, mode="valid")
        for i in np.argsort(-window)[:4]:
            candidates.append((float(window[i]), int(i), int(i + lag)))
    # A repetition must stand out from how similar bars of this song usually are.
    typical = float(np.median(sim[np.triu_indices(n, k=1)]))
    threshold = max(min_similarity, typical + 0.3)
    taken = np.zeros(n, dtype=bool)  # bars already used as a repetition
    repeats = []
    for score, src, dst in sorted(candidates, reverse=True):
        if score < threshold:
            break
        if taken[dst:dst + bars].any() or taken[src:src + bars].any():
            continue
        taken[dst:dst + bars] = True
        repeats.append(Repeat(float(starts[src]), float(starts[dst]), bars * bar_ms, score))
    return sorted(repeats, key=lambda r: r.target_ms)


def copy_rhythm(plan: list, repeats: list[Repeat], tick_ms: float, bar_ms: float) -> list:
    """Replace each repetition's objects with the first occurrence's, shifted in time."""
    plan = list(plan)
    for rep in repeats:
        shift = rep.target_ms - rep.source_ms
        source = [p for p in plan if rep.source_ms - 1 <= p.time < rep.source_ms + rep.length_ms - 1
                  and p.end_time < rep.source_ms + rep.length_ms - 1 and p.kind != "spinner"]
        if not source:
            continue
        end = rep.target_ms + rep.length_ms - 1
        kept = [p for p in plan if not (rep.target_ms - 1 <= p.time < end)]
        # Keep objects around the section from running into the copied ones.
        before = [p for p in kept if p.time < rep.target_ms]
        if before and before[-1].end_time > rep.target_ms - tick_ms:
            continue
        copies = []
        for p in source:
            q = copy.copy(p)
            q.time, q.end_time = p.time + shift, p.end_time + shift
            q.tick = p.tick + int(round(shift / tick_ms))
            q.measure = p.measure + int(round(shift / bar_ms))
            q.copy_of = p  # placement copies this object's position
            copies.append(q)
        plan = sorted(kept + copies, key=lambda p: p.time)
    return plan


TRANSFORMS = {
    "same": lambda x, y: (x, y),
    "mirror": lambda x, y: (PLAYFIELD_WIDTH - x, y),
    "flip": lambda x, y: (x, PLAYFIELD_HEIGHT - y),
    "rotate": lambda x, y: (PLAYFIELD_WIDTH - x, PLAYFIELD_HEIGHT - y),
}


def copy_patterns(objects: list[HitObject], plan: list, rng: np.random.Generator) -> list[HitObject]:
    """Give copied objects (see copy_rhythm) the positions of their originals, as the
    same pattern, mirrored, flipped or rotated -- whichever continues best from the
    object before the repetition."""
    index = {id(p): i for i, p in enumerate(plan)}
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(plan):
        src = getattr(p, "copy_of", None)
        if src is not None and id(src) in index:
            groups.setdefault(round(p.time - src.time), []).append(i)
    objects = list(objects)
    for members in groups.values():
        first = members[0]
        prev = objects[first - 1] if first > 0 else None
        wanted = None
        if prev is not None:
            # Aim for the distance the original section started with.
            src_first = index[id(plan[first].copy_of)]
            if src_first > 0:
                a, b = objects[src_first - 1], objects[src_first]
                wanted = math.dist(_end(a), (b.x, b.y))
        # ...and for the object after it to follow on as it did after the original.
        last = members[-1]
        after = objects[last + 1] if last + 1 < len(objects) else None
        src_last = index[id(plan[last].copy_of)]
        wanted_after = (math.dist(_end(objects[src_last]), _start(objects[src_last + 1]))
                        if after is not None and src_last + 1 < len(objects) else None)
        best = None
        names = list(TRANSFORMS)
        rng.shuffle(names)
        for name in names:
            f = TRANSFORMS[name]
            cost = 0.0
            if prev is not None and wanted is not None:
                start = f(*_start(objects[index[id(plan[first].copy_of)]]))
                cost += abs(math.dist(_end(prev), start) - wanted)
            if wanted_after is not None:
                cost += abs(math.dist(f(*_end(objects[src_last])), _start(after)) - wanted_after)
            if best is None or cost < best[0]:
                best = (cost, f)
        f = best[1]
        for i in members:
            src = objects[index[id(plan[i].copy_of)]]
            obj = copy.deepcopy(src)
            obj.time = objects[i].time
            obj.new_combo = objects[i].new_combo
            obj.x, obj.y = f(src.x, src.y)
            obj.curve_points = [f(px, py) for px, py in src.curve_points]
            if obj.kind != objects[i].kind:  # e.g. a slider that could not be placed
                continue
            objects[i] = obj
    return objects


def _start(obj: HitObject) -> tuple[float, float]:
    return obj.x, obj.y


def _end(obj: HitObject) -> tuple[float, float]:
    if obj.kind == "slider" and obj.curve_points and obj.slides % 2 == 1:
        return obj.curve_points[-1]
    return obj.x, obj.y


def kiai_sections(features: AudioFeatures, repeats: list[Repeat], min_seconds: float = 8.0
                  ) -> list[tuple[float, float]]:
    """(start, end) ms of kiai time: repeated sections that are louder than the song's
    average (usually the chorus). Songs without repeated sections get no kiai."""
    loud = features.rms
    mean = float(loud.mean())

    def level(start: float, end: float) -> float:
        a, b = int(start * FPS / 1000.0), max(int(end * FPS / 1000.0), int(start * FPS / 1000.0) + 1)
        return float(loud[a:b].mean())

    spans = []
    for rep in repeats:
        for start in (rep.source_ms, rep.target_ms):
            if level(start, start + rep.length_ms) > 1.1 * mean:
                spans.append((start, start + rep.length_ms))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_seconds * 1000.0]


def copy_sections(objects: list[HitObject], repeats: list[Repeat], rng: np.random.Generator,
                  end_time) -> list[HitObject]:
    """Map each repetition like its first occurrence: its objects are replaced by the
    source section's, shifted in time and placed as the same pattern, mirrored,
    flipped or rotated (whichever continues best). For maps whose rhythm and placement
    were drawn together (sequence model); ``end_time(obj)`` gives an object's end."""
    objects = sorted(objects, key=lambda o: o.time)
    for rep in repeats:
        shift = rep.target_ms - rep.source_ms
        src_end = rep.source_ms + rep.length_ms
        source = [o for o in objects if rep.source_ms - 1 <= o.time < src_end - 1
                  and end_time(o) < src_end - 1 and o.kind != "spinner"]
        dst_end = rep.target_ms + rep.length_ms
        before = [o for o in objects if o.time < rep.target_ms - 1]
        after = [o for o in objects if o.time >= dst_end - 1]
        if not source or (before and end_time(before[-1]) > source[0].time + shift - 1) \
                or (after and end_time(source[-1]) + shift > after[0].time - 1):
            continue
        # Pick the transform whose entry and exit distances match the original's.
        i0 = objects.index(source[0])
        i1 = objects.index(source[-1])
        wanted_in = math.dist(_end(objects[i0 - 1]), _start(source[0])) if i0 > 0 else None
        wanted_out = (math.dist(_end(source[-1]), _start(objects[i1 + 1]))
                      if i1 + 1 < len(objects) else None)
        best = None
        names = list(TRANSFORMS)
        rng.shuffle(names)
        for name in names:
            f = TRANSFORMS[name]
            cost = 0.0
            if before and wanted_in is not None:
                cost += abs(math.dist(_end(before[-1]), f(*_start(source[0]))) - wanted_in)
            if after and wanted_out is not None:
                cost += abs(math.dist(f(*_end(source[-1])), _start(after[0])) - wanted_out)
            if best is None or cost < best[0]:
                best = (cost, f)
        f = best[1]
        copies = []
        for o in source:
            c = copy.deepcopy(o)
            c.time = o.time + shift
            c.x, c.y = f(o.x, o.y)
            c.curve_points = [f(px, py) for px, py in o.curve_points]
            copies.append(c)
        objects = before + copies + after
    return objects
