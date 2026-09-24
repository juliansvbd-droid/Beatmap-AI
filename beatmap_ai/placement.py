"""Spatial placement: deciding *where* hit objects go on the 512x384 playfield.

Objects follow a flowing path. The distance to the next object grows with the time gap
(distance snapping) and, on harder difficulties, with the music's intensity (jumps).
The direction of travel curves smoothly within a combo and turns sharply at combo
starts; candidates that leave the playfield or overlap recent objects are rejected.
"""

from __future__ import annotations

import math

import numpy as np

from .difficulty import DifficultyPreset
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, HitObject
from .rhythm import PlannedObject

MAX_JUMP = 300.0


def circle_radius(cs: float) -> float:
    return 54.4 - 4.48 * cs


def slider_path(start, heading: float, length: float, bend: float):
    """Circular arc of ``length`` starting at ``start`` in direction ``heading`` that turns
    by ``bend`` radians in total. Returns (curve type, control points, sampler, end heading)."""
    sx, sy = start
    if abs(bend) < 1e-3:
        def point(s):
            return sx + s * math.cos(heading), sy + s * math.sin(heading)
        return "L", [point(length)], point, heading
    k = bend / length

    def point(s):
        return (
            sx + (math.sin(heading + k * s) - math.sin(heading)) / k,
            sy - (math.cos(heading + k * s) - math.cos(heading)) / k,
        )
    return "P", [point(length / 2), point(length)], point, heading + bend


class Placer:
    def __init__(self, preset: DifficultyPreset, rng: np.random.Generator, beat_length: float):
        self.preset = preset
        self.rng = rng
        self.beat_length = beat_length
        self.radius = circle_radius(preset.cs)
        self.margin = self.radius * 0.6

    def spacing(self, gap_beats: float) -> float:
        """Distance to the next object in beats of slider velocity. Gaps longer than a
        beat don't push objects further apart."""
        quarter, half, full = self.preset.spacing
        if gap_beats <= 0.25:
            return quarter * gap_beats / 0.25
        return float(np.interp(gap_beats, [0.25, 0.5, 1.0], [quarter, half, full]))

    def in_bounds(self, x: float, y: float) -> bool:
        m = self.margin
        return m <= x <= PLAYFIELD_WIDTH - m and m <= y <= PLAYFIELD_HEIGHT - m

    def place(self, plan: list[PlannedObject]) -> list[HitObject]:
        preset, rng = self.preset, self.rng
        velocity = preset.slider_multiplier * 100.0  # pixels per beat
        objects: list[HitObject] = []
        recent: list[tuple[float, float, float]] = []  # (x, y, time) of recent objects
        pos = (PLAYFIELD_WIDTH / 2, PLAYFIELD_HEIGHT / 2)
        heading = rng.uniform(0, 2 * math.pi)
        curl = rng.choice([-1.0, 1.0])
        prev_end = None

        for item in plan:
            if item.kind == "spinner":
                objects.append(HitObject(256, 192, item.time, "spinner", True, end_time=item.end_time))
                pos, prev_end = (PLAYFIELD_WIDTH / 2, PLAYFIELD_HEIGHT / 2), item.end_time
                recent.clear()
                continue

            gap_beats = (item.time - prev_end) / self.beat_length if prev_end is not None else 8.0
            distance = velocity * self.spacing(gap_beats)
            if gap_beats >= 0.5:
                distance *= 1.0 + preset.jump_scale * (item.intensity - 0.5)
            distance = min(distance, MAX_JUMP)
            if gap_beats > 4:  # After a break, start somewhere comfortable.
                distance = min(distance, 120.0)

            # Smooth curvature normally, sharp turns at combo starts and for jumps.
            if rng.random() < 0.2:
                curl = -curl
            if gap_beats <= 0.25:
                turn = curl * rng.normal(0.25, 0.1)
            elif item.new_combo:
                turn = curl * rng.normal(1.9, 0.4)
            else:
                turn = curl * abs(rng.normal(0.6, 0.35))
            desired = heading + turn

            x, y, heading = self._find_position(pos, desired, distance, recent, item.time)
            obj = HitObject(x, y, item.time, "circle", item.new_combo)
            end = (x, y)
            body = []
            if item.kind == "slider":
                length = (item.end_time - item.time) / self.beat_length * velocity
                path = self._find_slider(end, heading, length, recent, item.time)
                if path is None:
                    item.end_time = item.time
                else:
                    curve_type, points, point, heading = path
                    end = points[-1]
                    body = [(*point(length * k / 4), item.time) for k in (1, 2, 3)]
                    obj.kind, obj.curve_type, obj.curve_points, obj.length = (
                        "slider", curve_type, points, length)
            objects.append(obj)
            # The last entry is always the previous object's end, which the next object
            # is placed relative to.
            recent = (recent + [(x, y, item.time), *body, (*end, item.end_time)])[-16:]
            pos, prev_end = end, item.end_time
        return objects

    def _find_position(self, pos, desired, distance, recent, time):
        # Prefer the intended spacing without overlaps; then allow overlaps before
        # shrinking the spacing, which would misrepresent the rhythm.
        attempts = [(scale, True) for scale in (1.0, 0.85, 0.7)]
        attempts += [(scale, False) for scale in (1.0, 0.8, 0.6, 0.4, 0.2)]
        for scale, avoid_overlap in attempts:
            d = distance * scale
            for delta in _alternating(math.radians(15), 12):
                angle = desired + delta
                x = pos[0] + d * math.cos(angle)
                y = pos[1] + d * math.sin(angle)
                if self.in_bounds(x, y) and not (avoid_overlap and self._overlaps(x, y, recent, time, d)):
                    return x, y, angle
        x = min(max(pos[0], self.margin), PLAYFIELD_WIDTH - self.margin)
        y = min(max(pos[1], self.margin), PLAYFIELD_HEIGHT - self.margin)
        return x, y, desired

    def _overlaps(self, x, y, recent, time, distance, skip_previous=True) -> bool:
        """Whether (x, y) covers a recent object. The previous object is skipped by
        default, since spacing already keeps new circles away from it."""
        if distance < self.radius:  # Intentional stacks/overlaps in tight streams are fine.
            return False
        for rx, ry, rt in (recent[:-1] if skip_previous else recent):
            if time - rt < 2000 and math.hypot(x - rx, y - ry) < 1.2 * self.radius:
                return True
        return False

    def _find_slider(self, start, heading, length, recent, time):
        """A slider path from ``start`` that stays on the playfield, preferably without
        covering recent objects. Returns (curve type, points, sampler, end heading)."""
        bends = [self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.3, 1.2), 0.0]
        fallback = None
        for bend in bends:
            for delta in _alternating(math.radians(20), 9):
                curve_type, points, point, end_heading = slider_path(start, heading + delta, length, bend)
                samples = [point(length * s / 8) for s in range(1, 9)]
                if not all(self.in_bounds(*p) for p in samples):
                    continue
                if not any(self._overlaps(*p, recent, time, self.radius, skip_previous=False)
                           for p in samples):
                    return curve_type, points, point, end_heading
                fallback = fallback or (curve_type, points, point, end_heading)
        return fallback


def _alternating(step: float, n: int):
    yield 0.0
    for i in range(1, n + 1):
        yield i * step
        yield -i * step
