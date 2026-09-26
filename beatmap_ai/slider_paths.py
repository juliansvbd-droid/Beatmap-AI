"""osu! slider path sampling used by the v3 placement targets."""

from __future__ import annotations

import math

import numpy as np

PATH_POINTS = 8


def _bezier(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    controls = np.asarray(points, dtype=np.float64)
    values = []
    for t in np.linspace(0.0, 1.0, max(count, 2)):
        work = controls.copy()
        while len(work) > 1:
            work = work[:-1] * (1.0 - t) + work[1:] * t
        values.append(tuple(work[0]))
    return values


def _bezier_segments(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Split osu! Bezier control points at repeated anchors."""
    anchors = [0]
    for i in range(1, len(points)):
        if np.allclose(points[i], points[i - 1]):
            anchors.append(i)
    anchors.append(len(points) - 1)
    segments = []
    start = 0
    for end in anchors[1:]:
        if end > start:
            segments.append(points[start:end + 1])
            start = end
    return segments or [points]


def _perfect_arc(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]] | None:
    if len(points) != 3:
        return None
    (ax, ay), (bx, by), (cx, cy) = points
    determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) < 1e-8:
        return None
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ox = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / determinant
    oy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / determinant
    angles = [math.atan2(y - oy, x - ox) for x, y in points]
    ccw = (angles[2] - angles[0]) % (2 * math.pi)
    middle = (angles[1] - angles[0]) % (2 * math.pi)
    direction = 1.0 if middle <= ccw else -1.0
    sweep = ccw if direction > 0 else ccw - 2 * math.pi
    radius = math.hypot(ax - ox, ay - oy)
    return [(ox + radius * math.cos(angles[0] + sweep * t),
             oy + radius * math.sin(angles[0] + sweep * t))
            for t in np.linspace(0.0, 1.0, max(count, 2))]


def _polyline(obj) -> list[tuple[float, float]]:
    start = (float(obj.x), float(obj.y))
    controls = [start, *[(float(x), float(y)) for x, y in obj.curve_points]]
    if obj.curve_type == "L":
        return controls
    if obj.curve_type == "P":
        return _perfect_arc(controls, 100) or controls
    points: list[tuple[float, float]] = []
    for segment in _bezier_segments(controls):
        sampled = _bezier(segment, max(12, 12 * len(segment)))
        points.extend(sampled if not points else sampled[1:])
    return points


def sample_slider_path(obj, heading: float, count: int = PATH_POINTS) -> np.ndarray:
    """Return K local-frame points, normalized by slider pixel length.

    The curve is arc-length fitted to osu!'s requested slider length before points are
    sampled. A short control path is extended along its last tangent.
    """
    if count < 2:
        raise ValueError("slider paths need at least two samples")
    points = np.asarray(_polyline(obj), dtype=np.float64)
    if len(points) < 2:
        points = np.asarray([(obj.x, obj.y), (obj.x + obj.length, obj.y)], dtype=np.float64)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    keep = np.concatenate([[True], lengths > 1e-8])
    points = points[keep]
    if len(points) < 2:
        points = np.asarray([(obj.x, obj.y), (obj.x + obj.length, obj.y)], dtype=np.float64)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target_length = max(float(obj.length), 1.0)
    if cumulative[-1] < target_length:
        tangent = delta[-1] / max(lengths[-1], 1e-8)
        points = np.vstack([points, points[-1] + tangent * (target_length - cumulative[-1])])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distances = np.linspace(0.0, target_length, count)
    x = np.interp(distances, cumulative, points[:, 0]) - float(obj.x)
    y = np.interp(distances, cumulative, points[:, 1]) - float(obj.y)
    c, s = math.cos(heading), math.sin(heading)
    u, v = (c * x + s * y) / target_length, (-s * x + c * y) / target_length
    return np.stack([u, v], axis=-1).astype(np.float32)
