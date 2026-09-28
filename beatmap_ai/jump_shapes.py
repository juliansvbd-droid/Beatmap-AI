"""Which geometric shape each jump belongs to (star, polygon, zigzag, line, arc, flow ...).

Used to measure how mappers build jump sections and, as labels, to teach the placement
model shapes on purpose instead of hoping it finds them. Works on the object arrays of
``placement_data.map_objects`` (so on training data directly). Torch-free.

A jump run is a stretch of hit circles at the same rhythm (gaps within 12 %) whose moves
are all jumps (more than 1.5 circle radii). Windows of four moves (five circles) along the
run are classified; each circle takes the most specific shape among its windows.
"""

from __future__ import annotations

import math

import numpy as np

from .placement_data import BEAT, CIRCLE, END, EX, EY, KIND, T, X, Y

SHAPES = ("none", "other", "flow", "arc", "line", "back_and_forth", "zigzag",
          "triangle", "square", "pentagon", "hexagon", "star")
# Specific shapes win over vague ones when windows of one circle disagree.
PRIORITY = {name: i for i, name in enumerate(SHAPES)}
POLYGONS = (("hexagon", 60.0), ("pentagon", 72.0), ("square", 90.0), ("triangle", 120.0),
            ("star", 144.0))
WINDOW = 4  # moves per window


def jump_runs(objects: np.ndarray, cs: float = 4.0) -> list[np.ndarray]:
    """Indices of each run of >= WINDOW + 1 circles joined by same-rhythm jumps."""
    radius = 54.4 - 4.48 * float(cs)
    runs, current = [], []
    last_gap = None
    for i in range(1, len(objects)):
        a, b = objects[i - 1], objects[i]
        gap = float(b[T] - a[END])
        is_jump = (a[KIND] == CIRCLE and b[KIND] == CIRCLE
                   and math.hypot(b[X] - a[EX], b[Y] - a[EY]) > 1.5 * radius
                   and gap <= 2.0 * max(float(b[BEAT]), 1.0))
        same = last_gap is not None and abs(gap - last_gap) <= 0.12 * max(last_gap, 1.0)
        if is_jump and current and same:
            current.append(i)
        else:
            if len(current) >= WINDOW + 1:
                runs.append(np.asarray(current))
            current = [i - 1, i] if is_jump else []
        last_gap = gap if is_jump else None
    if len(current) >= WINDOW + 1:
        runs.append(np.asarray(current))
    return runs


def classify(points: np.ndarray) -> str:
    """Shape of a window of five points (four moves)."""
    moves = np.diff(points, axis=0)
    sides = np.hypot(moves[:, 0], moves[:, 1])
    if sides.min() <= 1e-6:
        return "other"
    cv = float(sides.std() / sides.mean())
    cross = moves[:-1, 0] * moves[1:, 1] - moves[:-1, 1] * moves[1:, 0]
    dot = (moves[:-1] * moves[1:]).sum(axis=1)
    turns = np.degrees(np.arctan2(cross, dot))
    size = np.abs(turns)
    if np.all(size < 25) and cv < 0.3:
        return "line"
    if np.all(size > 150) and cv < 0.3:
        return "back_and_forth"
    signs = np.sign(turns)
    if np.all(signs[1:] != signs[:-1]) and size.std() < 25 and cv < 0.3 and size.min() >= 25:
        return "zigzag"
    if np.all(signs == signs[0]):
        if cv < 0.2 and size.std() < 15:
            mean = float(size.mean())
            name, angle = min(POLYGONS, key=lambda item: abs(mean - item[1]))
            if abs(mean - angle) <= 10:
                return name
            if mean < 50:
                return "arc"
        if cv < 0.35:
            return "flow"
    return "other"


def object_shapes(objects: np.ndarray, cs: float = 4.0) -> np.ndarray:
    """Shape index (into SHAPES) for every object; 0 ("none") outside jump runs."""
    out = np.zeros(len(objects), dtype=np.int64)
    for run in jump_runs(objects, cs):
        points = objects[run][:, [X, Y]].astype(np.float64)
        for start in range(len(run) - WINDOW):
            label = PRIORITY[classify(points[start:start + WINDOW + 1])]
            span = run[start:start + WINDOW + 1]
            out[span] = np.maximum(out[span], label)
    return out
