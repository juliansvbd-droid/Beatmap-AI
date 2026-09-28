"""Which geometric shape each jump belongs to (star, polygon, zigzag, line, arc, flow ...).

Used to measure how mappers build jump sections and, as labels, to teach the placement
model shapes on purpose instead of hoping it finds them. Works on the object arrays of
``placement_data.map_objects`` (so on training data directly). Torch-free.

A jump run is a stretch of hit circles at the same rhythm (gaps within 12 %) whose moves
are all jumps (more than 1.5 circle radii). Windows of four moves (five circles) along the
run are classified; each circle takes the most specific shape among its windows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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


# Same-sign shapes, for which the rotation direction is part of the shape.
TURNING = frozenset(("arc", "flow", "triangle", "square", "pentagon", "hexagon", "star"))
SHAPE_FEATURES = len(SHAPES) + 1  # one-hot shape, then rotation direction (+1 / -1 / 0)


def run_direction(points: np.ndarray) -> float:
    """+1 or -1 for the run's main turning direction, 0 when it does not keep one."""
    moves = np.diff(points, axis=0)
    cross = moves[:-1, 0] * moves[1:, 1] - moves[:-1, 1] * moves[1:, 0]
    total = float(np.sign(cross).sum())
    return float(np.sign(total)) if abs(total) >= 0.6 * max(len(cross), 1) else 0.0


def shape_features(objects: np.ndarray, cs: float = 4.0) -> np.ndarray:
    """Model input per object: one-hot shape and the rotation direction of its run.

    An all-zero row means "shape unknown" (hidden in training, no plan at generation);
    one-hot "none" means explicitly "not in a jump run"."""
    out = np.zeros((len(objects), SHAPE_FEATURES), dtype=np.float32)
    if not len(objects):
        return out
    labels = np.zeros(len(objects), dtype=np.int64)
    for run in jump_runs(objects, cs):
        points = objects[run][:, [X, Y]].astype(np.float64)
        for start in range(len(run) - WINDOW):
            label = PRIORITY[classify(points[start:start + WINDOW + 1])]
            span = run[start:start + WINDOW + 1]
            labels[span] = np.maximum(labels[span], label)
        direction = run_direction(points)
        turning = np.isin(labels[run], [PRIORITY[s] for s in TURNING])
        out[run[turning], -1] = direction
    out[np.arange(len(objects)), labels] = 1.0
    return out


def rhythm_runs(objects: np.ndarray) -> list[np.ndarray]:
    """Indices of each run of >= WINDOW + 1 circles at one rhythm, judged on timing only:
    the places where a jump run *could* go (at generation, before anything is placed)."""
    runs, current = [], []
    last_gap = None
    for i in range(1, len(objects)):
        a, b = objects[i - 1], objects[i]
        gap = float(b[T] - a[END])
        fits = (a[KIND] == CIRCLE and b[KIND] == CIRCLE
                and 0 < gap <= 2.0 * max(float(b[BEAT]), 1.0))
        same = last_gap is not None and abs(gap - last_gap) <= 0.12 * max(last_gap, 1.0)
        if fits and current and same:
            current.append(i)
        else:
            if len(current) >= WINDOW + 1:
                runs.append(np.asarray(current))
            current = [i - 1, i] if fits else []
        last_gap = gap if fits else None
    if len(current) >= WINDOW + 1:
        runs.append(np.asarray(current))
    return runs


def gap_class(objects: np.ndarray, run: np.ndarray) -> int:
    """Rhythm of a run in quarter beats (1, 2, 3, 4; 4 also for anything slower)."""
    a, b = objects[run[0]], objects[run[1]]
    return int(np.clip(round(float(b[T] - a[END]) / max(float(b[BEAT]), 1.0) * 4), 1, 4))


REFERENCE = Path(__file__).with_name("shape_reference.json")


def load_reference(path=REFERENCE) -> dict:
    """Table of scripts/shape_reference.py: {"<stars>/<gap>": {"jump_share", "shapes"}}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["table"]


MIN_RUNS = 30  # thinner table cells borrow from the neighbouring star levels


def _reference_row(reference: dict, level: int, gap: int) -> dict | None:
    rows = [reference.get(f"{level}/{gap}")]
    if not rows[0] or rows[0]["runs"] < MIN_RUNS:
        rows += [reference.get(f"{level - 1}/{gap}"), reference.get(f"{level + 1}/{gap}")]
    rows = [r for r in rows if r and r["runs"]]
    if not rows:
        return None
    runs = sum(r["runs"] for r in rows)
    jump = sum(r["runs"] * r["jump_share"] for r in rows)
    shapes: dict[str, float] = {}
    for r in rows:
        weight = r["runs"] * r["jump_share"]
        for name, share in r["shapes"].items():
            shapes[name] = shapes.get(name, 0.0) + weight * share
    total = sum(shapes.values()) or 1.0
    return {"runs": runs, "jump_share": jump / runs, "shapes": {k: v / total for k, v in shapes.items()}}


def plan_shapes(objects: np.ndarray, stars: float, rng: np.random.Generator,
                reference: dict | None = None, jump_scale: float = 1.0) -> np.ndarray:
    """shape_features for a rhythm that is not placed yet: every same-rhythm run of
    circles becomes a jump run as often as in ranked maps of this star level and rhythm
    (times ``jump_scale``), with a shape drawn from their shapes. Everything else is
    "none" (not a jump run)."""
    reference = reference if reference is not None else load_reference()
    out = np.zeros((len(objects), SHAPE_FEATURES), dtype=np.float32)
    out[:, 0] = 1.0
    level = int(np.clip(stars, 0, 9))
    for run in rhythm_runs(objects):
        row = _reference_row(reference, level, gap_class(objects, run))
        if not row or not row["shapes"] or rng.random() >= min(row["jump_share"] * jump_scale, 1.0):
            continue
        names = list(row["shapes"])
        p = np.asarray([row["shapes"][n] for n in names], dtype=np.float64)
        shape = names[int(rng.choice(len(names), p=p / p.sum()))]
        out[run] = 0.0
        out[run, PRIORITY[shape]] = 1.0
        if shape in TURNING:
            out[run, -1] = 1.0 if rng.random() < 0.5 else -1.0
    return out
