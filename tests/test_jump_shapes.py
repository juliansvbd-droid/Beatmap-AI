import math

import numpy as np

from beatmap_ai.jump_shapes import SHAPES, classify, object_shapes
from beatmap_ai.placement_data import BEAT, CIRCLE, END, EX, EY, KIND, N_COLUMNS, T, X, Y


def _polygon(k: int, step: int = 1, n: int = 5):
    return np.array([(256 + 120 * math.cos(2 * math.pi * step * i / k),
                      192 + 120 * math.sin(2 * math.pi * step * i / k)) for i in range(n)])


def test_ideal_shapes_are_recognised():
    assert classify(_polygon(3)) == "triangle"
    assert classify(_polygon(4)) == "square"
    assert classify(_polygon(5)) == "pentagon"
    assert classify(_polygon(6)) == "hexagon"
    assert classify(_polygon(5, 2)) == "star"
    assert classify(np.array([(0, 0), (100, 60), (200, 0), (300, 60), (400, 0)], float)) == "zigzag"
    assert classify(np.array([(0, 0), (100, 4), (200, 0), (300, 4), (400, 0)], float)) == "line"


def test_every_circle_of_a_star_run_is_labelled_star():
    points = _polygon(5, 2, n=10)
    o = np.zeros((10, N_COLUMNS), dtype=np.float32)
    o[:, T] = np.arange(10) * 250.0
    o[:, END] = o[:, T]
    o[:, BEAT] = 500.0
    o[:, KIND] = CIRCLE
    o[:, X], o[:, Y] = points[:, 0], points[:, 1]
    o[:, EX], o[:, EY] = points[:, 0], points[:, 1]
    labels = object_shapes(o, 4.0)
    assert [SHAPES[i] for i in labels] == ["star"] * 10
