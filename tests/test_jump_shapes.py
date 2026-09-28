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


def _run_objects(points, gap=250.0):
    o = np.zeros((len(points), N_COLUMNS), dtype=np.float32)
    o[:, T] = np.arange(len(points)) * gap
    o[:, END] = o[:, T]
    o[:, BEAT] = 500.0
    o[:, KIND] = CIRCLE
    o[:, X], o[:, Y] = points[:, 0], points[:, 1]
    o[:, EX], o[:, EY] = points[:, 0], points[:, 1]
    return o


def test_shape_features_carry_rotation_direction():
    from beatmap_ai.jump_shapes import SHAPE_FEATURES, shape_features
    o = _run_objects(_polygon(5, 2, n=10))
    f = shape_features(o, 4.0)
    assert f.shape == (10, SHAPE_FEATURES)
    assert np.all(f[:, SHAPES.index("star")] == 1)
    mirrored = o.copy()
    mirrored[:, X] = 512 - mirrored[:, X]
    mirrored[:, EX] = mirrored[:, X]
    g = shape_features(mirrored, 4.0)
    assert set(f[:, -1]) == {f[0, -1]} and f[0, -1] != 0
    assert np.all(g[:, -1] == -f[:, -1])


def test_plan_shapes_uses_the_reference_table():
    from beatmap_ai.jump_shapes import plan_shapes
    o = _run_objects(np.zeros((8, 2)) + 256)  # positions do not matter for the plan
    table = {"5/2": {"jump_share": 1.0, "runs": 1, "shapes": {"square": 1.0}}}
    f = plan_shapes(o, 5.2, np.random.default_rng(0), table)
    assert np.all(f[:, SHAPES.index("square")] == 1) and np.all(np.abs(f[:, -1]) == 1)
    none = plan_shapes(o, 5.2, np.random.default_rng(0), table, jump_scale=0.0)
    assert np.all(none[:, 0] == 1) and np.all(none[:, 1:] == 0)


def test_v4_warm_start_matches_v3_without_shapes():
    import torch
    from beatmap_ai.sequence_data import V3_FEATURES, V4_FEATURES
    from beatmap_ai.sequence_model import SequenceV3Net, save_sequence, warm_start
    torch.manual_seed(0)
    old = SequenceV3Net(features=V3_FEATURES, hidden=32, layers=1, heads=2, context=16).eval()
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "v3.pt"
        save_sequence(old, path)
        new = SequenceV3Net(features=V4_FEATURES, hidden=32, layers=1, heads=2, context=16).eval()
        warm_start(new, path)
    x = torch.randn(1, 5, V3_FEATURES)
    extra = torch.randn(1, 5, V4_FEATURES - V3_FEATURES)
    with torch.no_grad():
        a = old(x)["offset"]
        b = new(torch.cat([x, extra], dim=2))["offset"]
    assert torch.allclose(a, b, atol=1e-6)


def test_rhythm_runs_accept_swung_half_notes():
    from beatmap_ai.jump_shapes import jump_runs, rhythm_runs
    o = _run_objects(_polygon(5, 2, n=8))
    o[:, BEAT] = 343.6
    o[:, T] = np.cumsum([0] + [160.0, 183.6] * 3 + [160.0])
    o[:, END] = o[:, T]
    assert [len(r) for r in rhythm_runs(o)] == [8]
    assert jump_runs(o, 4.0) == []  # training labels: strict rhythm
