import math

import numpy as np

from beatmap_ai.osu import Beatmap, HitObject, TimingPoint
from beatmap_ai.patterns import (
    DIFFICULTY_FEATURES,
    _shape_fingerprint,
    analyze_map,
    apply_reference,
    build_reference,
    star_bucket,
)


def make_map(points, times=None):
    times = times or [i * 500 for i in range(len(points))]
    return Beatmap(
        cs=4.0,
        slider_multiplier=1.4,
        timing_points=[TimingPoint(time=0, beat_length=500, meter=4)],
        hit_objects=[HitObject(x=x, y=y, time=time, kind="circle")
                     for (x, y), time in zip(points, times)],
    )


def test_detects_triangle_quadrilateral_and_zigzag():
    triangle = make_map([(100, 100), (200, 100), (150, 186.6), (100, 100)])
    square = make_map([(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)])
    zigzag = make_map([(100, 100), (200, 100), (100, 100), (200, 100)])

    assert analyze_map(triangle)["metrics"]["pattern_triangle_per_100"] > 0
    assert analyze_map(square)["metrics"]["pattern_quadrilateral_per_100"] > 0
    assert analyze_map(zigzag)["metrics"]["pattern_zigzag_per_100"] > 0


def test_counts_double_triple_burst_and_stream_groups():
    subdivisions = [1, 4, 1, 1, 4, 1, 1, 1, 4, *([1] * 8)]
    times = [0]
    for spacing in subdivisions:
        times.append(times[-1] + spacing * 125)
    points = [(80 + i * 19, 100 + (i % 2) * 30) for i in range(len(times))]
    bm = make_map(points, times)

    metrics = analyze_map(bm)["metrics"]

    expected = 100.0 / len(points)
    assert math.isclose(metrics["double_1_4"], expected)
    assert math.isclose(metrics["triple_1_4"], expected)
    assert math.isclose(metrics["burst_4_8_1_4"], expected)
    assert math.isclose(metrics["stream_9p_1_4"], expected)


def test_shape_fingerprint_ignores_rotation_mirror_translation_and_scale():
    base = np.asarray([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)], dtype=float)
    angle = math.radians(37)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    transformed = (base @ rotation.T) * 1.7 + np.asarray([320, -90])
    mirrored = base * np.asarray([-1, 1]) + np.asarray([500, 20])

    reference = _shape_fingerprint(base)
    assert _shape_fingerprint(transformed) == reference
    assert _shape_fingerprint(mirrored) == reference


def test_star_buckets_use_half_open_edges():
    assert star_bucket(2.99) == "<3"
    assert star_bucket(3.0) == "3-4.5"
    assert star_bucket(4.5) == "4.5-6"
    assert star_bucket(6.0) == "6+"
    assert star_bucket(None) is None


def test_pattern_deviation_is_zero_for_the_human_median_profile():
    human_maps = [
        {"star_bucket": "<3", "metrics": {"slider_share_pct": value},
         "_difficulty_values": {feature: [0.0] for feature in DIFFICULTY_FEATURES}}
        for value in (0.0, 2.0, 4.0)
    ]
    reference = build_reference(human_maps)
    report = {
        "star_bucket": "<3",
        "object_count": 1,
        "metrics": {"slider_share_pct": 2.0},
        "_difficulty_values": {feature: [0.0] for feature in DIFFICULTY_FEATURES},
    }

    apply_reference(report, reference)

    assert math.isclose(report["pattern_deviation"]["score"], 0.0)
