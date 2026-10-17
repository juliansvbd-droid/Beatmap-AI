import numpy as np
from pytest import approx

from beatmap_ai.rhythm_audit import summary
from beatmap_ai.timing import TimingEstimate


def _event(time, end=None, kind="circle", slides=1):
    return {"time": float(time), "end": float(time if end is None else end), "kind": kind,
            "slides": slides}


def test_summary_counts_figures_sliders_gaps_and_pauses():
    timing = TimingEstimate(bpm=120.0, offset_ms=0.0)  # 500 ms per beat
    items = [
        _event(0), _event(125),  # double
        _event(1000), _event(1125), _event(1250),  # triple
        _event(2000, end=2250, kind="slider", slides=2),
        _event(3000),  # 1.5 beats free after the slider's end (2 after its head)
        _event(5000),  # 4 beats after the previous start: a long pause
    ]
    out = summary(items, timing, truth=np.array([0.0, 125.0, 1000.0, 4000.0]))
    assert out["objects"] == 8
    assert out["double_1_4_per_100"] == approx(100.0 / 8)
    assert out["triple_1_4_per_100"] == approx(100.0 / 8)
    assert out["slider_pct"] == approx(100.0 / 8)
    assert out["repeat_slider_per_100"] == approx(100.0 / 8)
    assert out["gap_1_4_pct"] == approx(100.0 * 3 / 7)
    # 125 -> 1000 (1.75 beats), 1250 -> 2000 (1.5), 2250 -> 3000 (1.5), 3000 -> 5000 (4)
    assert out["pause_1_per_100"] == approx(100.0 * 4 / 8)
    assert out["pause_2_per_100"] == approx(100.0 / 8)
    assert out["rhythm_f1"] == approx(2 * 3 / (8 + 4))
