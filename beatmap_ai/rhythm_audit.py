"""Rhythm counts for the pipeline audit and its human reference (no PyTorch).

Objects are reduced to events (start, end, kind, slider runs). Figures are counted with
``patterns._rhythm_metrics`` (runs of starts exactly 1/4 beat apart, slider heads
included), so generated and human maps use the same definition. Gaps are measured start
to start, pauses from an object's end to the next start.
"""

from __future__ import annotations

from bisect import bisect_right
from types import SimpleNamespace

import numpy as np

from .patterns import _rhythm_metrics

# Start-to-start gaps in beats; a gap matches within 10 % (at least 0.012 beats), as in
# patterns._rhythm_metrics.
GAP_CLASSES = {"1_4": 0.25, "1_2": 0.5, "3_4": 0.75, "1": 1.0}


def events(objects, end_time=None) -> list[dict]:
    """Non-spinner objects as dicts. ``end_time(obj)`` gives a slider's end when the
    object does not carry it (rendered HitObjects need their Beatmap for that)."""
    out = []
    for obj in objects:
        if obj.kind == "spinner":
            continue
        end = getattr(obj, "end_time", None)
        if end is None or (obj.kind == "slider" and end <= obj.time and end_time is not None):
            end = end_time(obj) if end_time is not None else obj.time
        out.append({"time": round(float(obj.time), 3), "end": round(float(end), 3),
                    "kind": obj.kind, "slides": int(getattr(obj, "slides", 1) or 1)})
    return out


def beat_lookup(timing):
    """``beat_length(time)`` for a Beatmap (``timing_at``) or a TimingEstimate."""
    if hasattr(timing, "timing_at"):
        return lambda time: float(timing.timing_at(time)[0])
    points = timing.tempo_points or [(timing.offset_ms, timing.beat_length)]
    starts = [float(p[0]) for p in points]
    beats = [float(p[1]) for p in points]
    return lambda time: beats[max(0, bisect_right(starts, time) - 1)]


class _View:
    def __init__(self, beat_length):
        self.beat_length = beat_length

    def timing_at(self, time):
        return self.beat_length(time), 1.0


def summary(items: list[dict], timing, truth: np.ndarray | None = None) -> dict:
    """Figures, sliders, gaps and pauses per 100 objects (``truth``: human start times
    for rhythm F1 within 30 ms)."""
    beat_length = beat_lookup(timing)
    items = sorted(items, key=lambda e: e["time"])
    n = len(items)
    metrics = _rhythm_metrics(_View(beat_length),
                              [(SimpleNamespace(**item), None) for item in items])
    per100 = 100.0 / max(n, 1)
    out = {
        "objects": n,
        "double_1_4_per_100": metrics["double_1_4"],
        "triple_1_4_per_100": metrics["triple_1_4"],
        "burst_4_8_1_4_per_100": metrics["burst_4_8_1_4"],
        "stream_9p_1_4_per_100": metrics["stream_9p_1_4"],
        "slider_pct": per100 * sum(e["kind"] == "slider" for e in items),
        "repeat_slider_per_100": per100 * sum(e["kind"] == "slider" and e["slides"] > 1
                                              for e in items),
    }
    gaps, rests = [], []
    for a, b in zip(items, items[1:]):
        beat = max(beat_length(b["time"]), 1.0)
        gaps.append((b["time"] - a["time"]) / beat)
        rests.append((b["time"] - max(a["end"], a["time"])) / beat)
    gaps, rests = np.asarray(gaps), np.asarray(rests)
    share = 100.0 / max(len(gaps), 1)
    matched = np.zeros(len(gaps), dtype=bool)
    for name, value in GAP_CLASSES.items():
        hit = np.abs(gaps - value) <= max(0.012, 0.10 * value)
        out[f"gap_{name}_pct"] = share * float(hit.sum())
        matched |= hit
    out["gap_longer_pct"] = share * float(((gaps > 1.1) & ~matched).sum())
    out["gap_other_pct"] = share * float(((gaps <= 1.1) & ~matched).sum())
    # Pauses: free time between an object's end and the next start.
    out["pause_1_per_100"] = per100 * float((rests >= 0.95).sum())
    out["pause_2_per_100"] = per100 * float((rests >= 1.95).sum())
    if truth is not None:
        from .evaluate import match_f1
        out["rhythm_f1"] = match_f1(np.array([e["time"] for e in items]), np.asarray(truth))
    return out
