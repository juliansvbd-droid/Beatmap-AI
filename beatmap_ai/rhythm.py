"""Rhythm generation: deciding *when* hit objects happen and whether they are circles,
sliders or spinners.

Every candidate time is a tick on the beat grid (``divisor`` ticks per beat). Each tick
gets a score -- from onset strength (heuristic mode) or from the neural network's note
probability (model mode) -- and ticks are then picked greedily under spacing rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from .audio import FPS, AudioFeatures, sample_peak
from .difficulty import DifficultyPreset
from .timing import ONSET_LATENCY, TimingEstimate

ACTIVE_RMS = 0.08  # loudness below which the music counts as silent
ONSET_LAG_MS = ONSET_LATENCY * 1000.0  # onset-curve peaks come this long after the hit


@dataclass
class TickGrid:
    times: np.ndarray  # ms
    index: np.ndarray  # tick number counted from the timing offset (can be negative)
    divisor: int

    @property
    def pos_in_beat(self) -> np.ndarray:
        return self.index % self.divisor

    @property
    def beat(self) -> np.ndarray:
        return self.index // self.divisor

    @property
    def is_downbeat(self) -> np.ndarray:
        return (self.pos_in_beat == 0) & (self.beat % 4 == 0)


@dataclass
class PlannedObject:
    time: float
    kind: str  # "circle", "slider" or "spinner"
    tick: int  # index into the TickGrid
    intensity: float  # 0..1, how strongly the music hits here
    end_time: float  # equals time for circles
    measure: int
    new_combo: bool = False


def make_tick_grid(timing: TimingEstimate, duration_ms: float, divisor: int) -> TickGrid:
    tick_ms = timing.beat_length / divisor
    first = -int(np.floor(timing.offset_ms / tick_ms))
    last = int(np.floor((duration_ms - 100.0 - timing.offset_ms) / tick_ms))
    index = np.arange(first, max(last, first) + 1)
    return TickGrid(times=timing.offset_ms + index * tick_ms, index=index, divisor=divisor)


def heuristic_scores(features: AudioFeatures, grid: TickGrid, lag_ms: float = ONSET_LAG_MS,
                     radius: int = 2) -> np.ndarray:
    """Score ticks by onset strength, favouring strong metrical positions."""
    onset = sample_peak(features.onset, grid.times + lag_ms, radius=radius)
    rms = sample_peak(features.rms, grid.times, radius=0)
    pos = grid.pos_in_beat
    weight = np.full(len(pos), 0.9, dtype=np.float32)
    weight[pos * 2 == grid.divisor] = 1.05
    weight[pos == 0] = 1.2
    weight[grid.is_downbeat] = 1.3
    score = onset * weight + 0.15 * rms * (pos == 0)
    # Blend in loudness-independent scores so quiet sections still get notes.
    local_max = maximum_filter1d(score, size=8 * grid.divisor * 4 + 1)
    return 0.5 * score + 0.5 * score / (local_max + 1e-3)


def select_ticks(
    scores: np.ndarray,
    active: np.ndarray,
    preset: DifficultyPreset,
    threshold: float,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Pick ticks scoring at least ``threshold``, strongest first, while respecting the
    minimum gap between objects and the maximum stream length. Ticks already set in
    ``fixed`` are kept and constrain the new picks. Returns a boolean mask."""
    min_gap = max(int(round(preset.min_gap_beats * preset.divisor)), 1)
    taken = np.zeros(len(scores), dtype=bool) if fixed is None else fixed.copy()
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < threshold:
            break
        if not active[i] or taken[max(i - min_gap + 1, 0): i + min_gap].any():
            continue
        if _run_length(taken, i, min_gap) + 1 > preset.max_run:
            continue
        taken[i] = True
    return taken


def select_ticks_by_density(
    scores: np.ndarray,
    active: np.ndarray,
    grid: TickGrid,
    preset: DifficultyPreset,
    beat_length: float,
    loudness: np.ndarray,
    window_beats: int = 16,
) -> np.ndarray:
    """Choose ticks window by window so each stretch of the song gets roughly the
    preset's density, scaled up in loud sections and down in quiet ones."""
    window = window_beats * grid.divisor
    tick_seconds = beat_length / grid.divisor / 1000.0
    reference = np.median(loudness[active]) if active.any() else 1.0
    taken = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), window):
        span = slice(start, min(start + window, len(scores)))
        n_active = active[span].sum()
        if n_active == 0:
            continue
        factor = np.clip(0.6 + 0.4 * loudness[span][active[span]].mean() / (reference + 1e-6), 0.5, 1.4)
        count = max(int(round(preset.density * factor * n_active * tick_seconds)), 1)
        local = np.full(len(scores), -np.inf)
        local[span] = scores[span]
        taken = _select_count(local, active, preset, count, taken)
    return np.flatnonzero(taken)


def _select_count(scores, active, preset, count, fixed) -> np.ndarray:
    """Binary-search the score threshold that adds about ``count`` ticks to ``fixed``."""
    base = fixed.sum()
    finite = scores[np.isfinite(scores)]
    lo, hi = float(finite.min()) - 1e-6, float(finite.max()) + 1e-6
    best = fixed
    for _ in range(20):
        mid = (lo + hi) / 2
        taken = select_ticks(scores, active, preset, mid, fixed)
        if taken.sum() - base > count:
            lo = mid
        else:
            hi, best = mid, taken
    # Equal scores (e.g. a repeated drum loop) make the count jump in steps; take
    # whichever side of the step is closer to the target in ratio.
    over = select_ticks(scores, active, preset, lo, fixed)
    n_best, n_over = best.sum() - base, over.sum() - base
    if n_best == 0 or n_over / count < count / n_best:
        return over
    return best


def _run_length(taken: np.ndarray, i: int, step: int) -> int:
    """Number of taken ticks chained to ``i`` at exactly ``step`` spacing, on both sides."""
    n = 0
    for direction in (-1, 1):
        j = i + direction * step
        while 0 <= j < len(taken) and taken[j]:
            n += 1
            j += direction * step
    return n


def fill_gaps(ticks: np.ndarray, grid: TickGrid, active: np.ndarray, beats: int = 8) -> np.ndarray:
    """Add on-beat notes where the music plays but nothing was picked for ``beats`` beats,
    so quiet-but-active passages are not left empty."""
    span = beats * grid.divisor
    step = 2 * grid.divisor
    added = []
    bounds = np.concatenate([[-1], ticks, [len(grid.times)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= span:
            continue
        candidates = [
            t for t in range(a + step, b - step + 1)
            if grid.pos_in_beat[t] == 0 and active[t]
        ]
        last = a
        for t in candidates:
            if t - last >= step:
                added.append(t)
                last = t
    return np.array(sorted(set(ticks.tolist()) | set(added)), dtype=int)


def plan_objects(
    features: AudioFeatures,
    timing: TimingEstimate,
    preset: DifficultyPreset,
    rng: np.random.Generator,
    note_probs: np.ndarray | None = None,
    slider_probs: np.ndarray | None = None,
    threshold: float = 0.5,
) -> list[PlannedObject]:
    """Build the full rhythm for one difficulty.

    ``note_probs``/``slider_probs`` are per-frame probabilities from the neural model;
    when omitted, onset-strength heuristics are used instead.
    """
    duration_ms = features.duration * 1000.0
    grid = make_tick_grid(timing, duration_ms, preset.divisor)
    if len(grid.times) == 0:
        return []
    rms = sample_peak(features.rms, grid.times, radius=2)
    active = rms > ACTIVE_RMS
    beat_length = timing.beat_length

    if note_probs is None:
        scores = heuristic_scores(features, grid)
        ticks = select_ticks_by_density(scores, active, grid, preset, beat_length, rms)
        ticks = fill_gaps(ticks, grid, active)
    else:
        scores = sample_peak(note_probs, grid.times, radius=1)
        ticks = np.flatnonzero(select_ticks(scores, np.ones_like(active), preset, threshold))
    if len(ticks) == 0:
        return []
    intensity = np.clip(sample_peak(features.onset, grid.times + ONSET_LAG_MS, radius=2), 0.0, 1.0)

    objects: list[PlannedObject] = []
    recovery = max(int(round(preset.slider_recovery_beats * preset.divisor)), 1)
    min_slider = max(int(round(preset.min_slider_beats * preset.divisor)), 1)
    max_slider = max(int(round(preset.max_slider_beats * preset.divisor)), min_slider)
    for n, tick in enumerate(ticks):
        time = float(grid.times[tick])
        obj = PlannedObject(time, "circle", int(tick), float(intensity[tick]), time,
                            int(grid.beat[tick] // 4))
        if n + 1 < len(ticks):
            next_tick = ticks[n + 1]
            gap_beats = (next_tick - tick) / preset.divisor
            length = min(next_tick - tick - recovery, max_slider) // min_slider * min_slider
            if (6 <= gap_beats <= 16 and active[tick:next_tick].mean() > 0.8
                    and (intensity[tick + 1:next_tick] > 0.5).mean() < 0.15):
                # A long gap in sustained music without strong hits: fill it with a spinner.
                lead = max(preset.min_gap_beats, 1.0) * beat_length
                start, end = time + lead, float(grid.times[next_tick]) - lead
                if end - start >= 2 * beat_length:
                    objects.append(obj)
                    objects.append(PlannedObject(start, "spinner", int(tick), 1.0, end, obj.measure))
                    continue
            if length >= min_slider and _wants_slider(
                features, grid, scores, tick, next_tick, preset, rng, slider_probs
            ):
                obj.kind = "slider"
                obj.end_time = time + length * beat_length / preset.divisor
        objects.append(obj)
    return objects


def _wants_slider(features, grid, scores, tick, next_tick, preset, rng, slider_probs) -> bool:
    if slider_probs is not None:
        return bool(sample_peak(slider_probs, grid.times[tick:tick + 1], radius=1)[0] > 0.5)
    a = int(grid.times[tick] * FPS / 1000.0)
    b = int(grid.times[next_tick] * FPS / 1000.0)
    held = features.rms[a:max(b, a + 1)]
    sustain = float(np.clip(held.mean() / (features.rms[min(a, len(features.rms) - 1)] + 1e-3), 0, 1))
    inner = scores[tick + 1:next_tick]
    # Strong unpicked onsets inside the gap argue against holding a slider over them.
    busy = float(inner.max() / (scores[tick] + 1e-3)) if len(inner) else 0.0
    p = preset.slider_rate * (0.5 + sustain) * (1.0 - 0.5 * min(busy, 1.0))
    return bool(rng.random() < p)
