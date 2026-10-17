"""Targets and constrained decoding for short quarter-beat figures."""

from __future__ import annotations

import heapq

import numpy as np

from .audio import FPS


def quarter_link_targets(example, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """At circle starts, label whether the next circle follows one quarter beat later.

    The mask includes only circle starts. Song timing points supply the local beat
    length, so variable-tempo maps do not require a cached extra label column.
    """
    target = np.zeros(length, dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    frames = np.unique(example.note_frames)
    if len(frames) == 0:
        return target, mask
    is_slider = np.isin(frames, example.slider_frames)
    local = frames - start
    valid = (local >= 0) & (local < length) & ~is_slider
    mask[local[valid]] = 1.0
    if len(frames) < 2:
        return target, mask
    points = sorted(example.grid)
    starts = np.asarray([point[0] for point in points], dtype=float)
    beat_lengths = np.asarray([point[1] for point in points], dtype=float)
    which = np.clip(np.searchsorted(starts, frames[:-1] * 1000.0 / FPS,
                                    side="right") - 1, 0, len(starts) - 1)
    quarter = beat_lengths[which] * FPS / 4000.0
    gap = np.diff(frames)
    linked = (np.abs(gap - quarter) <= np.maximum(2.0, 0.10 * quarter))
    linked &= ~is_slider[:-1] & ~is_slider[1:] & valid[:-1]
    target[local[:-1][linked]] = 1.0
    return target, mask


def reinforce_figures(ticks: np.ndarray, note_scores: np.ndarray,
                      link_scores: np.ndarray, *, divisor: int,
                      min_gap_beats: float, max_run: int, threshold: float,
                      link_threshold: float = 0.5,
                      min_note_ratio: float = 0.45,
                      max_extra_ratio: float = 0.05) -> np.ndarray:
    """Add a predicted quarter-beat successor when it fits the existing rhythm.

    The link head proposes *where a figure belongs*. The ordinary note head must
    still hear the new note, and the usual gap/run limits still apply. A small
    addition budget avoids replacing the song's rhythm with an enforced quota.
    """
    chosen = set(map(int, ticks))
    if not chosen or divisor < 4:
        return np.asarray(sorted(chosen), dtype=int)
    step = max(divisor // 4, 1)
    min_gap = max(int(round(min_gap_beats * divisor)), 1)
    budget = max(0, int(round(len(chosen) * max_extra_ratio)))
    heap: list[tuple[float, int]] = []

    def offer(anchor: int) -> None:
        next_tick = anchor + step
        if (next_tick >= len(note_scores) or next_tick in chosen
                or link_scores[anchor] < link_threshold
                or note_scores[next_tick] < min_note_ratio * threshold):
            return
        heapq.heappush(heap, (-float(link_scores[anchor] * note_scores[next_tick]),
                              next_tick))

    for tick in sorted(chosen):
        offer(tick)
    added = 0
    while heap and added < budget:
        _, tick = heapq.heappop(heap)
        if tick in chosen or not any(tick - step == prior for prior in chosen):
            continue
        if any(abs(tick - prior) < min_gap for prior in chosen):
            continue
        run = 1
        for direction in (-step, step):
            neighbor = tick + direction
            while neighbor in chosen:
                run += 1
                neighbor += direction
        if run > max_run:
            continue
        chosen.add(tick)
        added += 1
        offer(tick)
    return np.asarray(sorted(chosen), dtype=int)
