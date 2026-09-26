"""Beat-grid features shared by the dataset and the model (no PyTorch needed)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .audio import FPS

# (time ms, ms per beat, beats per bar) for each uninherited timing point.
BeatGrid = Sequence[tuple[float, float, int]]


def beat_phase_features(n_frames: int, grid: BeatGrid) -> np.ndarray:
    """sin/cos of the position within the current beat and bar for every frame, (T, 4)."""
    times = np.arange(n_frames) * 1000.0 / FPS
    grid = sorted(grid)
    starts = np.array([g[0] for g in grid])
    which = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(grid) - 1)
    t0 = starts[which]
    beat_length = np.array([g[1] for g in grid])[which]
    meter = np.array([g[2] for g in grid])[which]
    beats = (times - t0) / beat_length
    beat_phase = 2 * np.pi * (beats % 1.0)
    bar_phase = 2 * np.pi * ((beats % meter) / meter)
    return np.stack(
        [np.sin(beat_phase), np.cos(beat_phase), np.sin(bar_phase), np.cos(bar_phase)], axis=1
    ).astype(np.float32)
