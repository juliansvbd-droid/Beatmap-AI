from types import SimpleNamespace

import numpy as np

from beatmap_ai.audio import FPS
from beatmap_ai.rhythm_figures import quarter_link_targets, reinforce_figures


def test_quarter_targets_use_local_tempo_and_exclude_sliders():
    quarter_a = round(500 * FPS / 4000)
    quarter_b = round(400 * FPS / 4000)
    start_b = round(2000 * FPS / 1000)
    example = SimpleNamespace(
        note_frames=np.array([10, 10 + quarter_a, 60, start_b + 4,
                              start_b + 4 + quarter_b, start_b + 4 + 2 * quarter_b]),
        slider_frames=np.array([60, start_b + 4 + 2 * quarter_b]),
        grid=[(0.0, 500.0, 4), (2000.0, 400.0, 4)],
    )
    target, mask = quarter_link_targets(example, 0, start_b + 40)
    assert target[10] == 1
    assert target[start_b + 4] == 1
    assert target[start_b + 4 + quarter_b] == 0  # next object is a slider
    assert mask[60] == 0
    assert mask[start_b + 4 + 2 * quarter_b] == 0


def test_reinforcement_requires_audio_support_and_respects_run_limit():
    note = np.zeros(64)
    link = np.zeros(64)
    note[[11, 21, 31, 41]] = [0.5, 0.5, 0.5, 0.1]
    link[[10, 20, 30, 40]] = 0.9
    selected = np.array([10, 20, 30, 40])
    result = reinforce_figures(selected, note, link, divisor=4,
                               min_gap_beats=0.25, max_run=2, threshold=0.8,
                               max_extra_ratio=0.75)
    assert result.tolist() == [10, 11, 20, 21, 30, 31, 40]
    # A note one tick away from a selected note still must satisfy the model's
    # own note probability. A third note would exceed max_run=2.
    note[12] = 0.8
    link[11] = 0.9
    result = reinforce_figures(selected, note, link, divisor=4,
                               min_gap_beats=0.25, max_run=2, threshold=0.8,
                               max_extra_ratio=0.75)
    assert 12 not in result
