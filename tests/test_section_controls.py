import numpy as np

from beatmap_ai.placement_data import BEAT, END, EX, EY, KIAI, KIND, N_COLUMNS, SLIDER, T, X, Y
from beatmap_ai.sequence_data import _section_controls_reference, section_controls


def _random_map(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    o = np.zeros((n, N_COLUMNS), dtype=np.float32)
    o[:, T] = np.cumsum(rng.choice([62.5, 125.0, 250.0, 500.0], size=n))
    o[:, BEAT] = rng.choice([300.0, 500.0], size=n)
    o[:, X] = rng.uniform(0, 512, size=n)
    o[:, Y] = rng.uniform(0, 384, size=n)
    stacked = rng.random(n) < 0.1  # some stacks: moves under 5 px are no turns
    o[stacked, X] = np.roll(o[:, X], 1)[stacked]
    o[stacked, Y] = np.roll(o[:, Y], 1)[stacked]
    sliders = rng.random(n) < 0.4
    o[:, KIND] = np.where(sliders, SLIDER, 0)
    o[:, EX] = np.where(sliders, np.clip(o[:, X] + rng.normal(0, 60, n), 0, 512), o[:, X])
    o[:, EY] = np.where(sliders, np.clip(o[:, Y] + rng.normal(0, 60, n), 0, 384), o[:, Y])
    o[:, END] = o[:, T] + np.where(sliders, 100.0, 0.0)
    o[n // 3:n // 2, KIAI] = 1.0
    return o


def test_vectorised_section_values_match_the_loop_version():
    for seed, (n, start, count, stars) in enumerate([(300, 0, 300, 4.5), (120, 40, 30, 2.2),
                                                     (80, 70, 50, 6.4), (2, 0, 2, 3.0)]):
        objects = _random_map(n, seed)
        ref = _section_controls_reference(objects, start, count, stars, 4.0)
        new = section_controls(objects, start, count, stars, 4.0)
        assert new.shape == ref.shape
        assert np.allclose(new, ref, atol=1e-5)
