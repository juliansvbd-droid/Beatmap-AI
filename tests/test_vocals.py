import numpy as np
import pytest

from beatmap_ai.audio import SAMPLE_RATE
from beatmap_ai.vocals import _held_vowel_candidates, analyze_vocals


def _synthetic_syllables() -> np.ndarray:
    sr = SAMPLE_RATE
    y = np.zeros(int(2.5 * sr), dtype=np.float32)
    t = np.arange(len(y), dtype=np.float32) / sr
    carrier = 0.18 * np.sin(2 * np.pi * 220.0 * t) + 0.12 * np.sin(2 * np.pi * 2640.0 * t)
    for start_s in (0.25, 0.95, 1.65):
        start = int(start_s * sr)
        length = int(0.24 * sr)
        ramp = max(1, int(0.015 * sr))
        envelope = np.ones(length, dtype=np.float32)
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
        y[start:start + length] = carrier[start:start + length] * envelope
    return y


def test_separated_sine_syllables_have_expected_onsets():
    result = analyze_vocals(_synthetic_syllables(), estimate_pitch=False)

    assert len(result.syllables) == 3
    expected = np.asarray([250.0, 950.0, 1650.0])
    detected = np.asarray([row["time_ms"] for row in result.syllables])
    assert np.all(np.abs(detected - expected) < 60.0)
    assert result.frame_rate == pytest.approx(SAMPLE_RATE / 256.0)


def test_white_noise_is_classified_as_unvoiced():
    rng = np.random.default_rng(17)
    y = rng.normal(0.0, 0.1, SAMPLE_RATE).astype(np.float32)

    result = analyze_vocals(y, estimate_pitch=False)

    assert result.n_frames > 50
    assert np.mean(result.labels == 2) > 0.70
    assert np.mean(result.labels == 1) < 0.20


def test_silence_is_silent_and_has_no_syllables():
    result = analyze_vocals(np.zeros(SAMPLE_RATE, dtype=np.float32), estimate_pitch=False)

    assert np.all(result.labels == 0)
    assert result.syllables == []
    assert result.slider_candidates == []


def test_long_stable_vowel_becomes_a_slider_candidate():
    syllable = {
        "frame": 0, "end_frame": 19, "time_ms": 0.0, "end_ms": 220.0,
        "duration_ms": 220.0, "type": "VOWEL",
    }
    candidates = _held_vowel_candidates(
        [syllable],
        np.ones(20, dtype=np.uint8),
        np.full(20, 220.0, dtype=np.float32),
        np.zeros((20, 13), dtype=np.float32),
        quarter_ms=500.0,
    )

    assert len(candidates) == 1
    assert candidates[0]["pitch_stable"]
    assert candidates[0]["duration_ms"] >= 150.0
