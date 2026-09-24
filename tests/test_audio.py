import numpy as np
import pytest

from beatmap_ai.audio import compute_features, load_audio
from beatmap_ai.timing import ONSET_LATENCY, estimate_timing, fit_offset

from .conftest import drum_loop


# The synthetic drums attack instantly, so the onset curve peaks only ~18 ms after them.
# ONSET_LATENCY is calibrated on real music, whose attacks are slower, so offsets detected
# on synthetic audio land early by the difference.
SYNTHETIC_BIAS_MS = 18.0 - ONSET_LATENCY * 1000.0


def phase_error_ms(offset_ms: float, true_offset_ms: float, bpm: float) -> float:
    beat = 60000.0 / bpm
    return abs((offset_ms - true_offset_ms - SYNTHETIC_BIAS_MS + beat / 2) % beat - beat / 2)


@pytest.mark.parametrize("bpm, offset", [(150, 0.3), (174, 0.51), (128, 0.0955), (200, 1.0)])
def test_detects_bpm_and_offset(bpm, offset):
    features = compute_features(drum_loop(bpm, offset, 30))
    timing = estimate_timing(features)
    assert timing.bpm == pytest.approx(bpm, abs=0.05)
    assert phase_error_ms(timing.offset_ms, offset * 1000, bpm) <= 5
    # The offset must land on a kick (beat 1 or 3), not the snare.
    beats = (timing.offset_ms - offset * 1000) / timing.beat_length
    assert round(beats) % 2 == 0


def test_fit_offset_with_known_bpm():
    features = compute_features(drum_loop(140, 0.25, 20))
    assert phase_error_ms(fit_offset(features, 140), 250, 140) <= 5


def test_loads_mp3(song_mp3):
    features = compute_features(load_audio(song_mp3))
    assert features.duration == pytest.approx(30, abs=0.1)
    assert features.mel.shape[0] == 80
    assert np.isfinite(features.mel).all()
    assert estimate_timing(features).bpm == pytest.approx(160, abs=0.05)
