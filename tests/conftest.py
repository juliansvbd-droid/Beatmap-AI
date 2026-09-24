import numpy as np
import pytest
import soundfile as sf

from beatmap_ai.audio import SAMPLE_RATE


def drum_loop(bpm: float, offset: float, duration: float, seed: int = 0) -> np.ndarray:
    """A synthetic rock beat: kick on 1 and 3, snare on 2 and 4, hi-hat on eighths."""
    rng = np.random.default_rng(seed)
    sr = SAMPLE_RATE
    t = np.arange(int(sr * duration)) / sr
    y = 0.03 * np.sin(2 * np.pi * 220 * t)
    n = int(0.25 * sr)
    tt = np.arange(n) / sr
    sounds = {
        "kick": np.sin(2 * np.pi * (50 + 100 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12),
        "snare": (0.5 * np.sin(2 * np.pi * 190 * tt) + rng.normal(0, 0.5, n)) * np.exp(-tt * 25),
        "hat": rng.normal(0, 0.3, n) * np.exp(-tt * 60),
    }
    half_beat = 30.0 / bpm
    k = 0
    while offset + k * half_beat < duration - 0.3:
        i = int(round((offset + k * half_beat) * sr))
        hits = [("hat", 0.4)]
        if k % 4 == 0:
            hits.append(("kick", 1.0))
        elif k % 4 == 2:
            hits.append(("snare", 0.7))
        for name, amp in hits:
            seg = amp * sounds[name][: len(y) - i]
            y[i:i + len(seg)] += seg
        k += 1
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


@pytest.fixture(scope="session")
def song_mp3(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "Test Artist - Drum Loop.mp3"
    sf.write(path, drum_loop(bpm=160, offset=0.4, duration=30), SAMPLE_RATE)
    return path
