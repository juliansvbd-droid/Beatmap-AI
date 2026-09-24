"""Audio loading and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_FFT = 2048
N_MELS = 80
FPS = SAMPLE_RATE / HOP_LENGTH  # feature frames per second (~86)


@dataclass
class AudioFeatures:
    mel: np.ndarray  # (N_MELS, T) log-mel spectrogram scaled to roughly [-1, 1]
    onset: np.ndarray  # (T,) onset strength in [0, 1]
    rms: np.ndarray  # (T,) loudness in [0, 1]
    bass_onset: np.ndarray  # (T,) onset strength of the bass range (kicks), in [0, 1]
    duration: float  # seconds

    @property
    def n_frames(self) -> int:
        return self.mel.shape[1]

    def frame_times_ms(self) -> np.ndarray:
        return np.arange(self.n_frames) * 1000.0 / FPS


def load_audio(path: str | Path) -> np.ndarray:
    """Decode to mono at SAMPLE_RATE."""
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        # libsndfile rejects some slightly broken MP3s that miniaudio decodes fine
        # (both trim the encoder delay, so timing is identical).
        import miniaudio
        decoded = miniaudio.decode_file(str(path), output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1)
        y, sr = np.frombuffer(decoded.samples, dtype=np.float32), decoded.sample_rate
    return y if sr == SAMPLE_RATE else librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)


def compute_features(y: np.ndarray) -> AudioFeatures:
    power = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(power, ref=np.max, top_db=80.0)
    onset = librosa.onset.onset_strength(S=mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    n_bass = int(np.searchsorted(librosa.mel_frequencies(N_MELS + 2, fmax=SAMPLE_RATE / 2), 200.0))
    bass_onset = librosa.onset.onset_strength(
        S=mel_db[: max(n_bass, 2)], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    n = mel_db.shape[1]
    return AudioFeatures(
        mel=((mel_db + 40.0) / 40.0).astype(np.float32),
        onset=_normalize(_fit_length(onset, n)),
        rms=_normalize(_fit_length(rms, n)),
        bass_onset=_normalize(_fit_length(bass_onset, n)),
        duration=len(y) / SAMPLE_RATE,
    )


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _normalize(x: np.ndarray) -> np.ndarray:
    scale = np.percentile(x, 99) if len(x) else 0.0
    return np.clip(x / (scale + 1e-8), 0.0, 1.0).astype(np.float32)


def sample_peak(signal: np.ndarray, times_ms: np.ndarray, radius: int = 1) -> np.ndarray:
    """Max of a per-frame ``signal`` within ``radius`` frames of each time."""
    frames = np.rint(np.asarray(times_ms) * FPS / 1000.0).astype(int)
    out = np.zeros(len(frames), dtype=np.float32)
    for offset in range(-radius, radius + 1):
        idx = frames + offset
        valid = (idx >= 0) & (idx < len(signal))
        out[valid] = np.maximum(out[valid], signal[idx[valid]])
    return out


def preview_time_ms(features: AudioFeatures, window_s: float = 10.0) -> int:
    """Start of the loudest ``window_s`` second stretch, used as the song-select preview."""
    window = max(int(window_s * FPS), 1)
    if features.n_frames <= window:
        return 0
    energy = np.convolve(features.rms, np.ones(window), mode="valid")
    return int(np.argmax(energy) * 1000.0 / FPS)

