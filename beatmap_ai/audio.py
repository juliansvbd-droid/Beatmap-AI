"""Audio loading, feature extraction and tempo/offset estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_FFT = 2048
N_MELS = 80
FPS = SAMPLE_RATE / HOP_LENGTH  # feature frames per second (~86)
# Onset-strength peaks trail the actual attack by this much (measured on synthetic drums).
ONSET_LATENCY = 0.018


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


@dataclass
class TimingEstimate:
    bpm: float
    offset_ms: float  # time of the first downbeat

    @property
    def beat_length(self) -> float:
        return 60000.0 / self.bpm


def load_audio(path: str | Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y


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


def _grid_score(onset: np.ndarray, period: float, phases: np.ndarray, duration: float) -> np.ndarray:
    """Mean onset strength on beat grids with the given period and phases (seconds)."""
    n_beats = max(int(duration / period) - 1, 1)
    times = phases[:, None] + np.arange(n_beats)[None, :] * period
    values = np.interp(times * FPS, np.arange(len(onset)), onset, right=0.0)
    return values.mean(axis=1)


def estimate_timing(features: AudioFeatures, min_bpm: float = 90.0, max_bpm: float = 300.0) -> TimingEstimate:
    """Estimate a single constant BPM and the offset of the first downbeat."""
    onset = gaussian_filter1d(features.onset.astype(np.float64), 1.0)
    duration = features.duration
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=features.onset, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    if len(beat_times) < 4:
        return TimingEstimate(bpm=120.0, offset_ms=0.0)

    # Fit a straight line through the tracked beats: time = phase + index * period.
    # Beat indices come from rounding each inter-beat interval, so tempo drift in the
    # initial guess cannot accumulate into wrong indices.
    period = float(np.median(np.diff(beat_times)))
    index = np.concatenate([[0], np.cumsum(np.maximum(np.rint(np.diff(beat_times) / period), 1))])
    keep = np.ones(len(beat_times), dtype=bool)
    for _ in range(3):
        period, phase = np.polyfit(index[keep], beat_times[keep], 1)
        residual = np.abs(beat_times - (phase + index * period))
        keep = residual < max(0.1 * period, np.percentile(residual, 80))
        if keep.sum() < 4:
            break
    bpm = 60.0 / period
    while bpm < min_bpm:
        bpm *= 2
    while bpm > max_bpm:
        bpm /= 2
    period = 60.0 / bpm

    # Refine BPM and phase by maximising onset strength on the beat grid. Integer BPMs
    # are always candidates, and win if they fit (almost) as well: mappers prefer them.
    def best_phase(bpm: float) -> tuple[float, float]:
        p = 60.0 / bpm
        phases = np.arange(0.0, p, 0.002)
        scores = _grid_score(onset, p, phases, duration)
        i = int(np.argmax(scores))
        return float(scores[i]), float(phases[i])

    candidates = np.concatenate([
        np.linspace(bpm * 0.98, bpm * 1.02, 161),
        np.arange(np.ceil(bpm * 0.98), np.floor(bpm * 1.02) + 1),
    ])
    fits = {float(c): best_phase(c) for c in candidates}
    bpm = max(fits, key=lambda c: fits[c][0])
    best_score = fits[bpm][0]
    integers = [c for c in fits if c.is_integer() and fits[c][0] >= 0.97 * best_score]
    if integers:
        bpm = max(integers, key=lambda c: fits[c][0])
    phase = fits[bpm][1]
    period = 60.0 / bpm

    # Hi-hats can make the off-beats as strong as the beats; the bass decides.
    bass = gaussian_filter1d(features.bass_onset.astype(np.float64), 1.0)
    candidates = np.array([phase, phase + period / 2]) % period
    phase = float(candidates[np.argmax(
        _grid_score(onset, period, candidates, duration) + _grid_score(bass, period, candidates, duration)
    )])

    # Beat trackers often lock onto half the real tempo. If the half-beats are as strong
    # as the beats themselves, the real beat is twice as fast.
    beat_score, half_score = _grid_score(onset, period, np.array([phase, phase + period / 2]), duration)
    if half_score >= beat_score and 2 * bpm <= max_bpm:
        bpm, period = 2 * bpm, period / 2

    phase -= ONSET_LATENCY
    return TimingEstimate(bpm=round(float(bpm), 3), offset_ms=_pick_downbeat(bass, period, phase, duration))


def preview_time_ms(features: AudioFeatures, window_s: float = 10.0) -> int:
    """Start of the loudest ``window_s`` second stretch, used as the song-select preview."""
    window = max(int(window_s * FPS), 1)
    if features.n_frames <= window:
        return 0
    energy = np.convolve(features.rms, np.ones(window), mode="valid")
    return int(np.argmax(energy) * 1000.0 / FPS)


def fit_offset(features: AudioFeatures, bpm: float) -> float:
    """Best downbeat offset (ms) for a known BPM."""
    onset = gaussian_filter1d(features.onset.astype(np.float64), 1.0)
    bass = gaussian_filter1d(features.bass_onset.astype(np.float64), 1.0)
    period = 60.0 / bpm
    phases = np.arange(0.0, period, 0.001)
    scores = _grid_score(onset, period, phases, features.duration)
    scores += _grid_score(bass, period, phases, features.duration)
    phase = float(phases[np.argmax(scores)]) - ONSET_LATENCY
    return _pick_downbeat(bass, period, phase, features.duration)


def _pick_downbeat(onset: np.ndarray, period: float, phase: float, duration: float) -> float:
    """Choose which beat of the 4/4 bar is the downbeat (strongest in ``onset``); returns ms."""
    measure = 4 * period
    scores = [_grid_score(onset, measure, np.array([phase + j * period]), duration)[0] for j in range(4)]
    return float(round(((phase + int(np.argmax(scores)) * period) % measure) * 1000.0))
