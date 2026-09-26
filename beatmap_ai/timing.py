"""Tempo (BPM) and offset estimation.

1. librosa's beat tracker gives a rough tempo. Beat trackers often land on a related
   tempo instead (half, double, 2/3, 3/2, ...), so each of those ratios is a candidate.
2. Every candidate's BPM and phase are refined against the onset curve of the whole song.
3. A small learned model (a softmax over hand-made features of each candidate) picks
   the right one. Its weights were fitted on ranked beatmaps with
   ``scripts/fit_timing.py``.
4. An integer BPM that fits (almost) as well replaces the pick, bass onsets decide
   between the beat and the off-beat, and the offset is corrected for onset-detector lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .audio import FPS, HOP_LENGTH, SAMPLE_RATE, AudioFeatures

RATIOS = (1 / 2, 2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0)
MIN_BPM, MAX_BPM = 60.0, 330.0
FEATURE_NAMES = (
    "log(score / best score)",
    "onsets on 1/4 grid",
    "onsets on 1/2 grid",
    "onsets on beats",
    "onsets on 1/3 grid",
    "half-beat score / score",
    "log2(bpm / 170)",
    "log2(bpm / 170)^2",
)
# Fitted by scripts/fit_timing.py (see the README for the accuracy on held-out songs).
TEMPO_WEIGHTS = np.array([2.745, 1.736, 2.826, 1.073, -0.349, 0.651, -0.137, -2.701])
# The onset curve peaks this many seconds after the attack that mappers time notes to.
ONSET_LATENCY = 0.058


@dataclass
class TimingEstimate:
    bpm: float
    offset_ms: float  # time of the first downbeat
    swing_ratio: float = 0.5  # where the swung eighth-note falls within one beat
    swing_confidence: float = 0.0
    beat_times_ms: np.ndarray | None = None
    beat_indices: np.ndarray | None = None
    tempo_points: list[tuple[float, float]] | None = None

    @property
    def beat_length(self) -> float:
        return 60000.0 / self.bpm


@dataclass
class TempoCandidate:
    bpm: float
    phase: float  # seconds: time of some beat
    score: float  # mean onset strength on the beats
    features: np.ndarray | None = None


def grid_score(onset: np.ndarray, period: float, phases: np.ndarray, duration: float) -> np.ndarray:
    """Mean onset strength on beat grids with the given period and phases (seconds)."""
    n_beats = max(int(duration / period) - 1, 1)
    times = phases[:, None] + np.arange(n_beats)[None, :] * period
    values = np.interp(times * FPS, np.arange(len(onset)), onset, right=0.0)
    return values.mean(axis=1)


def _best_phase(onset: np.ndarray, duration: float, bpm: float, step: float = 0.002):
    period = 60.0 / bpm
    phases = np.arange(0.0, period, step)
    scores = grid_score(onset, period, phases, duration)
    i = int(np.argmax(scores))
    return float(scores[i]), float(phases[i])


def refine_tempo(onset: np.ndarray, duration: float, bpm: float, span: float = 0.02) -> TempoCandidate:
    """Best BPM within ``span`` (relative) of ``bpm``, searched coarse to fine."""
    best = None
    for width, n in ((span, 41), (span / 20, 21)):
        center = bpm if best is None else best.bpm
        for candidate in np.linspace(center * (1 - width), center * (1 + width), n):
            score, phase = _best_phase(onset, duration, candidate, step=0.003)
            if best is None or score > best.score:
                best = TempoCandidate(float(candidate), phase, score)
    return best


def _smooth(x: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(x.astype(np.float64), 1.0)


def tempo_candidates(features: AudioFeatures) -> list[TempoCandidate]:
    """Refined candidate tempos with their feature vectors."""
    onset = _smooth(features.onset)
    duration = features.duration
    tempo, _ = librosa.beat.beat_track(onset_envelope=features.onset, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    base = float(np.atleast_1d(tempo)[0]) or 120.0
    candidates = [refine_tempo(onset, duration, base * r) for r in RATIOS if MIN_BPM <= base * r <= MAX_BPM]

    peaks = librosa.util.peak_pick(features.onset, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                   delta=0.1, wait=4)
    weights = features.onset[peaks].astype(np.float64)
    times = peaks / FPS
    best_score = max(c.score for c in candidates)
    for c in candidates:
        period = 60.0 / c.bpm

        def on_grid(division: int) -> float:
            step = period / division
            error = (times - c.phase + step / 2) % step - step / 2
            return float((weights * (np.abs(error) < 0.015)).sum() / (weights.sum() + 1e-9))

        half = grid_score(onset, period, np.array([c.phase + period / 2]), duration)[0]
        octave = np.log2(c.bpm / 170.0)
        c.features = np.array([
            np.log(c.score / best_score + 1e-6), on_grid(4), on_grid(2), on_grid(1), on_grid(3),
            half / (c.score + 1e-9), octave, octave ** 2,
        ])
    return candidates


def pick_tempo(candidates: list[TempoCandidate], weights: np.ndarray = TEMPO_WEIGHTS) -> TempoCandidate:
    return max(candidates, key=lambda c: float(c.features @ weights))


def _mel_band_onsets(features: AudioFeatures) -> list[np.ndarray]:
    """Simple per-band positive spectral flux, used only to resolve close BPM choices."""
    mel = features.mel.astype(np.float32, copy=False)
    centers = librosa.mel_frequencies(n_mels=mel.shape[0], fmax=SAMPLE_RATE / 2)
    flux = np.maximum(np.diff(mel, axis=1, prepend=mel[:, :1]), 0.0)
    bands = []
    for low, high in ((20, 200), (200, 2000), (2000, 6000), (6000, SAMPLE_RATE / 2)):
        mask = (centers >= low) & (centers < high)
        if not mask.any():
            bands.append(np.zeros(mel.shape[1], dtype=np.float32))
            continue
        signal = flux[mask].mean(axis=0)
        scale = float(np.percentile(signal, 99)) if len(signal) else 0.0
        bands.append(_smooth(signal / (scale + 1e-8)).astype(np.float32))
    return bands


def _multiband_support(features: AudioFeatures, candidates: list[TempoCandidate]) -> np.ndarray:
    """How strongly each candidate's beat/phase is supported across audio bands."""
    bands = _mel_band_onsets(features)
    weights = np.array([3.0, 2.0, 0.5, 1.5], dtype=np.float64)  # bass, mid, vocal, treble
    evidence = np.zeros((len(candidates), len(bands)), dtype=np.float64)
    offsets = np.arange(-0.03, 0.031, 0.01)
    for i, candidate in enumerate(candidates):
        period = 60.0 / candidate.bpm
        phases = candidate.phase + offsets
        for b, signal in enumerate(bands):
            evidence[i, b] = float(np.max(grid_score(signal, period, phases, features.duration)))
    maxima = evidence.max(axis=0)
    usable = maxima > 1e-8
    if not usable.any():
        return np.zeros(len(candidates), dtype=np.float64)
    normalized = evidence[:, usable] / maxima[usable]
    band_weights = weights[usable]
    return normalized @ band_weights / band_weights.sum()


def estimate_swing(features: AudioFeatures, bpm: float, offset_ms: float) -> tuple[float, float]:
    """Estimate a conservative swing ratio from onset positions between quarter beats."""
    if bpm <= 0 or features.duration < 8.0:
        return 0.5, 0.0
    signal = _smooth(0.7 * features.onset + 0.3 * features.bass_onset)
    peaks = librosa.util.peak_pick(
        signal, pre_max=2, post_max=2, pre_avg=10, post_avg=10,
        delta=0.04, wait=max(5, int(round(0.08 * FPS))),
    )
    if len(peaks) < 8:
        return 0.5, 0.0
    period = 60.0 / bpm
    times = peaks / FPS - ONSET_LATENCY
    phases = ((times - offset_ms / 1000.0) % period) / period
    # Ignore obvious sixteenth positions; swing evidence should come from the
    # eighth-note slot moving away from the straight midpoint.
    selected = (phases > 0.42) & (phases < 0.72)
    positions = phases[selected]
    weights = signal[peaks[selected]].astype(np.float64)
    if len(positions) < 6 or weights.sum() <= 1e-8:
        return 0.5, 0.0
    ratio = float(np.average(positions, weights=weights))
    variance = float(np.average((positions - ratio) ** 2, weights=weights))
    consistency = 1.0 - min(variance * 20.0, 1.0)
    count_confidence = min(len(positions) / 20.0, 1.0)
    confidence = float(np.clip(consistency * count_confidence, 0.0, 1.0))
    if ratio < 0.535 or confidence < 0.25:
        return 0.5, 0.0
    # Shrink weak evidence toward a straight grid to keep false swing subtle.
    return 0.5 + (ratio - 0.5) * confidence, confidence


def _track_variable_beats(
    features: AudioFeatures, bpm: float, offset_ms: float,
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[float, float]] | None]:
    """Follow local beat peaks with a tempo-regularized DP; keep the global grid if stable."""
    if bpm <= 0 or features.duration < 20.0:
        return None, None, None
    bands = _mel_band_onsets(features)
    signal = 1.5 * bands[0] + 1.2 * bands[1] + 0.8 * bands[2] + bands[3]
    scale = float(np.percentile(signal, 99)) if len(signal) else 0.0
    signal = signal / (scale + 1e-8)
    n = len(signal)
    period_frames = 60.0 / bpm * FPS
    min_lag = max(2, int(round(period_frames * 0.72)))
    max_lag = max(min_lag + 1, int(round(period_frames * 1.28)))
    window = max(2, int(round(period_frames)))

    # Local-mean-subtracted positive flux emphasizes attacks and ignores sustained tone.
    prefix = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    onset = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        local_mean = (prefix[hi] - prefix[lo]) / (hi - lo)
        onset[i] = max(0.0, signal[i] - local_mean)
    peak_scale = float(np.percentile(onset, 99)) if len(onset) else 0.0
    if peak_scale <= 1e-9:
        return None, None, None
    onset = np.clip(onset / peak_scale, 0.0, 1.5)

    cumulative = onset.copy()
    previous = np.full(n, -1, dtype=np.int32)
    tightness = 18.0
    for i in range(n):
        lo, hi = max(0, i - max_lag), i - min_lag
        if hi < lo:
            continue
        best_score, best_j = 0.0, -1
        for j in range(lo, hi + 1):
            lag = i - j
            deviation = np.log(lag / period_frames)
            score = cumulative[j] - tightness * deviation * deviation
            if score > best_score:
                best_score, best_j = score, j
        if best_j >= 0:
            cumulative[i] += best_score
            previous[i] = best_j

    tail = int(np.argmax(cumulative))
    frames = []
    current = tail
    while current >= 0:
        frames.append(current)
        current = int(previous[current])
    frames.reverse()
    if len(frames) < 24:
        return None, None, None

    refined = []
    for frame in frames:
        lo, hi = max(1, frame - 2), min(n - 2, frame + 2)
        peak = lo + int(np.argmax(onset[lo:hi + 1]))
        fraction = 0.5 * (onset[peak - 1] - onset[peak + 1]) / (
            onset[peak - 1] - 2 * onset[peak] + onset[peak + 1] + 1e-12)
        fraction = float(np.clip(fraction, -0.5, 0.5))
        refined.append(max(0.0, (peak + fraction) * 1000.0 / FPS - ONSET_LATENCY * 1000.0))
    beat_times = np.asarray(refined, dtype=np.float64)
    intervals = np.diff(beat_times)
    if len(intervals) < 20 or np.any(intervals <= 0):
        return None, None, None
    typical = float(np.median(intervals))
    if typical <= 0 or np.any((intervals < typical * 0.55) | (intervals > typical * 1.65)):
        return None, None, None

    smoothed = gaussian_filter1d(intervals, sigma=2.0, mode="nearest")
    variation = float((np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) / typical)
    if variation < 0.05:
        return None, None, None

    first_index = int(round((beat_times[0] - offset_ms) / (60000.0 / bpm)))
    beat_indices = first_index + np.arange(len(beat_times), dtype=np.int32)
    downbeat_ids = [i for i, beat_index in enumerate(beat_indices[:-1])
                    if beat_index % 4 == 0 and beat_times[i] >= 0]
    first_downbeat = downbeat_ids[0] if downbeat_ids else None
    first_time = float(beat_times[first_downbeat]) if first_downbeat is not None else float(offset_ms)
    first_interval_index = first_downbeat or 0
    points: list[tuple[float, float]] = [
        (first_time, float(np.mean(smoothed[first_interval_index:min(first_interval_index + 4, len(smoothed))]))),
    ]
    for i, beat_index in enumerate(beat_indices[:-1]):
        if beat_index % 4 != 0 or beat_times[i] <= first_time + 1.0:
            continue
        local = smoothed[i:min(i + 4, len(smoothed))]
        if len(local):
            points.append((float(beat_times[i]), float(np.mean(local))))
    # A single global point cannot represent a meaningful tempo change.
    if len(points) < 2:
        return None, None, None
    return beat_times, beat_indices, points


def track_variable_timing(
    features: AudioFeatures, bpm: float, offset_ms: float,
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[float, float]] | None]:
    """Public wrapper for fitting a beat path after a user supplies timing overrides."""
    return _track_variable_beats(features, bpm, offset_ms)


def fit_tempo_weights(candidate_sets: list[list[TempoCandidate]], true_bpms: list[float],
                      l2: float = 0.01, steps: int = 3000, lr: float = 0.1) -> np.ndarray:
    """Fit the softmax tempo picker: maximise the probability of the candidate(s) that
    match the true BPM (within 0.5%)."""
    data = []
    for cands, bpm in zip(candidate_sets, true_bpms):
        y = np.array([abs(c.bpm - bpm) / bpm < 0.005 for c in cands], dtype=float)
        if y.sum() > 0:
            data.append((np.stack([c.features for c in cands]), y / y.sum()))
    w = np.zeros(len(FEATURE_NAMES))
    for _ in range(steps):
        grad = l2 * w
        for x, y in data:
            z = x @ w
            p = np.exp(z - z.max())
            grad += x.T @ (p / p.sum() - y) / len(data)
        w -= lr * grad
    return w


def estimate_timing(features: AudioFeatures, weights: np.ndarray = TEMPO_WEIGHTS,
                    latency: float = ONSET_LATENCY) -> TimingEstimate:
    """Estimate a single constant BPM and the offset of the first downbeat."""
    return timing_from_candidates(features, tempo_candidates(features), weights, latency)


def timing_from_candidates(features: AudioFeatures, candidates: list[TempoCandidate],
                           weights: np.ndarray = TEMPO_WEIGHTS,
                           latency: float = ONSET_LATENCY) -> TimingEstimate:
    onset = _smooth(features.onset)
    bass = _smooth(features.bass_onset)
    duration = features.duration
    if not candidates:
        return TimingEstimate(bpm=120.0, offset_ms=0.0)
    base_scores = np.array([float(c.features @ weights) for c in candidates])
    best_index = int(np.argmax(base_scores))
    if len(candidates) > 1:
        ordered = np.sort(base_scores)
        if ordered[-1] - ordered[-2] < 0.15:
            support = _multiband_support(features, candidates)
            best_index = int(np.argmax(base_scores + 0.15 * support))
    best = candidates[best_index]
    bpm, phase, score = best.bpm, best.phase, best.score

    # Mappers prefer round BPMs; take one if it fits (almost) as well.
    for integer in range(int(np.ceil(bpm * 0.99)), int(np.floor(bpm * 1.01)) + 1):
        s, ph = _best_phase(onset, duration, float(integer))
        if s >= 0.97 * score and (bpm != round(bpm) or s > score):
            bpm, phase, score = float(integer), ph, s
    period = 60.0 / bpm

    # Hi-hats can make the off-beats as strong as the beats; the bass decides.
    options = np.array([phase, phase + period / 2]) % period
    strength = grid_score(onset, period, options, duration) + grid_score(bass, period, options, duration)
    phase = float(options[np.argmax(strength)])
    offset = _pick_downbeat(bass, period, phase, duration, latency)
    swing_ratio, swing_confidence = estimate_swing(features, bpm, offset)
    beat_times, beat_indices, tempo_points = track_variable_timing(features, bpm, offset)
    return TimingEstimate(bpm=round(bpm, 3), offset_ms=offset,
                          swing_ratio=swing_ratio, swing_confidence=swing_confidence,
                          beat_times_ms=beat_times, beat_indices=beat_indices,
                          tempo_points=tempo_points)


def fit_offset(features: AudioFeatures, bpm: float, latency: float = ONSET_LATENCY) -> float:
    """Best downbeat offset (ms) for a known BPM."""
    onset = _smooth(features.onset)
    bass = _smooth(features.bass_onset)
    period = 60.0 / bpm
    phases = np.arange(0.0, period, 0.001)
    scores = grid_score(onset, period, phases, features.duration)
    scores += grid_score(bass, period, phases, features.duration)
    phase = float(phases[np.argmax(scores)])
    return _pick_downbeat(bass, period, phase, features.duration, latency)


def _pick_downbeat(onset: np.ndarray, period: float, phase: float, duration: float,
                   latency: float) -> float:
    """Choose which beat of the 4/4 bar is the downbeat (strongest in ``onset``), then
    correct for onset latency; returns the offset in ms."""
    measure = 4 * period
    scores = [grid_score(onset, measure, np.array([phase + j * period]), duration)[0] for j in range(4)]
    downbeat = phase + int(np.argmax(scores)) * period - latency
    return float(round((downbeat % measure) * 1000.0))
