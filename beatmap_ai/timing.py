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
    best = pick_tempo(candidates, weights)
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
    return TimingEstimate(bpm=round(bpm, 3), offset_ms=offset)


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
