"""CPU-only vocal activity, syllable, and sustained-vowel analysis.

This is a small NumPy/librosa implementation of the useful parts of the local
VocalRhythmEngine reference. It deliberately has no dependency on PyTorch or on the
beatmap generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import librosa
import numpy as np

from .audio import FPS, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE

VAD_ENERGY_FLOOR = 0.005
VAD_FLATNESS_VOICED = 0.35
VAD_FLATNESS_UNVOICED = 0.65
VAD_HARMONIC_RATIO = 0.55
VAD_MIN_VOICED_RATIO = 0.40
VAD_SMOOTH_FRAMES = 5
VAD_MIN_SEGMENT_FRAMES = 4

PITCH_MIN_HZ = 80.0
PITCH_MAX_HZ = 1200.0
PITCH_STABLE_MAX_CENTS = 80.0

SYLLABLE_MFCC_DELTA_MULT = 1.5
SYLLABLE_NOVELTY_MULT = 1.3
SYLLABLE_MIN_DURATION_MS = 60.0
SYLLABLE_MAX_GAP_MS = 300.0
PHRASE_BREAK_MS = 400.0

CONSONANT_FLATNESS_MIN = 0.4
PLOSIVE_RISE_FACTOR = 3.0
FRICATIVE_FLATNESS_MIN = 0.5
FRICATIVE_MIN_FRAMES = 3

HOLD_MIN_VOWEL_MS = 150.0
VIR_VOCAL_THRESHOLD = 1.1
VIR_INSTRUMENTAL_THRESHOLD = 0.8

VOCAL_FREQ_MIN = 2000.0
VOCAL_FREQ_MAX = 6000.0
HPSS_KERNEL_SIZE = (17, 17)


@dataclass(slots=True)
class VocalAnalysis:
    """Per-frame vocal signals and event-level results for one song."""

    frame_rate: float
    vocal_energy: np.ndarray
    other_energy: np.ndarray
    harmonic_energy: np.ndarray
    percussive_energy: np.ndarray
    flatness: np.ndarray
    spectral_novelty: np.ndarray
    mfcc: np.ndarray
    mfcc_delta_flux: np.ndarray
    labels: np.ndarray  # 0=silent, 1=voiced, 2=unvoiced
    onset_function: np.ndarray
    percussive_onsets_ms: np.ndarray
    syllables: list[dict]
    slider_candidates: list[dict]
    pitch_hz: np.ndarray
    sections: list[dict]
    runtime_seconds: float

    @property
    def n_frames(self) -> int:
        return len(self.labels)

    @property
    def frame_duration_ms(self) -> float:
        return 1000.0 / self.frame_rate


def analyze_vocals(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    *,
    section_ms: float = 8000.0,
    quarter_ms: float | None = None,
    estimate_pitch: bool = True,
) -> VocalAnalysis:
    """Analyze mono audio on the same ~86 FPS grid as :mod:`beatmap_ai.audio`.

    ``section_ms`` can be set to four or eight beatmap bars by the caller. ``quarter_ms``
    optionally snaps a suggested hold duration to a sixteenth-note grid. The returned
    ``runtime_seconds`` measures this call, including pitch estimation when enabled.
    """
    started = perf_counter()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != SAMPLE_RATE and len(y):
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    sr = SAMPLE_RATE
    if len(y) == 0:
        return _empty_analysis(section_ms, perf_counter() - started)

    # One STFT supplies the matched frame grid, spectral shape, and 2–6 kHz band.
    spectrum = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
    magnitude = np.abs(spectrum).astype(np.float32, copy=False)
    power = magnitude * magnitude
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    vocal_bins = (frequencies >= VOCAL_FREQ_MIN) & (frequencies <= VOCAL_FREQ_MAX)
    if not np.any(vocal_bins):
        vocal_bins = frequencies >= min(VOCAL_FREQ_MIN, sr / 2.0)
    vocal_energy = power[vocal_bins].sum(axis=0, dtype=np.float64).astype(np.float32)
    other_energy = power[~vocal_bins].sum(axis=0, dtype=np.float64).astype(np.float32)

    mel_power = librosa.feature.melspectrogram(
        S=power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_magnitude = np.sqrt(np.maximum(mel_power, 0.0))
    harmonic, percussive = librosa.decompose.hpss(
        mel_magnitude.astype(np.complex64), kernel_size=HPSS_KERNEL_SIZE
    )
    harmonic_energy = (np.abs(harmonic) ** 2).sum(axis=0).astype(np.float32)
    percussive_energy = (np.abs(percussive) ** 2).sum(axis=0).astype(np.float32)
    flatness = _spectral_flatness(mel_magnitude)
    spectral_novelty = _spectral_novelty(mel_magnitude)
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_power + 1e-12), sr=sr, n_mfcc=13
    ).T.astype(np.float32)
    delta_flux = _mfcc_delta_flux(mfcc)

    labels = _detect_voice_activity(vocal_energy, harmonic_energy, percussive_energy, flatness)
    onset_function = _syllable_onset_function(
        vocal_energy, delta_flux, spectral_novelty, labels
    )
    syllables = _segment_syllables(
        vocal_energy, flatness, labels, onset_function, delta_flux, spectral_novelty,
        frame_duration_ms=1000.0 * HOP_LENGTH / sr,
    )
    pitch_hz = _estimate_pitch(y, labels) if estimate_pitch else np.full(len(labels), np.nan)
    slider_candidates = _held_vowel_candidates(
        syllables, labels, pitch_hz, mfcc,
        quarter_ms=quarter_ms,
    )
    result = VocalAnalysis(
        frame_rate=FPS,
        vocal_energy=vocal_energy,
        other_energy=other_energy,
        harmonic_energy=harmonic_energy,
        percussive_energy=percussive_energy,
        flatness=flatness,
        spectral_novelty=spectral_novelty,
        mfcc=mfcc,
        mfcc_delta_flux=delta_flux,
        labels=labels,
        onset_function=onset_function,
        percussive_onsets_ms=_percussive_onsets(spectral_novelty),
        syllables=syllables,
        slider_candidates=slider_candidates,
        pitch_hz=pitch_hz,
        sections=[],
        runtime_seconds=0.0,
    )
    result.sections = summarize_sections(result, section_ms=section_ms)
    result.runtime_seconds = perf_counter() - started
    return result


def analyze_mel(
    mel: np.ndarray,
    *,
    sr: int = SAMPLE_RATE,
    section_ms: float = 8000.0,
) -> VocalAnalysis:
    """Approximate the same analysis from the project's cached log-mel array.

    Compact dataset folders can contain only ``mel.npy``. This fallback treats the
    existing 80-bin log-mel features as a coarse spectrum; it cannot estimate F0, so
    ``slider_candidates`` is empty. Raw waveform analysis should be preferred when it
    is available.
    """
    started = perf_counter()
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError("mel must have shape (n_mels, frames)")
    if mel.shape[1] == 0:
        return _empty_analysis(section_ms, perf_counter() - started)

    # audio.compute_features stores (dB + 40) / 40, with dB clipped to [-80, 0].
    mel_power = np.power(10.0, np.clip(mel * 40.0 - 40.0, -80.0, 0.0) / 10.0)
    mel_freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0.0, fmax=sr / 2.0)
    vocal_bins = (mel_freqs >= VOCAL_FREQ_MIN) & (mel_freqs <= min(VOCAL_FREQ_MAX, sr / 2.0))
    if not np.any(vocal_bins):
        vocal_bins = mel_freqs >= min(VOCAL_FREQ_MIN, sr / 2.0)
    band = mel_power[vocal_bins]
    other = mel_power[~vocal_bins]
    full_magnitude = np.sqrt(np.maximum(mel_power, 0.0))
    vocal_energy = band.sum(axis=0).astype(np.float32)
    other_energy = other.sum(axis=0).astype(np.float32) if len(other) else np.zeros(mel.shape[1], np.float32)

    # Median-filter separation is approximate on mel bins, but preserves full-spectrum
    # harmonic/percussive evidence when the source waveform is unavailable.
    harmonic, percussive = librosa.decompose.hpss(
        full_magnitude.astype(np.complex64), kernel_size=HPSS_KERNEL_SIZE
    )
    harmonic_energy = (np.abs(harmonic) ** 2).sum(axis=0).astype(np.float32)
    percussive_energy = (np.abs(percussive) ** 2).sum(axis=0).astype(np.float32)
    flatness = _spectral_flatness(full_magnitude)
    spectral_novelty = _spectral_novelty(full_magnitude)
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_power + 1e-12), sr=sr, n_mfcc=13
    ).T.astype(np.float32)
    delta_flux = _mfcc_delta_flux(mfcc)
    labels = _detect_voice_activity(vocal_energy, harmonic_energy, percussive_energy, flatness)
    onset_function = _syllable_onset_function(vocal_energy, delta_flux, spectral_novelty, labels)
    syllables = _segment_syllables(
        vocal_energy, flatness, labels, onset_function, delta_flux, spectral_novelty,
        frame_duration_ms=1000.0 / FPS,
    )
    result = VocalAnalysis(
        frame_rate=FPS,
        vocal_energy=vocal_energy,
        other_energy=other_energy,
        harmonic_energy=harmonic_energy,
        percussive_energy=percussive_energy,
        flatness=flatness,
        spectral_novelty=spectral_novelty,
        mfcc=mfcc,
        mfcc_delta_flux=delta_flux,
        labels=labels,
        onset_function=onset_function,
        percussive_onsets_ms=_percussive_onsets(spectral_novelty),
        syllables=syllables,
        slider_candidates=[],
        pitch_hz=np.full(len(labels), np.nan),
        sections=[],
        runtime_seconds=0.0,
    )
    result.sections = summarize_sections(result, section_ms=section_ms)
    result.runtime_seconds = perf_counter() - started
    return result


def summarize_sections(analysis: VocalAnalysis, *, section_ms: float) -> list[dict]:
    """Summarize fixed-width song sections using the reference engine's VIR limits."""
    if section_ms <= 0:
        raise ValueError("section_ms must be positive")
    width = max(1, int(round(section_ms * analysis.frame_rate / 1000.0)))
    rows: list[dict] = []
    for start in range(0, analysis.n_frames, width):
        end = min(start + width, analysis.n_frames)
        voice = float(analysis.vocal_energy[start:end].sum())
        other = float(analysis.other_energy[start:end].sum())
        vir = voice / max(other, 1e-12)
        active = analysis.labels[start:end]
        voiced_ratio = float(np.mean(active == 1)) if len(active) else 0.0
        unvoiced_ratio = float(np.mean(active == 2)) if len(active) else 0.0
        vocal_share = voice / max(voice + other, 1e-12)
        if vir > VIR_VOCAL_THRESHOLD and voiced_ratio >= VAD_MIN_VOICED_RATIO:
            section_type = "vocal_focus"
        elif vir < VIR_INSTRUMENTAL_THRESHOLD or voiced_ratio < 0.10:
            section_type = "instrumental_focus"
        else:
            section_type = "balanced"
        rows.append({
            "start_ms": start * 1000.0 / analysis.frame_rate,
            "end_ms": end * 1000.0 / analysis.frame_rate,
            "vir": vir,
            "vocal_share": vocal_share,
            "voiced_ratio": voiced_ratio,
            "unvoiced_ratio": unvoiced_ratio,
            "type": section_type,
        })

    # Copy the reference's persistence check so a single borderline window cannot
    # flip the song map's foreground label.
    if len(rows) > 2:
        raw = [row["type"] for row in rows]
        for i in range(1, len(rows)):
            if raw[i] != rows[i - 1]["type"]:
                persistent = i + 1 < len(rows) and raw[i + 1] == raw[i]
                if abs(rows[i]["vir"] - rows[i - 1]["vir"]) <= 0.5 and not persistent:
                    rows[i]["type"] = rows[i - 1]["type"]
    return rows


def _empty_analysis(section_ms: float, runtime: float) -> VocalAnalysis:
    empty = np.zeros(0, dtype=np.float32)
    result = VocalAnalysis(
        frame_rate=FPS, vocal_energy=empty, other_energy=empty,
        harmonic_energy=empty, percussive_energy=empty, flatness=empty,
        spectral_novelty=empty, mfcc=np.zeros((0, 13), np.float32),
        mfcc_delta_flux=empty, labels=np.zeros(0, np.uint8), onset_function=empty,
        percussive_onsets_ms=empty, syllables=[], slider_candidates=[], pitch_hz=empty,
        sections=[], runtime_seconds=runtime,
    )
    result.sections = summarize_sections(result, section_ms=section_ms)
    return result


def _spectral_flatness(magnitude: np.ndarray) -> np.ndarray:
    if magnitude.size == 0:
        return np.zeros(magnitude.shape[1] if magnitude.ndim == 2 else 0, np.float32)
    values = np.asarray(magnitude, dtype=np.float64)
    mean = values.mean(axis=0)
    result = np.zeros(values.shape[1], dtype=np.float32)
    active = mean > 1e-12
    if np.any(active):
        log_mean = np.log(np.maximum(values[:, active], 1e-12)).mean(axis=0)
        result[active] = np.clip(np.exp(log_mean) / mean[active], 0.0, 1.0)
    return result


def _spectral_novelty(magnitude: np.ndarray) -> np.ndarray:
    if magnitude.shape[1] == 0:
        return np.zeros(0, np.float32)
    norm = magnitude / np.maximum(magnitude.sum(axis=0, keepdims=True), 1e-12)
    flux = np.zeros(magnitude.shape[1], dtype=np.float32)
    if magnitude.shape[1] > 1:
        flux[1:] = np.maximum(norm[:, 1:] - norm[:, :-1], 0.0).sum(axis=0)
    return _smooth(flux, 3)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) <= 1:
        return np.asarray(values, dtype=np.float32)
    left = width // 2
    right = width - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width, dtype=np.float32) / width, mode="valid").astype(np.float32)


def _mfcc_delta_flux(mfcc: np.ndarray) -> np.ndarray:
    n_frames, n_coeff = mfcc.shape
    if n_frames == 0:
        return np.zeros(0, np.float32)
    padded = np.pad(mfcc, ((2, 2), (0, 0)), mode="edge")
    deltas = np.empty_like(mfcc, dtype=np.float32)
    for t in range(n_frames):
        # Regression delta from the JavaScript reference (N=2; norm=10).
        deltas[t] = (padded[t + 3] - padded[t + 1] + 2.0 * (padded[t + 4] - padded[t])) / 10.0
    flux = np.zeros(n_frames, np.float32)
    if n_frames > 1:
        flux[1:] = np.linalg.norm(np.diff(deltas, axis=0), axis=1)
    return flux


def _detect_voice_activity(
    energy: np.ndarray,
    harmonic_energy: np.ndarray,
    percussive_energy: np.ndarray,
    flatness: np.ndarray,
) -> np.ndarray:
    n = len(energy)
    if n == 0:
        return np.zeros(0, np.uint8)
    maximum = max(float(np.max(energy)), 1e-10)
    norm = energy / maximum
    harmonic_ratio = harmonic_energy / np.maximum(harmonic_energy + percussive_energy, 1e-12)
    labels = np.zeros(n, np.uint8)
    active = norm >= VAD_ENERGY_FLOOR
    harmonic = harmonic_ratio > VAD_HARMONIC_RATIO
    high_flat = flatness > VAD_FLATNESS_UNVOICED
    voiced = active & harmonic & ~high_flat
    unvoiced = active & ~voiced & (high_flat | (~harmonic & (norm > VAD_ENERGY_FLOOR * 2.0)))
    ambiguous_voiced = active & ~voiced & ~unvoiced & (norm > VAD_ENERGY_FLOOR * 5.0)
    labels[voiced | ambiguous_voiced] = 1
    labels[unvoiced] = 2

    # Modal median smoothing, with voiced then unvoiced winning ties as in the source.
    half = VAD_SMOOTH_FRAMES // 2
    padded = np.pad(labels, (half, half), mode="edge")
    smooth = np.empty_like(labels)
    for i in range(n):
        counts = np.bincount(padded[i:i + VAD_SMOOTH_FRAMES], minlength=3)
        if counts[1] >= counts[0] and counts[1] >= counts[2]:
            smooth[i] = 1
        elif counts[2] >= counts[0]:
            smooth[i] = 2
        else:
            smooth[i] = 0
    return smooth


def _local_mean_std(values: np.ndarray, half_window: int) -> tuple[np.ndarray, np.ndarray]:
    # Cumulative sums keep adaptive thresholds O(frames), including long songs.
    values64 = np.asarray(values, dtype=np.float64)
    sums = np.concatenate(([0.0], np.cumsum(values64)))
    squares = np.concatenate(([0.0], np.cumsum(values64 * values64)))
    idx = np.arange(len(values64))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(values64), idx + half_window + 1)
    count = np.maximum(hi - lo, 1)
    mean = (sums[hi] - sums[lo]) / count
    variance = np.maximum((squares[hi] - squares[lo]) / count - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _syllable_onset_function(
    energy: np.ndarray,
    delta_flux: np.ndarray,
    novelty: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    n = len(energy)
    onset = np.zeros(n, np.float32)
    if n < 3:
        return onset
    energy_scale = max(float(np.max(energy)), 1e-10)
    delta_scale = max(float(np.max(delta_flux)), 1e-10)
    novelty_scale = max(float(np.max(novelty)), 1e-10)
    normalized_energy = energy / energy_scale
    dnorm = delta_flux / delta_scale
    nnorm = novelty / novelty_scale
    for i in range(1, n):
        if labels[i] == 0 and labels[i - 1] == 0:
            continue
        rise = max(0.0, float(normalized_energy[i] - normalized_energy[i - 1]))
        onset[i] = 2.0 * rise + 3.0 * float(dnorm[i]) + 2.0 * float(nnorm[i])

    half_window = max(1, int(np.ceil(500.0 * FPS / 1000.0)))
    mean, std = _local_mean_std(onset, half_window)
    threshold = mean + SYLLABLE_MFCC_DELTA_MULT * std
    # Keep the two source thresholds meaningful: a candidate also needs a local
    # timbre/novelty event, unless its normalized energy jump is independently strong.
    dmean, dstd = _local_mean_std(dnorm, half_window)
    nmean, nstd = _local_mean_std(nnorm, half_window)
    delta_gate = dnorm > dmean + SYLLABLE_MFCC_DELTA_MULT * dstd
    novelty_gate = nnorm > nmean + SYLLABLE_NOVELTY_MULT * nstd
    energy_rise = np.maximum(np.diff(normalized_energy, prepend=normalized_energy[0]), 0.0)
    energy_gate = energy_rise >= 0.12
    non_falling = normalized_energy >= np.roll(normalized_energy, 1) * 0.98
    if n:
        non_falling[0] = True
    activity_restart = (labels > 0) & (np.r_[0, labels[:-1]] == 0)
    min_spacing = max(1, int(np.ceil(SYLLABLE_MIN_DURATION_MS * FPS / 1000.0)))
    peaks = _pick_peaks(onset, threshold, min_spacing)
    for i in peaks:
        if energy_gate[i] or activity_restart[i] or (
            non_falling[i] and (delta_gate[i] or novelty_gate[i])
        ):
            continue
        onset[i] = 0.0
    return onset


def _pick_peaks(signal: np.ndarray, threshold: np.ndarray, min_spacing: int) -> list[int]:
    candidates = [
        i for i in range(1, len(signal) - 1)
        if signal[i] > threshold[i] and signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
    ]
    # Select the strongest event in each minimum-spacing neighborhood.
    kept: list[int] = []
    for i in candidates:
        if not kept or i - kept[-1] >= min_spacing:
            kept.append(i)
        elif signal[i] > signal[kept[-1]]:
            kept[-1] = i
    return kept


def _segment_syllables(
    energy: np.ndarray,
    flatness: np.ndarray,
    labels: np.ndarray,
    onset: np.ndarray,
    delta_flux: np.ndarray,
    novelty: np.ndarray,
    *,
    frame_duration_ms: float,
) -> list[dict]:
    min_spacing = max(1, int(np.ceil(SYLLABLE_MIN_DURATION_MS / frame_duration_ms)))
    threshold_window = max(1, int(np.ceil(500.0 / frame_duration_ms)))
    mean, std = _local_mean_std(onset, threshold_window)
    peaks = _pick_peaks(onset, mean + SYLLABLE_MFCC_DELTA_MULT * std, min_spacing)
    maximum = max(float(np.max(energy)), 1e-10)
    normalized = energy / maximum
    dnorm = delta_flux / max(float(np.max(delta_flux)), 1e-10)
    nnorm = novelty / max(float(np.max(novelty)), 1e-10)
    dmean, dstd = _local_mean_std(dnorm, threshold_window)
    nmean, nstd = _local_mean_std(nnorm, threshold_window)
    delta_gate = dnorm > dmean + SYLLABLE_MFCC_DELTA_MULT * dstd
    novelty_gate = nnorm > nmean + SYLLABLE_NOVELTY_MULT * nstd
    energy_rise = np.maximum(np.diff(normalized, prepend=normalized[0]), 0.0)
    energy_gate = energy_rise >= 0.12
    non_falling = normalized >= np.roll(normalized, 1) * 0.98
    if len(non_falling):
        non_falling[0] = True
    activity_restart = (labels > 0) & (np.r_[0, labels[:-1]] == 0)
    evidence = energy_gate | activity_restart | (non_falling & (delta_gate | novelty_gate))
    accepted = [i for i in peaks if labels[i] > 0 and onset[i] > 0.0 and evidence[i]]
    syllables: list[dict] = []
    n = len(energy)
    for index, frame in enumerate(accepted):
        next_start = accepted[index + 1] if index + 1 < len(accepted) else n
        end = frame
        peak_energy = float(energy[frame])
        below = 0
        for f in range(frame + 1, next_start):
            if labels[f] == 0 or energy[f] < peak_energy * 0.15:
                below += 1
                if below >= 3:
                    end = f - 2
                    break
            else:
                below = 0
                end = f
        if end < frame:
            end = frame
        kind = _classify_onset(frame, energy, flatness, labels, maximum)
        start_ms = frame * frame_duration_ms
        end_ms = end * frame_duration_ms
        syllables.append({
            "frame": int(frame),
            "end_frame": int(end),
            "time_ms": float(start_ms),
            "end_ms": float(end_ms),
            "duration_ms": float(max(0.0, end_ms - start_ms)),
            "type": kind,
            "onset_strength": float(onset[frame]),
        })
    return syllables


def _classify_onset(
    frame: int, energy: np.ndarray, flatness: np.ndarray, labels: np.ndarray, maximum: float
) -> str:
    norm = float(energy[frame] / maximum)
    previous = float(energy[frame - 1] / maximum) if frame else 0.0
    rise = norm / max(previous, 1e-6)
    shape = float(flatness[frame])
    if rise > PLOSIVE_RISE_FACTOR and shape > CONSONANT_FLATNESS_MIN:
        return "PLOSIVE"
    if shape > FRICATIVE_FLATNESS_MIN:
        stop = min(frame + 8, len(flatness))
        if int(np.count_nonzero(flatness[frame:stop] > FRICATIVE_FLATNESS_MIN)) >= FRICATIVE_MIN_FRAMES:
            return "FRICATIVE"
    if labels[frame] == 1 and shape < VAD_FLATNESS_VOICED:
        return "NASAL" if norm < 0.15 else "VOWEL"
    if labels[frame] == 2:
        return "FRICATIVE" if shape > CONSONANT_FLATNESS_MIN else "MIXED"
    return "MIXED"


def _estimate_pitch(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if len(y) < N_FFT or len(labels) == 0:
        return np.full(len(labels), np.nan, np.float32)
    pitch = librosa.yin(
        y, fmin=PITCH_MIN_HZ, fmax=min(PITCH_MAX_HZ, SAMPLE_RATE / 2.0),
        sr=SAMPLE_RATE, frame_length=N_FFT, hop_length=HOP_LENGTH,
    )
    pitch = np.asarray(pitch, dtype=np.float32)
    if len(pitch) < len(labels):
        pitch = np.pad(pitch, (0, len(labels) - len(pitch)), constant_values=np.nan)
    else:
        pitch = pitch[:len(labels)]
    pitch[labels != 1] = np.nan
    return pitch


def _held_vowel_candidates(
    syllables: list[dict],
    labels: np.ndarray,
    pitch_hz: np.ndarray,
    mfcc: np.ndarray,
    *,
    quarter_ms: float | None,
) -> list[dict]:
    candidates: list[dict] = []
    for i, syllable in enumerate(syllables):
        duration = float(syllable["duration_ms"])
        if syllable["type"] == "BREATH" or duration < HOLD_MIN_VOWEL_MS:
            continue
        start, end = int(syllable["frame"]), int(syllable["end_frame"])
        segment_pitch = pitch_hz[start:end + 1]
        good_pitch = segment_pitch[np.isfinite(segment_pitch) & (segment_pitch >= PITCH_MIN_HZ)]
        deviation_cents = float("inf")
        pitch_stable = False
        if len(good_pitch) >= 3:
            midi = 69.0 + 12.0 * np.log2(good_pitch / 440.0)
            median_midi = float(np.median(midi))
            deviation_cents = float(100.0 * np.median(np.abs(midi - median_midi)))
            pitch_stable = deviation_cents <= PITCH_STABLE_MAX_CENTS
        phrase_final = i + 1 == len(syllables) or (
            float(syllables[i + 1]["time_ms"]) - float(syllable["end_ms"]) >= PHRASE_BREAK_MS
        )
        active = labels[start:end + 1]
        voiced_ratio = float(np.mean(active == 1)) if len(active) else 0.0
        confidence = min(0.3, duration / 2000.0)
        reasons = [f"duration {duration:.0f} ms"]
        if pitch_stable:
            confidence += 0.25
            reasons.append(f"pitch stable (MAD {deviation_cents:.0f} cents)")
        if phrase_final:
            confidence += 0.15
            reasons.append("phrase-final")
        if voiced_ratio > 0.85:
            confidence += 0.15
            reasons.append(f"voiced {voiced_ratio:.0%}")
        if end > start and mfcc.shape[0] > end:
            timbre_std = float(np.mean(np.std(mfcc[start:end + 1], axis=0)))
            if timbre_std < 4.0:
                confidence += 0.15
                reasons.append("timbre stable")
        suggested = duration if syllable["type"] == "VOWEL" else duration * 0.8
        if quarter_ms and quarter_ms > 0:
            grid = quarter_ms / 4.0
            suggested = round(suggested / grid) * grid
        if syllable["type"] == "PLOSIVE":
            confidence *= 0.3
            reasons.append("plosive penalty")
        elif syllable["type"] == "FRICATIVE":
            confidence *= 0.5
            reasons.append("fricative penalty")
        if confidence >= 0.3 and suggested >= HOLD_MIN_VOWEL_MS:
            candidates.append({
                "time_ms": float(syllable["time_ms"]),
                "duration_ms": float(suggested),
                "confidence": float(min(1.0, confidence)),
                "pitch_stable": pitch_stable,
                "pitch_deviation_cents": deviation_cents,
                "reasons": reasons,
            })
    return candidates


def _percussive_onsets(full_spectral_flux: np.ndarray) -> np.ndarray:
    """Peak-pick whole-spectrum spectral flux as a percussion/transient proxy."""
    if len(full_spectral_flux) < 3 or float(np.max(full_spectral_flux)) <= 1e-12:
        return np.zeros(0, np.float32)
    mean, std = _local_mean_std(full_spectral_flux, max(1, int(0.2 * FPS)))
    peaks = _pick_peaks(
        full_spectral_flux, mean + 1.2 * std,
        max(1, int(np.ceil(80.0 * FPS / 1000.0))),
    )
    return (np.asarray(peaks, dtype=np.float32) * (1000.0 / FPS))
