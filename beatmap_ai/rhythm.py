"""Rhythm generation: deciding *when* hit objects happen and whether they are circles,
sliders or spinners.

Every candidate time is a tick on the beat grid (``divisor`` ticks per beat). Each tick
gets a score -- from onset strength (heuristic mode) or from the neural network's note
probability (model mode) -- and ticks are then picked greedily under spacing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import maximum_filter1d

from .audio import FPS, AudioFeatures, sample_peak
from .difficulty import DifficultyPreset, spacing_spread, stars_for_density
from .timing import ONSET_LATENCY, TimingEstimate

ACTIVE_RMS = 0.08  # loudness below which the music counts as silent
ONSET_LAG_MS = ONSET_LATENCY * 1000.0  # onset-curve peaks come this long after the hit
RHYTHM_VARIETY = 0.5  # default for plan_objects, see variety_ratio


@dataclass
class TickGrid:
    times: np.ndarray  # ms
    index: np.ndarray  # tick number counted from the timing offset (can be negative)
    divisor: int
    score_times: np.ndarray | None = None  # straight grid for the model's trained beat phase
    beat_length_ms: np.ndarray | None = None  # local duration for each tick, when tempo varies

    @property
    def pos_in_beat(self) -> np.ndarray:
        return self.index % self.divisor

    @property
    def beat(self) -> np.ndarray:
        return self.index // self.divisor

    @property
    def is_downbeat(self) -> np.ndarray:
        return (self.pos_in_beat == 0) & (self.beat % 4 == 0)


@dataclass
class PlannedObject:
    time: float
    kind: str  # "circle", "slider" or "spinner"
    tick: int  # index into the TickGrid
    intensity: float  # 0..1, how strongly the music hits here
    end_time: float  # equals time for circles
    measure: int
    new_combo: bool = False
    beat_length: float | None = None
    spacing: float | None = None  # distance snap predicted by the model, if any
    slides: int = 1  # a slider's runs: 2+ goes back and forth


def make_tick_grid(timing: TimingEstimate, duration_ms: float, divisor: int) -> TickGrid:
    ratio = float(np.clip(timing.swing_ratio, 0.5, 2.0 / 3.0))
    use_swing = divisor >= 2 and divisor % 2 == 0 and timing.swing_confidence >= 0.25 and ratio > 0.515

    def swing_phase(phase: float) -> float:
        if phase <= 0.5:
            return phase * (ratio / 0.5)
        return ratio + (phase - 0.5) * ((1.0 - ratio) / 0.5)

    if timing.beat_times_ms is not None and timing.beat_indices is not None \
            and len(timing.beat_times_ms) >= 2 and len(timing.beat_indices) == len(timing.beat_times_ms):
        beat_times = timing.beat_times_ms.astype(np.float64).tolist()
        beat_indices = timing.beat_indices.astype(int).tolist()
        first_interval = beat_times[1] - beat_times[0]
        while beat_times[0] > 0 and first_interval > 0:
            beat_times.insert(0, beat_times[0] - first_interval)
            beat_indices.insert(0, beat_indices[0] - 1)
        while beat_times[-1] < duration_ms and len(beat_times) >= 2:
            last_interval = beat_times[-1] - beat_times[-2]
            if last_interval <= 0:
                break
            beat_times.append(beat_times[-1] + last_interval)
            beat_indices.append(beat_indices[-1] + 1)

        grid_times, score_times, indexes, beat_lengths = [], [], [], []
        for i, (start, end) in enumerate(zip(beat_times, beat_times[1:])):
            interval = end - start
            if interval <= 0:
                continue
            for subdivision in range(divisor):
                phase = subdivision / divisor
                straight = start + phase * interval
                if straight < 0 or straight > duration_ms - 100.0:
                    continue
                actual_phase = swing_phase(phase) if use_swing else phase
                score_times.append(straight)
                grid_times.append(start + actual_phase * interval)
                indexes.append(beat_indices[i] * divisor + subdivision)
                beat_lengths.append(interval)
        if grid_times:
            return TickGrid(
                times=np.asarray(grid_times, dtype=np.float64),
                index=np.asarray(indexes, dtype=np.int64),
                divisor=divisor,
                score_times=np.asarray(score_times, dtype=np.float64),
                beat_length_ms=np.asarray(beat_lengths, dtype=np.float32),
            )

    tick_ms = timing.beat_length / divisor
    first = -int(np.floor(timing.offset_ms / tick_ms))
    last = int(np.floor((duration_ms - 100.0 - timing.offset_ms) / tick_ms))
    index = np.arange(first, max(last, first) + 1)
    straight_times = timing.offset_ms + index * tick_ms
    times = straight_times.copy()
    if use_swing:
        beat_index = np.floor_divide(index, divisor)
        position = index - beat_index * divisor
        phase = position / divisor
        swung_phase = np.array([swing_phase(float(p)) for p in phase])
        times = timing.offset_ms + (beat_index + swung_phase) * timing.beat_length
    return TickGrid(times=times, index=index, divisor=divisor, score_times=straight_times)


def heuristic_scores(features: AudioFeatures, grid: TickGrid, lag_ms: float = ONSET_LAG_MS,
                     radius: int = 2) -> np.ndarray:
    """Score ticks by onset strength, favouring strong metrical positions."""
    onset = sample_peak(features.onset, grid.times + lag_ms, radius=radius)
    rms = sample_peak(features.rms, grid.times, radius=0)
    pos = grid.pos_in_beat
    weight = np.full(len(pos), 0.9, dtype=np.float32)
    weight[pos * 2 == grid.divisor] = 1.05
    weight[pos == 0] = 1.2
    weight[grid.is_downbeat] = 1.3
    score = onset * weight + 0.15 * rms * (pos == 0)
    # Blend in loudness-independent scores so quiet sections still get notes.
    local_max = maximum_filter1d(score, size=8 * grid.divisor * 4 + 1)
    return 0.5 * score + 0.5 * score / (local_max + 1e-3)


def select_ticks(
    scores: np.ndarray,
    active: np.ndarray,
    preset: DifficultyPreset,
    threshold: float,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Pick ticks scoring at least ``threshold``, strongest first, while respecting the
    minimum gap between objects and the maximum stream length. Ticks already set in
    ``fixed`` are kept and constrain the new picks. Returns a boolean mask."""
    min_gap = max(int(round(preset.min_gap_beats * preset.divisor)), 1)
    taken = np.zeros(len(scores), dtype=bool) if fixed is None else fixed.copy()
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < threshold:
            break
        if not active[i] or taken[max(i - min_gap + 1, 0): i + min_gap].any():
            continue
        if _run_length(taken, i, min_gap) + 1 > preset.max_run:
            continue
        taken[i] = True
    return taken


def select_ticks_by_density(
    scores: np.ndarray,
    active: np.ndarray,
    grid: TickGrid,
    preset: DifficultyPreset,
    beat_length: float,
    loudness: np.ndarray,
    window_beats: int = 16,
) -> np.ndarray:
    """Choose ticks window by window so each stretch of the song gets roughly the
    preset's density, scaled up in loud sections and down in quiet ones."""
    window = window_beats * grid.divisor
    tick_seconds = beat_length / grid.divisor / 1000.0
    reference = np.median(loudness[active]) if active.any() else 1.0
    taken = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), window):
        span = slice(start, min(start + window, len(scores)))
        n_active = active[span].sum()
        if n_active == 0:
            continue
        factor = np.clip(0.6 + 0.4 * loudness[span][active[span]].mean() / (reference + 1e-6), 0.5, 1.4)
        count = max(int(round(preset.density * factor * n_active * tick_seconds)), 1)
        local = np.full(len(scores), -np.inf)
        local[span] = scores[span]
        taken = _select_count(local, active, preset, count, taken)
    return np.flatnonzero(taken)


def _select_count(scores, active, preset, count, fixed) -> np.ndarray:
    """Binary-search the score threshold that adds about ``count`` ticks to ``fixed``."""
    base = fixed.sum()
    finite = scores[np.isfinite(scores)]
    lo, hi = float(finite.min()) - 1e-6, float(finite.max()) + 1e-6
    best = fixed
    for _ in range(20):
        mid = (lo + hi) / 2
        taken = select_ticks(scores, active, preset, mid, fixed)
        if taken.sum() - base > count:
            lo = mid
        else:
            hi, best = mid, taken
    # Equal scores (e.g. a repeated drum loop) make the count jump in steps; take
    # whichever side of the step is closer to the target in ratio.
    over = select_ticks(scores, active, preset, lo, fixed)
    n_best, n_over = best.sum() - base, over.sum() - base
    if n_best == 0 or n_over / count < count / n_best:
        return over
    return best


def _run_length(taken: np.ndarray, i: int, step: int) -> int:
    """Number of taken ticks chained to ``i`` at exactly ``step`` spacing, on both sides."""
    n = 0
    for direction in (-1, 1):
        j = i + direction * step
        while 0 <= j < len(taken) and taken[j]:
            n += 1
            j += direction * step
    return n


def fill_gaps(ticks: np.ndarray, grid: TickGrid, active: np.ndarray, beats: int = 8) -> np.ndarray:
    """Add on-beat notes where the music plays but nothing was picked for ``beats`` beats,
    so quiet-but-active passages are not left empty."""
    span = beats * grid.divisor
    step = 2 * grid.divisor
    added = []
    bounds = np.concatenate([[-1], ticks, [len(grid.times)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= span:
            continue
        candidates = [
            t for t in range(a + step, b - step + 1)
            if grid.pos_in_beat[t] == 0 and active[t]
        ]
        last = a
        for t in candidates:
            if t - last >= step:
                added.append(t)
                last = t
    return np.array(sorted(set(ticks.tolist()) | set(added)), dtype=int)


def plan_objects(
    features: AudioFeatures,
    timing: TimingEstimate,
    preset: DifficultyPreset,
    rng: np.random.Generator,
    note_probs: np.ndarray | None = None,
    slider_probs: np.ndarray | None = None,
    threshold: float = 0.5,
    selection: str = "threshold",
    sustain_probs: np.ndarray | None = None,
    spacing: np.ndarray | None = None,
    stars: float | None = None,
    variety: float = RHYTHM_VARIETY,
) -> list[PlannedObject]:
    """Build the full rhythm for one difficulty.

    ``note_probs``/``slider_probs`` are per-frame probabilities from the neural model;
    when omitted, onset-strength heuristics are used instead. With the model,
    ``selection`` picks ticks above ``threshold`` ("threshold"), or picks the model's
    most likely ticks window by window to reach the preset's density ("density").
    ``sustain_probs`` (slider held) sets slider lengths and ``spacing`` (per-frame
    distance snap) is passed on to the placement, when the model provides them; its
    variation is widened to what mappers use at ``stars`` (estimated from the density
    when not given).
    """
    duration_ms = features.duration * 1000.0
    grid = make_tick_grid(timing, duration_ms, preset.divisor)
    if len(grid.times) == 0:
        return []
    rms = sample_peak(features.rms, grid.times, radius=2)
    active = rms > ACTIVE_RMS
    beat_length = timing.beat_length

    if note_probs is None:
        scores = heuristic_scores(features, grid)
        ticks = select_ticks_by_density(scores, active, grid, preset, beat_length, rms)
        ticks = fill_gaps(ticks, grid, active)
    else:
        model_times = grid.score_times if grid.score_times is not None else grid.times
        scores = sample_peak(note_probs, model_times, radius=1)
        if timing.swing_confidence >= 0.25 and timing.swing_ratio > 0.515:
            # Existing checkpoints were trained with straight beat phases. Keep
            # their predictions while also accepting peaks at the swung positions.
            scores = np.maximum(scores, sample_peak(note_probs, grid.times, radius=1))
        if selection == "density":
            ticks = select_ticks_by_density(scores, active, grid, preset, beat_length, rms)
        else:
            # The model hears when a stream fits, so Hard and Insane get a looser run limit
            # (without any, it strings too many long streams together).
            rules = preset if preset.divisor < 4 else replace(preset, max_run=max(preset.max_run, 9 if preset.max_run >= 5 else 5))
            ticks = np.flatnonzero(select_ticks(scores, np.ones_like(active), rules, threshold))
            if preset.divisor >= 4:
                ticks = shape_triples(ticks, scores, preset.divisor)
    if len(ticks) == 0:
        return []
    intensity = np.clip(sample_peak(features.onset, grid.times + ONSET_LAG_MS, radius=2), 0.0, 1.0)

    objects: list[PlannedObject] = []
    recovery = max(int(round(preset.slider_recovery_beats * preset.divisor)), 1)
    min_slider = max(int(round(preset.min_slider_beats * preset.divisor)), 1)
    max_slider = max(int(round(preset.max_slider_beats * preset.divisor)), min_slider)
    held = None
    if sustain_probs is not None:
        # The model hears how long a sound is held, so allow the longer sliders mappers use.
        held = sample_peak(sustain_probs, grid.score_times if grid.score_times is not None
                           else grid.times, radius=0) > 0.5
        max_slider = max(max_slider, 4 * preset.divisor)
        if slider_probs is not None:
            ticks = absorb_into_sliders(ticks, scores, sustain_probs, slider_probs, grid,
                                        max_slider, variety_ratio(variety))
    frame_of = np.rint(grid.times * FPS / 1000.0).astype(int)
    level = stars if stars is not None else stars_for_density(preset.density)
    # Kick sliders (a quarter beat, no held sound): about a tenth of the sliders at 3-6
    # stars and a third from 6. A laxer bar made nearly all sliders of a 4-star map kicks.
    kick_confidence = float(np.interp(level, [4.0, 6.0, 7.0], [0.92, 0.8, 0.65]))
    # Mappers use back-and-forth sliders over quick notes, most on easier maps (about a
    # fifth of the sliders below 3 stars, a tenth from 5). Easy maps have more music
    # that qualifies, so they need a lower chance to end up at the same share.
    repeat_chance = float(np.interp(level, [2.0, 3.5, 5.0], [0.35, 0.9, 0.5]))
    # A note only a quarter beat after a slider's end (release and hit again at once) fits
    # the rhythm but plays awkwardly; mappers do it after 7 % of the sliders at 3.5-5.5
    # stars and a fifth from 5.5. Otherwise they leave at least half a beat.
    half_beat = max(int(round(0.5 * preset.divisor)), recovery)
    quick_release = float(np.interp(level, [4.0, 5.5, 6.5], [0.08, 0.15, 0.25]))
    # Songs held all the way through (slowed songs, pads) made nearly every note a slider;
    # keep the share within what ranked maps of these stars use (their upper quartile).
    slider_cut = 0.5
    if slider_probs is not None and len(ticks):
        at_notes = sample_peak(slider_probs, grid.times[ticks], radius=1)
        most = float(np.interp(level, [2.0, 3.5, 5.0, 6.0], [0.68, 0.66, 0.55, 0.48]))
        slider_cut = max(0.5, float(np.quantile(at_notes, 1.0 - most)))
    for n, tick in enumerate(ticks):
        time = float(grid.times[tick])
        local_beat_length = (
            float(grid.beat_length_ms[tick]) if grid.beat_length_ms is not None else beat_length
        )
        obj = PlannedObject(time, "circle", int(tick), float(intensity[tick]), time,
                            int(grid.beat[tick] // 4), beat_length=local_beat_length)
        if spacing is not None:
            obj.spacing = float(spacing[min(max(frame_of[tick], 0), len(spacing) - 1)])
        if n + 1 < len(ticks):
            next_tick = ticks[n + 1]
            gap_beats = (next_tick - tick) / preset.divisor
            rest = recovery if rng.random() < quick_release else half_beat
            room = min(next_tick - tick - rest, max_slider)
            if held is not None:
                # Hold for as long as the model hears the sound sustained.
                run = 1
                while run < room and held[tick + run]:
                    run += 1
                length = max(run, min_slider) if room >= min_slider else 0
                # No held sound at all: a kick slider only when the model is sure enough;
                # mappers use them rarely below 4 stars and often from 6.
                if run == 1 and slider_probs is not None and sample_peak(
                        slider_probs, grid.times[tick:tick + 1], radius=1)[0] < kick_confidence:
                    length = 0
            else:
                length = room // min_slider * min_slider
            if (6 <= gap_beats <= 16 and active[tick:next_tick].mean() > 0.8
                    and (intensity[tick + 1:next_tick] > 0.5).mean() < 0.15):
                # A long gap in sustained music without strong hits: fill it with a spinner.
                lead = max(preset.min_gap_beats, 1.0) * local_beat_length
                start, end = time + lead, float(grid.times[next_tick]) - lead
                if end - start >= 2 * local_beat_length:
                    objects.append(obj)
                    objects.append(PlannedObject(start, "spinner", int(tick), 1.0, end, obj.measure,
                                                 beat_length=local_beat_length))
                    continue
            if length >= min_slider and _wants_slider(
                features, grid, scores, tick, next_tick, preset, rng, slider_probs, slider_cut
            ):
                obj.kind = "slider"
                obj.end_time = time + length * local_beat_length / preset.divisor
                obj.slides = repeat_slides(intensity, tick, length, preset.divisor)
                if obj.slides > 1 and rng.random() > repeat_chance:
                    obj.slides = 1
        objects.append(obj)
    if spacing is not None:
        spread_spacing(objects, spacing_spread(stars if stars is not None
                                               else stars_for_density(preset.density)))
    return objects


def variety_ratio(variety: float) -> float:
    """How much likelier "still holding" must be than "a new note" before a note inside a
    slider is dropped. ``variety`` 0 never drops notes (closest to the beat, the best
    rhythm F1), 0.5 needs 1.5x (costs ~0.004 F1), 1 needs 1x (most varied rhythm,
    ~0.017 F1 less on held-out songs)."""
    variety = float(np.clip(variety, 0.0, 1.0))
    return float("inf") if variety <= 0.0 else variety ** -0.585


TRIPLE_RATIO = 0.8


def shape_triples(ticks: np.ndarray, scores: np.ndarray, divisor: int,
                  ratio: float | None = None) -> np.ndarray:
    """Turn a 1/4 triple into a double when its last note is much less certain.

    Choosing every tick on its own appends the next beat to a double tap (it scores
    high because it is on the beat), so triples come out twice as often and doubles
    half as often as in ranked maps. A triple's last note is dropped when it scores
    below ``ratio`` times the weaker of the first two."""
    ratio = TRIPLE_RATIO if ratio is None else ratio
    step = max(divisor // 4, 1)
    ticks = [int(t) for t in ticks]
    picked = set(ticks)
    keep = set(ticks)
    i = 0
    while i < len(ticks):
        j = i
        while j + 1 < len(ticks) and ticks[j + 1] - ticks[j] == step:
            j += 1
        if j - i + 1 == 3:
            a, b, c = ticks[i], ticks[i + 1], ticks[i + 2]
            if scores[c] < ratio * min(scores[a], scores[b]) and c + step not in picked:
                keep.discard(c)
        i = j + 1
    return np.array([t for t in ticks if t in keep], dtype=int)


def repeat_slides(intensity: np.ndarray, tick: int, length: int, divisor: int) -> int:
    """How many runs a slider held over ``length`` ticks should make: when the music
    hits at every half beat (or quarter, on 1/4 maps) inside it, the slider turns back
    on each hit, standing in for a burst or long double taps. 1 means no repeats."""
    for unit in sorted({max(divisor // 2, 1), 1}):
        if length < 2 * unit or length % unit:
            continue
        hits = [intensity[tick + j] > 0.35 for j in range(unit, length, unit)
                if tick + j < len(intensity)]
        if hits and all(hits):
            return length // unit
    return 1


def absorb_into_sliders(ticks: np.ndarray, scores: np.ndarray, sustain_probs: np.ndarray,
                        slider_probs: np.ndarray, grid: TickGrid, max_slider: int,
                        ratio: float = 1.5) -> np.ndarray:
    """Drop picked notes that fall inside a slider the model would rather hold.

    Where a mapper might either hold one slider or tap two circles, the per-tick
    probabilities pick both the slider start and the note inside it, which cuts every
    slider short and leaves a steady stream of half-beat notes. Going through the
    slider starts in time order, a later note is dropped when the model finds "still
    holding" more likely than "a new note" there."""
    times = grid.score_times if grid.score_times is not None else grid.times
    held = sample_peak(sustain_probs, times, radius=0)
    starts = sample_peak(slider_probs, grid.times, radius=1) > 0.5
    picked = set(int(t) for t in ticks)
    dropped: set[int] = set()
    for tick in ticks:
        tick = int(tick)
        if tick in dropped or not starts[tick]:
            continue
        for t in range(tick + 1, min(tick + max_slider + 1, len(held))):
            if held[t] <= 0.5:
                break
            if t in picked:
                if held[t] > ratio * scores[t]:
                    dropped.add(t)
                else:
                    break
    return np.array([t for t in ticks if int(t) not in dropped], dtype=int)


def spread_spacing(objects: list[PlannedObject], target_std: float) -> None:
    """Widen the model's distance snaps around the map's median.

    The model predicts the typical spacing for each moment, which pulls every value
    towards the middle; mappers make calm parts calmer and emphasised notes bigger.
    The log distance snaps are stretched (by at most 4x) until their spread matches
    ``target_std``, what mappers use at this difficulty."""
    values = np.array([o.spacing for o in objects if o.spacing is not None])
    if len(values) < 10:
        return
    logs = np.log(values[values > 0.2])
    current = float(logs.std()) if len(logs) >= 10 else 0.0
    contrast = float(np.clip(target_std / max(current, 1e-3), 1.0, 4.0))
    median = float(np.median(values))
    for o in objects:
        if o.spacing is not None:
            o.spacing = float(median * (o.spacing / median) ** contrast)


def _wants_slider(features, grid, scores, tick, next_tick, preset, rng, slider_probs,
                  cut: float = 0.5) -> bool:
    if slider_probs is not None:
        return bool(sample_peak(slider_probs, grid.times[tick:tick + 1], radius=1)[0] > cut)
    a = int(grid.times[tick] * FPS / 1000.0)
    b = int(grid.times[next_tick] * FPS / 1000.0)
    held = features.rms[a:max(b, a + 1)]
    sustain = float(np.clip(held.mean() / (features.rms[min(a, len(features.rms) - 1)] + 1e-3), 0, 1))
    inner = scores[tick + 1:next_tick]
    # Strong unpicked onsets inside the gap argue against holding a slider over them.
    busy = float(inner.max() / (scores[tick] + 1e-3)) if len(inner) else 0.0
    p = preset.slider_rate * (0.5 + sustain) * (1.0 - 0.5 * min(busy, 1.0))
    return bool(rng.random() < p)
