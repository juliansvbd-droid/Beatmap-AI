"""Difficulty presets controlling note density, rhythm complexity, spacing and map settings.

Values follow the medians of ~2,200 ranked osu!standard difficulties, grouped by note
density (see the README).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    density: float  # target hit objects per second of active music
    divisor: int  # rhythm snap: ticks per beat
    min_gap_beats: float  # shortest allowed time between two objects
    max_run: int  # longest run of objects at the minimum gap (streams)
    slider_rate: float  # how eager the generator is to turn objects into sliders
    slider_recovery_beats: float  # time between a slider end and the next object
    min_slider_beats: float  # shortest slider body
    max_slider_beats: float  # longest slider body
    # Distance to the next object, in beats of slider velocity, after a 1/4, 1/2 and
    # 1 beat gap (1.0 at every gap would be exact distance snapping).
    spacing: tuple[float, float, float]
    jump_scale: float  # spacing varies by ±jump_scale/2 with the music's intensity
    combo_measures: int  # measures per combo
    hp: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float


PRESETS: dict[str, DifficultyPreset] = {
    p.name.lower(): p
    for p in [
        DifficultyPreset(
            "Easy",
            density=1.08, divisor=2, min_gap_beats=1.0, max_run=64,
            slider_rate=1.0, slider_recovery_beats=1.0, min_slider_beats=0.5, max_slider_beats=2.0,
            spacing=(0.25, 0.5, 1.0), jump_scale=0.0, combo_measures=2,
            hp=2, cs=3, od=2, ar=3, slider_multiplier=0.9,
        ),
        DifficultyPreset(
            "Normal",
            density=1.72, divisor=2, min_gap_beats=0.5, max_run=4,
            slider_rate=0.77, slider_recovery_beats=0.5, min_slider_beats=0.5, max_slider_beats=1.5,
            spacing=(0.2, 0.47, 1.0), jump_scale=0.0, combo_measures=2,
            hp=4, cs=3, od=4, ar=5, slider_multiplier=1.1,
        ),
        DifficultyPreset(
            "Hard",
            density=2.87, divisor=4, min_gap_beats=0.25, max_run=3,
            slider_rate=0.9, slider_recovery_beats=0.5, min_slider_beats=0.5, max_slider_beats=1.0,
            spacing=(0.23, 0.64, 1.0), jump_scale=0.5, combo_measures=1,
            hp=5, cs=4, od=6.5, ar=8, slider_multiplier=1.5,
        ),
        DifficultyPreset(
            "Insane",
            density=3.94, divisor=4, min_gap_beats=0.25, max_run=5,
            slider_rate=0.9, slider_recovery_beats=0.25, min_slider_beats=0.5, max_slider_beats=1.0,
            spacing=(0.22, 0.96, 1.0), jump_scale=0.6, combo_measures=1,
            hp=6, cs=4, od=8, ar=9, slider_multiplier=1.7,
        ),
        DifficultyPreset(
            "Expert",
            density=5.2, divisor=4, min_gap_beats=0.25, max_run=64,
            slider_rate=0.36, slider_recovery_beats=0.25, min_slider_beats=0.25, max_slider_beats=0.5,
            spacing=(0.23, 1.0, 1.0), jump_scale=0.7, combo_measures=1,
            hp=6, cs=4, od=9, ar=9.4, slider_multiplier=1.8,
        ),
    ]
}


def get_preset(name: str) -> DifficultyPreset:
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown difficulty {name!r}; choose from {', '.join(PRESETS)}") from None
