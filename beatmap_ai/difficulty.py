"""Difficulty presets controlling note density, rhythm complexity, spacing and map settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    density: float  # target hit objects per second of active music
    divisor: int  # rhythm snap: ticks per beat
    min_gap_beats: float  # shortest allowed time between two objects
    max_run: int  # longest run of objects at the minimum gap (streams)
    slider_rate: float  # how eager the generator is to use sliders
    slider_recovery_beats: float  # time between a slider end and the next object
    spacing: float  # distance multiplier relative to slider velocity (distance snap)
    jump_scale: float  # extra distance on loud notes (0 = pure distance snap)
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
            density=1.0, divisor=2, min_gap_beats=1.0, max_run=64,
            slider_rate=0.55, slider_recovery_beats=1.0, spacing=1.0, jump_scale=0.0,
            combo_measures=2,
            hp=3, cs=3, od=3, ar=4, slider_multiplier=1.0,
        ),
        DifficultyPreset(
            "Normal",
            density=1.7, divisor=2, min_gap_beats=0.5, max_run=4,
            slider_rate=0.5, slider_recovery_beats=0.5, spacing=1.0, jump_scale=0.0,
            combo_measures=2,
            hp=4, cs=3.5, od=4.5, ar=5.5, slider_multiplier=1.3,
        ),
        DifficultyPreset(
            "Hard",
            density=2.8, divisor=4, min_gap_beats=0.25, max_run=3,
            slider_rate=0.45, slider_recovery_beats=0.5, spacing=1.1, jump_scale=0.3,
            combo_measures=1,
            hp=5, cs=4, od=6.5, ar=8, slider_multiplier=1.6,
        ),
        DifficultyPreset(
            "Insane",
            density=4.0, divisor=4, min_gap_beats=0.25, max_run=5,
            slider_rate=0.4, slider_recovery_beats=0.25, spacing=1.2, jump_scale=0.6,
            combo_measures=1,
            hp=6, cs=4, od=8, ar=9, slider_multiplier=1.8,
        ),
        DifficultyPreset(
            "Expert",
            density=5.5, divisor=4, min_gap_beats=0.25, max_run=64,
            slider_rate=0.35, slider_recovery_beats=0.25, spacing=1.3, jump_scale=0.9,
            combo_measures=1,
            hp=6, cs=4.2, od=9, ar=9.5, slider_multiplier=2.0,
        ),
    ]
}


def get_preset(name: str) -> DifficultyPreset:
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown difficulty {name!r}; choose from {', '.join(PRESETS)}") from None
