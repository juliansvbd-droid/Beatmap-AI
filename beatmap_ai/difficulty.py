"""Difficulty presets controlling note density, rhythm complexity, spacing and map settings.

Values follow the medians of ~2,200 ranked osu!standard difficulties, grouped by note
density (see the README).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


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
            slider_rate=0.9, slider_recovery_beats=0.25, min_slider_beats=0.25, max_slider_beats=1.0,
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


# Median settings of ranked maps by star rating (from ~3,000 difficulties):
# stars, AR, OD, CS, HP, slider multiplier, objects per second.
STAR_TABLE = [
    (1.5, 3.0, 2.0, 3.0, 2.0, 0.80, 1.1),
    (2.0, 3.5, 3.0, 3.0, 2.5, 0.95, 1.5),
    (2.5, 5.0, 4.0, 3.2, 4.0, 1.10, 1.9),
    (3.0, 7.0, 6.0, 4.0, 5.0, 1.40, 2.5),
    (3.5, 8.0, 6.0, 4.0, 5.0, 1.40, 2.9),
    (4.0, 8.0, 7.0, 4.0, 5.5, 1.60, 3.3),
    (4.5, 9.0, 7.7, 4.0, 6.0, 1.70, 3.8),
    (5.0, 9.0, 8.0, 4.0, 6.0, 1.70, 4.2),
    (5.5, 9.2, 8.5, 4.0, 6.0, 1.80, 4.6),
    (6.0, 9.3, 8.9, 4.0, 6.0, 1.80, 5.0),
    (6.5, 9.5, 9.0, 4.0, 6.0, 1.80, 5.3),
    (7.0, 9.6, 9.0, 4.0, 6.0, 1.90, 5.6),
    (8.0, 9.7, 9.3, 4.0, 6.0, 1.90, 6.2),
]
# Rhythm rules (snap, gaps, stream length, slider shapes) come from the nearest named
# preset below this star rating.
PRESET_STARS = [("Easy", 0.0), ("Normal", 1.9), ("Hard", 2.8), ("Insane", 3.9), ("Expert", 5.0)]


def preset_for_stars(stars: float) -> DifficultyPreset:
    """A preset for a star rating: rhythm rules from the matching named difficulty and
    map settings (AR, OD, CS, HP, slider speed, density) typical for those stars."""
    table = np.array(STAR_TABLE)
    s = float(np.clip(stars, table[0, 0], table[-1, 0]))
    ar, od, cs, hp, sm, density = (float(np.interp(s, table[:, 0], table[:, i])) for i in range(1, 7))
    base = [name for name, lowest in PRESET_STARS if stars >= lowest][-1]
    preset = PRESETS[base.lower()]
    return replace(preset, name=f"{base} {stars:.1f}*", ar=round(ar, 1), od=round(od, 1),
                   cs=round(cs, 1), hp=round(hp, 1), slider_multiplier=round(sm, 2),
                   density=density)


def parse_difficulty(value: str) -> DifficultyPreset | float:
    """A preset name, or a star rating such as "4.5" (returned as float)."""
    try:
        return float(value.replace("*", "").replace(",", "."))
    except ValueError:
        return get_preset(value)


# How much mappers vary their spacing within one map: standard deviation of the log
# distance snap (stacks excluded), median over ranked maps at each star rating.
SPACING_SPREAD = [(1.75, 0.07), (2.25, 0.12), (2.75, 0.14), (3.25, 0.22), (3.75, 0.26),
                  (4.25, 0.38), (4.75, 0.42), (5.25, 0.44), (5.75, 0.47), (6.25, 0.52),
                  (7.0, 0.53)]


def stars_for_density(density: float) -> float:
    """Typical star rating of maps with this many objects per second."""
    table = np.array(STAR_TABLE)
    return float(np.interp(density, table[:, 6], table[:, 0]))


def spacing_spread(stars: float) -> float:
    table = np.array(SPACING_SPREAD)
    return float(np.interp(stars, table[:, 0], table[:, 1]))
