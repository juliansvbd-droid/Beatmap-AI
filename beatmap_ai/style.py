"""Describing a beatmap's difficulty and style with a few numbers.

The rhythm model is trained with these as conditions, so a map can be requested with
a star rating and a style ("more jumps", "more streams", ...). During training each
condition is sometimes hidden from the model, which is what "Auto" uses: the model
then picks what fits the music.
"""

from __future__ import annotations

import numpy as np

from .osu import Beatmap

# Order of the model's condition inputs. Each has a value and a "given" flag.
CONDITIONS = ("density", "stars", "jump", "stream", "tech", "sliders")

# Style presets for the UI: which conditions they raise (to a high percentile of the
# training maps) and which they lower.
STYLES = {
    "jump": {"jump": 1.0, "stream": 0.0},
    "stream": {"stream": 1.0, "jump": 0.0},
    "tech": {"tech": 1.0},
    "flow": {"jump": 0.0, "stream": 0.0, "sliders": 1.0},
    "circles": {"sliders": 0.0},
    "cross-screen": {"jump": 1.0},
    "doubles": {},
    "alt": {},
    "geometric": {},
    "simple": {"jump": 0.0},
}
# Community tags each style asks the sequence model for (value = share of top votes).
STYLE_TAGS = {
    "jump": {"skillset/jumps": 1.0, "jumps/sharp": 0.6},
    "stream": {"skillset/streams": 1.0, "streams/bursts": 0.8},
    "tech": {"skillset/tech": 1.0, "tech/aim control": 0.7, "tech/slider tech": 0.6},
    "flow": {"streams/flow aim": 1.0, "style/clean": 0.6},
    "circles": {},
    "cross-screen": {"jumps/cross-screen": 1.0, "skillset/jumps": 0.8, "jumps/wide": 0.6},
    "doubles": {"streams/doubles": 1.0, "streams/bursts": 0.6},
    "alt": {"skillset/alt": 1.0, "streams/bursts": 0.5},
    "geometric": {"style/geometric": 1.0, "style/symmetrical": 0.7},
    "simple": {"expression/simple": 1.0, "style/clean": 0.7},
}


def style_tags(style: dict[str, float] | None) -> dict[str, float] | None:
    """Tag values for the chosen styles, scaled by their strength (None for Auto)."""
    if not style:
        return None
    tags: dict[str, float] = {}
    for name, strength in style.items():
        for tag, value in STYLE_TAGS.get(name, {}).items():
            tags[tag] = max(tags.get(tag, 0.0), value * float(strength))
    return tags or None


def star_rating(text: str) -> float | None:
    """osu!standard star rating (no mods) from the .osu file contents, if rosu-pp is
    installed (``pip install rosu-pp-py``)."""
    try:
        import rosu_pp_py as rosu
    except ImportError:
        return None
    try:
        return float(rosu.Difficulty().calculate(rosu.Beatmap(content=text)).stars)
    except Exception:
        return None


def map_style(bm: Beatmap, text: str | None = None) -> dict[str, float]:
    """Density, stars and style descriptors of one difficulty.

    * jump: median distance snap between objects half a beat or more apart
      (1.0 = exact distance snapping, 2.0 = spaced twice as far)
    * stream: share of objects inside runs of 4+ objects at 1/4 beat or faster
    * tech: rhythm variety -- entropy of the gaps between objects, plus the share of
      objects on 1/4 and 3/4 beat positions outside streams (0..1)
    * sliders: share of sliders among circles and sliders
    """
    objs = [o for o in bm.hit_objects if o.kind != "spinner"]
    style = {name: float("nan") for name in CONDITIONS}
    if len(objs) < 2 or not any(tp.uninherited for tp in bm.timing_points):
        return style
    times = np.array([o.time for o in objs])
    style["density"] = len(objs) / max((times[-1] - times[0]) / 1000.0, 1.0)
    if text is not None:
        stars = star_rating(text)
        style["stars"] = float("nan") if stars is None else stars

    beat_lengths = np.array([bm.timing_at(t)[0] for t in times])
    gaps = np.diff(times) / beat_lengths[1:]
    fast = gaps <= 0.26
    in_stream = np.zeros(len(objs), dtype=bool)
    run_start = 0
    for i in range(1, len(objs) + 1):
        if i == len(objs) or not fast[i - 1]:
            if i - run_start >= 4:
                in_stream[run_start:i] = True
            run_start = i
    style["stream"] = float(in_stream.mean())

    velocity = bm.slider_multiplier * 100.0
    snaps = []
    for prev, obj in zip(objs, objs[1:]):
        beat_length = bm.timing_at(obj.time)[0]
        gap = (obj.time - bm.end_time(prev)) / beat_length
        if 0.45 <= gap <= 1.05:
            end = (prev.x, prev.y)
            if prev.kind == "slider" and prev.slides % 2 == 1 and prev.curve_points:
                end = prev.curve_points[-1]
            snaps.append(np.hypot(obj.x - end[0], obj.y - end[1]) / (velocity * gap))
    style["jump"] = float(np.median(snaps)) if snaps else float("nan")

    snapped = np.clip(np.round(gaps * 4), 1, 9)
    counts = np.bincount(snapped.astype(int), minlength=10)[1:]
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(9))
    first = bm.timing_points[0]
    phase = ((times - first.time) / beat_lengths) % 1.0
    quarter = np.abs(((phase * 4) % 2) - 1) < 0.2  # on 1/4 or 3/4
    style["tech"] = 0.5 * entropy + 0.5 * min(1.0, 2.0 * float((quarter & ~in_stream).mean()))
    style["sliders"] = float(np.mean([o.kind == "slider" for o in objs]))
    return style


def encode_conditions(values: dict[str, float] | None, names=CONDITIONS) -> np.ndarray:
    """Model input for the given condition values: (value, given) per condition.
    Missing or NaN values count as not given."""
    values = values or {}
    out = np.zeros(2 * len(names), dtype=np.float32)
    for i, name in enumerate(names):
        value = values.get(name)
        if value is None or not np.isfinite(value):
            continue
        if name == "density":
            value = np.log1p(value)
        elif name == "stars":
            value = value / 5.0
        elif name == "jump":
            value = np.log(max(value, 0.05))
        out[2 * i], out[2 * i + 1] = value, 1.0
    return out
