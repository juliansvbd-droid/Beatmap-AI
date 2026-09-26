import numpy as np
import pytest

from beatmap_ai.dataset import build_examples, is_validation, song_key
from beatmap_ai.difficulty import preset_for_stars, spacing_spread
from beatmap_ai.generator import analyze, generate, generate_beatmap, style_values
from beatmap_ai.osu import Beatmap, HitObject, TimingPoint
from beatmap_ai.rhythm import PlannedObject, spread_spacing
from beatmap_ai.style import encode_conditions, map_style, star_rating


def test_song_key_ignores_versions_and_spacing():
    a = song_key(Beatmap(artist="TK from Ling tosite sigure", title="unravel (TV edit)"))
    b = song_key(Beatmap(artist="TK from Ling tosite sigure", title="unravel"))
    c = song_key(Beatmap(artist="Ikimonogakari", title="Blue Bird"))
    d = song_key(Beatmap(artist="Ikimono Gakari", title="Blue Bird (TV Size)"))
    assert a == b and c == d and a != c
    assert is_validation(a) == is_validation(b)


def test_timing_at_follows_timing_points():
    bm = Beatmap(timing_points=[
        TimingPoint(1000, 500.0), TimingPoint(3000, -50.0, uninherited=False),
        TimingPoint(5000, 400.0),
    ])
    assert bm.timing_at(0) == (500.0, 1.0)
    assert bm.timing_at(2000) == (500.0, 1.0)
    assert bm.timing_at(3000) == (500.0, 2.0)
    assert bm.timing_at(6000) == (400.0, 1.0)
    bm.timing_points.append(TimingPoint(7000, 300.0))
    assert bm.timing_at(8000) == (300.0, 1.0)  # The cache notices new points.


def test_map_style_measures_streams_and_jumps():
    bm = Beatmap(slider_multiplier=1.0, timing_points=[TimingPoint(0, 500.0)])
    # A stream of eight 1/4 notes, then half-beat jumps of 200 px (distance snap 4).
    bm.hit_objects = [HitObject(100 + 10 * i, 100, 125.0 * i, "circle") for i in range(8)]
    for i in range(8):
        bm.hit_objects.append(HitObject(100 + 200 * (i % 2), 300, 1500 + 250.0 * i, "circle"))
    style = map_style(bm)
    assert style["stream"] == pytest.approx(0.5)
    assert style["jump"] == pytest.approx(4.0)
    assert style["sliders"] == 0.0


def test_encode_conditions_marks_missing_values():
    x = encode_conditions({"density": 3.0, "stars": float("nan")})
    assert x[0] == pytest.approx(np.log1p(3.0)) and x[1] == 1.0
    assert x[2] == 0.0 and x[3] == 0.0


def test_duplicate_difficulties_are_used_once(song_mp3, tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        generate(song_mp3, tmp_path / folder / "set.osz", ["normal", "hard"], model_path=None,
                 log=lambda *_: None)
    logs = []
    examples = build_examples([tmp_path / "a", tmp_path / "b"], tmp_path / "cache", log=logs.append)
    assert len(examples) == 2
    assert any("2 duplicate" in line for line in logs)
    # The second run reads the parsed maps from the cache.
    again = build_examples([tmp_path / "a"], tmp_path / "cache2", log=lambda *_: None)
    cached = build_examples([tmp_path / "a"], tmp_path / "cache2", log=lambda *_: None)
    assert [e.name for e in again] == [e.name for e in cached]


def test_preset_for_stars_scales_settings():
    easy, hard = preset_for_stars(2.0), preset_for_stars(5.5)
    assert easy.ar < hard.ar and easy.od < hard.od and easy.density < hard.density
    assert hard.name.startswith("Expert")


def test_spread_spacing_reaches_target_variation():
    rng = np.random.default_rng(0)
    plan = [PlannedObject(i * 100.0, "circle", i, 0.5, i * 100.0, 0,
                          spacing=float(np.exp(rng.normal(0.4, 0.2)))) for i in range(200)]
    spread_spacing(plan, spacing_spread(6.0))
    logs = np.log([p.spacing for p in plan])
    assert logs.std() == pytest.approx(spacing_spread(6.0), rel=0.05)


def test_style_values_use_percentiles():
    class Model:
        config = {"cond_stats": {"4": {"jump": [1.0, 1.4, 1.7, 2.1, 2.8],
                                       "stream": [0.0, 0.0, 0.02, 0.1, 0.4]}}}
    values = style_values(Model(), 4.5, {"jump": 1.0})
    assert values["jump"] == pytest.approx(2.8)
    assert values["stream"] == pytest.approx(0.0)
    half = style_values(Model(), 4.5, {"stream": 0.5})
    assert half["stream"] == pytest.approx(0.21)


def test_star_rating_target_is_reached(song_mp3):
    pytest.importorskip("rosu_pp_py")
    features, timing = analyze(song_mp3)
    for stars in (2.0, 3.5):
        bm = generate_beatmap(features, timing, stars)
        assert star_rating(bm.to_osu_string()) == pytest.approx(stars, abs=0.3)


def test_spread_spacing_stretches_at_most_four_times():
    rng = np.random.default_rng(0)
    plan = [PlannedObject(i * 100.0, "circle", i, 0.5, i * 100.0, 0,
                          spacing=float(np.exp(rng.normal(0.4, 0.05)))) for i in range(200)]
    before = np.log([p.spacing for p in plan]).std()
    spread_spacing(plan, 1.0)
    assert np.log([p.spacing for p in plan]).std() == pytest.approx(4 * before, rel=0.01)
