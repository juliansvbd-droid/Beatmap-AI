import math
import zipfile

import pytest

from beatmap_ai.difficulty import PRESETS
from beatmap_ai.generator import analyze, approach_preempt, generate, generate_beatmap
from beatmap_ai.osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, parse_osu
from beatmap_ai.placement import slider_path
from beatmap_ai.style import star_rating


@pytest.fixture(scope="module")
def analysis(song_mp3):
    return analyze(song_mp3)


@pytest.mark.parametrize("difficulty", list(PRESETS))
def test_generated_map_is_valid(analysis, difficulty):
    features, timing = analysis
    bm = generate_beatmap(features, timing, difficulty)
    preset = PRESETS[difficulty]
    objs = bm.hit_objects
    assert len(objs) >= 10
    assert objs[0].new_combo

    tick = timing.beat_length / preset.divisor
    min_gap = preset.min_gap_beats * timing.beat_length
    for a, b in zip(objs, objs[1:]):
        assert b.time > a.time
        assert b.time - bm.end_time(a) >= min_gap - 2, "objects overlap in time"
    for o in objs:
        # Every object starts on the beat grid.
        ticks = (o.time - timing.offset_ms) / tick
        assert abs(ticks - round(ticks)) * tick < 1.5
        assert 0 <= o.x <= PLAYFIELD_WIDTH and 0 <= o.y <= PLAYFIELD_HEIGHT
        if o.kind == "slider":
            assert o.length > 0
            end = bm.end_time(o)
            end_ticks = (end - timing.offset_ms) / tick
            assert abs(end_ticks - round(end_ticks)) * tick < 1.5, "slider must end on the grid"
            for px, py in o.curve_points:
                assert -5 <= px <= PLAYFIELD_WIDTH + 5 and -5 <= py <= PLAYFIELD_HEIGHT + 5


def test_harder_difficulties_have_more_stars(analysis):
    # Harder is not necessarily more notes (an Expert may reach its stars with jumps), but
    # every named difficulty must be harder than the one before.
    features, timing = analysis
    stars = [star_rating(generate_beatmap(features, timing, d).to_osu_string()) for d in PRESETS]
    if None in stars:
        pytest.skip("rosu-pp-py not installed")
    assert stars == sorted(stars)


def test_generation_is_deterministic(analysis):
    features, timing = analysis
    a = generate_beatmap(features, timing, "hard", seed=3).to_osu_string()
    b = generate_beatmap(features, timing, "hard", seed=3).to_osu_string()
    assert a == b


def test_slider_path_has_requested_length():
    for bend in (0.0, 0.5, -1.2):
        curve_type, points, point, _ = slider_path((100, 100), 0.3, 150.0, bend)
        samples = [point(150.0 * i / 200) for i in range(201)]
        length = sum(math.dist(p, q) for p, q in zip(samples, samples[1:]))
        assert length == pytest.approx(150.0, rel=1e-3)
        assert points[-1] == pytest.approx(point(150.0))
        assert curve_type == ("L" if bend == 0 else "P")


def test_generate_writes_osz(song_mp3, tmp_path):
    out = generate(song_mp3, tmp_path / "out.osz", ["easy", "insane"], model_path=None,
                   log=lambda *_: None)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "audio.mp3" in names
        osu_files = [n for n in names if n.endswith(".osu")]
        assert len(osu_files) == 2
        maps = [parse_osu(zf.read(n).decode()) for n in osu_files]
    assert {m.version for m in maps} == {"Easy", "Insane"}
    assert all(m.artist == "Test Artist" and m.title == "Drum Loop" for m in maps)
    assert all(m.audio_filename == "audio.mp3" for m in maps)
    assert all(m.timing_points[0].bpm == pytest.approx(160, abs=0.05) for m in maps)


def test_early_first_object_gets_lead_in(analysis):
    features, timing = analysis
    bm = generate_beatmap(features, timing, "normal")
    first = bm.hit_objects[0].time
    assert bm.audio_lead_in == max(0, round(approach_preempt(bm.ar) + 500 - first))


def test_bundled_model_is_optional(monkeypatch, tmp_path):
    import beatmap_ai.generator as gen

    monkeypatch.setattr(gen, "BUNDLED_MODEL", tmp_path / "missing.pt")
    assert gen.bundled_model() is None
