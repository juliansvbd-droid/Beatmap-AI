import zipfile

import pytest

from beatmap_ai.osu import Beatmap, HitObject, TimingPoint, parse_osu, write_osz


def sample_map() -> Beatmap:
    return Beatmap(
        title="Song", artist="Artist", version="Hard", slider_multiplier=1.6,
        timing_points=[TimingPoint(100, 500.0), TimingPoint(2100, -50.0, uninherited=False)],
        hit_objects=[
            HitObject(100, 200, 100, "circle", new_combo=True),
            HitObject(200, 200, 600, "slider", curve_type="P",
                      curve_points=[(250, 150), (300, 200)], length=160.0),
            HitObject(300, 100, 2100, "slider", curve_type="L", curve_points=[(300, 260)],
                      slides=2, length=160.0),
            HitObject(256, 192, 4000, "spinner", end_time=6000),
        ],
        breaks=[(7000.0, 9000.0)],
    )


def test_round_trip():
    bm = sample_map()
    parsed = parse_osu(bm.to_osu_string())
    assert (parsed.title, parsed.artist, parsed.version) == ("Song", "Artist", "Hard")
    assert parsed.slider_multiplier == 1.6
    assert parsed.breaks == [(7000.0, 9000.0)]
    assert [tp.uninherited for tp in parsed.timing_points] == [True, False]
    kinds = [o.kind for o in parsed.hit_objects]
    assert kinds == ["circle", "slider", "slider", "spinner"]
    slider = parsed.hit_objects[1]
    assert slider.curve_type == "P" and slider.curve_points == [(250, 150), (300, 200)]
    assert parsed.hit_objects[0].new_combo
    assert parsed.hit_objects[3].end_time == 6000


def test_slider_duration_uses_timing_and_velocity():
    bm = sample_map()
    # 160 px at 1.6 * 100 px/beat = 1 beat of 500 ms.
    assert bm.slider_duration(bm.hit_objects[1]) == pytest.approx(500.0)
    # Inherited point doubles the velocity; two slides.
    assert bm.slider_duration(bm.hit_objects[2]) == pytest.approx(500.0)
    assert bm.end_time(bm.hit_objects[3]) == 6000


def test_write_osz(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"not really audio")
    bm = sample_map()
    bm.audio_filename = "audio.mp3"
    path = write_osz(tmp_path / "set.osz", audio, [bm])
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == sorted(["audio.mp3", bm.filename])
        assert parse_osu(zf.read(bm.filename).decode()).version == "Hard"
