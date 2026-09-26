from types import SimpleNamespace

import numpy as np
import pytest

from beatmap_ai.osu import Beatmap, HitObject, TimingPoint
from beatmap_ai.placement_data import (BEAT, END, KIAI, N_COLUMNS, PATH_END, PATH_START,
                                      T, augment_objects, map_objects)
from beatmap_ai.sequence_data import (TAGS, V3_FEATURES, SequenceMap, _v3_sampling_weights)
from beatmap_ai.slider_paths import sample_slider_path


def test_cli_keeps_v2_one_pass_and_experimental_models_opt_in(monkeypatch):
    from beatmap_ai import cli, generator

    captured = {}
    monkeypatch.setattr(generator, "generate",
                        lambda *args, **kwargs: captured.update(kwargs))
    cli.main(["generate", "song.mp3"])

    assert captured["placement_path"] == "auto"
    assert captured["passes"] == 1
    assert captured["planner_path"] is None
    assert captured["songfit_path"] is None
    assert captured["crutches"] is True


def test_slider_path_has_eight_length_normalized_local_samples_and_mirrors():
    bm = Beatmap(slider_multiplier=1.4,
                 timing_points=[TimingPoint(0, 500, effects=1),
                                TimingPoint(1500, 500, effects=0)],
                 hit_objects=[HitObject(100, 100, 0, "slider", curve_type="L",
                                        curve_points=[(300, 100)], length=200)])
    rows = map_objects(bm)
    path = rows[0, PATH_START:PATH_END].reshape(8, 2)
    assert rows.shape == (1, N_COLUMNS)
    assert rows[0, KIAI] == 1
    assert path.shape == (8, 2)
    assert path[0] == pytest.approx((0, 0))
    assert path[-1] == pytest.approx((1, 0))
    assert np.diff(path[:, 0]) == pytest.approx(np.full(7, 1 / 7))

    mirrored = augment_objects(rows, flip_x=True, flip_y=False)
    assert mirrored[0, PATH_START:PATH_END].reshape(8, 2)[:, 1] == pytest.approx(-path[:, 1])
    assert rows[0, PATH_START:PATH_END].reshape(8, 2)[:, 1] == pytest.approx(path[:, 1])


def test_slider_path_is_heading_local_and_curved_path_stays_finite():
    vertical = HitObject(100, 100, 0, "slider", curve_type="L",
                         curve_points=[(100, 300)], length=200)
    samples = sample_slider_path(vertical, np.pi / 2)
    assert samples.shape == (8, 2)
    assert samples[-1] == pytest.approx((1, 0), abs=1e-6)

    curved = HitObject(100, 100, 0, "slider", curve_type="B",
                       curve_points=[(100, 220), (220, 220)], length=180)
    curve_samples = sample_slider_path(curved, 0)
    assert curve_samples.shape == (8, 2)
    assert np.isfinite(curve_samples).all()
    assert curve_samples[3, 1] > 0


def test_v3_sampler_balances_available_star_and_style_buckets():
    maps = []
    for _ in range(10):
        tags = np.zeros(len(TAGS) + 1, dtype=np.float32)
        tags[TAGS.index("skillset/jumps")] = 1.0
        maps.append(SimpleNamespace(style={"stars": 2.0}, tags=tags))
    tags = np.zeros(len(TAGS) + 1, dtype=np.float32)
    tags[TAGS.index("skillset/streams")] = 1.0
    maps.append(SimpleNamespace(style={"stars": 2.0}, tags=tags))

    weights = _v3_sampling_weights(maps)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:10].sum() == pytest.approx(weights[-1])


def test_sequence_v2_still_loads_and_v3_emits_form_head(tmp_path):
    torch = pytest.importorskip("torch")
    from beatmap_ai.sequence_model import (SequenceNet, SequenceV3Net, load_sequence,
                                           save_sequence)
    from beatmap_ai.sequence_data import FEATURES

    v2 = SequenceNet(features=FEATURES, hidden=16, layers=1, heads=2, context=8,
                     offset_mixtures=2, chord_mixtures=2).eval()
    v2_path = tmp_path / "v2.pt"
    save_sequence(v2, v2_path)
    loaded_v2 = load_sequence(v2_path)
    assert "form" not in loaded_v2(torch.zeros(1, 8, FEATURES))

    v3 = SequenceV3Net(features=V3_FEATURES, hidden=16, layers=1, heads=2, context=8,
                       offset_mixtures=2, chord_mixtures=2, form_points=16).eval()
    v3_path = tmp_path / "v3.pt"
    save_sequence(v3, v3_path)
    loaded_v3 = load_sequence(v3_path)
    assert loaded_v3(torch.zeros(1, 8, V3_FEATURES))["form"].shape == (1, 8, 16)


def test_planner_transformer_reports_validation_metrics_and_target_sizes():
    torch = pytest.importorskip("torch")
    from beatmap_ai.planner import PlannerNet, _validation_metrics
    from beatmap_ai.planner_data import PLANNER_FEATURES, SECTION_TYPES, PlannerExample, STAR_CLASSES

    small, large = PlannerNet(hidden=128), PlannerNet(hidden=256)
    small_count = sum(parameter.numel() for parameter in small.parameters())
    large_count = sum(parameter.numel() for parameter in large.parameters())
    assert 0.55e6 < small_count < 0.65e6
    assert 2.1e6 < large_count < 2.4e6
    assert isinstance(small.encoder, torch.nn.TransformerEncoder)

    stars = np.zeros(STAR_CLASSES, dtype=np.float32)
    stars[7] = 1.0
    examples = []
    for index in range(2):
        controls = np.zeros((6, 8), dtype=np.float32)
        controls[2:4, 7] = 1.0
        kinds = np.full(6, SECTION_TYPES.index("verse"), dtype=np.int64)
        kinds[2:4] = SECTION_TYPES.index("main")
        examples.append(PlannerExample(
            song=f"song-{index}", features=np.zeros((6, PLANNER_FEATURES), dtype=np.float32),
            controls=controls, section_type=kinds, boundary=np.zeros(6, dtype=np.float32),
            stars=stars, styles=np.zeros(30, dtype=np.float32), styles_known=False, max_stars=4.0))
    metrics = _validation_metrics(small.eval(), examples, "cpu", batch_size=2)
    assert 0 <= metrics["kiai_recall"] <= 1
    assert metrics["max_stars_mae"] >= 0
    assert metrics["section_controls_mae"] >= 0
    assert np.isfinite(metrics["selection_score"])


def test_songfit_sampler_builds_each_negative_type(tmp_path):
    from beatmap_ai.placement_data import PlacementMap
    from beatmap_ai.songfit import SongFitSampler

    mel_path = tmp_path / "mel.npy"
    np.save(mel_path, np.zeros((80, 1400), dtype=np.float32))
    maps = []
    for map_index, song in enumerate(("song-a", "song-a", "song-b")):
        objects = np.zeros((80, N_COLUMNS), dtype=np.float32)
        objects[:, T] = np.arange(80) * 500.0
        objects[:, END] = objects[:, T]
        objects[:, BEAT] = 500.0
        objects[:, 2] = 70.0 + map_index * 20.0
        objects[:, 3] = 100.0
        maps.append(PlacementMap(f"map-{map_index}", song, mel_path, 4.0,
                                 {"stars": 4.0}, objects))

    batch = SongFitSampler(maps, seed=7).batch(size=4)
    assert batch["negative_type"].tolist() == [-1, 0, 1, 2]
    assert batch["label"].ravel().tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch["x"].shape[0] == 4
    assert np.isfinite(batch["x"]).all()
