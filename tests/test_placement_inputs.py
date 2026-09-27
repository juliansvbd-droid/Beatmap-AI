import numpy as np

from beatmap_ai.audio import AudioFeatures
from beatmap_ai.difficulty import get_preset
from beatmap_ai.placement_data import HCOS, HSIN
from beatmap_ai.placement_model import through_points
from beatmap_ai.rhythm import PlannedObject
from beatmap_ai.sequence_model import SequenceNet, SequencePlacer


def _placer():
    preset = get_preset("normal")
    features = AudioFeatures(np.zeros((80, 200), dtype=np.float32), np.zeros(200),
                             np.zeros(200), np.zeros(200), 5.0)

    class Timing:
        offset_ms = 0.0
        beat_length = 500.0

    class Grid:
        times = np.arange(200) * 125.0
        score_times = times
        beat = np.arange(200) // 4
        beat_length_ms = None

    model = SequenceNet(hidden=32, layers=1, heads=2, offset_mixtures=2, chord_mixtures=2).eval()
    return SequencePlacer(model, preset, features, Timing(), {"stars": 3.0, "density": 2.0},
                          None, np.ones(200), Grid(), 0.5, np.random.default_rng(1))


def test_model_sees_the_heading_it_was_trained_with():
    """Training rows carry the heading each offset is measured against (a unit vector).
    When placing, the row being placed must carry it already when the model reads it --
    it used to be 0/0 there, an input the model never saw in training."""
    placer = _placer()
    seen = []
    original = placer._model_features

    def spy(rows):
        seen.append(float(np.hypot(rows[-1, HCOS], rows[-1, HSIN])))
        return original(rows)

    placer._model_features = spy
    plan = [PlannedObject(i * 500.0, "circle", i * 4, 0.5, i * 500.0, i, i % 4 == 0, 500.0)
            for i in range(12)]
    placer.follow(plan)
    assert seen and np.allclose(seen, 1.0, atol=1e-5)


def test_slider_curve_passes_through_the_predicted_points():
    points = [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, -5.0)]
    controls = through_points(points)
    # one cubic segment per pair: two handles and the anchor, anchors repeated between
    # segments (osu! starts a new Bezier segment at a repeated point)
    assert len(controls) == 3 * 3 + 2
    anchors = [controls[2], controls[6], controls[10]]
    assert anchors == [(10.0, 5.0), (20.0, 0.0), (30.0, -5.0)]
    assert controls[3] == controls[2] and controls[7] == controls[6]


def test_combos_follow_the_models_prediction_but_long_pauses_force_one():
    rng = np.random.default_rng(0)

    def item(time, logit=None, new=False):
        obj = PlannedObject(time, "circle", 0, 0.5, time, 0, new, 500.0)
        if logit is not None:
            obj.next_combo_logit = logit
        return obj

    history = [item(0.0, new=True), item(500.0, logit=-20.0)]
    nxt = item(1000.0, new=True)
    SequencePlacer._decide_combo(nxt, history, rng)
    assert not nxt.new_combo  # model is sure: no new combo, the bar rule is overridden

    history = [item(0.0, new=True), item(500.0, logit=20.0)]
    nxt = item(1000.0)
    SequencePlacer._decide_combo(nxt, history, rng)
    assert nxt.new_combo

    history = [item(0.0, new=True), item(500.0, logit=-20.0)]
    nxt = item(4000.0)  # 7 beats of rest
    SequencePlacer._decide_combo(nxt, history, rng)
    assert nxt.new_combo
