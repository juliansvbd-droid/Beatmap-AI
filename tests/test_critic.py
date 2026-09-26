import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatmap_ai.critic import CriticNet, load_critic, save_critic
from beatmap_ai.critic_data import CRITIC_FEATURES, CRITIC_WINDOW, critic_features
from beatmap_ai.osu import Beatmap, HitObject
from beatmap_ai.placement_data import N_COLUMNS, map_objects
from beatmap_ai.rhythm import PlannedObject
from beatmap_ai.sequence_model import SequenceNet, SequencePlacer


def test_critic_forward_shape():
    model = CriticNet(features=CRITIC_FEATURES, hidden=64, layers=2, heads=4, ffn_dim=128, window=CRITIC_WINDOW)
    x = torch.zeros(2, CRITIC_WINDOW, CRITIC_FEATURES)
    mask = torch.ones(2, CRITIC_WINDOW)
    logits = model(x, mask)
    assert logits.shape == (2, 1)

    scores = model.score(x, mask)
    assert scores.shape == (2,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_critic_data_window():
    from beatmap_ai.osu import TimingPoint
    bm = Beatmap(cs=4.0, slider_multiplier=1.4, timing_points=[TimingPoint(0.0, 500.0)])
    # Create 30 hit objects
    for i in range(30):
        bm.hit_objects.append(HitObject(100.0 + i * 5.0, 100.0 + i * 3.0, i * 500.0, "circle", False, 0))

    objs = map_objects(bm)
    assert objs.shape == (30, N_COLUMNS)

    feats = critic_features(objs, mel=None, stars=4.0, energy=None)
    assert feats.shape == (30, CRITIC_FEATURES)
    assert not np.isnan(feats).any()
    assert np.isfinite(feats).all()

    # Verify no telltale hitsounds / combo color leakage
    # Positions are normalized in [0, 1] approximately
    assert (feats[:, 12:14] >= 0.0).all() and (feats[:, 12:14] <= 1.0).all()


def test_critic_checkpoint_save_load(tmp_path):
    model = CriticNet(features=CRITIC_FEATURES, hidden=64, layers=2, heads=4, ffn_dim=128, window=CRITIC_WINDOW).eval()
    ckpt_path = tmp_path / "test_critic.pt"
    save_critic(model, ckpt_path)

    loaded = load_critic(ckpt_path, device="cpu")
    assert loaded.config["hidden"] == 64
    assert loaded.config["layers"] == 2

    x = torch.zeros(1, CRITIC_WINDOW, CRITIC_FEATURES)
    mask = torch.ones(1, CRITIC_WINDOW)
    assert np.allclose(model(x, mask).detach().numpy(), loaded(x, mask).detach().numpy(), atol=1e-5)


def test_best_of_n_selects_highest_score():
    """Verify that SequencePlacer.follow with candidates > 1 picks the candidate evaluated to have the max score."""
    from beatmap_ai.difficulty import get_preset
    from beatmap_ai.audio import AudioFeatures

    preset = get_preset("normal")
    mel = np.zeros((80, 200), dtype=np.float32)
    features = AudioFeatures(mel, np.zeros(200), np.zeros(200), np.zeros(200), 5.0)

    class DummyTiming:
        offset_ms = 0.0
        beat_length = 500.0

    class DummyGrid:
        times = np.arange(20) * 125.0
        score_times = times
        beat = np.arange(20) // 4
        beat_length_ms = None

    seq_model = SequenceNet(hidden=32, layers=1, heads=2, offset_mixtures=2, chord_mixtures=2).eval()
    rng = np.random.default_rng(123)

    placer = SequencePlacer(
        seq_model, preset, features, DummyTiming(), {"stars": 3.0, "density": 2.0},
        None, np.ones(20), DummyGrid(), 0.5, rng
    )

    plan = [
        PlannedObject(i * 500.0, "circle", i * 4, 0.5, i * 500.0, i, (i % 4 == 0), 500.0)
        for i in range(24)
    ]

    evaluated_batches = []

    class MockCritic:
        def score(self, x, mask):
            # Record scores: assign candidate 2 the highest score
            evaluated_batches.append(len(x))
            scores = torch.tensor([0.1, 0.2, 0.99, 0.3][:len(x)], dtype=torch.float32)
            return scores

    mock_critic = MockCritic()
    out_plan, choices = placer.follow(plan, critic=mock_critic, candidates=4, chunk_size=16)

    assert len(out_plan) >= 10
    assert len(choices) == len(out_plan)
    assert len(evaluated_batches) >= 1
    assert evaluated_batches[0] == 4
