import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatmap_ai.dataset import build_examples, targets  # noqa: E402
from beatmap_ai.generator import generate  # noqa: E402
from beatmap_ai.model import BeatmapNet, beat_phase_features, load_checkpoint  # noqa: E402
from beatmap_ai.train import onset_f1, train  # noqa: E402


def test_forward_shape():
    model = BeatmapNet(hidden=32)
    logits = model(torch.zeros(2, 80, 50), torch.zeros(2, 50, 4), torch.tensor([2.0, 4.0]))
    assert logits.shape == (2, 50, 2)


def test_beat_phase_features():
    # 120 BPM from t=0: every 43rd frame or so is a beat; phase features stay on the unit circle.
    feats = beat_phase_features(200, [(0.0, 500.0, 4)])
    assert feats.shape == (200, 4)
    assert np.allclose(feats[:, 0] ** 2 + feats[:, 1] ** 2, 1.0, atol=1e-5)
    assert feats[0, 1] == pytest.approx(1.0)


def test_onset_f1():
    probs = np.zeros(100)
    probs[[10, 50, 90]] = 0.9
    assert onset_f1(probs, np.array([11, 50, 89]), 0.5) == pytest.approx(1.0)
    assert onset_f1(probs, np.array([30]), 0.5) == 0.0


def test_train_on_generated_maps_and_generate(song_mp3, tmp_path):
    # Use heuristic maps as a stand-in training set to exercise the whole pipeline.
    data = tmp_path / "data"
    data.mkdir()
    generate(song_mp3, data / "set.osz", ["normal", "hard"], model_path=None, log=lambda *_: None)
    examples = build_examples(data, tmp_path / "cache", log=lambda *_: None)
    assert len(examples) == 2
    note, slider, mask = targets(examples[0], 0, 500)
    assert note.max() == 1.0 and mask.sum() > 0

    ckpt = train(data, tmp_path / "model.pt", epochs=2, steps_per_epoch=3, batch_size=2,
                 chunk_seconds=4, hidden=16, cache_dir=tmp_path / "cache", log=lambda *_: None)
    model, threshold = load_checkpoint(ckpt)
    assert 0 < threshold < 1

    out = generate(song_mp3, tmp_path / "ai.osz", ["hard"], model_path=ckpt, log=lambda *_: None)
    assert out.exists()


def test_figure_head_warmstart_keeps_existing_outputs(song_mp3, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    generate(song_mp3, data / "set.osz", ["hard", "insane"],
             model_path=None, placement_path=None, critic_path=None, log=lambda *_: None)
    base = train(data, tmp_path / "base.pt", epochs=1, steps_per_epoch=1, batch_size=2,
                 chunk_seconds=4, hidden=16, cache_dir=tmp_path / "cache",
                 workers=0, device="cpu", log=lambda *_: None)
    original, original_threshold = load_checkpoint(base)
    updated = train(data, tmp_path / "figures.pt", epochs=1, steps_per_epoch=2,
                    batch_size=2, cache_dir=tmp_path / "cache", init_model=base,
                    workers=0, device="cpu", figure_head=True, log=lambda *_: None)
    model, updated_threshold = load_checkpoint(updated)
    assert model.outputs == 5
    assert updated_threshold == original_threshold
    for name, value in original.state_dict().items():
        new_value = model.state_dict()[name]
        if name.startswith("head."):
            new_value = new_value[:4]
        assert torch.equal(new_value, value), name
    out = generate(song_mp3, tmp_path / "figures.osz", ["hard"], model_path=updated,
                   placement_path=None, critic_path=None, log=lambda *_: None)
    assert out.exists()
