"""Map-to-audio matching model and multi-pass scoring support."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .audio import FPS
from .critic import CriticNet, evaluate_critic, roc_auc_score_np, save_critic
from .critic_data import CRITIC_FEATURES, CRITIC_WINDOW, critic_features
from .dataset import is_validation
from .placement_data import BEAT, T, PlacementMap, build_placement_maps
from .train import resolve_device

NEGATIVE_TYPES = ("other_song", "time_shift", "other_section")


class SongFitSampler:
    """On-the-fly matched and deliberately misaligned map/audio windows."""

    def __init__(self, maps: list[PlacementMap], window=CRITIC_WINDOW, seed=0):
        self.maps = [m for m in maps if len(m.objects) >= 16 and Path(m.mel_path).is_file()]
        self.window = window
        self.rng = np.random.default_rng(seed)
        self._mels: dict[Path, np.ndarray] = {}
        self._energy: dict[Path, np.ndarray] = {}
        bpm = np.array([60000.0 / max(float(np.median(m.objects[:, BEAT])), 1.0) for m in self.maps])
        self.bpm = bpm
        self.matches = []
        for i, m in enumerate(self.maps):
            mask = (np.abs(bpm - bpm[i]) / max(bpm[i], 1.0) <= 0.08)
            mask &= np.array([x.song != m.song for x in self.maps])
            self.matches.append(np.flatnonzero(mask))

    def mel(self, path):
        path = Path(path)
        if path not in self._mels:
            self._mels[path] = np.load(path, mmap_mode="r")
        if path not in self._energy:
            from .placement_data import section_energy
            self._energy[path] = section_energy(self._mels[path])
        return self._mels[path], self._energy[path]

    def _example(self, ref, start, offset=0.0, audio_ref=None):
        objects = ref.objects[start:start + self.window]
        audio_ref = audio_ref or ref
        mel, energy = self.mel(audio_ref.mel_path)
        features = critic_features(objects, mel, float(ref.style.get("stars", 4.0)), energy,
                                   audio_offset_ms=offset)
        n = len(features)
        return (np.pad(features, ((0, self.window - n), (0, 0))),
                np.pad(np.ones(n, dtype=np.float32), (0, self.window - n)))

    def batch(self, size=32):
        features, masks, labels, kinds = [], [], [], []
        for i in range(size):
            ref_index = int(self.rng.integers(len(self.maps)))
            ref = self.maps[ref_index]
            max_start = max(len(ref.objects) - self.window, 0)
            start = int(self.rng.integers(max_start + 1))
            label, kind, audio_ref, offset = 1.0, -1, None, 0.0
            if i % 4:
                kind = (i - 1) % len(NEGATIVE_TYPES)
                label = 0.0
                mel, _ = self.mel(ref.mel_path)
                if kind == 0:  # Another song with a close BPM.
                    choices = self.matches[ref_index]
                    if len(choices):
                        audio_ref = self.maps[int(self.rng.choice(choices))]
                    else:
                        offset = min(float(np.median(ref.objects[:, BEAT])), 600.0) / 2
                elif kind == 1:  # Half-beat or one-beat phase displacement.
                    beat = float(np.median(ref.objects[:, BEAT]))
                    offset = beat * (0.5 if self.rng.random() < 0.5 else 1.0)
                else:  # Same song, but another musical section.
                    seconds = mel.shape[1] / FPS
                    shift = self.rng.uniform(8.0, max(8.1, seconds * 0.6)) if seconds > 16 else 4.0
                    offset = shift * 1000.0
            x, mask = self._example(ref, start, offset, audio_ref)
            features.append(x)
            masks.append(mask)
            labels.append([label])
            kinds.append(kind)
        return {"x": np.stack(features), "mask": np.stack(masks),
                "label": np.asarray(labels, dtype=np.float32),
                "negative_type": np.asarray(kinds, dtype=np.int64)}


def _metrics(model, sampler, device, batches=8):
    labels, scores, groups, losses = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            raw = sampler.batch(32)
            x = torch.as_tensor(raw["x"], device=device)
            mask = torch.as_tensor(raw["mask"], device=device)
            y = torch.as_tensor(raw["label"], device=device)
            logits = model(x, mask)
            losses.append(float(F.binary_cross_entropy_with_logits(logits, y).cpu()))
            scores.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
            labels.extend(y.cpu().numpy().ravel().tolist())
            groups.extend(raw["negative_type"].tolist())
    y, p, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    result = {"loss": float(np.mean(losses)), "accuracy": float(np.mean((p >= 0.5) == y)),
              "auc": roc_auc_score_np(y, p), "auc_by_negative": {}}
    for mode, name in enumerate(NEGATIVE_TYPES):
        keep = (g == -1) | (g == mode)
        result["auc_by_negative"][name] = roc_auc_score_np(y[keep], p[keep])
    return result


def train_songfit(data_dirs, out_path, epochs=30, steps_per_epoch=300, batch_size=32,
                  lr=3e-4, hidden=256, layers=6, device=None, seed=42, log=print,
                  deadline=None):
    device = resolve_device(device)
    torch.manual_seed(seed)
    maps = build_placement_maps(data_dirs, log=log)
    train_maps = [m for m in maps if not is_validation(m.song)]
    val_maps = [m for m in maps if is_validation(m.song)]
    if not train_maps or not val_maps:
        raise ValueError("song-fit training requires train and validation songs")
    train_sampler, val_sampler = (SongFitSampler(train_maps, seed=seed),
                                  SongFitSampler(val_maps, seed=seed + 1))
    model = CriticNet(features=CRITIC_FEATURES, hidden=hidden, layers=layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_name(out_path.stem + ".best.pt")
    last_path = out_path.with_name(out_path.stem + ".last.pt")
    start, best = 0, -1.0
    if last_path.exists():
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, best = int(state["epoch"]), float(state["best_auc"])
            log(f"resuming song-fit at epoch {start}")
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            log(f"song-fit resume ignored: {exc}")
    log(f"song-fit data: {len(train_maps)} train maps, {len(val_maps)} validation maps; device={device}")
    for epoch in range(start + 1, epochs + 1):
        if deadline is not None and time.time() >= deadline:
            log("song-fit time budget reached at an epoch boundary")
            break
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            raw = train_sampler.batch(batch_size)
            x = torch.as_tensor(raw["x"], device=device)
            mask = torch.as_tensor(raw["mask"], device=device)
            y = torch.as_tensor(raw["label"], device=device)
            logits = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _metrics(model, val_sampler, device, batches=6)
        log(f"song-fit epoch {epoch}/{epochs}: loss={np.mean(losses):.4f}, val AUC={metrics['auc']:.4f}; "
            + ", ".join(f"{key}={value:.4f}" for key, value in metrics["auc_by_negative"].items()))
        if metrics["auc"] > best:
            best = metrics["auc"]
            save_critic(model, best_path)
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            checkpoint["kind"] = "songfit"
            checkpoint["metrics"] = metrics
            torch.save(checkpoint, out_path)
        tmp = last_path.with_suffix(last_path.suffix + ".tmp")
        torch.save({"epoch": epoch, "best_auc": best,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict()}, tmp)
        tmp.replace(last_path)
    return out_path


class SongFitScorer:
    """Blend a human-likeness critic with map-to-audio alignment."""

    def __init__(self, critic=None, songfit=None, critic_weight=0.6):
        self.critic, self.songfit = critic, songfit
        self.critic_weight = float(np.clip(critic_weight, 0.0, 1.0))

    @torch.no_grad()
    def score(self, x, mask=None):
        if self.critic is None:
            return self.songfit.score(x, mask)
        if self.songfit is None:
            return self.critic.score(x, mask)
        return self.critic_weight * self.critic.score(x, mask) + (1.0 - self.critic_weight) * self.songfit.score(x, mask)


def load_songfit(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CriticNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _smoke(path):
    from .critic_data import CRITIC_WINDOW
    model = CriticNet(hidden=32, layers=1, heads=4, ffn_dim=64, window=CRITIC_WINDOW)
    x = torch.randn(4, CRITIC_WINDOW, CRITIC_FEATURES)
    mask = torch.ones(4, CRITIC_WINDOW)
    y = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    loss = F.binary_cross_entropy_with_logits(model(x, mask), y)
    loss.backward()
    save_critic(model, path)
    restored = load_songfit(path)
    if not torch.isfinite(loss) or restored.score(x, mask).shape != (4,):
        raise RuntimeError("song-fit smoke check failed")
    print(f"song-fit smoke passed: loss={float(loss.detach()):.4f}; checkpoint reload ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="*")
    parser.add_argument("--out", default="songfit.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.out)
    else:
        train_songfit(args.data, args.out, epochs=args.epochs, steps_per_epoch=args.steps)


if __name__ == "__main__":
    main()
