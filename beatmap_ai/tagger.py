"""Map-level style tagger; feature extraction is NumPy-only for worker safety."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import is_validation
from .placement_data import (BEAT, END, EX, EY, KIND, SLIDER, T, X, Y)
from .sequence_data import TAGS, build_sequence_maps
from .style import star_rating

TAGGER_FEATURES = 64


def _mean(values) -> float:
    """Mean that is 0 for an empty selection (a NaN would poison the whole training)."""
    return float(np.mean(values)) if len(values) else 0.0


def tagger_features(objects: np.ndarray, stars: float, cs: float = 4.0) -> np.ndarray:
    """Compact whole-map summary with eight local sections and global context."""
    stars = float(stars) if np.isfinite(stars) else 4.0
    n = len(objects)
    if n == 0:
        return np.zeros(TAGGER_FEATURES, dtype=np.float32)
    times = objects[:, T]
    duration = max((times[-1] - times[0]) / 1000.0, 1.0)
    global_values = [stars / 7.0, np.log1p(duration) / 10.0, np.log1p(n / duration) / 4.0]
    gaps = np.diff(times) / np.maximum(objects[:-1, BEAT], 1.0)
    vectors = np.stack([objects[1:, X] - objects[:-1, EX],
                        objects[1:, Y] - objects[:-1, EY]], axis=1) if n > 1 else np.zeros((0, 2))
    distances = np.linalg.norm(vectors, axis=1)
    turns = np.zeros(max(n - 2, 0), dtype=np.float32)
    if len(turns):
        a, b = vectors[:-1], vectors[1:]
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide((a * b).sum(axis=1), norms, out=np.ones_like(norms), where=norms > 1)
        turns[:] = np.arccos(np.clip(cosine, -1, 1)) > np.deg2rad(120)
    if len(gaps):
        global_values.extend(np.quantile(gaps, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
        global_values.extend(np.quantile(distances / 640.0, [0.25, 0.5, 0.75, 0.95]).tolist())
        global_values.extend([float(np.mean(gaps <= 0.5)), float(np.mean(gaps <= 0.25)),
                              float(np.mean(objects[1:, KIND] == SLIDER)),
                              float(np.mean(turns)) if len(turns) else 0.0])
    else:
        global_values.extend([0.0] * 13)
    global_values = np.pad(np.asarray(global_values, dtype=np.float32), (0, 16 - len(global_values)))[:16]
    local = np.zeros((8, 6), dtype=np.float32)
    bins = np.clip(((times - times[0]) / max(times[-1] - times[0], 1.0) * 8).astype(int), 0, 7)
    for k in range(8):
        ids = np.flatnonzero(bins == k)
        if not len(ids):
            continue
        lo, hi = ids[0], ids[-1] + 1
        local[k, 0] = np.log1p(len(ids) / max((times[hi - 1] - times[lo]) / 1000.0, 0.5)) / 4.0
        local[k, 1] = _mean(distances[max(lo - 1, 0):max(hi - 1, 0)]) / 400.0
        local[k, 2] = _mean(objects[lo:hi, KIND] == SLIDER)
        local[k, 3] = _mean(turns[max(lo - 1, 0):max(hi - 2, 0)])
        local[k, 4] = _mean(gaps[max(lo - 1, 0):max(hi - 1, 0)] <= 0.5)
        local[k, 5] = _mean(distances[max(lo - 1, 0):max(hi - 1, 0)] > 0.7 * 640.0)
    features = np.concatenate([global_values, local.ravel()]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class TaggerNet(nn.Module):
    def __init__(self, features: int = TAGGER_FEATURES, tags: int = len(TAGS), hidden: int = 128):
        super().__init__()
        self.config = {"features": features, "tags": tags, "hidden": hidden}
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, tags))

    def forward(self, x):
        return self.net(x)


def save_tagger(model: TaggerNet, path: str | Path, metrics=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "tagger", "config": model.config,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": metrics or {}}, path)


def load_tagger(path: str | Path, device="cpu") -> TaggerNet:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TaggerNet(**state["config"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def tag_probabilities(model: TaggerNet, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(features, dtype=torch.float32,
                                                    device=next(model.parameters()).device))).cpu().numpy()


def _auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    result = np.full(y.shape[1], 0.5, dtype=np.float32)
    for i in range(y.shape[1]):
        pos, neg = p[y[:, i] >= 0.5, i], p[y[:, i] < 0.5, i]
        if len(pos) and len(neg):
            ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
            result[i] = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return result


def train_tagger(data_dirs, tag_files, out_path, epochs=24, batch_size=256, lr=5e-4,
                 device=None, seed=42, log=print, deadline=None):
    from .train import resolve_device
    device = resolve_device(device)
    maps = build_sequence_maps(data_dirs, tag_files, log=log)
    labeled = [m for m in maps if m.tags is not None and m.tags[-1] > 0]
    train = [m for m in labeled if not is_validation(m.song)]
    val = [m for m in labeled if is_validation(m.song)]
    if not train or not val:
        raise ValueError("tagger requires tagged maps in both train and validation splits")
    x_train = np.stack([tagger_features(m.objects, float(m.style.get("stars", 4.0)), m.cs) for m in train])
    y_train = np.nan_to_num(np.stack([m.tags[:-1] for m in train]).astype(np.float32))
    x_val = np.stack([tagger_features(m.objects, float(m.style.get("stars", 4.0)), m.cs) for m in val])
    y_val = np.nan_to_num(np.stack([m.tags[:-1] for m in val]).astype(np.float32))
    low_star = np.asarray([float(m.style.get("stars", 99.0)) < 4.0 for m in val])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = TaggerNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out_path, best_path = Path(out_path), Path(out_path).with_name(Path(out_path).stem + ".best.pt")
    last_path = Path(out_path).with_name(Path(out_path).stem + ".last.pt")
    start, best_auc = 0, -1.0
    if last_path.exists():
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, best_auc = int(state["epoch"]), float(state["best_auc"])
            log(f"resuming tagger at epoch {start}")
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            log(f"tagger resume ignored: {exc}")
    xb = torch.as_tensor(x_train, device=device)
    yb = torch.as_tensor(y_train, device=device)
    xv = torch.as_tensor(x_val, device=device)
    yv = torch.as_tensor(y_val, device=device)
    log(f"tagger data: {len(train)} train, {len(val)} validation, {len(TAGS)} tags; device={device}")
    for epoch in range(start + 1, epochs + 1):
        if deadline is not None and time.time() >= deadline:
            log("tagger time budget reached at an epoch boundary")
            break
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for at in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[at:at + batch_size], device=device)
            loss = F.binary_cross_entropy_with_logits(model(xb[ids]), yb[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(xv)).cpu().numpy()
        auc = _auc(y_val, prob)
        mean_auc = float(auc.mean())
        low_auc = _auc(y_val[low_star], prob[low_star]) if low_star.any() else np.full(len(TAGS), 0.5)
        log(f"tagger epoch {epoch}/{epochs}: loss={np.mean(losses):.4f}, val AUC={mean_auc:.4f}; "
            f"1-4* AUC={float(low_auc.mean()):.4f}")
        metrics = {"mean_auc": mean_auc, "auc_by_tag": dict(zip(TAGS, auc.tolist())),
                   "low_star_auc_by_tag": dict(zip(TAGS, low_auc.tolist()))}
        if mean_auc > best_auc:
            best_auc = mean_auc
            save_tagger(model, best_path, metrics)
            save_tagger(model, out_path, metrics)
        tmp = last_path.with_suffix(last_path.suffix + ".tmp")
        torch.save({"epoch": epoch, "best_auc": best_auc,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict()}, tmp)
        tmp.replace(last_path)
    return out_path


def _smoke(out_path: Path):
    torch.manual_seed(1)
    model = TaggerNet(hidden=16)
    x = torch.randn(8, TAGGER_FEATURES)
    y = torch.rand(8, len(TAGS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = float(F.binary_cross_entropy_with_logits(model(x), y).detach())
    for _ in range(12):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        opt.step()
    after = float(F.binary_cross_entropy_with_logits(model(x), y).detach())
    save_tagger(model, out_path)
    restored = load_tagger(out_path)
    if not after < before or tag_probabilities(restored, x.numpy()).shape != y.shape:
        raise RuntimeError("tagger smoke check failed")
    print(f"tagger smoke passed: loss {before:.4f} -> {after:.4f}; checkpoint reload ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="*")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--out", default="tagger.pt")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke(Path(args.out))
    else:
        train_tagger(args.data, args.tags, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
