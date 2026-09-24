"""Training loop for the rhythm model."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audio import FPS
from .dataset import ChunkSampler, MapExample, build_examples, is_validation, spread
from .model import BeatmapNet, beat_phase_features, load_checkpoint, save_checkpoint


def onset_f1(probs: np.ndarray, true_frames: np.ndarray, threshold: float, tolerance: int = 2) -> float:
    """F1 of peak-picked note predictions against true note frames (± ``tolerance`` frames)."""
    peaks = np.flatnonzero(
        (probs >= threshold)
        & (probs >= np.roll(probs, 1))
        & (probs > np.roll(probs, -1))
    )
    truth = np.unique(true_frames)
    if len(peaks) == 0 or len(truth) == 0:
        return 0.0
    matched_truth = np.zeros(len(truth), dtype=bool)
    hits = 0
    for p in peaks:
        j = np.searchsorted(truth, p - tolerance)
        while j < len(truth) and truth[j] <= p + tolerance:
            if not matched_truth[j]:
                matched_truth[j] = True
                hits += 1
                break
            j += 1
    precision, recall = hits / len(peaks), hits / len(truth)
    return 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate(model: BeatmapNet, examples: list[MapExample], device: str) -> tuple[float, float]:
    """Return (best F1, threshold achieving it) over the validation maps."""
    model.eval()
    thresholds = np.linspace(0.1, 0.9, 17)
    scores = np.zeros(len(thresholds))
    for ex in examples:
        mel = torch.from_numpy(np.load(ex.mel_path).astype(np.float32)).unsqueeze(0).to(device)
        beat = torch.from_numpy(beat_phase_features(ex.n_frames, ex.grid)).unsqueeze(0).to(device)
        dens = torch.tensor([ex.density], dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(mel, beat, dens))[0, :, 0].cpu().numpy()
        scores += [onset_f1(probs, ex.note_frames, th) for th in thresholds]
    scores /= max(len(examples), 1)
    i = int(np.argmax(scores))
    return float(scores[i]), float(thresholds[i])


def resolve_device(device: str | None):
    """Default to the GPU when there is one. NVIDIA (CUDA) and AMD (ROCm) builds of
    PyTorch both show up as "cuda"; "directml" selects torch-directml on Windows."""
    if device == "directml":
        import torch_directml
        return torch_directml.device()
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train(
    data_dir: str | Path,
    out_path: str | Path,
    epochs: int = 30,
    steps_per_epoch: int = 200,
    batch_size: int = 16,
    chunk_seconds: float = 12.0,
    lr: float = 1e-3,
    hidden: int = 128,
    val_fraction: float = 0.1,
    max_val_maps: int = 40,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    workers: int = 1,
    init: str | Path | None = None,
    seed: int = 0,
    log=partial(print, flush=True),
) -> Path:
    device = resolve_device(device)
    torch.manual_seed(seed)
    cache_dir = Path(cache_dir) if cache_dir else Path(data_dir) / ".beatmap_ai_cache"
    examples = build_examples(data_dir, cache_dir, workers=workers, log=log)
    if not examples:
        raise SystemExit(f"no usable osu!standard beatmaps found under {data_dir}")

    # Split by song so validation measures generalisation to unseen audio.
    songs = {ex.mel_path for ex in examples}
    train_ex = [ex for ex in examples if not is_validation(ex, val_fraction)]
    val_ex = [ex for ex in examples if is_validation(ex, val_fraction)]
    if not train_ex:
        train_ex = val_ex
    val_ex = spread(val_ex or train_ex, max_val_maps)
    log(f"{len(examples)} difficulties from {len(songs)} songs "
        f"({len(train_ex)} train, {len(val_ex)} validation); device={device}")

    sampler = ChunkSampler(train_ex, int(chunk_seconds * FPS), seed=seed)
    if init is not None:  # Continue from an earlier checkpoint (its width wins).
        model, _ = load_checkpoint(init)
        model = model.to(device)
        log(f"starting from {init}")
    else:
        model = BeatmapNet(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch)

    out_path = Path(out_path)
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            batch = {k: torch.from_numpy(v).to(device) for k, v in sampler.batch(batch_size).items()}
            logits = model(batch["mel"], batch["beat"], batch["density"])
            note_loss = F.binary_cross_entropy_with_logits(logits[..., 0], batch["note"])
            slider_loss = (
                F.binary_cross_entropy_with_logits(logits[..., 1], batch["slider"], reduction="none")
                * batch["mask"]
            ).sum() / batch["mask"].sum().clamp(min=1.0)
            loss = note_loss + 0.1 * slider_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        f1, threshold = evaluate(model, val_ex, device)
        log(f"epoch {epoch:3d}  loss {total / steps_per_epoch:.4f}  "
            f"val note F1 {f1:.3f} @ threshold {threshold:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, out_path, threshold)
    log(f"saved best model (F1 {best_f1:.3f}) to {out_path}")
    return out_path
