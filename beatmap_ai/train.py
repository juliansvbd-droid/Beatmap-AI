"""Training loop for the rhythm model."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audio import FPS
from .dataset import ChunkSampler, MapExample, build_examples, is_validation, spread
from .insights import build_learning_report, save_learning_report
from .model import BeatmapNet, beat_phase_features, predict_probs, save_checkpoint
from .style import CONDITIONS


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
        mel = np.load(ex.mel_path).astype(np.float32)
        beat = beat_phase_features(ex.n_frames, ex.grid)
        probs = predict_probs(model, mel, beat, generation_conditions(ex), device=device)[:, 0]
        scores += [onset_f1(probs, ex.note_frames, th) for th in thresholds]
    scores /= max(len(examples), 1)
    i = int(np.argmax(scores))
    return float(scores[i]), float(thresholds[i])


def generation_conditions(ex: MapExample) -> dict:
    """What the generator knows when asked for a map like ``ex``: its star rating and
    density, but not its style."""
    return {"density": ex.density, "stars": (ex.style or {}).get("stars", float("nan"))}


def condition_stats(examples: list[MapExample]) -> dict:
    """Percentiles (5, 25, 50, 75, 95) of each style condition per whole-star level, so
    "more jumps" at 4 stars means a lot of jumps for a 4-star map."""
    stats: dict[str, dict[str, list[float]]] = {}
    stars = np.array([(ex.style or {}).get("stars", np.nan) for ex in examples], dtype=float)
    for level in range(0, 11):
        near = [ex for ex, s in zip(examples, stars) if np.isfinite(s) and abs(s - level - 0.5) <= 1.0]
        if len(near) < 20:
            continue
        stats[str(level)] = {}
        for name in CONDITIONS:
            values = np.array([(ex.style or {}).get(name, np.nan) for ex in near], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) >= 20:
                stats[str(level)][name] = np.percentile(values, [5, 25, 50, 75, 95]).round(4).tolist()
    return stats


def resolve_device(device: str | None):
    """Default to the GPU when there is one. NVIDIA (CUDA) and AMD (ROCm) builds of
    PyTorch both show up as "cuda"; "directml" selects torch-directml on Windows."""
    if device == "directml":
        import torch_directml
        return torch_directml.device()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and torch.version.hip and os.name == "nt":
        # MIOpen cannot compile its kernels on Windows (missing C++ headers for hiprtc),
        # so let PyTorch use its own HIP kernels for convolutions and batch norm.
        torch.backends.cudnn.enabled = False
    return device


def augment(batch: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random loudness changes and frequency masking, so the model relies on the
    rhythm of the music rather than on a song's mix or mastering."""
    mel = batch["mel"]
    mel += rng.uniform(-0.3, 0.3, size=(len(mel), 1, 1)).astype(np.float32)
    for m in mel:
        width = int(rng.integers(0, 12))
        start = int(rng.integers(0, m.shape[0] - width + 1))
        m[start:start + width] = -1.0
    return batch


def train(
    data_dir: str | Path | list[str | Path],
    out_path: str | Path,
    epochs: int = 30,
    steps_per_epoch: int = 200,
    batch_size: int = 16,
    chunk_seconds: float = 12.0,
    lr: float = 1e-3,
    hidden: int = 128,
    arch: str = "conformer",
    layers: int = 4,
    val_fraction: float = 0.1,
    max_val_maps: int = 40,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    workers: int = 1,
    seed: int = 0,
    init_model: str | Path | None = None,
    log=partial(print, flush=True),
) -> Path:
    use_directml = device == "directml"
    device = resolve_device(device)
    torch.manual_seed(seed)
    data_dirs = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
    examples = build_examples(data_dirs, cache_dir, workers=workers, log=log)
    if not examples:
        raise SystemExit(f"no usable osu!standard beatmaps found under {data_dir}")

    # Split by song so validation measures generalisation to unseen audio.
    songs = {ex.song for ex in examples}
    train_ex = [ex for ex in examples if not is_validation(ex, val_fraction)]
    val_ex = [ex for ex in examples if is_validation(ex, val_fraction)]
    if not train_ex:
        train_ex = val_ex
    val_ex = spread(val_ex or train_ex, max_val_maps)
    log(f"{len(examples)} difficulties from {len(songs)} songs "
        f"({len(train_ex)} train, {len(val_ex)} validation); device={device}")

    chunk = int(chunk_seconds * FPS)
    sampler = ChunkSampler(train_ex, chunk, seed=seed)
    has_styles = any(np.isfinite((ex.style or {}).get("stars", np.nan)) for ex in train_ex)
    rng = np.random.default_rng(seed + 1)
    if init_model is not None:
        # Continue from a checkpoint: its architecture wins over hidden/arch/layers.
        checkpoint = torch.load(init_model, map_location="cpu", weights_only=True)
        model = BeatmapNet(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        if model.window is not None:
            sampler.chunk = model.window
    else:
        conditions = list(CONDITIONS) if arch != "gru" and has_styles else None
        model = BeatmapNet(hidden=hidden, arch=arch, layers=layers, window=chunk,
                           outputs=2 if arch == "gru" else 4, conditions=conditions,
                           cond_stats=condition_stats(train_ex) if conditions else None).to(device)
    sampler.conditions = model.conditions
    log(f"model: {model.arch}, width {model.config['hidden']}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    # DirectML does not implement PyTorch's fused GRU kernel. Keep a GRU on the CPU
    # and train the other layers on the GPU.
    if use_directml and model.arch == "gru":
        model.rnn.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch)

    out_path = Path(out_path)
    best_f1, best_threshold = evaluate(model, val_ex, device)
    log(f"starting val note F1 {best_f1:.3f} @ threshold {best_threshold:.2f}")
    save_checkpoint(model, out_path, best_threshold)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            batch = augment(sampler.batch(batch_size), rng)
            batch = {k: torch.from_numpy(v).to(device) for k, v in batch.items()}
            logits = model(batch["mel"], batch["beat"], batch["cond"])
            note_loss = F.binary_cross_entropy_with_logits(logits[..., 0], batch["note"])
            slider_loss = (
                F.binary_cross_entropy_with_logits(logits[..., 1], batch["slider"], reduction="none")
                * batch["mask"]
            ).sum() / batch["mask"].sum().clamp(min=1.0)
            loss = note_loss + 0.1 * slider_loss
            if model.outputs >= 4:
                sustain_loss = F.binary_cross_entropy_with_logits(logits[..., 2], batch["sustain"])
                spacing_mask = batch["spacing_mask"]
                spacing_loss = (F.smooth_l1_loss(logits[..., 3], batch["spacing"], reduction="none")
                                * spacing_mask).sum() / spacing_mask.sum().clamp(min=1.0)
                loss = loss + 0.2 * sustain_loss + 0.2 * spacing_loss
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
            best_threshold = threshold
            save_checkpoint(model, out_path, threshold)
    log(f"saved best model (F1 {best_f1:.3f}) to {out_path}")
    try:
        report = build_learning_report(
            examples, train_ex, val_ex, best_f1, best_threshold, out_path,
        )
        report_path = save_learning_report(report, out_path.with_suffix(".learning.json"))
        log(f"learning summary saved to {report_path}")
    except OSError as exc:
        log(f"could not save learning summary: {exc}")
    return out_path
