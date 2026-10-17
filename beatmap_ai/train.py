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
def evaluate(model: BeatmapNet, examples: list[MapExample], device: str,
             *, figures: bool = False):
    """Return best note F1/threshold, optionally also quarter-link F1."""
    model.eval()
    thresholds = np.linspace(0.1, 0.9, 17)
    scores = np.zeros(len(thresholds))
    figure_thresholds = np.linspace(0.05, 0.95, 19)
    figure_tp = np.zeros(len(figure_thresholds), dtype=int)
    figure_fp = np.zeros(len(figure_thresholds), dtype=int)
    figure_fn = np.zeros(len(figure_thresholds), dtype=int)
    for ex in examples:
        mel = np.load(ex.mel_path).astype(np.float32)
        beat = beat_phase_features(ex.n_frames, ex.grid)
        probs = predict_probs(model, mel, beat, generation_conditions(ex), device=device)
        scores += [onset_f1(probs[:, 0], ex.note_frames, th) for th in thresholds]
        if figures:
            from .rhythm_figures import quarter_link_targets
            truth, mask = quarter_link_targets(ex, 0, ex.n_frames)
            chosen = mask > 0
            predicted = probs[chosen, 4, None] >= figure_thresholds
            actual = (truth[chosen] > 0)[:, None]
            figure_tp += np.sum(predicted & actual, axis=0)
            figure_fp += np.sum(predicted & ~actual, axis=0)
            figure_fn += np.sum(~predicted & actual, axis=0)
    scores /= max(len(examples), 1)
    i = int(np.argmax(scores))
    if figures:
        figure_f1 = 2 * figure_tp / np.maximum(2 * figure_tp + figure_fp + figure_fn, 1)
        j = int(np.argmax(figure_f1))
        return float(scores[i]), float(thresholds[i]), float(figure_f1[j]), float(figure_thresholds[j])
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
    figure_head: bool = False,
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
    sampler = ChunkSampler(train_ex, chunk, seed=seed, figures=figure_head)
    has_styles = any(np.isfinite((ex.style or {}).get("stars", np.nan)) for ex in train_ex)
    rng = np.random.default_rng(seed + 1)
    frozen_base = False
    generation_threshold = None
    if init_model is not None:
        # Continue from a checkpoint: its architecture wins over hidden/arch/layers.
        checkpoint = torch.load(init_model, map_location="cpu", weights_only=True)
        config = dict(checkpoint["config"])
        old_outputs = config.get("outputs", 2)
        if figure_head and old_outputs == 4:
            frozen_base = True
            generation_threshold = float(checkpoint.get("threshold", 0.5))
            config["outputs"] = 5
            model = BeatmapNet(**config)
            original = checkpoint["state_dict"]
            trunk = {name: value for name, value in original.items()
                     if name not in ("head.weight", "head.bias")}
            model.load_state_dict(trunk, strict=False)
            with torch.no_grad():
                model.head.weight[:4].copy_(original["head.weight"])
                model.head.bias[:4].copy_(original["head.bias"])
                model.head.weight[4].zero_()
                model.head.bias[4].fill_(-2.0)
        else:
            model = BeatmapNet(**config)
            model.load_state_dict(checkpoint["state_dict"])
        model.thresholds_by_stars = checkpoint.get("thresholds_by_stars") or {}
        model.figure_threshold = float(checkpoint.get("figure_threshold", 0.5))
        model.to(device)
        if model.window is not None:
            sampler.chunk = model.window
    else:
        conditions = list(CONDITIONS) if arch != "gru" and has_styles else None
        model = BeatmapNet(hidden=hidden, arch=arch, layers=layers, window=chunk,
                           outputs=2 if arch == "gru" else 5 if figure_head else 4,
                           conditions=conditions,
                           cond_stats=condition_stats(train_ex) if conditions else None).to(device)
    if figure_head and model.outputs < 5:
        raise ValueError("quarter-link training requires a five-output Conformer model")
    if frozen_base:
        # An aggregate note F1 can stay flat while individual songs change sharply.
        # Train only the new row so the shipped rhythm remains bit-for-bit intact.
        for name, parameter in model.named_parameters():
            if not name.startswith("head."):
                parameter.requires_grad_(False)
        model.head.weight.register_hook(
            lambda grad: torch.cat((torch.zeros_like(grad[:4]), grad[4:]), dim=0))
        model.head.bias.register_hook(
            lambda grad: torch.cat((torch.zeros_like(grad[:4]), grad[4:]), dim=0))
    sampler.conditions = model.conditions
    log(f"model: {model.arch}, width {model.config['hidden']}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    # DirectML does not implement PyTorch's fused GRU kernel. Keep a GRU on the CPU
    # and train the other layers on the GPU.
    if use_directml and model.arch == "gru":
        model.rnn.to("cpu")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr, weight_decay=0.0 if frozen_base else 1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch)

    out_path = Path(out_path)
    initial = evaluate(model, val_ex, device, figures=figure_head)
    best_f1, best_threshold = initial[:2]
    best_figure_f1 = initial[2] if figure_head else 0.0
    best_figure_threshold = initial[3] if figure_head else 0.5
    best_score = best_f1 + (0.05 * best_figure_f1 if figure_head else 0.0)
    initial_note_f1 = best_f1
    log(f"starting val note F1 {best_f1:.3f} @ threshold {best_threshold:.2f}; "
        f"figure F1 {best_figure_f1:.3f} @ threshold {best_figure_threshold:.2f}" if figure_head else
        f"starting val note F1 {best_f1:.3f} @ threshold {best_threshold:.2f}")
    save_checkpoint(model, out_path, generation_threshold if frozen_base else best_threshold,
                    figure_threshold=best_figure_threshold if figure_head else None)
    for epoch in range(1, epochs + 1):
        # BatchNorm running statistics are part of the base model too.
        model.eval() if frozen_base else model.train()
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
            if figure_head:
                figure_mask = batch["figure_mask"]
                figure_loss = F.binary_cross_entropy_with_logits(
                    logits[..., 4], batch["figure_link"], reduction="none",
                    pos_weight=torch.tensor(3.0, device=device),
                )
                figure_loss = (figure_loss * figure_mask).sum() / figure_mask.sum().clamp(min=1.0)
                loss = loss + 0.2 * figure_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        validation = evaluate(model, val_ex, device, figures=figure_head)
        f1, threshold = validation[:2]
        figure_f1 = validation[2] if figure_head else 0.0
        figure_threshold = validation[3] if figure_head else 0.5
        score = f1 + (0.05 * figure_f1 if figure_head else 0.0)
        log(f"epoch {epoch:3d}  loss {total / steps_per_epoch:.4f}  "
            f"val note F1 {f1:.3f} @ threshold {threshold:.2f}"
            + (f"  quarter-link F1 {figure_f1:.3f} @ threshold {figure_threshold:.2f}"
               if figure_head else ""))
        if score > best_score and (not figure_head or f1 >= initial_note_f1 - 0.01):
            best_score = score
            best_f1 = f1
            best_figure_f1 = figure_f1
            best_figure_threshold = figure_threshold
            best_threshold = threshold
            save_checkpoint(model, out_path, generation_threshold if frozen_base else threshold,
                            figure_threshold=figure_threshold if figure_head else None)
    log(f"saved best model (note F1 {best_f1:.3f}, quarter-link F1 "
        f"{best_figure_f1:.3f}) to {out_path}" if figure_head else
        f"saved best model (F1 {best_f1:.3f}) to {out_path}")
    try:
        report = build_learning_report(
            examples, train_ex, val_ex, best_f1, best_threshold, out_path,
        )
        report_path = save_learning_report(report, out_path.with_suffix(".learning.json"))
        log(f"learning summary saved to {report_path}")
    except OSError as exc:
        log(f"could not save learning summary: {exc}")
    return out_path
