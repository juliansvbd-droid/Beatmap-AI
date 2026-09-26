"""Bidirectional Transformer Critic for osu!standard beatmaps.

Classifies whether a 64-object sequence is human-mapped or AI-generated.
During generation, used to pick the most natural placement candidate (Best-of-N).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .critic_data import (CRITIC_FEATURES, CRITIC_WINDOW, CriticSampler,
                          critic_batch_stream, load_critic_dataset, share_critic_maps)


class CriticBlock(nn.Module):
    """Pre-LN Transformer Encoder layer without causal masking."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        # key_padding_mask is True for padded positions in nn.MultiheadAttention
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CriticNet(nn.Module):
    """Bidirectional Transformer Encoder scoring human likeness of beatmap sections."""

    def __init__(
        self,
        features: int = CRITIC_FEATURES,
        hidden: int = 256,
        layers: int = 6,
        heads: int = 8,
        ffn_dim: int = 1024,
        window: int = CRITIC_WINDOW,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(
            features=features,
            hidden=hidden,
            layers=layers,
            heads=heads,
            ffn_dim=ffn_dim,
            window=window,
            dropout=dropout,
        )
        self.inp = nn.Sequential(
            nn.Linear(features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.pos = nn.Parameter(torch.zeros(window, hidden))
        nn.init.normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList(
            CriticBlock(hidden, heads, ffn_dim, dropout=dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, T, D), mask (B, T) where 1=valid, 0=padding -> logits (B, 1)."""
        t = x.shape[1]
        h = self.inp(x) + self.pos[:t]

        # In PyTorch MultiheadAttention, key_padding_mask is True for positions to IGNORE
        pad_mask = (mask <= 0.5) if mask is not None else None

        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)

        h = self.norm(h)

        if mask is not None:
            weights = mask.unsqueeze(-1)
            pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        logits = self.head(pooled)  # (B, 1)
        return logits

    @torch.no_grad()
    def score(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Probability (B,) that the sequences are human-made."""
        self.eval()
        logits = self.forward(x, mask)
        return torch.sigmoid(logits).squeeze(-1)


def save_critic(model: CriticNet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.config, "kind": "critic", "state_dict": state}, path)


def load_critic(path: str | Path, device="cpu") -> CriticNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CriticNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def is_critic_checkpoint(path: str | Path) -> bool:
    try:
        return torch.load(path, map_location="cpu", weights_only=True).get("kind") == "critic"
    except Exception:
        return False


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U statistic for ROC AUC calculation."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Combine and rank
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_ranks = ranks[:n_pos]
    u = pos_ranks.sum() - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def evaluate_critic(model: CriticNet, batches: list[dict[str, torch.Tensor]], device) -> dict[str, float]:
    """Score model on validation batches, returning loss, accuracy, and ROC AUC."""
    model.eval()
    all_logits = []
    all_labels = []
    losses = []

    for b in batches:
        x = b["x"].to(device).float()
        mask = b["mask"].to(device).float()
        label = b["label"].to(device).float()

        logits = model(x, mask)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        losses.append(loss.item())

        all_logits.append(logits.cpu().numpy().ravel())
        all_labels.append(label.cpu().numpy().ravel())

    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    preds = (probs_arr >= 0.5).astype(float)
    acc = float(np.mean(preds == labels_arr))
    auc = roc_auc_score_np(labels_arr, probs_arr)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": acc,
        "auc": auc,
    }


def train_critic(
    human_data_dirs: list[str | Path],
    negative_manifest: str | Path,
    out_path: str | Path = "critic.pt",
    cache_dir: str | Path | None = None,
    epochs: int = 15,
    steps_per_epoch: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    hidden: int = 256,
    layers: int = 6,
    heads: int = 8,
    ffn_dim: int = 1024,
    window: int = CRITIC_WINDOW,
    device: str | None = None,
    seed: int = 42,
    workers: int = 4,
    log=partial(print, flush=True),
) -> Path:
    """Train the critic on human and negative beatmaps."""
    from .train import resolve_device
    device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_maps, val_maps = load_critic_dataset(human_data_dirs, negative_manifest, log=log)
    if not train_maps:
        raise ValueError("No training maps found for critic!")
    if not val_maps:
        raise ValueError("No validation maps found for critic!")

    log("Preparing validation batches...")
    val_sampler = CriticSampler(val_maps, window=window, seed=seed + 1, augment=False)
    val_batches = [
        {k: torch.from_numpy(v) for k, v in val_sampler.batch(batch_size).items()}
        for _ in range(16)
    ]

    cache_path = Path(cache_dir or out_path.parent) / "critic_train_objects.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Sharing {len(train_maps)} training maps to {cache_path}...")
    shared = share_critic_maps(train_maps, cache_path)
    del train_maps

    model = CriticNet(
        features=CRITIC_FEATURES,
        hidden=hidden,
        layers=layers,
        heads=heads,
        ffn_dim=ffn_dim,
        window=window,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Critic model: {param_count:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    loader = critic_batch_stream(shared, window, batch_size, seed, workers=workers)
    log("Training with 50% per-window dropout on slider feature columns 9:12.")

    init_val = evaluate_critic(model, val_batches, device)
    log(f"Initial val loss: {init_val['loss']:.4f}, acc: {init_val['accuracy']:.3%}, auc: {init_val['auc']:.4f}")

    best_auc = init_val["auc"]
    best_loss = init_val["loss"]

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for _ in range(steps_per_epoch):
            batch_data = next(loader)
            x = torch.from_numpy(batch_data["x"]).to(device).float()
            mask = torch.from_numpy(batch_data["mask"]).to(device).float()
            label = torch.from_numpy(batch_data["label"]).to(device).float()

            # Slider timing/length is matched within each pair; dropout prevents
            # the classifier from depending on slider fields as a shortcut.
            drop_slider = torch.rand((x.shape[0], 1, 1), device=device) < 0.5
            x[:, :, 9:12] = x[:, :, 9:12].masked_fill(drop_slider, 0.0)

            logits = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        val_metrics = evaluate_critic(model, val_batches, device)
        log(f"Epoch {epoch:2d}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3%} | "
            f"Val AUC: {val_metrics['auc']:.4f}")

        # Save best model based on validation AUC
        if val_metrics["auc"] > best_auc or (val_metrics["auc"] == best_auc and val_metrics["loss"] < best_loss):
            best_auc = val_metrics["auc"]
            best_loss = val_metrics["loss"]
            save_critic(model, out_path)
            log(f"  -> New best model saved (AUC: {best_auc:.4f})")

    if not out_path.exists():
        save_critic(model, out_path)

    # Diagnostics must describe the selected checkpoint, not necessarily the last epoch.
    model = load_critic(out_path, device=device)
    clean_metrics = evaluate_critic(model, val_batches, device)
    log(f"Selected checkpoint clean validation: Acc {clean_metrics['accuracy']:.3%}, "
        f"AUC {clean_metrics['auc']:.4f}")

    log("Running shortcut resistance test (adding spatial jitter to validation inputs)...")
    jittered_batches = []
    for b in val_batches:
        x_jit = b["x"].clone()
        noise = torch.randn_like(x_jit[:, :, 12:23]) * 0.015
        x_jit[:, :, 12:23] += noise
        jittered_batches.append({"x": x_jit, "mask": b["mask"], "label": b["label"]})
    jit_metrics = evaluate_critic(model, jittered_batches, device)
    log(f"Shortcut test (spatial jitter +-5px): Val Acc: {jit_metrics['accuracy']:.3%}, "
        f"Val AUC: {jit_metrics['auc']:.4f} (Clean Val Acc: {clean_metrics['accuracy']:.3%})")

    log("Running audio sensitivity tests (scrambled and zeroed audio)...")
    all_audio = torch.cat([b["x"][:, :, 23:104] for b in val_batches], dim=0)
    shuffled_audio = all_audio[torch.randperm(all_audio.shape[0])]
    shuffled_audio_batches = []
    at = 0
    for b in val_batches:
        x_aud = b["x"].clone()
        x_aud[:, :, 23:104] = shuffled_audio[at:at + len(x_aud)]
        at += len(x_aud)
        shuffled_audio_batches.append({"x": x_aud, "mask": b["mask"], "label": b["label"]})
    aud_metrics = evaluate_critic(model, shuffled_audio_batches, device)
    log(f"Audio test (shuffled audio):   Val Acc: {aud_metrics['accuracy']:.3%}, "
        f"Val AUC: {aud_metrics['auc']:.4f} (Clean Val Acc: {clean_metrics['accuracy']:.3%})")

    zeroed_audio_batches = []
    for b in val_batches:
        x_zero = b["x"].clone()
        x_zero[:, :, 23:104] = 0.0
        zeroed_audio_batches.append({"x": x_zero, "mask": b["mask"], "label": b["label"]})
    zero_metrics = evaluate_critic(model, zeroed_audio_batches, device)
    log(f"Audio test (zeroed audio):     Val Acc: {zero_metrics['accuracy']:.3%}, "
        f"Val AUC: {zero_metrics['auc']:.4f} (Clean Val Acc: {clean_metrics['accuracy']:.3%})")

    log("Running feature ablation tests (slider features vs placement features)...")
    no_slider_batches = []
    no_position_batches = []
    for b in val_batches:
        x_slider = b["x"].clone()
        x_slider[:, :, 9:12] = 0.0
        no_slider_batches.append({"x": x_slider, "mask": b["mask"], "label": b["label"]})

        x_position = b["x"].clone()
        x_position[:, :, 12:23] = 0.0
        no_position_batches.append({"x": x_position, "mask": b["mask"], "label": b["label"]})

    slider_metrics = evaluate_critic(model, no_slider_batches, device)
    position_metrics = evaluate_critic(model, no_position_batches, device)
    log(f"Feature test (slider columns 9:12 zeroed): Val Acc {slider_metrics['accuracy']:.3%}, "
        f"AUC {slider_metrics['auc']:.4f} (Clean Acc {clean_metrics['accuracy']:.3%}, "
        f"AUC {clean_metrics['auc']:.4f})")
    log(f"Feature test (position columns 12:23 zeroed): Val Acc {position_metrics['accuracy']:.3%}, "
        f"AUC {position_metrics['auc']:.4f} (Clean Acc {clean_metrics['accuracy']:.3%}, "
        f"AUC {clean_metrics['auc']:.4f})")

    log(f"Training complete. Best checkpoint saved at {out_path} (Best Val AUC: {best_auc:.4f})")
    return out_path
