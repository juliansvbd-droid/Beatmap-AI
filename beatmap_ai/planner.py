"""Bidirectional full-song pre-planner for star, style and section recommendations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .planner_data import (PLANNER_FEATURES, SECTION_TYPES, STAR_CLASSES,
                           audio_section_features, build_planner_examples, split_examples)
from .sequence_data import TAGS
from .train import resolve_device


class PlannerNet(nn.Module):
    def __init__(self, features: int = PLANNER_FEATURES, hidden: int = 128,
                 styles: int = len(TAGS), layers: int = 4, heads: int = 4,
                 ffn_multiplier: int = 2):
        super().__init__()
        self.config = {"features": features, "hidden": hidden, "styles": styles,
                       "layers": layers, "heads": heads, "ffn_multiplier": ffn_multiplier}
        self.input = nn.Sequential(nn.Linear(features + 1, hidden), nn.GELU(), nn.LayerNorm(hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * ffn_multiplier,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.section_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8))
        self.type_head = nn.Linear(hidden, len(SECTION_TYPES))
        self.boundary_head = nn.Linear(hidden, 1)
        self.song_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.stars_head = nn.Linear(hidden, STAR_CLASSES)
        self.max_stars_head = nn.Linear(hidden, 1)
        self.styles_head = nn.Linear(hidden, styles)

    def forward(self, x, stars_condition, mask=None):
        condition = stars_condition.reshape(-1, 1, 1).expand(-1, x.shape[1], 1) / 7.0
        h = self.input(torch.cat([x, condition], dim=-1))
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, h.shape[-1], 2, device=x.device, dtype=x.dtype)
                                * (-math.log(10000.0) / h.shape[-1]))
        encoding = torch.zeros((x.shape[1], h.shape[-1]), device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:encoding[:, 1::2].shape[1]])
        h = self.encoder(h + encoding[None],
                         src_key_padding_mask=(mask <= 0) if mask is not None else None)
        h = self.norm(h)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1)
            pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        song = self.song_head(pooled)
        return {"section": self.section_head(h).sigmoid(), "type": self.type_head(h),
                "boundary": self.boundary_head(h)[..., 0], "stars": self.stars_head(song),
                "max_stars": self.max_stars_head(song)[..., 0], "styles": self.styles_head(song)}


def save_planner(model: PlannerNet, path, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "planner", "config": model.config,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": metrics or {}}, path)


def load_planner(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PlannerNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def _collate(examples, indices, target_device):
    batch = [examples[int(i)] for i in indices]
    width = max(len(ex.features) for ex in batch)
    x = np.zeros((len(batch), width, PLANNER_FEATURES), dtype=np.float32)
    section = np.zeros((len(batch), width, 8), dtype=np.float32)
    kinds = np.zeros((len(batch), width), dtype=np.int64)
    boundary = np.zeros((len(batch), width), dtype=np.float32)
    mask = np.zeros((len(batch), width), dtype=np.float32)
    for i, ex in enumerate(batch):
        n = len(ex.features)
        x[i, :n], section[i, :n], kinds[i, :n] = ex.features, ex.controls, ex.section_type
        boundary[i, :n], mask[i, :n] = ex.boundary, 1.0
    return {
        "x": torch.as_tensor(x, device=target_device),
        "section": torch.as_tensor(section, device=target_device),
        "type": torch.as_tensor(kinds, device=target_device),
        "boundary": torch.as_tensor(boundary, device=target_device),
        "mask": torch.as_tensor(mask, device=target_device),
        "stars": torch.as_tensor(np.stack([ex.stars for ex in batch]), device=target_device),
        "max_stars": torch.as_tensor([ex.max_stars for ex in batch], device=target_device),
        "styles": torch.as_tensor(np.stack([ex.styles for ex in batch]), device=target_device),
        "styles_known": torch.as_tensor([ex.styles_known for ex in batch], device=target_device),
        "condition": torch.as_tensor([ex.max_stars for ex in batch], device=target_device),
    }


def _loss(model, batch):
    out = model(batch["x"], batch["condition"], batch["mask"])
    mask = batch["mask"]
    section = F.smooth_l1_loss(out["section"], batch["section"], reduction="none").mean(-1)
    section = (section * mask).sum() / mask.sum().clamp(min=1.0)
    type_loss = F.cross_entropy(out["type"].flatten(0, 1), batch["type"].flatten(), reduction="none")
    type_loss = (type_loss.view_as(mask) * mask).sum() / mask.sum().clamp(min=1.0)
    boundary = F.binary_cross_entropy_with_logits(out["boundary"], batch["boundary"], reduction="none")
    boundary = (boundary * mask).sum() / mask.sum().clamp(min=1.0)
    stars = -(batch["stars"] * F.log_softmax(out["stars"], dim=-1)).sum(-1).mean()
    style_values = F.binary_cross_entropy_with_logits(out["styles"], batch["styles"], reduction="none").mean(-1)
    known = batch["styles_known"].float()
    styles = (style_values * known).sum() / known.sum().clamp(min=1.0)
    max_stars = F.smooth_l1_loss(out["max_stars"], batch["max_stars"])
    total = stars + 0.4 * styles + 0.8 * section + 0.2 * type_loss + 0.2 * boundary + 0.5 * max_stars
    return total, {"stars": stars.detach(), "styles": styles.detach(), "section": section.detach(),
                   "type": type_loss.detach(), "boundary": boundary.detach(),
                   "max_stars": max_stars.detach()}


def _validation_metrics(model, examples, device, batch_size=16):
    """Measure full-song and section quality on held-out songs."""
    model.eval()
    levels = torch.arange(0.5, 7.01, 0.5, device=device)
    totals = {key: 0.0 for key in ("main_tp", "main_fp", "main_fn", "kiai_tp", "kiai_fp", "kiai_fn",
                                   "star_abs", "max_star_abs", "section_abs", "section_count",
                                   "boundary_tp", "boundary_fp", "boundary_fn")}
    with torch.no_grad():
        for at in range(0, len(examples), batch_size):
            batch = _collate(examples, np.arange(at, min(at + batch_size, len(examples))), device)
            output = model(batch["x"], batch["condition"], batch["mask"])
            valid = batch["mask"] > 0
            predicted_type = output["type"].argmax(-1)
            main_pred = (predicted_type == SECTION_TYPES.index("main")) & valid
            main_target = (batch["type"] == SECTION_TYPES.index("main")) & valid
            kiai_target = (batch["section"][..., 7] >= 0.5) & valid
            kiai_pred = (output["section"][..., 7] >= 0.5) & valid
            totals["main_tp"] += float((main_pred & main_target).sum())
            totals["main_fp"] += float((main_pred & ~main_target & valid).sum())
            totals["main_fn"] += float((~main_pred & main_target).sum())
            totals["kiai_tp"] += float((kiai_pred & kiai_target).sum())
            totals["kiai_fp"] += float((kiai_pred & ~kiai_target & valid).sum())
            totals["kiai_fn"] += float((~kiai_pred & kiai_target).sum())
            star_prediction = (torch.softmax(output["stars"], dim=-1) * levels).sum(-1)
            star_target = (batch["stars"] * levels).sum(-1)
            totals["star_abs"] += float((star_prediction - star_target).abs().sum())
            totals["max_star_abs"] += float((output["max_stars"] - batch["max_stars"]).abs().sum())
            totals["section_abs"] += float(((output["section"] - batch["section"]).abs()
                                             * valid.unsqueeze(-1)).sum())
            totals["section_count"] += float(valid.sum()) * batch["section"].shape[-1]
            boundary_pred = (torch.sigmoid(output["boundary"]) >= 0.5) & valid
            boundary_target = (batch["boundary"] >= 0.5) & valid
            totals["boundary_tp"] += float((boundary_pred & boundary_target).sum())
            totals["boundary_fp"] += float((boundary_pred & ~boundary_target & valid).sum())
            totals["boundary_fn"] += float((~boundary_pred & boundary_target).sum())
    song_count = max(len(examples), 1)
    main_precision = totals["main_tp"] / max(totals["main_tp"] + totals["main_fp"], 1.0)
    main_recall = totals["main_tp"] / max(totals["main_tp"] + totals["main_fn"], 1.0)
    main_f1 = 2 * main_precision * main_recall / max(main_precision + main_recall, 1e-9)
    kiai_precision = totals["kiai_tp"] / max(totals["kiai_tp"] + totals["kiai_fp"], 1.0)
    kiai_recall = totals["kiai_tp"] / max(totals["kiai_tp"] + totals["kiai_fn"], 1.0)
    kiai_f1 = 2 * kiai_precision * kiai_recall / max(kiai_precision + kiai_recall, 1e-9)
    boundary_precision = totals["boundary_tp"] / max(totals["boundary_tp"] + totals["boundary_fp"], 1.0)
    boundary_recall = totals["boundary_tp"] / max(totals["boundary_tp"] + totals["boundary_fn"], 1.0)
    boundary_f1 = 2 * boundary_precision * boundary_recall / max(boundary_precision + boundary_recall, 1e-9)
    result = {
        "main_precision": main_precision, "main_recall": main_recall, "main_f1": main_f1,
        "kiai_precision": kiai_precision, "kiai_recall": kiai_recall, "kiai_f1": kiai_f1,
        "star_distribution_mae": totals["star_abs"] / song_count,
        "max_stars_mae": totals["max_star_abs"] / song_count,
        "section_controls_mae": totals["section_abs"] / max(totals["section_count"], 1.0),
        "boundary_f1": boundary_f1,
    }
    result["selection_score"] = (0.25 * main_f1 + 0.25 * kiai_f1
                                 + 0.20 / (1.0 + result["max_stars_mae"])
                                 + 0.15 / (1.0 + result["star_distribution_mae"])
                                 + 0.15 / (1.0 + result["section_controls_mae"]))
    model.train()
    return result


def train_planner(data_dirs, tag_files, out_path, tagger_path=None, epochs=30, batch_size=16,
                  lr=3e-4, hidden=128, device=None, seed=42, log=print, deadline=None,
                  examples=None):
    device = resolve_device(device)
    torch.manual_seed(seed)
    if examples is None:
        examples = build_planner_examples(data_dirs, tag_files, tagger_path=tagger_path, log=log)
    train, validation = split_examples(examples)
    if not train or not validation:
        raise ValueError("planner needs train and held-out validation songs")
    model = PlannerNet(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out_path = Path(out_path)
    best_path = out_path.with_name(out_path.stem + ".best.pt")
    last_path = out_path.with_name(out_path.stem + ".last.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start, best, best_score = 0, float("inf"), -1.0
    if last_path.exists():
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, best = int(state["epoch"]), float(state["best"])
            best_score = float(state.get("best_score", -1.0))
            log(f"resuming planner at epoch {start}")
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            log(f"planner resume ignored: {exc}")
    rng = np.random.default_rng(seed)
    log(f"planner data: {len(train)} train songs, {len(validation)} validation songs; device={device}")
    for epoch in range(start + 1, epochs + 1):
        if deadline is not None and time.time() >= deadline:
            log("planner time budget reached at an epoch boundary")
            break
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for at in range(0, len(order), batch_size):
            batch = _collate(train, order[at:at + batch_size], device)
            loss, _ = _loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for at in range(0, len(validation), batch_size):
                batch = _collate(validation, np.arange(at, min(at + batch_size, len(validation))), device)
                vals.append(float(_loss(model, batch)[0].cpu()))
        val = float(np.mean(vals))
        quality = _validation_metrics(model, validation, device, batch_size=batch_size)
        log(f"planner epoch {epoch}/{epochs}: train={np.mean(losses):.4f}, validation={val:.4f}; "
            f"main P/R={quality['main_precision']:.3f}/{quality['main_recall']:.3f}, "
            f"Kiai P/R={quality['kiai_precision']:.3f}/{quality['kiai_recall']:.3f}, "
            f"stars MAE={quality['star_distribution_mae']:.3f}/"
            f"{quality['max_stars_mae']:.3f}, sections MAE={quality['section_controls_mae']:.3f}, "
            f"boundary F1={quality['boundary_f1']:.3f}")
        metrics = {"val_loss": val, "train_loss": float(np.mean(losses)), **quality}
        if quality["selection_score"] > best_score:
            best, best_score = val, quality["selection_score"]
            save_planner(model, best_path, metrics)
            save_planner(model, out_path, metrics)
        tmp = last_path.with_suffix(last_path.suffix + ".tmp")
        torch.save({"epoch": epoch, "best": best, "best_score": best_score,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict()}, tmp)
        tmp.replace(last_path)
    return out_path


@torch.no_grad()
def plan_song(model: PlannerNet, mel: np.ndarray, beat_length: float, stars: float):
    """Return variable-length musical sections, local controls and whole-song advice."""
    features, intensity = audio_section_features(mel, beat_length)
    device = next(model.parameters()).device
    x = torch.as_tensor(features[None], dtype=torch.float32, device=device)
    condition = torch.as_tensor([stars], dtype=torch.float32, device=device)
    out = model(x, condition)
    controls = out["section"][0].cpu().numpy()
    types = out["type"][0].argmax(-1).cpu().numpy()
    boundaries = torch.sigmoid(out["boundary"][0]).cpu().numpy()
    indices = [0] + [i for i in range(1, len(features)) if boundaries[i] >= 0.55] + [len(features)]
    chunk_ms = beat_length * 16.0
    sections = []
    for a, b in zip(indices, indices[1:]):
        if b <= a:
            continue
        kind = int(np.bincount(types[a:b], minlength=len(SECTION_TYPES)).argmax())
        sections.append({"start": a * chunk_ms, "end": b * chunk_ms,
                         "type": SECTION_TYPES[kind], "intensity": float(np.clip(np.mean(controls[a:b, 0]), 0, 1)),
                         "controls": controls[a:b].mean(axis=0).tolist()})
    star_probs = torch.softmax(out["stars"][0], dim=-1).cpu().numpy()
    style_probs = torch.sigmoid(out["styles"][0]).cpu().numpy()
    max_stars = float(out["max_stars"][0].clamp(0.5, 7.0).cpu())
    return {"max_stars": max_stars, "star_distribution": star_probs,
            "styles": dict(zip(TAGS, style_probs.tolist())), "sections": sections}


def _smoke(path, device="cpu"):
    device = torch.device(resolve_device(device))
    model = PlannerNet(hidden=16).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 12, PLANNER_FEATURES, device=device)
    cond = torch.tensor([3.0, 5.0], device=device)
    target = torch.rand(2, 12, 8, device=device)
    out = model(x, cond)
    assert out["section"].shape == (2, 12, 8)
    assert out["styles"].shape[-1] == len(TAGS)
    started = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, cond)["section"] - target).square().mean()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = (time.perf_counter() - started) / 3
    save_planner(model, path)
    restored = load_planner(path, device=device)
    with torch.no_grad():
        if restored(x, cond)["section"].shape != (2, 12, 8):
            raise RuntimeError("planner checkpoint reload failed")
    print(f"planner smoke passed: Transformer output/checkpoint ok; {step_seconds:.4f} sec/step on {device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="*")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--tagger")
    parser.add_argument("--out", default="planner.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, choices=(128, 256), default=128)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.out, args.device or "cpu")
    else:
        train_planner(args.data, args.tags, args.out, args.tagger, args.epochs,
                      hidden=args.hidden, device=args.device)


if __name__ == "__main__":
    main()
