"""Neural rhythm model.

A convolutional front end reads the log-mel spectrogram and conformer blocks (or, in
older checkpoints, a bidirectional GRU) add context. For every ~11.6 ms audio frame the
model predicts:

* the probability that a hit object starts there,
* the probability that such an object is a slider,
* (newer models) the probability that a slider is being held, which sets slider
  lengths, and
* (newer models) the distance snap mappers would use for an object there: large for
  emphasised notes (jumps), small for calm passages.

The network is conditioned on the beat grid (phase within the beat and the bar) and on
the target note density, so one model can produce every difficulty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audio import N_MELS, AudioFeatures
from .beatgrid import BeatGrid, beat_phase_features  # noqa: F401  (re-exported)
from .style import encode_conditions



class ConformerBlock(nn.Module):
    """Feed-forward, self-attention and depthwise convolution, each with a residual.

    Unlike a GRU, every frame is processed in parallel, so it runs entirely on the GPU
    with DirectML and ROCm on Windows (neither provides a usable fused GRU kernel)."""

    def __init__(self, dim: int, heads: int = 4, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.ff1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_in = nn.Conv1d(dim, 2 * dim, 1)
        self.conv_depth = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.conv_out = nn.Conv1d(dim, dim, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.drop(self.ff1(x))
        b, t, d = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B, heads, T, d/heads)
        # Plain matmul/softmax attention: supported by every backend, and chunks are short.
        weights = torch.softmax(q @ k.transpose(-1, -2) * (d // self.heads) ** -0.5, dim=-1)
        attn = (self.drop(weights) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.attn_out(attn))
        h = self.conv_in(self.conv_norm(x).transpose(1, 2))
        h = nn.functional.glu(h, dim=1)
        h = self.conv_out(nn.functional.silu(self.conv_depth(h)))
        x = x + self.drop(h.transpose(1, 2))
        x = x + 0.5 * self.drop(self.ff2(x))
        return self.out_norm(x)


class BeatmapNet(nn.Module):
    """``arch="conformer"`` (used for new training) or ``"gru"`` (the original design;
    the default so that old checkpoints without an ``arch`` entry still load)."""

    def __init__(self, n_mels: int = N_MELS, hidden: int = 128, arch: str = "gru",
                 layers: int = 4, window: int = 1032, outputs: int = 2,
                 conditions: list[str] | None = None, cond_stats: dict | None = None):
        super().__init__()
        self.arch = arch
        self.config = {"n_mels": n_mels, "hidden": hidden}
        if arch != "gru":
            # Attention only sees ``window`` frames at a time; longer inputs go through
            # predict_probs in overlapping windows.
            self.config.update(arch=arch, layers=layers, window=window, outputs=outputs)
        if conditions:
            # Stars and style (see style.py); cond_stats holds the training maps'
            # percentiles of each, used to turn "more jumps" into a value.
            self.config.update(conditions=list(conditions), cond_stats=cond_stats or {})
        conv_layers: list[nn.Module] = []
        channels = 1
        for out in (16, 32, 32):
            conv_layers += [
                nn.Conv2d(channels, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            ]
            channels = out
        self.conv = nn.Sequential(*conv_layers)
        self.proj = nn.Linear(channels * (n_mels // 8) + 4 + self.cond_size, hidden)
        if arch == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 2)
        else:
            self.blocks = nn.ModuleList(ConformerBlock(hidden) for _ in range(layers))
            # note, slider start, and with 4 outputs also slider held (sustain) and
            # log distance snap (spacing, a regression output)
            self.head = nn.Linear(hidden, outputs)

    @property
    def window(self) -> int | None:
        return self.config.get("window")

    @property
    def outputs(self) -> int:
        return self.config.get("outputs", 2)

    @property
    def conditions(self) -> list[str] | None:
        return self.config.get("conditions")

    @property
    def cond_size(self) -> int:
        return 2 * len(self.conditions) if self.conditions else 1

    def encode(self, values: float | dict) -> np.ndarray:
        """Condition input for a density (older models) or a dict of style values."""
        if not isinstance(values, dict):
            values = {"density": float(values)}
        if self.conditions:
            return encode_conditions(values, self.conditions)
        return np.array([np.log1p(values.get("density", 3.0))], dtype=np.float32)

    def forward(self, mel: torch.Tensor, beat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """mel (B, n_mels, T), beat (B, T, 4), cond (B, cond_size) from ``encode`` -- or
        a density (B,) for models without style conditions -> logits (B, T, outputs)."""
        h = self.conv(mel.unsqueeze(1))
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        if cond.dim() == 1:
            cond = torch.log1p(cond).view(b, 1)
        cond = cond.view(b, 1, -1).expand(b, t, cond.shape[-1])
        h = torch.relu(self.proj(torch.cat([h, beat, cond], dim=-1)))
        if self.arch != "gru":
            for block in self.blocks:
                h = block(h)
            return self.head(h)
        rnn_device = next(self.rnn.parameters()).device
        head_device = next(self.head.parameters()).device
        if h.device != rnn_device:
            h = h.to(rnn_device)
        h, _ = self.rnn(h)
        if h.device != head_device:
            h = h.to(head_device)
        return self.head(h)


def save_checkpoint(model: BeatmapNet, path: str | Path, threshold: float = 0.5) -> None:
    # Checkpoints stay portable even when training uses a mixed CPU/DirectML model.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": model.config, "state_dict": state_dict,
                "threshold": threshold}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[BeatmapNet, float]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BeatmapNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    # Thresholds tuned on generated maps per star range (scripts/tune_threshold.py).
    model.thresholds_by_stars = ckpt.get("thresholds_by_stars") or {}
    return model, float(ckpt.get("threshold", 0.5))


def threshold_for(model: BeatmapNet, stars: float, default: float) -> float:
    """The tuned note threshold for maps of ``stars``, or ``default``."""
    table = getattr(model, "thresholds_by_stars", None) or {}
    lower = [float(k) for k in table if float(k) <= stars]
    return float(table[f"{max(lower):g}"]) if lower else default


@torch.no_grad()
def predict_probs(model: BeatmapNet, mel: np.ndarray, beat: np.ndarray, cond: float | dict,
                  device=None, batch: int = 16) -> np.ndarray:
    """Per-frame outputs, (T, outputs), for a whole song: probabilities, except the
    spacing output (index 3), which stays a log distance snap.

    Windowed models see the song in overlapping windows; each frame takes its value
    from the window where it lies closest to the centre."""
    device = device or next(model.parameters()).device
    model.eval()
    n = mel.shape[1]
    window = model.window
    if window is None or n <= window:
        logits = model(torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))[None].to(device),
                       torch.from_numpy(beat)[None].to(device),
                       torch.from_numpy(model.encode(cond))[None].to(device))
        return _activate(logits)[0].cpu().numpy()
    hop = window // 2
    starts = list(range(0, n - window, hop)) + [n - window]
    out = np.zeros((n, model.outputs), dtype=np.float32)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        mels = torch.from_numpy(np.stack([mel[:, s:s + window] for s in chunk]).astype(np.float32))
        beats = torch.from_numpy(np.stack([beat[s:s + window] for s in chunk]))
        dens = torch.from_numpy(model.encode(cond))[None].expand(len(chunk), -1)
        probs = _activate(model(mels.to(device), beats.to(device), dens.to(device))).cpu().numpy()
        for s, p in zip(chunk, probs):
            # Keep the middle half of each window (the edges of the song keep theirs).
            a = 0 if s == 0 else s + window // 4
            b = n if s == starts[-1] else s + 3 * window // 4
            if s == starts[-1] and len(starts) > 1:
                a = max(starts[-2] + 3 * window // 4, 0)
            out[a:b] = p[a - s:b - s]
    return out


def _activate(logits: torch.Tensor) -> torch.Tensor:
    out = torch.sigmoid(logits)
    if logits.shape[-1] > 3:
        out[..., 3] = logits[..., 3]
    return out


def predict(
    model: BeatmapNet,
    features: AudioFeatures,
    grid: BeatGrid,
    density: float | dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (note probability, slider probability) for a whole song."""
    outputs = predict_all(model, features, grid, density)
    return outputs["note"], outputs["slider"]


def predict_all(model: BeatmapNet, features: AudioFeatures, grid: BeatGrid,
                density: float | dict) -> dict[str, np.ndarray]:
    """Every per-frame output of the model: "note", "slider" and, for models trained
    with them, "sustain" (slider held) and "spacing" (distance snap, not logged).
    ``density`` is a note density, or a dict of conditions (see ``style.CONDITIONS``)."""
    beat = beat_phase_features(features.n_frames, grid)
    probs = predict_probs(model, features.mel, beat, density)
    out = {"note": probs[:, 0], "slider": probs[:, 1]}
    if probs.shape[1] >= 4:
        out["sustain"] = probs[:, 2]
        out["spacing"] = np.exp(probs[:, 3])
    return out
