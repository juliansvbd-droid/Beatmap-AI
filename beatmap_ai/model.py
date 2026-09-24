"""Neural rhythm model.

A convolutional front end reads the log-mel spectrogram, a bidirectional GRU adds
song-level context, and two heads predict, for every ~11.6 ms audio frame:

* the probability that a hit object starts there, and
* the probability that such an object is a slider.

The network is conditioned on the beat grid (phase within the beat and the bar) and on
the target note density, so one model can produce every difficulty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .audio import FPS, N_MELS, AudioFeatures

# (time ms, ms per beat, beats per bar) for each uninherited timing point.
BeatGrid = Sequence[tuple[float, float, int]]


def beat_phase_features(n_frames: int, grid: BeatGrid) -> np.ndarray:
    """sin/cos of the position within the current beat and bar for every frame, (T, 4)."""
    times = np.arange(n_frames) * 1000.0 / FPS
    grid = sorted(grid)
    starts = np.array([g[0] for g in grid])
    which = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(grid) - 1)
    t0 = starts[which]
    beat_length = np.array([g[1] for g in grid])[which]
    meter = np.array([g[2] for g in grid])[which]
    beats = (times - t0) / beat_length
    beat_phase = 2 * np.pi * (beats % 1.0)
    bar_phase = 2 * np.pi * ((beats % meter) / meter)
    return np.stack(
        [np.sin(beat_phase), np.cos(beat_phase), np.sin(bar_phase), np.cos(bar_phase)], axis=1
    ).astype(np.float32)


class BeatmapNet(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden: int = 128):
        super().__init__()
        self.config = {"n_mels": n_mels, "hidden": hidden}
        layers: list[nn.Module] = []
        channels = 1
        for out in (16, 32, 32):
            layers += [
                nn.Conv2d(channels, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            ]
            channels = out
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(channels * (n_mels // 8) + 4 + 1, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, mel: torch.Tensor, beat: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """mel (B, n_mels, T), beat (B, T, 4), density (B,) -> logits (B, T, 2)."""
        h = self.conv(mel.unsqueeze(1))
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        cond = torch.log1p(density).view(b, 1, 1).expand(b, t, 1)
        h = torch.relu(self.proj(torch.cat([h, beat, cond], dim=-1)))
        h, _ = self.rnn(h)
        return self.head(h)


def save_checkpoint(model: BeatmapNet, path: str | Path, threshold: float = 0.5) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "threshold": threshold}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[BeatmapNet, float]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BeatmapNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("threshold", 0.5))


@torch.no_grad()
def predict(
    model: BeatmapNet,
    features: AudioFeatures,
    grid: BeatGrid,
    density: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (note probability, slider probability) for a whole song."""
    device = next(model.parameters()).device
    model.eval()
    mel = torch.from_numpy(features.mel).unsqueeze(0).to(device)
    beat = torch.from_numpy(beat_phase_features(features.n_frames, grid)).unsqueeze(0).to(device)
    dens = torch.tensor([density], dtype=torch.float32, device=device)
    probs = torch.sigmoid(model(mel, beat, dens))[0].cpu().numpy()
    return probs[:, 0], probs[:, 1]
