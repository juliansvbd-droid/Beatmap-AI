"""Compare planner encoder step times on the configured training device."""

from __future__ import annotations

import argparse
import time

import torch
from beatmap_ai.planner import PlannerNet
from beatmap_ai.train import resolve_device


def time_step(model, device, steps, batch, length, features):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(batch, length, features, device=device)
    stars = torch.full((batch,), 4.5, device=device)
    target = torch.rand(batch, length, 8, device=device)
    durations = []
    for index in range(steps + 3):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = model(x, stars)["section"]
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if index >= 3:
            durations.append(elapsed)
    return sum(durations) / len(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sections", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    torch.manual_seed(42)
    for hidden in (128, 256):
        model = PlannerNet(hidden=hidden).to(device).train()
        seconds = time_step(model, device, args.steps, args.batch, args.sections, 164)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"hidden={hidden}: {parameters / 1e6:.2f}M parameters, "
              f"{seconds:.4f} sec/step ({device})", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
