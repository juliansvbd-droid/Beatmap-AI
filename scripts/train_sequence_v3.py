"""Train the experimental v3 placement model without changing the bundled model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from beatmap_ai.placement_data import (BEAT, CMASK, KIND, N_COLUMNS, OMASK, PATH_END,
                                       PATH_START, SLIDER, T, X)
from beatmap_ai.sequence_data import V3_FEATURES
from beatmap_ai.sequence_model import (SequenceV3Net, load_sequence, save_sequence,
                                       sequence_losses, train_sequence_v3)


def smoke(path: Path):
    model = SequenceV3Net(features=V3_FEATURES, hidden=32, layers=1, heads=4, context=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {"x": torch.randn(2, 8, V3_FEATURES),
             "y": torch.zeros(2, 8, N_COLUMNS),
             "gap": torch.ones(2, 8, dtype=torch.long),
             "kind": torch.zeros(2, 8, dtype=torch.long),
             "duration": torch.zeros(2, 8, dtype=torch.long),
             "repeat": torch.zeros(2, 8, dtype=torch.long),
             "combo": torch.zeros(2, 8),
             "rhythm_mask": torch.ones(2, 8)}
    batch["y"][..., T] = torch.arange(1, 9).float()[None]
    batch["y"][..., X] = 100.0
    batch["y"][..., BEAT] = 500.0
    batch["y"][..., KIND] = SLIDER
    batch["y"][..., OMASK] = 1.0
    batch["y"][..., CMASK] = 1.0
    batch["y"][..., PATH_START:PATH_END] = torch.randn(2, 8, PATH_END - PATH_START) * 0.1
    loss, parts = sequence_losses(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_sequence(model, path)
    restored = load_sequence(path)
    with torch.no_grad():
        output = restored(batch["x"])
    if "form" not in parts or output["form"].shape != (2, 8, PATH_END - PATH_START):
        raise RuntimeError("sequence v3 smoke check failed")
    print(f"sequence v3 smoke passed: loss={float(loss.detach()):.4f}; form={tuple(output['form'].shape)}; "
          "checkpoint reload ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="*")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--tagger")
    parser.add_argument("--out", default="sequence_v3.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(Path(args.out))
        return
    train_sequence_v3(args.data, args.out, args.tags, epochs=args.epochs,
                      steps_per_epoch=args.steps, batch_size=args.batch_size,
                      hidden=args.hidden, layers=args.layers, context=args.context,
                      workers=args.workers, device=args.device, resume=not args.no_resume,
                      tagger_path=args.tagger)


if __name__ == "__main__":
    main()
