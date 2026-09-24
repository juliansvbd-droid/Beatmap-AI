"""Command line interface: ``beatmap-ai generate|analyze|train``."""

from __future__ import annotations

import argparse
import os

from .difficulty import PRESETS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="beatmap-ai", description="Generate osu! beatmaps from audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create an .osz beatmap set from an audio file")
    gen.add_argument("audio", help="input song (.mp3 or .ogg)")
    gen.add_argument("-o", "--output", help="output .osz path (default: next to the audio)")
    gen.add_argument("-d", "--difficulty", action="append", choices=list(PRESETS),
                     help="difficulty to create; repeat for several (default: normal, hard, insane)")
    gen.add_argument("-m", "--model", default="auto",
                     help="rhythm model checkpoint (.pt); default: the bundled model when "
                          "PyTorch is installed")
    gen.add_argument("--no-model", dest="model", action="store_const", const=None,
                     help="use onset heuristics instead of the rhythm model")
    gen.add_argument("--bpm", type=float, help="override the detected BPM")
    gen.add_argument("--offset", type=float, help="override the detected offset (ms)")
    gen.add_argument("--title")
    gen.add_argument("--artist")
    gen.add_argument("--seed", type=int, default=0)

    ana = sub.add_parser("analyze", help="print the detected BPM and offset")
    ana.add_argument("audio")

    tr = sub.add_parser("train", help="train the rhythm model on existing beatmaps")
    tr.add_argument("data", help="directory of .osz files and/or song folders (e.g. osu!/Songs)")
    tr.add_argument("-o", "--output", default="beatmap_model.pt")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--steps-per-epoch", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--cache-dir")
    tr.add_argument("--device")
    tr.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes for computing spectrograms")
    tr.add_argument("--seed", type=int, default=0)

    ev = sub.add_parser("evaluate", help="compare generated rhythm/timing with human maps")
    ev.add_argument("data", help="directory of .osz files and/or song folders")
    ev.add_argument("-m", "--model", help="also score this rhythm model checkpoint")
    ev.add_argument("--max-songs", type=int, default=40)

    args = parser.parse_args(argv)
    if args.command == "generate":
        from .generator import generate
        generate(args.audio, args.output, args.difficulty or ["normal", "hard", "insane"],
                 model_path=args.model, bpm=args.bpm, offset=args.offset,
                 title=args.title, artist=args.artist, seed=args.seed)
    elif args.command == "analyze":
        from .generator import analyze
        features, timing = analyze(args.audio)
        print(f"duration {features.duration:.2f}s  BPM {timing.bpm:g}  offset {timing.offset_ms:g} ms")
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.data, args.model, max_songs=args.max_songs)
    else:
        from .train import train
        train(args.data, args.output, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
              batch_size=args.batch_size, lr=args.lr, cache_dir=args.cache_dir,
              device=args.device, workers=args.workers, seed=args.seed)


if __name__ == "__main__":
    main()
