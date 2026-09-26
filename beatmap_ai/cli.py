"""Command line interface: ``beatmap-ai generate|analyze|train``."""

from __future__ import annotations

import argparse
import os

from .difficulty import PRESETS, parse_difficulty
from .style import STYLES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="beatmap-ai", description="Generate osu! beatmaps from audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create an .osz beatmap set from an audio file")
    gen.add_argument("audio", help="input song (.mp3 or .ogg)")
    gen.add_argument("-o", "--output", help="output .osz path (default: next to the audio)")
    gen.add_argument("-d", "--difficulty", action="append", type=parse_difficulty,
                     help=f"difficulty to create: {', '.join(PRESETS)} or a star rating such "
                          "as 4.5; repeat for several (default: normal, hard, insane)")
    gen.add_argument("--style", action="append", default=[], metavar="NAME[=STRENGTH]",
                     help=f"map style: {', '.join(STYLES)} (strength 0..1, default 0.7); "
                          "repeat to combine. Without it the model chooses (Auto).")
    gen.add_argument("-m", "--model", default="auto",
                     help="rhythm model checkpoint (.pt); default: the bundled model when "
                          "PyTorch is installed")
    gen.add_argument("--no-model", dest="model", action="store_const", const=None,
                     help="use onset heuristics instead of the rhythm model")
    gen.add_argument("--placement", default="auto",
                     help="placement or sequence model checkpoint (.pt); default: the bundled "
                          "sequence model if present, otherwise placement model")
    gen.add_argument("--sequence", default=None,
                     help="sequence model checkpoint (.pt); alias for --placement")
    gen.add_argument("--rule-placement", dest="placement", action="store_const", const=None,
                     help="place objects with the built-in rules instead of the placement model")
    gen.add_argument("--critic", default="auto",
                     help="critic model checkpoint (.pt) for Best-of-N ranking; "
                          "default: bundled critic")
    gen.add_argument("--no-critic", dest="critic", action="store_const", const=None,
                     help="disable critic model ranking")
    gen.add_argument("--candidates", type=int, default=4,
                     help="candidates to evaluate per chunk with critic (default: 4)")
    gen.add_argument("--variety", type=float, default=0.5,
                     help="rhythm variety from 0 (close to the beat) to 1 (varied, longer "
                          "sliders); default 0.5")
    gen.add_argument("--no-repeats", dest="repeats", action="store_false",
                     help="do not copy patterns into repeated sections or add kiai time")
    gen.add_argument("--bpm", type=float, help="override the detected BPM")
    gen.add_argument("--offset", type=float, help="override the detected offset (ms)")
    gen.add_argument("--title")
    gen.add_argument("--artist")
    gen.add_argument("--seed", type=int, default=0)

    ana = sub.add_parser("analyze", help="print the detected BPM and offset")
    ana.add_argument("audio")

    tr = sub.add_parser("train", help="train the rhythm model on existing beatmaps")
    tr.add_argument("data", nargs="+",
                    help="directories of .osz files and/or song folders (e.g. osu!/Songs)")
    tr.add_argument("-o", "--output", default="beatmap_model.pt")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--steps-per-epoch", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--hidden", type=int, default=192, help="model width")
    tr.add_argument("--arch", choices=("conformer", "gru"), default="conformer",
                    help="model design; gru is the original, slower on AMD GPUs")
    tr.add_argument("--layers", type=int, default=4, help="conformer blocks")
    tr.add_argument("--val-maps", type=int, default=40,
                    help="validation difficulties scored after each epoch")
    tr.add_argument("--chunk-seconds", type=float, default=12.0,
                    help="length of the audio excerpts the model trains on")
    tr.add_argument("--cache-dir")
    tr.add_argument("--init-model", help="initialize from an existing checkpoint and fine-tune it")
    tr.add_argument("--device", help='e.g. "cuda" (NVIDIA or AMD ROCm), "cpu" or "directml"; '
                                     "default: GPU if available")
    tr.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes for computing spectrograms")
    tr.add_argument("--seed", type=int, default=0)

    pl = sub.add_parser("train-placement",
                        help="train the placement model (where objects go) on existing beatmaps")
    pl.add_argument("data", nargs="+", help="directories of .osz files and/or song folders")
    pl.add_argument("-o", "--output", default="placement.pt")
    pl.add_argument("--epochs", type=int, default=30)
    pl.add_argument("--steps-per-epoch", type=int, default=1000)
    pl.add_argument("--batch-size", type=int, default=64)
    pl.add_argument("--lr", type=float, default=5e-4)
    pl.add_argument("--hidden", type=int, default=256)
    pl.add_argument("--layers", type=int, default=6)
    pl.add_argument("--device")

    sq = sub.add_parser("train-sequence",
                        help="train the sequence model (rhythm and placement object by object)")
    sq.add_argument("data", nargs="+", help="directories of .osz files and/or song folders")
    sq.add_argument("-o", "--output", default="sequence.pt")
    sq.add_argument("--tags", nargs="*", default=[], help="tags.json files from scripts/fetch_tags.py")
    sq.add_argument("--epochs", type=int, default=30)
    sq.add_argument("--steps-per-epoch", type=int, default=1000)
    sq.add_argument("--batch-size", type=int, default=48)
    sq.add_argument("--lr", type=float, default=4e-4)
    sq.add_argument("--hidden", type=int, default=320)
    sq.add_argument("--layers", type=int, default=8)
    sq.add_argument("--workers", type=int, default=4, help="processes that build batches")
    sq.add_argument("--device")

    cr = sub.add_parser("train-critic",
                        help="train the critic model (distinguish human vs AI beatmaps)")
    cr.add_argument("data", nargs="+", help="directories of .osz files and/or song folders")
    cr.add_argument("--negatives", required=True, help="manifest.jsonl from scripts/make_critic_negatives.py")
    cr.add_argument("-o", "--output", default="critic.pt")
    cr.add_argument("--epochs", type=int, default=15)
    cr.add_argument("--steps-per-epoch", type=int, default=300)
    cr.add_argument("--batch-size", type=int, default=64)
    cr.add_argument("--lr", type=float, default=3e-4)
    cr.add_argument("--hidden", type=int, default=256)
    cr.add_argument("--layers", type=int, default=6)
    cr.add_argument("--heads", type=int, default=8)
    cr.add_argument("--ffn-dim", type=int, default=1024)
    cr.add_argument("--cache-dir")
    cr.add_argument("--workers", type=int, default=4, help="processes that build batches")
    cr.add_argument("--device")
    cr.add_argument("--seed", type=int, default=42)

    ui_cmd = sub.add_parser("ui", help="open the desktop interface")
    ui_cmd.add_argument("audio", nargs="?", default=None, help="optional audio file to open in the UI")

    ev = sub.add_parser("evaluate", help="compare generated rhythm/timing with human maps")
    ev.add_argument("data", nargs="+", help="directories of .osz files and/or song folders")
    ev.add_argument("-m", "--model", help="also score this rhythm model checkpoint")
    ev.add_argument("--max-songs", type=int, default=40)

    args = parser.parse_args(argv)
    if args.command == "generate":
        from .generator import generate
        style = {}
        for item in args.style:
            name, _, strength = item.partition("=")
            if name not in STYLES:
                parser.error(f"unknown style {name!r}; choose from {', '.join(STYLES)}")
            style[name] = float(strength) if strength else 0.7
        difficulties = [d if isinstance(d, float) else d.name.lower() for d in args.difficulty or []]
        placement_arg = args.sequence or args.placement
        generate(args.audio, args.output, difficulties or ["normal", "hard", "insane"],
                 model_path=args.model, bpm=args.bpm, offset=args.offset,
                 title=args.title, artist=args.artist, seed=args.seed, style=style or None,
                 placement_path=placement_arg, repeat_sections=args.repeats,
                 variety=args.variety, critic_path=args.critic, candidates=args.candidates)
    elif args.command == "analyze":
        from .generator import analyze
        features, timing = analyze(args.audio)
        print(f"duration {features.duration:.2f}s  BPM {timing.bpm:g}  offset {timing.offset_ms:g} ms")
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.data, args.model, max_songs=args.max_songs)
    elif args.command == "train-placement":
        from .placement_model import train_placement
        train_placement(args.data, args.output, epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
                        lr=args.lr, hidden=args.hidden, layers=args.layers, device=args.device)
    elif args.command == "train-sequence":
        from .sequence_model import train_sequence
        train_sequence(args.data, args.output, args.tags, epochs=args.epochs,
                       steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size, lr=args.lr,
                       hidden=args.hidden, layers=args.layers, device=args.device,
                       workers=args.workers)
    elif args.command == "train-critic":
        from .critic import train_critic
        train_critic(args.data, args.negatives, out_path=args.output,
                     cache_dir=args.cache_dir, epochs=args.epochs,
                     steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
                     lr=args.lr, hidden=args.hidden, layers=args.layers,
                     heads=args.heads, ffn_dim=args.ffn_dim,
                     device=args.device, seed=args.seed, workers=args.workers)
    elif args.command == "ui":
        from .ui import main as ui_main
        ui_main(args.audio)
    else:
        from .train import train
        train(args.data, args.output, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
              batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
              arch=args.arch, layers=args.layers, max_val_maps=args.val_maps,
              chunk_seconds=args.chunk_seconds, cache_dir=args.cache_dir, device=args.device,
              workers=args.workers, seed=args.seed, init_model=args.init_model)


if __name__ == "__main__":
    main()
