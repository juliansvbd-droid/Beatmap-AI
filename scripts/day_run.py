"""Daytime training run: sequence model v3 (large) for a fixed time, then the planner.

The overnight run (scripts/night_run.py) spent most of its time preparing data; with the
object cache, the vectorised section values and the trained tagger in place this only
trains. Everything goes to D:\\BeatMap-AI-Dataset\\day\\<date>; nothing is copied into
beatmap_ai/models (that is decided after measuring).

    python scripts/day_run.py --v3-hours 3
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from night_phase import data_dirs, tag_files  # noqa: E402

TAGGER = Path("D:/BeatMap-AI-Dataset/prepared/tagger.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-hours", type=float, default=3.0)
    parser.add_argument("--planner-minutes", type=float, default=15.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out or f"D:/BeatMap-AI-Dataset/day/{datetime.now():%Y-%m-%d}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "day_run.log", "a", encoding="utf-8")
    start = time.time()

    def log(message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    tagger = TAGGER if TAGGER.is_file() else None
    log(f"day run: v3 {args.v3_hours:g} h, then planner {args.planner_minutes:g} min; "
        f"tagger={tagger}; out={out}")
    from beatmap_ai.sequence_model import train_sequence_v3
    v3_deadline = start + args.v3_hours * 3600.0
    try:
        train_sequence_v3(data_dirs(), out / "sequence-v3.pt", tag_files(), epochs=1000,
                          steps_per_epoch=500, batch_size=8, hidden=512, layers=8, context=256,
                          device="cuda", workers=4, tagger_path=tagger, deadline=v3_deadline,
                          log=log)
    except Exception as exc:  # keep going to the planner; the last v3 state is on disk
        log(f"v3 stopped with an error: {exc!r}")

    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    from beatmap_ai.planner import train_planner
    planner_deadline = time.time() + args.planner_minutes * 60.0
    try:
        train_planner(data_dirs(), tag_files(), out / "planner.pt", tagger_path=tagger,
                      epochs=40, device="cuda", deadline=planner_deadline, log=log)
    except TypeError:  # older signature without deadline
        train_planner(data_dirs(), tag_files(), out / "planner.pt", tagger_path=tagger,
                      epochs=12, device="cuda", log=log)
    except Exception as exc:
        log(f"planner stopped with an error: {exc!r}")
    log(f"day run finished after {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
