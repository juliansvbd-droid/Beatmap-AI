"""Pick untouched validation songs for rhythm comparisons (CPU only, no PyTorch).

Lists validation songs (``dataset.is_validation(song_key)``) in the numeric set folders
of the dataset that still have their audio, with star rating, constant timing and the
human rhythm counts (``beatmap_ai.rhythm_audit.summary``) of every osu!standard map.

A song counts as *used* (and is left out of the holdout) when
- a standard tool picks it with its default settings: the evenly spaced songs of
  ``beatmap-ai evaluate``/``eval_sequence.py`` (40) and ``tune_threshold.py`` (80), the
  first 60 songs in folder order (``eval_ab_critic.py`` takes the first 40), and the
  songs of the 16/40 maps a rhythm training run validates on (``dataset.spread``),
  each with its neighbours in that order (the dataset changed since those runs), or
- its set folder or title appears in a log, manifest or report on the dataset drive,
  in ``docs/`` or in ``prompts/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatmap_ai.dataset import MIN_OBJECTS, is_validation, song_key  # noqa: E402
from beatmap_ai.evaluate import constant_timing  # noqa: E402
from beatmap_ai.osu import parse_osu  # noqa: E402
from beatmap_ai.rhythm_audit import events, summary  # noqa: E402
from beatmap_ai.style import star_rating  # noqa: E402

AUDIO_LINE = re.compile(r"^AudioFilename:\s*(.+?)\s*$", re.MULTILINE)
TEXT_SUFFIXES = {".log", ".json", ".jsonl", ".md", ".csv", ".txt"}
# Lists of every song (catalogues, community tags) say nothing about use.
NOT_USAGE = {"catalog.json", "loved_catalog.json", "tags.json"}


def _evenly(ordered: list, n: int) -> set:
    return {ordered[int(i * len(ordered) / n)] for i in range(min(n, len(ordered)))}


def _with_neighbours(ordered: list, chosen: set, margin: int = 1) -> set:
    index = {key: i for i, key in enumerate(ordered)}
    out = set()
    for key in chosen:
        i = index[key]
        out.update(ordered[max(i - margin, 0): i + margin + 1])
    return out


def _usage_text(dataset: Path) -> str:
    """Lower-cased text of every log, manifest and report outside the set folders."""
    parts = []
    roots = [p for p in dataset.iterdir() if not p.name.isdecimal()]
    roots += [ROOT / "docs", ROOT / "prompts"]
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if (path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
                    and path.name.lower() not in NOT_USAGE):
                try:
                    parts.append(path.read_text(encoding="utf-8", errors="replace").lower())
                except OSError:
                    continue
    return "\n".join(parts)


def scan(dataset: Path, log=print) -> list[dict]:
    """Every validation map with audio in the numeric set folders."""
    maps = []
    folders = sorted(p for p in dataset.iterdir() if p.is_dir() and p.name.isdecimal())
    for n, folder in enumerate(folders):
        if n % 2000 == 0:
            log(f"  {n}/{len(folders)} folders", flush=True)
        files = sorted(folder.glob("*.osu"))
        if not files:
            continue
        first = files[0].read_text(encoding="utf-8", errors="replace")
        if not is_validation(song_key(parse_osu(first))):
            continue
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            match = AUDIO_LINE.search(text)
            if not match or not (folder / match.group(1)).is_file():
                continue
            bm = parse_osu(text)
            key = song_key(bm)
            if bm.mode != 0 or len(bm.hit_objects) < MIN_OBJECTS or not is_validation(key):
                continue
            timing = constant_timing(bm)
            objects = [o for o in bm.hit_objects if o.kind != "spinner"]
            maps.append({
                "set": folder.name, "file": str(path.relative_to(dataset)), "song": key,
                "artist": bm.artist, "title": bm.title, "version": bm.version,
                "audio": str(folder / match.group(1)), "stars": star_rating(text),
                "constant_timing": timing is not None,
                "bpm": None if timing is None else round(timing.bpm, 3),
                "duration_s": round((objects[-1].time - objects[0].time) / 1000.0, 1)
                if objects else 0.0,
                "human": summary(events(bm.hit_objects, bm.end_time), bm),
            })
    return maps


def used_songs(maps: list[dict], usage: str) -> dict[str, str]:
    """song key -> why it counts as used."""
    reasons: dict[str, str] = {}
    keys = sorted({m["song"] for m in maps})
    for key in _with_neighbours(keys, _evenly(keys, 40) | _evenly(keys, 80)):
        reasons.setdefault(key, "evenly spaced pick of evaluate/tune_threshold")
    in_folder_order = list(dict.fromkeys(m["song"] for m in sorted(maps, key=lambda m: m["file"])))
    for key in in_folder_order[:60]:
        reasons.setdefault(key, "first songs in folder order (eval_ab_critic)")
    examples = sorted(((m["song"], m["file"]) for m in maps))
    for count in (16, 40):
        picked = {examples[int(i * len(examples) / count)][0] for i in range(count)}
        for key in _with_neighbours(keys, picked):
            reasons.setdefault(key, f"rhythm training validation maps (spread {count})")
    for m in maps:
        title = m["title"].lower().strip()
        if f"\\{m['set']}\\" in usage or f"/{m['set']}/" in usage or (
                len(title) >= 6 and title in usage):
            reasons.setdefault(m["song"], "named in a log/manifest/report")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("D:/BeatMap-AI-Dataset"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    maps = scan(args.dataset)
    reasons = used_songs(maps, _usage_text(args.dataset))
    for m in maps:
        m["used"] = reasons.get(m["song"])
    songs = sorted({m["song"] for m in maps})
    holdout = sorted({m["song"] for m in maps if m["used"] is None},
                     key=lambda key: hashlib.sha1(key.encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"validation_songs": len(songs), "maps": maps,
                                       "holdout_order": holdout}, indent=1), encoding="utf-8")
    print(f"{len(maps)} validation maps from {len(songs)} songs; "
          f"{len(holdout)} songs unused, {len(songs) - len(holdout)} used")


if __name__ == "__main__":
    main()
