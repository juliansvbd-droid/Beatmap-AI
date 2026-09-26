"""Fetch osu!'s community tags (player-voted, per difficulty) for the training sets.

Players vote tags such as "skillset/jumps", "streams/bursts" or "style/tech" on ranked
difficulties. They describe a map's style much better than anything measured from the
map itself, so the models can learn styles from them. Tags come from a public mirror
of the osu! API and are stored as one JSON file:

    {"names": {tag id: name}, "sets": {set id: {difficulty name: {tag id: votes}}}}

    python scripts/fetch_tags.py data best_maps D:/maps -o D:/maps/tags.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SET_URL = "https://catboy.best/api/v2/s/{id}"
HEADERS = {"User-Agent": "beatmap-ai dataset downloader"}


def set_ids(folder: Path) -> set[int]:
    """Beatmap set ids from folder names ("123456", "123456 Artist - Title") and .osz
    names ("001 123456 Artist - Title.osz")."""
    ids = set()
    for path in folder.iterdir():
        numbers = re.findall(r"\d+", path.stem)
        if path.is_dir() and numbers and path.name.split()[0].isdigit():
            ids.add(int(path.name.split()[0]))
        elif path.suffix == ".osz" and len(numbers) >= 2:
            ids.add(int(numbers[1]))
    return ids


def fetch(set_id: int) -> dict | None:
    for attempt in range(4):
        try:
            request = urllib.request.Request(SET_URL.format(id=set_id), headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    store = {"names": {}, "sets": {}}
    if args.output.exists():
        store = json.loads(args.output.read_text(encoding="utf-8"))
    ids = sorted(set().union(*(set_ids(f) for f in args.folders)) - {int(k) for k in store["sets"]})
    print(f"{len(ids)} sets to fetch ({len(store['sets'])} already known)", flush=True)

    def save() -> None:
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(store), encoding="utf-8")
        tmp.replace(args.output)

    with ThreadPoolExecutor(args.threads) as pool:
        for n, (set_id, data) in enumerate(zip(ids, pool.map(fetch, ids)), 1):
            if data is None:
                store["sets"][str(set_id)] = {}
                print(f"  no data for {set_id}", file=sys.stderr, flush=True)
                continue
            for tag in data.get("related_tags") or []:
                store["names"][str(tag["id"])] = tag["name"]
            store["sets"][str(set_id)] = {
                b["version"]: {str(t["tag_id"]): t["count"] for t in (b.get("top_tag_ids") or [])}
                for b in data.get("beatmaps") or [] if b.get("mode_int", 0) == 0
            }
            if n % 250 == 0:
                save()
                tagged = sum(1 for s in store["sets"].values() for v in s.values() if v)
                print(f"[{n}/{len(ids)}] {tagged} tagged difficulties so far", flush=True)
    save()
    tagged = sum(1 for s in store["sets"].values() for v in s.values() if v)
    print(f"done: {len(store['sets'])} sets, {tagged} tagged difficulties, "
          f"{len(store['names'])} tag names", flush=True)


if __name__ == "__main__":
    main()
