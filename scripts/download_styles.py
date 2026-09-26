"""Add ranked sets that players tagged with less common styles (tech, alt, jump
patterns, stream types, ...) to a dataset made by download_dataset.py.

1. Page through every ranked osu!standard set on a public mirror of the osu! API,
   keeping each difficulty's community tags (cached in ``catalog.json``).
2. Pick sets the dataset does not have yet whose difficulties carry one of the wanted
   tags (at least ``--min-votes`` votes), rarest tags first.
3. Download and store them like download_dataset.py, and add their tags to tags.json.

    python scripts/download_styles.py D:/maps --count 3000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_dataset import compact, download  # noqa: E402

from beatmap_ai.dataset import iter_beatmap_texts, song_key  # noqa: E402
from beatmap_ai.osu import Beatmap, parse_osu  # noqa: E402

SEARCH = "https://catboy.best/api/v2/search?query=&mode=0&status=1&limit=50&offset={offset}"
HEADERS = {"User-Agent": "beatmap-ai dataset downloader"}
WANTED_PREFIXES = ("tech/", "streams/", "jumps/", "sliders/", "style/", "reading/")
WANTED_TAGS = {"skillset/tech", "skillset/alt", "skillset/streams", "skillset/jumps",
               "expression/repetition", "expression/chaotic", "expression/simple"}


def fetch_json(url: str):
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())
        except Exception as exc:
            print(f"  retry {url}: {exc}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def build_catalog(path: Path) -> dict:
    """Every ranked osu!standard set with its difficulties' tags (id -> entry)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    catalog = {"names": {}, "sets": {}}
    offset = 0
    while True:
        page = fetch_json(SEARCH.format(offset=offset))
        page = page if isinstance(page, list) else (page or {}).get("beatmapsets", [])
        if not page:
            break
        for s in page:
            for tag in s.get("related_tags") or []:
                catalog["names"][str(tag["id"])] = tag["name"]
            diffs = [b for b in s.get("beatmaps") or [] if b.get("mode_int", 0) == 0]
            if not diffs:
                continue
            catalog["sets"][str(s["id"])] = {
                "artist": s.get("artist", ""), "title": s.get("title", ""),
                "length": max(b.get("total_length", 0) for b in diffs),
                "download_disabled": bool((s.get("availability") or {}).get("download_disabled")),
                "tags": {b["version"]: {str(t["tag_id"]): t["count"] for t in b.get("top_tag_ids") or []}
                         for b in diffs},
            }
        offset += len(page)
        if offset % 2500 < 50:
            print(f"  catalog: {offset} sets read, {len(catalog['sets'])} kept", flush=True)
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the dataset folder (with tags.json)")
    parser.add_argument("--count", type=int, default=3000, help="sets to add at most")
    parser.add_argument("--skip", nargs="*", default=["data", "best_maps"])
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    catalog = build_catalog(args.out / "catalog.json")
    names = catalog["names"]
    print(f"catalog: {len(catalog['sets'])} ranked osu!standard sets", flush=True)
    have_ids = {p.name for p in args.out.iterdir() if p.is_dir()}
    songs = set()
    for folder in [*args.skip, args.out]:
        for _, text, _, _ in iter_beatmap_texts(folder):
            songs.add(song_key(parse_osu(text)))

    def wanted(tag_name: str) -> bool:
        return tag_name in WANTED_TAGS or tag_name.startswith(WANTED_PREFIXES)

    counts = Counter(names.get(t, t) for s in catalog["sets"].values() for tags in s["tags"].values()
                     for t, v in tags.items() if v >= args.min_votes)
    candidates = []
    for set_id, s in catalog["sets"].items():
        if set_id in have_ids or s["download_disabled"] or not 45 <= s["length"] <= 420:
            continue
        tags = {names.get(t, t) for d in s["tags"].values() for t, v in d.items() if v >= args.min_votes}
        tags = [t for t in tags if wanted(t)]
        if not tags:
            continue
        key = song_key(Beatmap(artist=s["artist"], title=s["title"]))
        if key in songs:
            continue
        rarity = min(counts[t] for t in tags)  # rarest style first
        candidates.append((rarity, set_id, key, s))
    candidates.sort(key=lambda c: c[0])
    print(f"{len(candidates)} new sets with wanted style tags", flush=True)

    tags_path = args.out / "tags.json"
    store = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path.exists() else {"names": {}, "sets": {}}
    store["names"].update(names)
    done = 0
    with ThreadPoolExecutor(4) as downloads, ProcessPoolExecutor(args.workers) as mels:
        jobs = []
        for _, set_id, key, s in candidates:
            if done + len(jobs) >= args.count:
                break
            if key in songs:
                continue
            songs.add(key)
            jobs.append((set_id, key, s, downloads.submit(download, int(set_id), args.out / set_id)))
        for set_id, key, s, future in jobs:
            if shutil.disk_usage(args.out).free / 1e9 < args.min_free_gb:
                print("stopping: disk almost full", flush=True)
                break
            try:
                n = future.result()
            except Exception as exc:
                print(f"  failed {set_id}: {exc}", file=sys.stderr, flush=True)
                continue
            if not n:
                continue
            error = mels.submit(compact, str(args.out / set_id), True).result()
            if error:
                print(f"  skipped {set_id}: {error}", file=sys.stderr, flush=True)
                continue
            store["sets"][set_id] = s["tags"]
            done += 1
            if done % 50 == 0:
                tags_path.write_text(json.dumps(store), encoding="utf-8")
                print(f"[{done}] {s['artist']} - {s['title']}", flush=True)
    tags_path.write_text(json.dumps(store), encoding="utf-8")
    print(f"finished: added {done} style sets", flush=True)


if __name__ == "__main__":
    main()
