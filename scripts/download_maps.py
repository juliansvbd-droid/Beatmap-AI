"""Download ranked osu!standard beatmap sets from a public mirror to use as training data.

Only the .osu files and the audio are kept (no videos, backgrounds or hitsounds).
The maps and music belong to their creators: use them for local training only.

    python scripts/download_maps.py data/ --count 400
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SEARCH = "https://api.nerinyan.moe/search?q=&s=ranked&m=0&ps=50&p={page}&sort=plays_desc"
DOWNLOAD = [
    "https://osu.direct/api/d/{id}?noVideo=1",
    "https://api.nerinyan.moe/d/{id}?noVideo=true&noBg=true&noHitsound=true&noStoryboard=true",
]
HEADERS = {"User-Agent": "beatmap-ai dataset downloader"}


def fetch(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def wanted(beatmapset: dict, min_length: int, max_length: int) -> bool:
    maps = beatmapset.get("beatmaps") or []
    return (
        bool(maps)
        and all(b.get("mode_int") == 0 for b in maps)
        and min_length <= maps[0].get("total_length", 0) <= max_length
        and not beatmapset.get("availability", {}).get("download_disabled")
    )


def keep_osu_and_audio(data: bytes, dest: Path) -> int:
    """Extract the .osu files and the audio file(s) they reference. Returns #difficulties."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = {n.lower(): n for n in zf.namelist()}
    osu_files = [n for n in zf.namelist() if n.lower().endswith(".osu")]
    audio = set()
    for name in osu_files:
        for line in zf.read(name).decode("utf-8", errors="replace").splitlines():
            if line.startswith("AudioFilename:"):
                audio.add(line.split(":", 1)[1].strip().lower())
                break
    dest.mkdir(parents=True, exist_ok=True)
    for name in osu_files + [names[a] for a in audio if a in names]:
        (dest / Path(name).name).write_bytes(zf.read(name))
    return len(osu_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=400, help="number of beatmap sets")
    parser.add_argument("--min-length", type=int, default=45, help="seconds")
    parser.add_argument("--max-length", type=int, default=360, help="seconds")
    parser.add_argument("--delay", type=float, default=0.5, help="pause between downloads")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in args.out.iterdir() if p.is_dir()}
    done, page = len(have), 0
    print(f"{done} sets already present", flush=True)
    while done < args.count:
        results = json.loads(fetch(SEARCH.format(page=page)))
        if not results:
            break
        page += 1
        for s in results:
            if done >= args.count:
                break
            set_id = str(s["id"])
            if set_id in have or not wanted(s, args.min_length, args.max_length):
                continue
            for url in DOWNLOAD:
                try:
                    n = keep_osu_and_audio(fetch(url.format(id=set_id)), args.out / set_id)
                    break
                except Exception as exc:  # Try the next mirror.
                    error = exc
            else:
                print(f"  failed {set_id}: {error}", file=sys.stderr, flush=True)
                continue
            have.add(set_id)
            done += 1
            print(f"[{done}/{args.count}] {s['artist']} - {s['title']} ({n} difficulties)", flush=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
