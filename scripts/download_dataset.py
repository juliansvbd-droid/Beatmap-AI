"""Build a large, compact training set of ranked osu!standard beatmap sets.

Each set becomes a folder with its osu!standard .osu files and ``mel.npy``, the
precomputed spectrogram the trainer needs (float16, about 2 MB). The audio is deleted
afterwards, except for songs in the validation split: evaluation needs their audio.
That keeps a few thousand sets within a few GB.

Sets come from a public mirror, mixing the most played, the most favourited and the
most recently ranked ones, skipping songs already present in ``--skip`` folders.
The maps and music belong to their creators: use them for local training only.

    python scripts/download_dataset.py dataset --count 3000 --skip data best_maps
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from beatmap_ai.dataset import is_validation, iter_beatmaps, song_key
from beatmap_ai.osu import Beatmap, parse_osu

SEARCH = "https://api.nerinyan.moe/search?q=&s=ranked&m=0&ps=50&p={page}&sort={sort}"
SORTS = ("plays_desc", "favourite_count_desc", "ranked_desc")
DOWNLOAD = [
    "https://osu.direct/api/d/{id}?noVideo=1",
    "https://api.nerinyan.moe/d/{id}?noVideo=true&noBg=true&noHitsound=true&noStoryboard=true",
    "https://catboy.best/d/{id}n",
]
HEADERS = {"User-Agent": "beatmap-ai dataset downloader"}
MEL_FILE = "mel.npy"


def fetch(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def candidates(sort: str, first_page: int = 0):
    page = first_page
    while True:
        try:
            results = json.loads(fetch(SEARCH.format(page=page, sort=sort)))
        except Exception as exc:
            print(f"  search {sort} page {page} failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(10)
            continue
        if not results:
            return
        yield from results
        page += 1


def interleave(*generators):
    generators = list(generators)
    while generators:
        for gen in list(generators):
            try:
                yield next(gen)
            except StopIteration:
                generators.remove(gen)


def wanted(s: dict, min_length: int, max_length: int) -> bool:
    maps = [b for b in s.get("beatmaps") or [] if b.get("mode_int") == 0]
    return (
        bool(maps)
        and min_length <= maps[0].get("total_length", 0) <= max_length
        and not s.get("availability", {}).get("download_disabled")
    )


def download(set_id: int, dest: Path) -> int:
    """Store the osu!standard .osu files and their audio in ``dest``; returns #difficulties."""
    error = None
    for url in DOWNLOAD:
        try:
            zf = zipfile.ZipFile(io.BytesIO(fetch(url.format(id=set_id))))
            break
        except Exception as exc:  # Try the next mirror.
            error = exc
    else:
        raise RuntimeError(error)
    names = {n.lower(): n for n in zf.namelist()}
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    count, audio = 0, set()
    for name in zf.namelist():
        if not name.lower().endswith(".osu"):
            continue
        data = zf.read(name)
        bm = parse_osu(data.decode("utf-8", errors="replace"))
        if bm.mode != 0 or bm.audio_filename.lower() not in names:
            continue
        (tmp / Path(name).name).write_bytes(data)
        audio.add(names[bm.audio_filename.lower()])
        count += 1
    if count == 0 or len(audio) != 1:  # Sets with several audio files are rare; skip them.
        shutil.rmtree(tmp, ignore_errors=True)
        return 0
    member = audio.pop()
    (tmp / Path(member).name).write_bytes(zf.read(member))
    tmp.rename(dest)
    return count


def compact(folder: str, keep_audio: bool) -> str | None:
    """Precompute the spectrogram; delete the audio unless it is kept for validation."""
    from beatmap_ai.audio import compute_features, load_audio
    folder = Path(folder)
    osu = next(folder.glob("*.osu"))
    bm = parse_osu(osu.read_text(encoding="utf-8", errors="replace"))
    audio = folder / bm.audio_filename
    try:
        mel = compute_features(load_audio(audio)).mel.astype(np.float16)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return f"{type(exc).__name__}: {exc}"
    np.save(folder / MEL_FILE, mel)
    if not keep_audio:
        audio.unlink()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=3000, help="number of beatmap sets")
    parser.add_argument("--skip", nargs="*", default=[], help="folders whose songs to leave out")
    parser.add_argument("--min-length", type=int, default=45, help="seconds")
    parser.add_argument("--max-length", type=int, default=420, help="seconds")
    parser.add_argument("--min-free-gb", type=float, default=5.0,
                        help="stop when the disk has less free space than this")
    parser.add_argument("--workers", type=int, default=6, help="processes for spectrograms")
    parser.add_argument("--keep-audio", action="store_true",
                        help="keep every song's audio, not only the validation songs'")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for part in args.out.glob("*.part"):
        shutil.rmtree(part, ignore_errors=True)
    songs = set()
    for folder in [*args.skip, args.out]:
        for _, bm, _ in iter_beatmaps(folder):
            songs.add(song_key(bm))
    for folder in args.out.iterdir():  # Compact sets have no audio for iter_beatmaps.
        for osu in folder.glob("*.osu") if folder.is_dir() else []:
            songs.add(song_key(parse_osu(osu.read_text(encoding="utf-8", errors="replace"))))
            break
    done = sum(1 for p in args.out.iterdir() if (p / MEL_FILE).exists())
    print(f"{done} sets present, {len(songs)} songs known", flush=True)

    streams = interleave(*(candidates(sort) for sort in SORTS))
    with ThreadPoolExecutor(4) as downloads, ProcessPoolExecutor(args.workers) as mels:
        pending_downloads, pending_mels = {}, {}
        for s in streams:
            if done + len(pending_mels) >= args.count:
                break
            if shutil.disk_usage(args.out).free / 1e9 < args.min_free_gb:
                print("stopping: disk almost full", flush=True)
                break
            key = song_key(Beatmap(artist=s.get("artist", ""), title=s.get("title", "")))
            if key in songs or not wanted(s, args.min_length, args.max_length):
                continue
            songs.add(key)
            dest = args.out / str(s["id"])
            future = downloads.submit(download, s["id"], dest)
            pending_downloads[future] = (s, dest, key)
            # Keep a few downloads in flight and hand finished ones to the spectrogram pool.
            while len(pending_downloads) >= 8:
                time.sleep(0.2)
                for f in [f for f in pending_downloads if f.done()]:
                    s_, dest_, key_ = pending_downloads.pop(f)
                    try:
                        n = f.result()
                    except Exception as exc:
                        print(f"  failed {s_['id']}: {exc}", file=sys.stderr, flush=True)
                        continue
                    if n:
                        pending_mels[mels.submit(compact, str(dest_), args.keep_audio or is_validation(key_))] = (s_, n)
                for f in [f for f in pending_mels if f.done()]:
                    s_, n = pending_mels.pop(f)
                    if f.result():
                        print(f"  skipped {s_['id']}: {f.result()}", file=sys.stderr, flush=True)
                        continue
                    done += 1
                    print(f"[{done}/{args.count}] {s_['artist']} - {s_['title']} ({n} difficulties)",
                          flush=True)
        for f in pending_downloads:
            s_, dest_, key_ = pending_downloads[f]
            try:
                if f.result():
                    pending_mels[mels.submit(compact, str(dest_), args.keep_audio or is_validation(key_))] = (s_, 1)
            except Exception as exc:
                print(f"  failed {s_['id']}: {exc}", file=sys.stderr, flush=True)
        for f, (s_, n) in pending_mels.items():
            if not f.result():
                done += 1
    print(f"finished: {done} sets in {args.out}", flush=True)


if __name__ == "__main__":
    main()
