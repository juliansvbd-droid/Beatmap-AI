"""Fit the tempo picker and calibrate the offset latency on ranked beatmaps.

Uses songs with a single constant BPM. Songs in the validation split (the same
song-level split the rhythm model uses) are held out and only used for reporting.

    python scripts/fit_timing.py data/ --workers 4

Paste the printed TEMPO_WEIGHTS / ONSET_LATENCY into beatmap_ai/timing.py.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beatmap_ai.dataset import AudioSource, is_validation, iter_beatmaps, song_id
from beatmap_ai.evaluate import constant_timing
from beatmap_ai.timing import (FEATURE_NAMES, TEMPO_WEIGHTS, fit_tempo_weights, tempo_candidates,
                               timing_from_candidates)


def analyse(source: AudioSource):
    warnings.filterwarnings("ignore")
    try:
        f = source.load_features()
    except Exception:
        return None
    light = SimpleNamespace(onset=f.onset, bass_onset=f.bass_onset, duration=f.duration)
    return light, tempo_candidates(f)


def evaluate(songs, weights, latency):
    bpm_ok, errors = [], []
    for s in songs:
        est = timing_from_candidates(s["features"], s["candidates"], weights, latency)
        ok = abs(est.bpm - s["bpm"]) < 0.5
        bpm_ok.append(ok)
        if ok:
            beat = 60000.0 / s["bpm"]
            errors.append((est.offset_ms - s["offset"] + beat / 2) % beat - beat / 2)
    return np.mean(bpm_ok), np.array(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, help="pickle of analysed songs to reuse")
    args = parser.parse_args()

    cache = args.cache or args.data / ".timing_cache.pkl"
    songs = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    truth = {}
    for _, bm, source in iter_beatmaps(args.data):
        t = constant_timing(bm) if bm.mode == 0 else None
        if t is not None:
            truth.setdefault(source, t)
    todo = [s for s in truth if song_id(s.key) not in songs]
    print(f"{len(truth)} constant-BPM songs, analysing {len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (source, result) in enumerate(zip(todo, pool.map(analyse, todo)), 1):
            if result is not None:
                songs[song_id(source.key)] = {
                    "features": result[0], "candidates": result[1],
                    "bpm": truth[source].bpm, "offset": truth[source].offset_ms,
                }
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                cache.write_bytes(pickle.dumps(songs))
    cache.write_bytes(pickle.dumps(songs))

    train = [s for sid, s in songs.items() if not is_validation(sid)]
    val = [s for sid, s in songs.items() if is_validation(sid)]
    weights = fit_tempo_weights([s["candidates"] for s in train], [s["bpm"] for s in train])
    _, raw = evaluate(train, weights, latency=0.0)
    latency = float(np.median(raw)) / 1000.0

    print(f"\ntrain songs {len(train)}, validation songs {len(val)}")
    for name, w, lat in (("previous", TEMPO_WEIGHTS, None), ("fitted", weights, latency)):
        for split, data in (("train", train), ("validation", val)):
            acc, err = evaluate(data, w, lat if lat is not None else latency)
            print(f"{name:>8} {split:>10}: BPM exact {acc:.1%}, offset |err| <= 10 ms "
                  f"{np.mean(np.abs(err) <= 10):.1%}, median err {np.median(err):+.1f} ms")
    print("\nTEMPO_WEIGHTS = np.array([" + ", ".join(f"{w:.3f}" for w in weights) + "])")
    print(f"ONSET_LATENCY = {latency:.3f}")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  {name:>26}: {w:+.3f}")


if __name__ == "__main__":
    main()
