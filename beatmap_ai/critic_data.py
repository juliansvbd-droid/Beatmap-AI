"""Training data and feature extraction for the beatmap critic.

Extracts objective, cheat-proof features for windows of consecutive hit objects:
- Relative timing to beat and measure
- Object types (circle, slider, spinner)
- Slider length, duration, repeats
- Positions, step deltas, movement vectors, heading
- Audio spectrogram and energy at object positions
- Map star rating

No PyTorch imports here so that worker processes can build batches cheaply without
allocating GPU runtime contexts. See critic.py for the neural network.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
import zipfile

import numpy as np

from .audio import FPS, AudioFeatures
from .dataset import AudioSource, MIN_OBJECTS, is_validation, iter_beatmap_texts, map_key, song_key
from .osu import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, parse_osu
from .placement_data import (BAR, BEAT, CIRCLE, CU, CV, END, EX, EY, HCOS, HSIN, KIND, LEN,
                             N_COLUMNS, OU, OV, PHASE, SLIDES, SPINNER, T, X, Y,
                             augment_objects, map_objects, section_energy)

CRITIC_WINDOW = 64
CRITIC_FEATURES = 105  # 6 time + 3 kind + 3 slider + 11 pos/move + 81 audio + 1 stars


def critic_features(
    objects: np.ndarray,
    mel: np.ndarray | None,
    stars: float,
    energy: np.ndarray | None = None,
) -> np.ndarray:
    """Extract (N, CRITIC_FEATURES) features from an object sequence (N, N_COLUMNS).

    Telltale features (hitsounds, curve type, combo colors, exact floats) are excluded
    to force the model to evaluate rhythm, geometry, flow and music alignment.
    """
    n = len(objects)
    if n == 0:
        return np.zeros((0, CRITIC_FEATURES), dtype=np.float32)

    o = objects
    prev = np.vstack([np.zeros((1, N_COLUMNS), dtype=np.float32), o[:-1]])
    first = np.zeros(n, dtype=bool)
    first[0] = True

    beat = np.maximum(o[:, BEAT], 1.0)
    gap_start = np.where(first, 8.0, np.clip((o[:, T] - prev[:, T]) / beat, 0.0, 16.0))
    gap_end = np.where(first, 8.0, np.clip((o[:, T] - prev[:, END]) / beat, 0.0, 16.0))

    phase = 2.0 * np.pi * o[:, PHASE]
    bar = 2.0 * np.pi * o[:, BAR]

    time_cols = [
        np.sin(phase)[:, None],
        np.cos(phase)[:, None],
        np.sin(bar)[:, None],
        np.cos(bar)[:, None],
        np.log1p(gap_start)[:, None],
        np.log1p(gap_end)[:, None],
    ]  # 6 cols

    # Object kind one-hot (3 cols)
    kind_idx = np.clip(o[:, KIND].astype(int), 0, 2)
    kind_one_hot = np.eye(3, dtype=np.float32)[kind_idx]

    # Slider attributes (3 cols)
    duration = np.clip((o[:, END] - o[:, T]) / beat, 0.0, 16.0) / 4.0
    slides = (o[:, SLIDES] / 4.0)
    slider_len = np.clip(o[:, LEN] / 200.0, 0.0, 10.0)
    slider_cols = [duration[:, None], slides[:, None], slider_len[:, None]]

    # Positions and movement vectors (11 cols)
    pos_x = o[:, X] / PLAYFIELD_WIDTH
    pos_y = o[:, Y] / PLAYFIELD_HEIGHT

    prev_end_x = np.where(first, o[:, X], prev[:, EX])
    prev_end_y = np.where(first, o[:, Y], prev[:, EY])
    dx = (o[:, X] - prev_end_x) / PLAYFIELD_WIDTH
    dy = (o[:, Y] - prev_end_y) / PLAYFIELD_HEIGHT
    dist = np.hypot(o[:, X] - prev_end_x, o[:, Y] - prev_end_y) / 100.0

    ou = o[:, OU]
    ov = o[:, OV]
    hcos = o[:, HCOS]
    hsin = o[:, HSIN]

    slider_dx = (o[:, EX] - o[:, X]) / PLAYFIELD_WIDTH
    slider_dy = (o[:, EY] - o[:, Y]) / PLAYFIELD_HEIGHT

    geom_cols = [
        pos_x[:, None], pos_y[:, None],
        dx[:, None], dy[:, None],
        dist[:, None],
        ou[:, None], ov[:, None],
        hcos[:, None], hsin[:, None],
        slider_dx[:, None], slider_dy[:, None],
    ]

    # Audio features around object time (81 cols)
    if mel is not None and mel.shape[1] > 0:
        frames = np.clip(np.rint(o[:, T] * FPS / 1000.0).astype(int), 0, mel.shape[1] - 1)
        window = np.clip(frames[:, None] + np.arange(-2, 3)[None, :], 0, mel.shape[1] - 1)
        audio = np.asarray(mel[:, window.ravel()], dtype=np.float32).reshape(mel.shape[0], n, 5).mean(axis=2).T
        if energy is None:
            energy = section_energy(mel)
        sec_energy = (energy[frames] * 4.0)[:, None]
    else:
        audio = np.zeros((n, 80), dtype=np.float32)
        sec_energy = np.zeros((n, 1), dtype=np.float32)

    stars_col = np.full((n, 1), float(stars) / 5.0, dtype=np.float32)

    all_cols = time_cols + [kind_one_hot] + slider_cols + geom_cols + [audio, sec_energy, stars_col]
    feats = np.concatenate(all_cols, axis=1).astype(np.float32)
    return feats


@dataclass
class CriticMap:
    name: str
    song: str
    mel_path: Path | None
    stars: float
    objects: np.ndarray
    is_human: bool
    pair_key: str | None = None


@dataclass
class CriticMapRef:
    start: int
    end: int
    song: str
    mel_path: Path | None
    stars: float
    is_human: bool
    name: str = ""
    pair_key: str | None = None


class SharedCriticMaps:
    def __init__(self, path: Path, refs: list[CriticMapRef]):
        self.path = path
        self.refs = refs
        self._objects = None

    def objects(self, ref: CriticMapRef) -> np.ndarray:
        if self._objects is None:
            self._objects = np.load(self.path, mmap_mode="r")
        return np.asarray(self._objects[ref.start:ref.end])

    def __getstate__(self):
        return {"path": self.path, "refs": self.refs, "_objects": None}


def share_critic_maps(maps: list[CriticMap], path: Path) -> SharedCriticMaps:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(m.objects) for m in maps)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, N_COLUMNS))
    refs, at = [], 0
    for m in maps:
        out[at:at + len(m.objects)] = m.objects
        refs.append(CriticMapRef(at, at + len(m.objects), m.song, m.mel_path, m.stars,
                                 m.is_human, m.name, m.pair_key))
        at += len(m.objects)
    out.flush()
    del out
    return SharedCriticMaps(path, refs)


class CriticSampler:
    """Balanced 50/50 sampler of 64-object windows from human and AI maps."""

    def __init__(
        self,
        maps: list[CriticMap] | SharedCriticMaps,
        window: int = CRITIC_WINDOW,
        seed: int = 42,
        augment: bool = True,
    ):
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.shared = isinstance(maps, SharedCriticMaps)
        all_refs = maps.refs if self.shared else maps
        self.maps_source = maps

        self.human_refs = [r for r in all_refs if (r.is_human if self.shared else r.is_human)
                           and ((r.end - r.start if self.shared else len(r.objects)) >= 16)]
        self.neg_refs = [r for r in all_refs if not (r.is_human if self.shared else r.is_human)
                         and ((r.end - r.start if self.shared else len(r.objects)) >= 16)]
        pair_groups = {}
        for ref in all_refs:
            pair_key = ref.pair_key if self.shared else ref.pair_key
            if not pair_key:
                continue
            group = pair_groups.setdefault(pair_key, {})
            group["human" if ref.is_human else "negative"] = ref
        self.paired_refs = [
            (group["human"], group["negative"])
            for group in pair_groups.values()
            if "human" in group and "negative" in group
            and (group["human"].end - group["human"].start if self.shared else len(group["human"].objects)) >= 16
            and (group["negative"].end - group["negative"].start if self.shared else len(group["negative"].objects)) >= 16
        ]

        self._mels: dict[Path, np.ndarray] = {}
        self._energy: dict[Path, np.ndarray] = {}

    def _get_objects(self, ref) -> np.ndarray:
        if self.shared:
            return self.maps_source.objects(ref)
        return ref.objects

    def mel(self, path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        if path not in self._mels:
            try:
                self._mels[path] = np.load(path, mmap_mode="r")
            except Exception:
                return None
        return self._mels[path]

    def energy(self, path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        if path not in self._energy:
            m = self.mel(path)
            self._energy[path] = section_energy(m) if m is not None else None
        return self._energy[path]

    def batch(self, size: int) -> dict[str, np.ndarray]:
        xs = []
        masks = []
        labels = []

        def append_example(ref, label: float, start: int, flips: tuple[bool, bool] | None = None) -> None:
            objs = self._get_objects(ref)
            chunk = objs[start:start + self.window]

            if self.augment:
                flip_x, flip_y = flips or (bool(self.rng.integers(2)), bool(self.rng.integers(2)))
                chunk = augment_objects(chunk, flip_x, flip_y)

            m_arr = self.mel(ref.mel_path)
            e_arr = self.energy(ref.mel_path)

            feats = critic_features(chunk, m_arr, ref.stars, e_arr)

            pad = self.window - len(chunk)
            if pad > 0:
                feats = np.pad(feats, ((0, pad), (0, 0)), mode="constant")
                mask = np.pad(np.ones(len(chunk), dtype=np.float32), (0, pad), mode="constant")
            else:
                mask = np.ones(self.window, dtype=np.float32)

            xs.append(feats)
            masks.append(mask)
            labels.append([label])

        if self.paired_refs:
            # Present each human/AI comparison from the same song, stars, rhythm and crop.
            for _ in range(size // 2):
                human_ref, neg_ref = self.paired_refs[int(self.rng.integers(len(self.paired_refs)))]
                n = min(len(self._get_objects(human_ref)), len(self._get_objects(neg_ref)))
                start = int(self.rng.integers(max(n - self.window, 0) + 1))
                flips = (bool(self.rng.integers(2)), bool(self.rng.integers(2)))
                append_example(human_ref, 1.0, start, flips)
                append_example(neg_ref, 0.0, start, flips)
            if size % 2:
                human_ref, neg_ref = self.paired_refs[int(self.rng.integers(len(self.paired_refs)))]
                label = float(self.rng.integers(2))
                ref = human_ref if label else neg_ref
                n = len(self._get_objects(ref))
                start = int(self.rng.integers(max(n - self.window, 0) + 1))
                append_example(ref, label, start)
        else:
            tasks = [1.0] * (size // 2) + [0.0] * (size - size // 2)
            self.rng.shuffle(tasks)
            for label in tasks:
                pool = self.human_refs if label else self.neg_refs
                if not pool:
                    continue
                ref = pool[int(self.rng.integers(len(pool)))]
                n = len(self._get_objects(ref))
                start = int(self.rng.integers(max(n - self.window, 0) + 1))
                append_example(ref, label, start)

        return {
            "x": np.stack(xs).astype(np.float32),
            "mask": np.stack(masks).astype(np.float32),
            "label": np.stack(labels).astype(np.float32),
        }


# Multiprocessing worker setup
_critic_worker = None


def _init_critic_worker(shared: SharedCriticMaps, window: int, batch_size: int, seed: int) -> None:
    global _critic_worker
    _critic_worker = (CriticSampler(shared, window=window, seed=seed + os.getpid(), augment=True), batch_size)


def _make_critic_batch(_=None) -> dict[str, np.ndarray]:
    sampler, size = _critic_worker
    batch = sampler.batch(size)
    batch["x"] = batch["x"].astype(np.float16)  # half the pipe transfer size
    return batch


def critic_batch_stream(
    shared: SharedCriticMaps,
    window: int,
    batch_size: int,
    seed: int,
    workers: int = 4,
    prefetch: int = 8,
):
    """Endless stream of training batches from lightweight worker processes."""
    if workers <= 0:
        _init_critic_worker(shared, window, batch_size, seed)
        while True:
            yield _make_critic_batch()
    pool = multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_critic_worker, initargs=(shared, window, batch_size, seed)
    )
    pending = deque(pool.apply_async(_make_critic_batch) for _ in range(prefetch))
    try:
        while True:
            result = pending.popleft()
            pending.append(pool.apply_async(_make_critic_batch))
            yield result.get()
    finally:
        pool.terminate()


def _fast_iter_beatmap_texts(root: str | Path):
    root = Path(root)
    if not (root / "tags.json").exists() and not (root / "catalog.json").exists():
        for item in iter_beatmap_texts(root):
            yield item
        return
    for entry in os.scandir(root):
        if entry.is_dir() and not entry.name.startswith("."):
            audio = None
            osus = []
            for sub in os.scandir(entry.path):
                sname = sub.name.lower()
                if sname in ("audio.mp3", "audio.ogg") or sname.endswith(".mp3"):
                    audio = Path(sub.path)
                elif sname.endswith(".osu"):
                    osus.append(Path(sub.path))
            if audio is not None:
                src = AudioSource(audio.resolve())
                for osu_path in osus:
                    try:
                        text = osu_path.read_text(encoding="utf-8", errors="replace")
                        yield str(osu_path.relative_to(root)), text, src, (0, 0)
                    except Exception:
                        pass


def resolve_human_osu_text(entry: dict, folders: list[Path]) -> str | None:
    """Resolve human source .osu text from manifest metadata across folders and .osz archives."""
    hname = entry.get("human_name", "")
    for base in folders:
        cand = base / hname
        if cand.is_file():
            try:
                return cand.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if ".osz" in hname:
        parts = hname.split(".osz", 1)
        osz_rel = parts[0] + ".osz"
        member = parts[1].lstrip("/\\")
        for base in folders:
            cand_osz = base / osz_rel
            if cand_osz.is_file():
                try:
                    with zipfile.ZipFile(cand_osz) as zf:
                        if member in zf.namelist():
                            return zf.read(member).decode("utf-8", errors="replace")
                except Exception:
                    pass
    hpath = entry.get("human_path", "")
    if hpath and hpath.lower().endswith(".osu"):
        p = Path(hpath)
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return None


def load_critic_dataset(
    human_folders: list[str | Path],
    manifest_path: str | Path | list[str | Path],
    max_human: int | None = None,
    paired: bool = True,
    log=print,
) -> tuple[list[CriticMap], list[CriticMap]]:
    """Load human and negative maps, partitioned into (train_maps, val_maps).

    When paired=True, each negative map is strictly paired 1-to-1 with its exact human source
    map (identical song, stars, timing and note counts), ensuring balanced distributions.
    """
    if isinstance(manifest_path, (str, Path)):
        manifest_paths = [Path(manifest_path)]
    else:
        manifest_paths = [Path(p) for p in manifest_path]

    folders = [Path(f) for f in human_folders]
    neg_maps = []
    human_maps = []

    human_stars = []
    neg_stars = []
    human_lens = []
    neg_lens = []
    human_buckets = Counter()
    neg_buckets = Counter()

    unpaired_skips = 0

    for mpath in manifest_paths:
        if not mpath.exists():
            log(f"Warning: Manifest not found: {mpath}")
            continue
        log(f"Loading paired maps from manifest: {mpath}...")
        with open(mpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    neg_path = Path(entry["negative_path"])
                    if not neg_path.exists():
                        continue
                    bm_neg = parse_osu(neg_path.read_text(encoding="utf-8"))
                    if len(bm_neg.hit_objects) < MIN_OBJECTS:
                        continue
                    objs_neg = map_objects(bm_neg)

                    # Resolve the exact source difficulty and the same sampled object window.
                    text_human = resolve_human_osu_text(entry, folders) if paired else None
                    objs_human = None
                    window_start = int(entry.get("window_start", 0))
                    window_count = int(entry.get("window_count", len(bm_neg.hit_objects)))
                    if text_human is not None:
                        bm_human = parse_osu(text_human)
                        source_window = bm_human.hit_objects[window_start:window_start + window_count]
                        if len(source_window) == len(bm_neg.hit_objects):
                            # Reset movement context at the exact window boundary like the negative map.
                            human_window = replace(bm_human, hit_objects=source_window)
                            objs_human = map_objects(human_window)

                    if paired and (objs_human is None or len(objs_human) != len(objs_neg)):
                        unpaired_skips += 1
                        continue

                    mel_path = Path(entry["mel_path"]) if entry.get("mel_path") else None
                    stars = float(entry.get("stars", 4.0))
                    cat = "<3" if stars < 3.0 else ("3-4.5" if stars < 4.5 else ("4.5-6" if stars < 6.0 else "6+"))
                    pair_key = str(entry.get("map_key_str") or f"{entry.get('song_key', '')}:{window_start}")

                    neg_maps.append(CriticMap(
                        name=neg_path.name,
                        song=entry["song_key"],
                        mel_path=mel_path if mel_path and mel_path.exists() else None,
                        stars=stars,
                        objects=objs_neg,
                        is_human=False,
                        pair_key=pair_key if paired else None,
                    ))
                    neg_stars.append(stars)
                    neg_lens.append(len(objs_neg))
                    neg_buckets[cat] += 1

                    if objs_human is not None:
                        human_maps.append(CriticMap(
                            name=entry.get("human_name", neg_path.name),
                            song=entry["song_key"],
                            mel_path=mel_path if mel_path and mel_path.exists() else None,
                            stars=stars,
                            objects=objs_human,
                            is_human=True,
                            pair_key=pair_key,
                        ))
                        human_stars.append(stars)
                        human_lens.append(len(objs_human))
                        human_buckets[cat] += 1
                except Exception:
                    continue

    log(f"Loaded {len(neg_maps)} negative maps and {len(human_maps)} paired human maps.")
    if unpaired_skips:
        log(f"Skipped {unpaired_skips} negatives without their exact source window.")

    # Only unpaired legacy datasets may use separately scanned positive examples.
    if not paired:
        log("Manifest is un-paired; falling back to scanning human folders...")
        seen_map_keys = {m.name for m in human_maps}
        target_human = len(neg_maps) if max_human is None else max_human
        for folder in folders:
            if not folder.exists():
                continue
            for name, text, source, _ in _fast_iter_beatmap_texts(folder):
                if source.path.suffix == ".npy":
                    continue
                try:
                    bm = parse_osu(text)
                    if bm.mode != 0 or len(bm.hit_objects) < MIN_OBJECTS:
                        continue
                    mkey = map_key(bm)
                    mkey_str = f"{mkey[0]}::{mkey[1]}::{mkey[2]}::{mkey[3]}"
                    if mkey_str in seen_map_keys:
                        continue
                    seen_map_keys.add(mkey_str)
                    from .style import star_rating
                    stars = star_rating(text)
                    if stars is None:
                        continue
                    objs = map_objects(bm)
                    mel_path = source.path.parent / "mel.npy" if source.path.is_file() else None
                    human_maps.append(CriticMap(
                        name=name,
                        song=song_key(bm),
                        mel_path=mel_path if mel_path and mel_path.exists() else None,
                        stars=float(stars),
                        objects=objs,
                        is_human=True,
                    ))
                    if len(human_maps) >= target_human:
                        break
                except Exception:
                    continue
            if len(human_maps) >= target_human:
                break

    # Log distributions
    log("\n" + "=" * 70)
    log("DATASET DISTRIBUTION REPORT (Positive vs Negative)")
    log("=" * 70)
    log(f"{'Metric':<25}{'Positive (Human)':>20}{'Negative (AI)':>20}")
    log("-" * 70)
    log(f"{'Total Count':<25}{len(human_maps):20d}{len(neg_maps):20d}")
    if human_stars and neg_stars:
        log(f"{'Stars Mean +- Std':<25}{f'{np.mean(human_stars):.2f} +- {np.std(human_stars):.2f}':>20}"
            f"{f'{np.mean(neg_stars):.2f} +- {np.std(neg_stars):.2f}':>20}")
        log(f"{'Objects Mean +- Std':<25}{f'{np.mean(human_lens):.1f} +- {np.std(human_lens):.1f}':>20}"
            f"{f'{np.mean(neg_lens):.1f} +- {np.std(neg_lens):.1f}':>20}")
        def window_density(objects: np.ndarray) -> float:
            times = objects[:, T]
            return float(len(times) / max((float(times.max()) - float(times.min())) / 1000.0, 1.0))

        human_density = np.asarray([window_density(m.objects) for m in human_maps], dtype=np.float64)
        neg_density = np.asarray([window_density(m.objects) for m in neg_maps], dtype=np.float64)
        log(f"{'Window density mean +- std':<25}{f'{human_density.mean():.3f} +- {human_density.std():.3f}':>20}"
            f"{f'{neg_density.mean():.3f} +- {neg_density.std():.3f}':>20}")
        if len(human_maps) == len(neg_maps) and human_maps:
            star_delta = max(abs(h.stars - n.stars) for h, n in zip(human_maps, neg_maps))
            length_delta = max(abs(len(h.objects) - len(n.objects)) for h, n in zip(human_maps, neg_maps))
            density_delta = max(abs(window_density(h.objects) - window_density(n.objects))
                                for h, n in zip(human_maps, neg_maps))
            log(f"Paired max deltas: stars={star_delta:.6f}, objects={length_delta}, density={density_delta:.6f}")
    for b in ("<3", "3-4.5", "4.5-6", "6+"):
        log(f"{f'Bucket {b}':<25}{human_buckets[b]:20d}{neg_buckets[b]:20d}")
    log("=" * 70 + "\n")

    all_maps = human_maps + neg_maps
    train_maps = [m for m in all_maps if not is_validation(m.song)]
    val_maps = [m for m in all_maps if is_validation(m.song)]

    log(f"Total dataset: {len(train_maps)} train ({sum(1 for m in train_maps if m.is_human)} human, "
        f"{sum(1 for m in train_maps if not m.is_human)} AI), "
        f"{len(val_maps)} val ({sum(1 for m in val_maps if m.is_human)} human, "
        f"{sum(1 for m in val_maps if not m.is_human)} AI)")

    return train_maps, val_maps
