"""Turning a collection of existing osu! beatmaps into training data.

Accepts a directory containing any mix of ``.osz`` archives and song folders (such as the
``osu!/Songs`` directory). Only osu!standard difficulties are used.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .audio import FPS, AudioFeatures, compute_features, load_audio
from .beatgrid import BeatGrid, beat_phase_features
from .osu import Beatmap, parse_osu
from .style import map_style

MIN_OBJECTS = 20
MEL_FILE = "mel.npy"  # precomputed spectrogram in compact set folders (no audio)


@dataclass
class MapExample:
    name: str
    mel_path: Path
    n_frames: int
    grid: BeatGrid
    note_frames: np.ndarray
    slider_frames: np.ndarray
    density: float  # hit objects per second
    song: str = ""  # song_key(): the same song in different beatmap sets shares it
    slider_spans: np.ndarray | None = None  # (n, 2) first and last frame of each slider
    spacing_frames: np.ndarray | None = None  # objects with a spacing target
    spacing: np.ndarray | None = None  # log of the distance snap at those objects
    style: dict | None = None  # map_style(): stars, jump, stream, ... (the model's conditions)
    key: tuple = ()  # map_key(), to skip copies of the same difficulty


@dataclass(frozen=True)
class AudioSource:
    """An audio file on disk, or a file inside an .osz archive."""

    path: Path
    member: str | None = None

    @property
    def key(self) -> str:
        return f"{self.path}::{self.member}" if self.member else str(self.path)

    def load_features(self) -> AudioFeatures:
        if self.member is None:
            return compute_features(load_audio(self.path))
        with zipfile.ZipFile(self.path) as zf, tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / ("audio" + Path(self.member).suffix)
            audio.write_bytes(zf.read(self.member))
            return compute_features(load_audio(audio))


_AUDIO_LINE = re.compile(r"^\s*AudioFilename\s*:(.*)$", re.MULTILINE)


def iter_beatmap_texts(root: str | Path) -> Iterator[tuple[str, str, AudioSource, tuple]]:
    """Yield (name, .osu text, audio source, stamp) for every .osu file under ``root``
    whose audio (or precomputed spectrogram) exists. ``stamp`` changes when the file does."""
    root = Path(root)
    for osz in sorted(root.rglob("*.osz")):
        try:
            with zipfile.ZipFile(osz) as zf:
                names = {n.lower(): n for n in zf.namelist()}
                entries = [(n, zf.read(n)) for n in names.values() if n.lower().endswith(".osu")]
        except zipfile.BadZipFile:
            continue
        stat = osz.stat()
        for entry, data in entries:
            text = data.decode("utf-8", errors="replace")
            match = _AUDIO_LINE.search(text)
            audio = names.get(match.group(1).strip().lower()) if match else None
            if audio is not None:
                yield (f"{osz.name}/{entry}", text, AudioSource(osz.resolve(), audio),
                       (stat.st_mtime, stat.st_size, entry))
    for osu_file in sorted(root.rglob("*.osu")):
        text = osu_file.read_text(encoding="utf-8", errors="replace")
        match = _AUDIO_LINE.search(text)
        audio = osu_file.parent / (match.group(1).strip() if match else "")
        mel = osu_file.parent / MEL_FILE
        stat = osu_file.stat()
        stamp = (stat.st_mtime, stat.st_size)
        if match and audio.is_file():
            yield str(osu_file.relative_to(root)), text, AudioSource(audio.resolve()), stamp
        elif mel.is_file():
            # Compact set (scripts/download_dataset.py): only the spectrogram is kept.
            yield str(osu_file.relative_to(root)), text, AudioSource(mel.resolve()), stamp


def iter_beatmaps(root: str | Path) -> Iterator[tuple[str, Beatmap, AudioSource]]:
    """Yield (name, beatmap, audio source) for every .osu file under ``root``."""
    for name, text, source, _ in iter_beatmap_texts(root):
        yield name, parse_osu(text), source


def song_id(audio_key: str) -> str:
    return hashlib.sha1(audio_key.encode()).hexdigest()


def song_key(bm: Beatmap) -> str:
    """Artist and title without version notes, so "unravel (TV Size)" by one mapper and
    "unravel" by another count as one song (for the train/validation split)."""
    def clean(text: str) -> str:
        text = re.sub(r"[(\[<].*?[)\]>]", "", text.lower())
        text = re.sub(r"\b(tv|size|ver|version|edit|cut|short|full|mix)\b", "", text)
        return re.sub(r"[^0-9a-zÀ-￿]+", "", text)
    return f"{clean(bm.artist)}-{clean(bm.title)}"


def map_key(bm: Beatmap) -> tuple:
    """Identifies one difficulty, to skip copies of the same map in several folders."""
    first = bm.hit_objects[0].time if bm.hit_objects else None
    return (bm.artist.lower(), bm.title.lower(), bm.creator.lower(), bm.version.lower(),
            len(bm.hit_objects), first)


def note_density(bm: Beatmap) -> float:
    """Hit objects per second between the first and the last object."""
    times = [o.time for o in bm.hit_objects]
    return len(times) / max((max(times) - min(times)) / 1000.0, 1.0) if times else 0.0


def make_example(name: str, bm: Beatmap, mel_path: Path | None, n_frames: int | None,
                 text: str | None = None) -> MapExample | None:
    """Training targets for one difficulty. Without ``n_frames`` notes past the end of
    the audio are kept; ``finish_example`` drops them once the length is known."""
    objects = bm.hit_objects
    grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points
            if tp.uninherited and tp.beat_length > 0]
    if bm.mode != 0 or len(objects) < MIN_OBJECTS or not grid:
        return None
    times = np.array([o.time for o in objects])
    frames = np.rint(times * FPS / 1000.0).astype(int)
    inside = (frames >= 0) & (frames < (n_frames if n_frames is not None else np.inf))
    is_slider = np.array([o.kind == "slider" for o in objects])
    ends = np.array([bm.end_time(o) for o in objects])
    spans = np.stack([frames, np.rint(ends * FPS / 1000.0).astype(int)], axis=1)[is_slider]
    spacing_frames, spacing = distance_snaps(bm, frames, ends)
    return MapExample(
        name=name,
        mel_path=mel_path,
        n_frames=n_frames,
        grid=grid,
        note_frames=frames[inside],
        slider_frames=frames[inside & is_slider],
        density=note_density(bm),
        song=song_key(bm),
        slider_spans=spans,
        spacing_frames=spacing_frames,
        spacing=spacing,
        style=map_style(bm, text),
        key=map_key(bm),
    )


def finish_example(example: MapExample, mel_path: Path, n_frames: int) -> MapExample:
    example.mel_path, example.n_frames = mel_path, n_frames
    example.note_frames = example.note_frames[example.note_frames < n_frames]
    example.slider_frames = example.slider_frames[example.slider_frames < n_frames]
    return example


def distance_snaps(bm: Beatmap, frames: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log(distance snap) for every object that follows another within about a beat.

    The distance snap is the distance from the previous object's end to this object,
    divided by the distance a slider would travel in the time between them. Mappers
    raise it for emphasis (jumps) and lower it for calm parts or stacks."""
    out_frames, out_values = [], []
    velocity = bm.slider_multiplier * 100.0  # osu! pixels per beat
    objects = bm.hit_objects
    for i in range(1, len(objects)):
        prev, obj = objects[i - 1], objects[i]
        if prev.kind == "spinner" or obj.kind == "spinner":
            continue
        beat_length, _ = bm.timing_at(obj.time)
        gap = (obj.time - ends[i - 1]) / beat_length
        if not 0.1 <= gap <= 1.05:
            continue
        end = (prev.x, prev.y)
        if prev.kind == "slider" and prev.slides % 2 == 1 and prev.curve_points:
            end = prev.curve_points[-1]
        distance = float(np.hypot(obj.x - end[0], obj.y - end[1]))
        out_frames.append(frames[i])
        out_values.append(np.log(np.clip(distance / (velocity * gap), 0.05, 8.0)))
    return np.asarray(out_frames, dtype=int), np.asarray(out_values, dtype=np.float32)


def _cache_mel(source: AudioSource, mel_path: Path) -> str | None:
    """Compute and store the mel spectrogram; returns an error message on failure."""
    try:
        np.save(mel_path, source.load_features().mel.astype(np.float16))
        return None
    except Exception as exc:  # Corrupt or unsupported audio: skip the song.
        error = f"{type(exc).__name__}: {exc}"
        mel_path.with_suffix(".failed").write_text(error)  # Don't retry on every run.
        return error


EXAMPLE_CACHE = "examples.pkl"
EXAMPLE_CACHE_VERSION = 2


def build_examples(root: str | Path | list[str | Path], cache_dir: str | Path | None = None,
                   workers: int = 1, log=print) -> list[MapExample]:
    """Parse every beatmap under ``root`` (one folder or a list), caching one mel
    spectrogram per audio file (computed in ``workers`` parallel processes). Without a
    ``cache_dir`` each folder keeps its cache in ``<folder>/.beatmap_ai_cache``.
    Copies of the same difficulty in several places are used once.

    Parsed examples are cached as well (``examples.pkl``), so later runs only parse new
    or changed files; this matters with tens of thousands of maps."""
    roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(r) for r in root]
    examples, seen, duplicates = [], set(), 0
    mel_paths, sources = {}, {}
    for folder in roots:
        folder_cache = Path(cache_dir) if cache_dir else folder / ".beatmap_ai_cache"
        folder_cache.mkdir(parents=True, exist_ok=True)
        cache_file = folder_cache / EXAMPLE_CACHE
        cached = {}
        try:
            with open(cache_file, "rb") as f:
                version, cached = pickle.load(f)
            if version != EXAMPLE_CACHE_VERSION:
                cached = {}
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError):
            cached = {}
        fresh, parsed = {}, 0
        for name, text, source, stamp in iter_beatmap_texts(folder):
            hit = cached.get(name)
            if hit is not None and hit[0] == stamp:
                example = hit[1]
            else:
                bm = parse_osu(text)
                example = make_example(name, bm, None, None, text) if bm.mode == 0 else None
                parsed += 1
                if parsed % 2000 == 0:
                    log(f"  parsed {parsed} beatmaps in {folder}")
            fresh[name] = (stamp, example)
            if example is None:
                continue
            if example.key in seen:
                duplicates += 1
                continue
            seen.add(example.key)
            if source not in mel_paths:
                beside = source.path.parent / MEL_FILE
                if source.path.name == MEL_FILE or (source.member is None and beside.is_file()):
                    mel_paths[source] = beside
                else:
                    mel_paths[source] = folder_cache / (song_id(source.key) + ".npy")
            sources[id(example)] = source
            examples.append(example)
        if parsed:
            tmp = cache_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump((EXAMPLE_CACHE_VERSION, fresh), f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_file)
    if duplicates:
        log(f"skipped {duplicates} duplicate difficulties")
    missing = [(s, p) for s, p in mel_paths.items()
               if not p.exists() and not p.with_suffix(".failed").exists()]
    if missing:
        log(f"computing spectrograms for {len(missing)} songs")
        with ProcessPoolExecutor(workers) as pool:
            for (source, _), error in zip(missing, pool.map(_cache_mel, *zip(*missing))):
                if error:
                    log(f"skipping {source.key}: {error}")
    n_frames = {s: np.load(p, mmap_mode="r").shape[1] for s, p in mel_paths.items() if p.exists()}
    return [finish_example(ex, mel_paths[sources[id(ex)]], n_frames[sources[id(ex)]])
            for ex in examples if sources[id(ex)] in n_frames]


def is_validation(example: MapExample | str, fraction: float = 0.1) -> bool:
    """Stable song-level split based on the hash of ``song_key`` (pass the example or
    the key), so every version of a song lands on the same side."""
    key = example if isinstance(example, str) else example.song
    sid = hashlib.sha1(key.encode()).hexdigest()
    return int(sid[:8], 16) % 1000 < fraction * 1000


def spread(examples: list[MapExample], n: int) -> list[MapExample]:
    """At most ``n`` examples, evenly spaced so they cover many songs."""
    examples = sorted(examples, key=lambda ex: (ex.song, ex.name))
    if len(examples) <= n:
        return examples
    return [examples[int(i * len(examples) / n)] for i in range(n)]


def targets(example: MapExample, start: int, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(note target, slider target, slider mask) for frames [start, start + length)."""
    note = np.zeros(length, dtype=np.float32)
    slider = np.zeros(length, dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    local = example.note_frames - start
    for offset, value in ((-1, 0.5), (1, 0.5), (0, 1.0)):
        idx = local + offset
        idx = idx[(idx >= 0) & (idx < length)]
        note[idx] = np.maximum(note[idx], value)
    idx = local[(local >= 0) & (local < length)]
    mask[idx] = 1.0
    idx = example.slider_frames - start
    slider[idx[(idx >= 0) & (idx < length)]] = 1.0
    return note, slider, mask


def extra_targets(example: MapExample, start: int, length: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sustain target, spacing target, spacing mask) for frames [start, start + length).
    Sustain is 1 while a slider is held, which teaches the model how long sliders last."""
    sustain = np.zeros(length, dtype=np.float32)
    spacing = np.zeros(length, dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    if example.slider_spans is not None:
        for a, b in example.slider_spans - start:
            sustain[max(a, 0):max(min(b + 1, length), 0)] = 1.0
    if example.spacing_frames is not None and len(example.spacing_frames):
        idx = example.spacing_frames - start
        keep = (idx >= 0) & (idx < length)
        spacing[idx[keep]] = example.spacing[keep]
        mask[idx[keep]] = 1.0
    return sustain, spacing, mask


class ChunkSampler:
    """Draws random fixed-length training crops from the examples."""

    def __init__(self, examples: list[MapExample], chunk_frames: int, seed: int = 0,
                 conditions: list[str] | None = None, balance: bool = True):
        self.examples = examples
        self.chunk = chunk_frames
        self.rng = np.random.default_rng(seed)
        self.conditions = conditions
        # Draw maps so every star level is trained on, not only the common ones: each
        # half-star bucket gets weight proportional to the square root of its size.
        stars = np.array([(ex.style or {}).get("stars", np.nan) for ex in examples], dtype=float)
        buckets = np.where(np.isfinite(stars), np.floor(np.nan_to_num(stars) * 2), -1)
        _, inverse, counts = np.unique(buckets, return_inverse=True, return_counts=True)
        weights = 1.0 / np.sqrt(counts[inverse]) if balance else np.ones(len(examples))
        self.weights = weights / weights.sum()
        self._mels: dict[Path, np.ndarray] = {}
        self._beats: dict[int, np.ndarray] = {}

    def mel(self, ex: MapExample) -> np.ndarray:
        if ex.mel_path not in self._mels:
            self._mels[ex.mel_path] = np.load(ex.mel_path, mmap_mode="r")
        return self._mels[ex.mel_path]

    def beats(self, i: int) -> np.ndarray:
        if i not in self._beats:
            ex = self.examples[i]
            self._beats[i] = beat_phase_features(ex.n_frames, ex.grid)
        return self._beats[i]

    def condition(self, ex: MapExample) -> np.ndarray:
        """Model condition input, with conditions randomly hidden: the model must also
        map well when only some (or none) are given, which is what "Auto" relies on."""
        if not self.conditions:
            return np.float32(ex.density)
        from .style import encode_conditions
        values = dict(ex.style or {}, density=ex.density)
        if self.rng.random() < 0.1:
            values = {}
        else:
            for name in self.conditions:
                hide = 0.5 if name == "density" else 0.3
                if self.rng.random() < hide:
                    values.pop(name, None)
        return encode_conditions(values, self.conditions)

    def batch(self, size: int) -> dict[str, np.ndarray]:
        out = {k: [] for k in ("mel", "beat", "cond", "note", "slider", "mask",
                               "sustain", "spacing", "spacing_mask")}
        for _ in range(size):
            i = int(self.rng.choice(len(self.examples), p=self.weights))
            ex = self.examples[i]
            start = int(self.rng.integers(max(ex.n_frames - self.chunk, 0) + 1))
            end = min(start + self.chunk, ex.n_frames)
            pad = self.chunk - (end - start)
            mel = np.asarray(self.mel(ex)[:, start:end], dtype=np.float32)
            beat = self.beats(i)[start:end]
            note, slider, mask = targets(ex, start, self.chunk)
            out["mel"].append(np.pad(mel, ((0, 0), (0, pad)), constant_values=-1.0))
            out["beat"].append(np.pad(beat, ((0, pad), (0, 0))))
            out["cond"].append(self.condition(ex))
            out["note"].append(note)
            out["slider"].append(slider)
            out["mask"].append(mask)
            sustain, spacing, spacing_mask = extra_targets(ex, start, self.chunk)
            out["sustain"].append(sustain)
            out["spacing"].append(spacing)
            out["spacing_mask"].append(spacing_mask)
        return {k: np.stack(v) for k, v in out.items()}
