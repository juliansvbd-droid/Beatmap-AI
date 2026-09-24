"""Turning a collection of existing osu! beatmaps into training data.

Accepts a directory containing any mix of ``.osz`` archives and song folders (such as the
``osu!/Songs`` directory). Only osu!standard difficulties are used.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .audio import FPS, AudioFeatures, compute_features, load_audio
from .model import BeatGrid, beat_phase_features
from .osu import Beatmap, parse_osu

MIN_OBJECTS = 20


@dataclass
class MapExample:
    name: str
    mel_path: Path
    n_frames: int
    grid: BeatGrid
    note_frames: np.ndarray
    slider_frames: np.ndarray
    density: float  # hit objects per second


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


def iter_beatmaps(root: str | Path) -> Iterator[tuple[str, Beatmap, AudioSource]]:
    """Yield (name, beatmap, audio source) for every .osu file under ``root``."""
    root = Path(root)
    for osz in sorted(root.rglob("*.osz")):
        try:
            with zipfile.ZipFile(osz) as zf:
                names = {n.lower(): n for n in zf.namelist()}
                entries = [(n, zf.read(n)) for n in names.values() if n.lower().endswith(".osu")]
        except zipfile.BadZipFile:
            continue
        for entry, data in entries:
            bm = parse_osu(data.decode("utf-8", errors="replace"))
            audio = names.get(bm.audio_filename.lower())
            if audio is not None:
                yield f"{osz.name}/{entry}", bm, AudioSource(osz.resolve(), audio)
    for osu_file in sorted(root.rglob("*.osu")):
        bm = parse_osu(osu_file.read_text(encoding="utf-8", errors="replace"))
        audio = osu_file.parent / bm.audio_filename
        if audio.is_file():
            yield str(osu_file.relative_to(root)), bm, AudioSource(audio.resolve())


def song_id(audio_key: str) -> str:
    return hashlib.sha1(audio_key.encode()).hexdigest()


def note_density(bm: Beatmap) -> float:
    """Hit objects per second between the first and the last object."""
    times = [o.time for o in bm.hit_objects]
    return len(times) / max((max(times) - min(times)) / 1000.0, 1.0) if times else 0.0


def make_example(name: str, bm: Beatmap, mel_path: Path, n_frames: int) -> MapExample | None:
    objects = bm.hit_objects
    grid = [(tp.time, tp.beat_length, tp.meter) for tp in bm.timing_points
            if tp.uninherited and tp.beat_length > 0]
    if bm.mode != 0 or len(objects) < MIN_OBJECTS or not grid:
        return None
    times = np.array([o.time for o in objects])
    frames = np.rint(times * FPS / 1000.0).astype(int)
    inside = (frames >= 0) & (frames < n_frames)
    is_slider = np.array([o.kind == "slider" for o in objects])
    return MapExample(
        name=name,
        mel_path=mel_path,
        n_frames=n_frames,
        grid=grid,
        note_frames=frames[inside],
        slider_frames=frames[inside & is_slider],
        density=note_density(bm),
    )


def _cache_mel(source: AudioSource, mel_path: Path) -> str | None:
    """Compute and store the mel spectrogram; returns an error message on failure."""
    try:
        np.save(mel_path, source.load_features().mel.astype(np.float16))
        return None
    except Exception as exc:  # Corrupt or unsupported audio: skip the song.
        error = f"{type(exc).__name__}: {exc}"
        mel_path.with_suffix(".failed").write_text(error)  # Don't retry on every run.
        return error


def build_examples(root: str | Path, cache_dir: str | Path, workers: int = 1,
                   log=print) -> list[MapExample]:
    """Parse every beatmap under ``root``, caching one mel spectrogram per audio file
    (computed in ``workers`` parallel processes)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    maps = [(name, bm, source) for name, bm, source in iter_beatmaps(root) if bm.mode == 0]
    mel_paths = {source: cache_dir / (song_id(source.key) + ".npy") for _, _, source in maps}
    missing = [(s, p) for s, p in mel_paths.items()
               if not p.exists() and not p.with_suffix(".failed").exists()]
    if missing:
        log(f"computing spectrograms for {len(missing)} songs")
        with ProcessPoolExecutor(workers) as pool:
            for (source, _), error in zip(missing, pool.map(_cache_mel, *zip(*missing))):
                if error:
                    log(f"skipping {source.key}: {error}")
    n_frames = {s: np.load(p, mmap_mode="r").shape[1] for s, p in mel_paths.items() if p.exists()}
    examples = []
    for name, bm, source in maps:
        if source in n_frames:
            example = make_example(name, bm, mel_paths[source], n_frames[source])
            if example is not None:
                examples.append(example)
    return examples


def is_validation(example: MapExample | str, fraction: float = 0.1) -> bool:
    """Stable song-level split based on the hash of the audio source (see ``song_id``)."""
    sid = example if isinstance(example, str) else example.mel_path.stem
    return int(sid[:8], 16) % 1000 < fraction * 1000


def spread(examples: list[MapExample], n: int) -> list[MapExample]:
    """At most ``n`` examples, evenly spaced so they cover many songs."""
    examples = sorted(examples, key=lambda ex: (str(ex.mel_path), ex.name))
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


class ChunkSampler:
    """Draws random fixed-length training crops from the examples."""

    def __init__(self, examples: list[MapExample], chunk_frames: int, seed: int = 0):
        self.examples = examples
        self.chunk = chunk_frames
        self.rng = np.random.default_rng(seed)
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

    def batch(self, size: int) -> dict[str, np.ndarray]:
        out = {k: [] for k in ("mel", "beat", "density", "note", "slider", "mask")}
        for _ in range(size):
            i = int(self.rng.integers(len(self.examples)))
            ex = self.examples[i]
            start = int(self.rng.integers(max(ex.n_frames - self.chunk, 0) + 1))
            end = min(start + self.chunk, ex.n_frames)
            pad = self.chunk - (end - start)
            mel = np.asarray(self.mel(ex)[:, start:end], dtype=np.float32)
            beat = self.beats(i)[start:end]
            note, slider, mask = targets(ex, start, self.chunk)
            out["mel"].append(np.pad(mel, ((0, 0), (0, pad)), constant_values=-1.0))
            out["beat"].append(np.pad(beat, ((0, pad), (0, 0))))
            out["density"].append(np.float32(ex.density))
            out["note"].append(note)
            out["slider"].append(slider)
            out["mask"].append(mask)
        return {k: np.stack(v) for k, v in out.items()}
