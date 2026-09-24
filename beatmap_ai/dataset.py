"""Turning a collection of existing osu! beatmaps into training data.

Accepts a directory containing any mix of ``.osz`` archives and song folders (such as the
``osu!/Songs`` directory). Only osu!standard difficulties are used.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .audio import FPS, compute_features, load_audio
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


def iter_beatmaps(root: str | Path) -> Iterator[tuple[str, Beatmap, str, Callable[[Path], None]]]:
    """Yield (name, beatmap, audio key, function that writes the audio to a path)."""
    root = Path(root)
    for osz in sorted(root.rglob("*.osz")):
        try:
            zf = zipfile.ZipFile(osz)
        except zipfile.BadZipFile:
            continue
        names = {n.lower(): n for n in zf.namelist()}
        for entry in names.values():
            if not entry.lower().endswith(".osu"):
                continue
            bm = parse_osu(zf.read(entry).decode("utf-8", errors="replace"))
            audio = names.get(bm.audio_filename.lower())
            if audio is None:
                continue

            def extract(dest: Path, zf=zf, audio=audio) -> None:
                dest.write_bytes(zf.read(audio))
            yield f"{osz.name}/{entry}", bm, f"{osz.resolve()}::{audio}", extract
    for osu_file in sorted(root.rglob("*.osu")):
        bm = parse_osu(osu_file.read_text(encoding="utf-8", errors="replace"))
        audio = osu_file.parent / bm.audio_filename
        if not audio.is_file():
            continue

        def copy(dest: Path, audio=audio) -> None:
            dest.write_bytes(audio.read_bytes())
        yield str(osu_file.relative_to(root)), bm, str(audio.resolve()), copy


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
    drain = (times.max() - times.min()) / 1000.0
    return MapExample(
        name=name,
        mel_path=mel_path,
        n_frames=n_frames,
        grid=grid,
        note_frames=frames[inside],
        slider_frames=frames[inside & is_slider],
        density=len(objects) / max(drain, 1.0),
    )


def build_examples(root: str | Path, cache_dir: str | Path, log=print) -> list[MapExample]:
    """Parse every beatmap under ``root``, caching one mel spectrogram per audio file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    frames_by_key: dict[str, tuple[Path, int] | None] = {}
    for name, bm, key, write_audio in iter_beatmaps(root):
        if bm.mode != 0:
            continue
        if key not in frames_by_key:
            mel_path = cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".npy")
            try:
                if not mel_path.exists():
                    suffix = Path(key.split("::")[-1]).suffix
                    with tempfile.TemporaryDirectory() as tmp:
                        audio = Path(tmp) / f"audio{suffix}"
                        write_audio(audio)
                        mel = compute_features(load_audio(audio)).mel
                    np.save(mel_path, mel.astype(np.float16))
                n_frames = np.load(mel_path, mmap_mode="r").shape[1]
                frames_by_key[key] = (mel_path, n_frames)
            except Exception as exc:  # Corrupt or unsupported audio: skip the song.
                log(f"skipping {name}: {exc}")
                frames_by_key[key] = None
        cached = frames_by_key[key]
        if cached is None:
            continue
        example = make_example(name, bm, *cached)
        if example is not None:
            examples.append(example)
    return examples


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
