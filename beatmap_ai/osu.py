"""Reading and writing osu! beatmaps (.osu file format v14) and .osz packages."""

from __future__ import annotations

import bisect
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

PLAYFIELD_WIDTH = 512
PLAYFIELD_HEIGHT = 384

TYPE_CIRCLE = 1
TYPE_SLIDER = 2
TYPE_NEW_COMBO = 4
TYPE_SPINNER = 8
TYPE_HOLD = 128


@dataclass
class TimingPoint:
    time: float
    # ms per beat when uninherited; for inherited points, -100 / slider-velocity.
    beat_length: float
    meter: int = 4
    sample_set: int = 2
    sample_index: int = 0
    volume: int = 60
    uninherited: bool = True
    effects: int = 0

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length

    def to_line(self) -> str:
        return (
            f"{_num(self.time)},{self.beat_length!r},{self.meter},{self.sample_set},"
            f"{self.sample_index},{self.volume},{int(self.uninherited)},{self.effects}"
        )


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    kind: str  # "circle", "slider" or "spinner"
    new_combo: bool = False
    hit_sound: int = 0
    # Sliders
    curve_type: str = "L"
    curve_points: list[tuple[float, float]] = field(default_factory=list)
    slides: int = 1
    length: float = 0.0
    # Spinners (sliders get their end time from Beatmap.end_time)
    end_time: float | None = None

    def to_line(self) -> str:
        combo = TYPE_NEW_COMBO if self.new_combo else 0
        x, y, t = round(self.x), round(self.y), round(self.time)
        if self.kind == "circle":
            return f"{x},{y},{t},{TYPE_CIRCLE | combo},{self.hit_sound},0:0:0:0:"
        if self.kind == "slider":
            points = "|".join(f"{round(px)}:{round(py)}" for px, py in self.curve_points)
            # A slider's hitsound plays on its head (the first edge), not along its body.
            edges = "|".join([str(self.hit_sound)] + ["0"] * self.slides)
            edge_sets = "|".join(["0:0"] * (self.slides + 1))
            return (
                f"{x},{y},{t},{TYPE_SLIDER | combo},0,"
                f"{self.curve_type}|{points},{self.slides},{self.length:.2f},"
                f"{edges},{edge_sets},0:0:0:0:"
            )
        if self.kind == "spinner":
            return (
                f"256,192,{t},{TYPE_SPINNER | TYPE_NEW_COMBO},{self.hit_sound},"
                f"{round(self.end_time)},0:0:0:0:"
            )
        raise ValueError(f"unknown hit object kind {self.kind!r}")


@dataclass
class Beatmap:
    title: str = "Unknown"
    artist: str = "Unknown"
    creator: str = "Beatmap-AI"
    version: str = "Normal"
    audio_filename: str = "audio.mp3"
    preview_time: int = -1
    audio_lead_in: int = 0
    mode: int = 0
    hp: float = 5.0
    cs: float = 4.0
    od: float = 5.0
    ar: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    beat_divisor: int = 4
    tags: str = "beatmap-ai generated"
    timing_points: list[TimingPoint] = field(default_factory=list)
    hit_objects: list[HitObject] = field(default_factory=list)
    breaks: list[tuple[float, float]] = field(default_factory=list)

    def timing_at(self, time: float) -> tuple[float, float]:
        """Return (ms per beat, slider velocity multiplier) in effect at ``time``."""
        times, values = self._timing_table()
        i = bisect.bisect_right(times, time + 1e-6) - 1
        return values[i]

    def _timing_table(self) -> tuple[list[float], list[tuple[float, float]]]:
        """(times, (ms per beat, slider velocity)) after each timing point, cached until
        the timing points change."""
        key = (id(self.timing_points), len(self.timing_points))
        cached = self.__dict__.get("_timing_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        uninherited = [tp for tp in self.timing_points if tp.uninherited]
        if not uninherited:
            raise ValueError("beatmap has no uninherited timing points")
        beat_length, sv = uninherited[0].beat_length, 1.0
        times, values = [], []
        for tp in sorted(self.timing_points, key=lambda tp: tp.time):
            if tp.uninherited:
                beat_length, sv = tp.beat_length, 1.0
            elif tp.beat_length < 0:
                sv = min(max(-100.0 / tp.beat_length, 0.1), 10.0)
            times.append(tp.time)
            values.append((beat_length, sv))
        # Before the first timing point, the first uninherited one applies.
        values.insert(0, (uninherited[0].beat_length, 1.0))
        times.insert(0, float("-inf"))
        self.__dict__["_timing_cache"] = (key, (times, values))
        return times, values

    def slider_duration(self, obj: HitObject) -> float:
        beat_length, sv = self.timing_at(obj.time)
        velocity = self.slider_multiplier * 100.0 * sv  # osu! pixels per beat
        return obj.length / velocity * beat_length * obj.slides

    def end_time(self, obj: HitObject) -> float:
        if obj.kind == "slider":
            return obj.time + self.slider_duration(obj)
        if obj.kind == "spinner":
            return obj.end_time
        return obj.time

    @property
    def filename(self) -> str:
        name = f"{self.artist} - {self.title} ({self.creator}) [{self.version}].osu"
        return re.sub(r'[\\/:*?"<>|]', "", name)

    def to_osu_string(self) -> str:
        lines = [
            "osu file format v14",
            "",
            "[General]",
            f"AudioFilename: {self.audio_filename}",
            f"AudioLeadIn: {self.audio_lead_in}",
            f"PreviewTime: {self.preview_time}",
            "Countdown: 0",
            "SampleSet: Soft",
            "StackLeniency: 0.7",
            f"Mode: {self.mode}",
            "LetterboxInBreaks: 0",
            "WidescreenStoryboard: 0",
            "",
            "[Editor]",
            "DistanceSpacing: 1",
            f"BeatDivisor: {self.beat_divisor}",
            "GridSize: 32",
            "TimelineZoom: 1",
            "",
            "[Metadata]",
            f"Title:{self.title}",
            f"TitleUnicode:{self.title}",
            f"Artist:{self.artist}",
            f"ArtistUnicode:{self.artist}",
            f"Creator:{self.creator}",
            f"Version:{self.version}",
            "Source:",
            f"Tags:{self.tags}",
            "BeatmapID:0",
            "BeatmapSetID:-1",
            "",
            "[Difficulty]",
            f"HPDrainRate:{_num(self.hp)}",
            f"CircleSize:{_num(self.cs)}",
            f"OverallDifficulty:{_num(self.od)}",
            f"ApproachRate:{_num(self.ar)}",
            f"SliderMultiplier:{_num(self.slider_multiplier)}",
            f"SliderTickRate:{_num(self.slider_tick_rate)}",
            "",
            "[Events]",
            "//Background and Video events",
            "//Break Periods",
            *(f"2,{round(start)},{round(end)}" for start, end in self.breaks),
            "//Storyboard Layer 0 (Background)",
            "//Storyboard Layer 1 (Fail)",
            "//Storyboard Layer 2 (Pass)",
            "//Storyboard Layer 3 (Foreground)",
            "//Storyboard Sound Samples",
            "",
            "[TimingPoints]",
            *(tp.to_line() for tp in self.timing_points),
            "",
            "",
            "[HitObjects]",
            *(obj.to_line() for obj in self.hit_objects),
        ]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_osu_string(), encoding="utf-8")


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def parse_osu(text: str) -> Beatmap:
    """Parse the contents of a .osu file. Unknown sections and keys are ignored."""
    bm = Beatmap(title="", artist="", creator="", version="", audio_filename="")
    section = None
    keys = {
        "AudioFilename": ("audio_filename", str),
        "PreviewTime": ("preview_time", int),
        "AudioLeadIn": ("audio_lead_in", int),
        "Mode": ("mode", int),
        "Title": ("title", str),
        "Artist": ("artist", str),
        "Creator": ("creator", str),
        "Version": ("version", str),
        "Tags": ("tags", str),
        "HPDrainRate": ("hp", float),
        "CircleSize": ("cs", float),
        "OverallDifficulty": ("od", float),
        "ApproachRate": ("ar", float),
        "SliderMultiplier": ("slider_multiplier", float),
        "SliderTickRate": ("slider_tick_rate", float),
        "BeatDivisor": ("beat_divisor", int),
    }
    has_ar = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Editor", "Metadata", "Difficulty"):
            key, sep, value = line.partition(":")
            if sep and key.strip() in keys:
                attr, cast = keys[key.strip()]
                try:
                    setattr(bm, attr, cast(value.strip()))
                    has_ar |= attr == "ar"
                except ValueError:
                    pass
        elif section == "Events":
            parts = line.split(",")
            if parts[0] in ("2", "Break") and len(parts) >= 3:
                bm.breaks.append((float(parts[1]), float(parts[2])))
        elif section == "TimingPoints":
            tp = _parse_timing_point(line)
            if tp is not None:
                bm.timing_points.append(tp)
        elif section == "HitObjects":
            obj = _parse_hit_object(line)
            if obj is not None:
                bm.hit_objects.append(obj)
    if not has_ar:  # Old maps use OD for AR.
        bm.ar = bm.od
    return bm


def _parse_timing_point(line: str) -> TimingPoint | None:
    parts = line.split(",")
    if len(parts) < 2:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    defaults = [0, 0, 4, 1, 0, 100, 1, 0]
    values += defaults[len(values):]
    return TimingPoint(
        time=values[0],
        beat_length=values[1],
        meter=int(values[2]) or 4,
        sample_set=int(values[3]),
        sample_index=int(values[4]),
        volume=int(values[5]),
        uninherited=bool(int(values[6])),
        effects=int(values[7]),
    )


def _parse_hit_object(line: str) -> HitObject | None:
    parts = line.split(",")
    if len(parts) < 4:
        return None
    try:
        x, y, time = float(parts[0]), float(parts[1]), float(parts[2])
        obj_type = int(parts[3])
        hit_sound = int(parts[4]) if len(parts) > 4 and parts[4] else 0
    except ValueError:
        return None
    new_combo = bool(obj_type & TYPE_NEW_COMBO)
    if obj_type & TYPE_CIRCLE:
        return HitObject(x, y, time, "circle", new_combo, hit_sound)
    if obj_type & TYPE_SLIDER and len(parts) >= 8:
        curve_type, *raw_points = parts[5].split("|")
        points = []
        for p in raw_points:
            px, _, py = p.partition(":")
            points.append((float(px), float(py)))
        # Mappers put slider hitsounds on the edges; keep the head's as the hitsound.
        if len(parts) >= 9 and parts[8]:
            try:
                hit_sound = int(parts[8].split("|")[0])
            except ValueError:
                pass
        return HitObject(
            x, y, time, "slider", new_combo, hit_sound,
            curve_type=curve_type, curve_points=points,
            slides=int(parts[6]), length=float(parts[7]),
        )
    if obj_type & TYPE_SPINNER and len(parts) >= 6:
        return HitObject(x, y, time, "spinner", True, hit_sound, end_time=float(parts[5]))
    return None


def write_osz(
    path: str | Path,
    audio_path: str | Path,
    beatmaps: list[Beatmap],
) -> Path:
    """Package ``beatmaps`` together with the audio file into an .osz archive."""
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(audio_path, beatmaps[0].audio_filename)
        for bm in beatmaps:
            zf.writestr(bm.filename, bm.to_osu_string())
    return path
