"""Human-readable summaries of the beatmaps used for model training."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .audio import FPS
from .dataset import MapExample


_RHYTHM_DIVISIONS = (
    (1, "beat"),
    (2, "half_beat"),
    (3, "triplet"),
    (4, "quarter_beat"),
    (6, "sixth_beat"),
)


def _rhythm_distribution(examples: list[MapExample]) -> dict[str, int]:
    counts = {name: 0 for _, name in _RHYTHM_DIVISIONS}
    counts["off_grid"] = 0

    for example in examples:
        if not len(example.note_frames) or not example.grid:
            continue
        grid = sorted(example.grid)
        starts = np.asarray([point[0] for point in grid], dtype=np.float64)
        beat_lengths = np.asarray([point[1] for point in grid], dtype=np.float64)
        times = example.note_frames.astype(np.float64) * 1000.0 / FPS
        active = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        phases = np.mod((times - starts[active]) / beat_lengths[active], 1.0)
        unassigned = np.ones(len(phases), dtype=bool)

        # Assign each note to the simplest subdivision it closely matches.
        # This avoids counting a beat note again as a half-beat or triplet.
        for divisions, name in _RHYTHM_DIVISIONS:
            anchors = np.arange(divisions, dtype=np.float64) / divisions
            distance = np.abs(phases[:, None] - anchors[None, :])
            distance = np.minimum(distance, 1.0 - distance).min(axis=1)
            matched = unassigned & (distance <= 0.045)
            counts[name] += int(matched.sum())
            unassigned[matched] = False
        counts["off_grid"] += int(unassigned.sum())

    return counts


def build_learning_report(
    examples: list[MapExample],
    train_examples: list[MapExample],
    validation_examples: list[MapExample],
    validation_f1: float,
    threshold: float,
    checkpoint: str | Path,
) -> dict[str, object]:
    """Summarize the patterns present in the exact usable examples seen by training."""
    densities = np.asarray([example.density for example in examples], dtype=np.float64)
    tempos = []
    for example in examples:
        bpm_values = [60000.0 / beat_length for _, beat_length, _ in example.grid
                      if beat_length > 0]
        if bpm_values:
            tempos.append(float(np.median(bpm_values)))

    note_count = sum(len(example.note_frames) for example in examples)
    slider_count = sum(len(example.slider_frames) for example in examples)
    rhythm_counts = _rhythm_distribution(examples)
    rhythm_total = sum(rhythm_counts.values())
    rhythm_labels = {
        "beat": "Auf dem Beat",
        "half_beat": "Halbe Beats",
        "triplet": "Triolen",
        "quarter_beat": "Viertelbeats",
        "sixth_beat": "Sechstelbeats",
        "off_grid": "Nicht eng am Beat-Raster",
    }

    density_summary = {
        "median": float(np.median(densities)) if len(densities) else 0.0,
        "p25": float(np.percentile(densities, 25)) if len(densities) else 0.0,
        "p75": float(np.percentile(densities, 75)) if len(densities) else 0.0,
    }
    tempo_summary = {
        "median": float(np.median(tempos)) if tempos else None,
        "p10": float(np.percentile(tempos, 10)) if tempos else None,
        "p90": float(np.percentile(tempos, 90)) if tempos else None,
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "maps": len(examples),
        "songs": len({example.mel_path for example in examples}),
        "training_maps": len(train_examples),
        "validation_maps": len(validation_examples),
        "note_starts": note_count,
        "slider_starts": slider_count,
        "slider_share": slider_count / max(note_count, 1),
        "tempo_bpm": tempo_summary,
        "density_objects_per_second": density_summary,
        "rhythm_grid": [
            {
                "key": key,
                "label": rhythm_labels[key],
                "count": count,
                "share": count / max(rhythm_total, 1),
            }
            for key, count in rhythm_counts.items()
        ],
        "validation_f1": float(validation_f1),
        "decision_threshold": float(threshold),
    }


def save_learning_report(report: dict[str, object], path: str | Path) -> Path:
    """Atomically write a JSON sidecar next to the trained checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
