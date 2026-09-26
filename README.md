# Beatmap-AI

Generate osu!standard beatmaps from an MP3.

```bash
pip install -e .                      # add ".[train]" for the neural model
beatmap-ai generate "Artist - Title.mp3"
# -> "Artist - Title.osz" with Normal, Hard and Insane difficulties
```

Double-click the `.osz` (or drop it into osu!) to import it.

## Desktop-Oberfläche unter Windows

Starte **`Start BeatMap AI.bat`** per Doppelklick. In der Oberfläche kannst du Songs
auswählen, Schwierigkeiten oder genaue Sternzahlen (z. B. `3.8; 4.5; 6`) festlegen,
einen Stil wählen (Auto, Jump, Stream, Tech, Flow, Circles – mit Stärke-Regler),
Beatmaps erzeugen und die KI mit lokalen Beatmap-Sammlungen weitertrainieren (mehrere
Ordner mit `;` trennen). Die Trainings-Voreinstellungen richten sich nach dem erkannten
Rechengerät (ROCm/CUDA, DirectML oder CPU). Alle Dateien bleiben auf deinem PC.

Alternativ lässt sich die Oberfläche aus einer aktivierten Python-Umgebung mit
`python -m beatmap_ai ui` öffnen.

## How it works

Beatmap-AI splits the job into four steps:

1. **Audio analysis** (`beatmap_ai/audio.py`): computes a log-mel spectrogram, onset
   strength (overall and bass-only) and loudness. The tempo comes from librosa's beat
   tracker, then gets refined against the onset curve over the whole song. Integer BPMs
   win when they fit about as well as a fractional one. Bass onsets decide between beat
   and off-beat, and multiband spectral flux breaks close tempo ties. A tempo-regularized
   beat tracker follows clear local tempo changes; swing is applied only when enough
   onset evidence supports it. Variable-tempo maps receive timing points at bar boundaries.
2. **Rhythm** (`beatmap_ai/rhythm.py`): decides *when* notes happen. Every 1/2 or 1/4
   tick of the beat grid gets a score, and ticks are picked one 4-bar window at a time
   so each part of the song gets the difficulty's note density. Loud sections get more
   notes, quiet ones fewer. The picks follow each difficulty's rules: shortest gap
   between notes, longest stream, and how long a slider needs before the next note.
   Sustained sounds become sliders, and long held passages become spinners.
   * **Heuristic mode** (default) scores ticks by onset strength and beat position.
   * **Model mode** (`--model`) scores ticks with a trained neural network. Newer
     models also set slider lengths (how long a sound is held) and the spacing of
     every note (see below).
3. **Placement** (`beatmap_ai/placement.py`): decides *where* notes go. Spacing grows
   with the time between notes (distance snapping). With a newer model the distance
   snap of each note comes from the model -- jumps where mappers emphasise the music,
   tight spacing in calm parts -- widened to the variation mappers use at that star
   rating. Otherwise loud notes get jumps on Hard and above. The path curves smoothly within a combo and turns sharply on new combos.
   Sliders are circular arcs whose length is exact. Any position that would leave the
   playfield or cover a recent note is rejected.
4. **Output** (`beatmap_ai/osu.py`): writes `.osu` v14 files with combos, break periods,
   preview time and audio lead-in, and packs them into an `.osz`.

### Difficulties

The presets follow the medians of ~2,200 ranked osu!standard difficulties (grouped by
note density): density, slider share, slider speed, spacing, AR/OD/CS/HP.

| Name   | Notes/sec | Snap | Sliders | Slider speed | AR  | OD  | CS | HP |
|--------|-----------|------|---------|--------------|-----|-----|----|----|
| easy   | 1.1       | 1/2  | ~55%    | 0.9          | 3   | 2   | 3  | 2  |
| normal | 1.7       | 1/2  | ~65%    | 1.1          | 5   | 4   | 3  | 4  |
| hard   | 2.9       | 1/4  | ~57%    | 1.5          | 8   | 6.5 | 4  | 5  |
| insane | 3.9       | 1/4  | ~42%    | 1.7          | 9   | 8   | 4  | 6  |
| expert | 5.2       | 1/4  | ~32%    | 1.8          | 9.4 | 9   | 4  | 6  |

Tweak them in `beatmap_ai/difficulty.py`.

### Star ratings and styles

Instead of a name, `-d` also takes a star rating (`-d 3.8 -d 4.5 -d 6`). AR, OD, CS,
HP, slider speed and density then follow the medians of ranked maps with that rating,
and the finished map is checked with osu!'s star calculation (`rosu-pp-py`): spacing,
and if needed the note density, are adjusted until the rating matches (usually within
±0.05 stars).

`--style` gives the map a character on top of "Auto" (no style: the model decides):

| Style     | Effect                                              |
|-----------|-----------------------------------------------------|
| `jump`    | larger spacing, fewer streams                        |
| `stream`  | more runs of 1/4 notes, smaller jumps                |
| `tech`    | more varied rhythms, notes on 1/4 and 3/4 positions  |
| `flow`    | calm spacing, more sliders                           |
| `circles` | few sliders                                          |

Add a strength from 0 to 1 (`--style jump=0.8`) and repeat to combine styles. The
strength moves the style's measures from the median of ranked maps at the chosen star
rating towards their 95th (or 5th) percentile. Styles need a model trained with style
conditions (any model trained with this version when `rosu-pp-py` is installed).

## Usage

```bash
# Choose difficulties
beatmap-ai generate song.mp3 -d easy -d hard -d expert -o song.osz

# Set metadata or timing by hand if detection gets it wrong
beatmap-ai generate song.mp3 --artist "Artist" --title "Title" --bpm 174 --offset 312

# Only print the detected timing
beatmap-ai analyze song.mp3

# Use a trained rhythm model
beatmap-ai generate song.mp3 --model beatmap_model.pt

# Star ratings and a style
beatmap-ai generate song.mp3 -d 3.8 -d 5.2 --style jump=0.7
```

If the file name looks like `Artist - Title.mp3`, the metadata is filled in from it.

## Training the AI on real beatmaps

The neural rhythm model (`beatmap_ai/model.py`) learns from existing ranked maps where
notes go in the music:

* a CNN reads the mel spectrogram
* conformer blocks (self-attention and convolution) add context; they see 12 seconds
  at a time and longer songs are processed in overlapping windows. Older checkpoints
  use a bidirectional GRU instead and still load.
* four outputs give, for every ~11.6 ms frame, the chance a note starts there, the
  chance it is a slider, the chance a slider is being held, and the distance snap a
  mapper would use there

It is conditioned on the beat grid (position in the beat and bar), on note density,
star rating and style measures (`beatmap_ai/style.py`). Each condition is hidden from
the model at random during training, so it also maps well when only some of them are
given ("Auto"). Maps are drawn so every star rating is trained on, not just the
common ones.

```bash
pip install -e ".[train]"
beatmap-ai train ~/osu!/Songs -o beatmap_model.pt --epochs 30
```

To continue training an existing checkpoint, add `--init-model path/to/model.pt`
(its architecture is kept). `--arch gru` trains the original GRU design.

The data folders can hold `.osz` files, extracted song folders, or both; pass several
(`beatmap-ai train data best_maps`). Your osu! `Songs` folder works as is. Only
osu!standard difficulties are used, and copies of the same difficulty count once. The
trainer:

* caches spectrograms and parsed maps in `.beatmap_ai_cache/`
* keeps about 10% of the songs aside for validation, by artist and title, so another
  mapset of the same song never ends up on the other side
* reports note F1 on those songs after each epoch
* saves the best checkpoint with its tuned decision threshold
* writes a `.learning.json` report beside the checkpoint with the maps and rhythm
  patterns used, plus the validation result; the desktop app opens this summary when
  training finishes

A GPU is used when one is available. For good results, train on a few hundred ranked
maps in the style you want.

## Training on your own GPU

Training uses the GPU automatically when PyTorch can see one:

* **NVIDIA:** the regular `pip install torch` (CUDA build).
* **AMD on Linux:** install the ROCm build of PyTorch. Pick "ROCm" in the selector on
  pytorch.org to get the `pip install` command. ROCm builds report the GPU as `cuda`, so
  no extra flag is needed. Some RDNA3 cards need the environment variable
  `HSA_OVERRIDE_GFX_VERSION=11.0.0`, including the RX 7700 XT/7800 XT (gfx1101).
* **AMD on Windows:** AMD's ROCm build of PyTorch for Windows (Python 3.12; see
  "Install PyTorch for Radeon GPUs" in AMD's ROCm docs) works with the conformer
  model. MIOpen cannot compile its kernels on Windows, so the trainer switches it off
  and PyTorch's own GPU kernels are used; the GRU design is slow there. On an
  RX 7700 XT the default conformer trains at ~40 twelve-second excerpts per second.
  `Start BeatMap AI.bat` prefers a `.venv-rocm` environment when it exists.
  Alternatively `pip install torch-directml` and pass `--device directml` (about half
  the speed).

Check with `python -c "import torch; print(torch.cuda.is_available())"`. Then, for example:

```bash
python scripts/download_dataset.py D:/maps --count 10000 --keep-audio  # ranked sets
beatmap-ai train D:/maps -o rhythm.pt --epochs 60 --steps-per-epoch 500 \
    --batch-size 16 --hidden 256 --layers 6
python scripts/tune_threshold.py D:/maps -m rhythm.pt --write  # thresholds per star range
beatmap-ai evaluate D:/maps -m rhythm.pt                # compare with the heuristics
python scripts/map_stats.py D:/maps -m rhythm.pt        # rhythm/spacing vs. human maps
cp rhythm.pt beatmap_ai/models/rhythm.pt                # make it the default model
python scripts/fit_timing.py data/                      # optional: refit BPM detection
```

`scripts/download_dataset.py` mixes the most played, most favourited and most recently
ranked sets, skips songs you already have (`--skip data`), precomputes each song's
spectrogram (`mel.npy`) and, without `--keep-audio`, deletes the audio of training
songs (about 2 MB per set instead of 7). It stops before the disk drops below
`--min-free-gb`.

`--hidden` sets the model width (default 192), `--layers` the number of conformer
blocks (default 4) and `--chunk-seconds` the length of each training excerpt (default
12). A bigger model needs more data and more epochs.

`beatmap-ai evaluate` reports the rhythm F1 (generated vs. human note times, ±30 ms)
overall and per star range. For scale: two human mappers of the same song reach about
0.80 against each other (median of 134 pairs from 6 songs).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests build synthetic drum tracks with known BPM and offset. They check that timing
is detected correctly and that every generated difficulty is valid:

* objects are on the grid, in order and inside the playfield
* sliders end on the grid
* objects don't overlap in time

They also run a short train → generate cycle for the model.

## Limitations and next steps

* Generated beat grids assume 4/4. Subtle tempo changes or weak beat signals may still need
  manual correction with `--bpm`/`--offset`.
* The model sets the spacing, but directions, patterns and slider shapes are still
  rule-based. A good next step is a learned placement model, for example a sequence
  model over (Δtime, distance, angle) trained on the same data.
* No hitsounds, kiai or slider-velocity changes yet.
* osu!stable plays only `.mp3`/`.ogg` audio.
