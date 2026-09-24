# Beatmap-AI

Generate osu!standard beatmaps from an MP3.

```bash
pip install -e .                      # add ".[train]" for the neural model
beatmap-ai generate "Artist - Title.mp3"
# -> "Artist - Title.osz" with Normal, Hard and Insane difficulties
```

Double-click the `.osz` (or drop it into osu!) to import it.

## How it works

Beatmap-AI splits the job into four steps:

1. **Audio analysis** (`beatmap_ai/audio.py`): computes a log-mel spectrogram, onset
   strength (overall and bass-only) and loudness. The tempo comes from librosa's beat
   tracker, then gets refined against the onset curve over the whole song. Integer BPMs
   win when they fit about as well as a fractional one. Bass onsets decide between beat
   and off-beat, and between tempo octaves. On test tracks the offset is accurate to
   about 1 ms.
2. **Rhythm** (`beatmap_ai/rhythm.py`): decides *when* notes happen. Every 1/2 or 1/4
   tick of the beat grid gets a score, and ticks are picked one 4-bar window at a time
   so each part of the song gets the difficulty's note density. Loud sections get more
   notes, quiet ones fewer. The picks follow each difficulty's rules: shortest gap
   between notes, longest stream, and how long a slider needs before the next note.
   Sustained sounds become sliders, and long held passages become spinners.
   * **Heuristic mode** (default) scores ticks by onset strength and beat position.
   * **Model mode** (`--model`) scores ticks with a trained neural network.
3. **Placement** (`beatmap_ai/placement.py`): decides *where* notes go. Spacing grows
   with the time between notes (distance snapping). On Hard and above, loud notes also
   get jumps. The path curves smoothly within a combo and turns sharply on new combos.
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
```

If the file name looks like `Artist - Title.mp3`, the metadata is filled in from it.

## Training the AI on real beatmaps

The neural rhythm model (`beatmap_ai/model.py`) learns from existing ranked maps where
notes go in the music:

* a CNN reads the mel spectrogram
* a 2-layer bidirectional GRU adds context from the whole song
* two outputs give, for every ~11.6 ms frame, the chance a note starts there and the
  chance it is a slider

It is conditioned on the beat grid (position in the beat and bar) and on note density,
so one model covers every difficulty.

```bash
pip install -e ".[train]"
beatmap-ai train ~/osu!/Songs -o beatmap_model.pt --epochs 30
```

The data folder can hold `.osz` files, extracted song folders, or both. Your osu!
`Songs` folder works as is. Only osu!standard difficulties are used. The trainer:

* caches spectrograms in `.beatmap_ai_cache/`
* keeps about 10% of the songs aside for validation
* reports note F1 on those songs after each epoch
* saves the best checkpoint with its tuned decision threshold

A GPU is used when one is available. For good results, train on a few hundred ranked
maps in the style you want.

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

* One BPM per song. Songs with tempo changes need `--bpm`/`--offset` or a
  variable-timing detector, and only 4/4 is assumed.
* Placement is rule-based. A good next step is a learned placement model, for example
  a sequence model over (Δtime, distance, angle) trained on the same data.
* No hitsounds, kiai or slider-velocity changes yet.
* osu!stable plays only `.mp3`/`.ogg` audio.
