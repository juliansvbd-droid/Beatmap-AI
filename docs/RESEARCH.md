# Research scope and evaluation

## Question and current stage

Beatmap-AI investigates whether generative deep learning can capture the decisions
that make osu!standard mapping feel intentional: selecting a musical voice, varying
rhythm, emphasizing phrases and designing movement that is comfortable to play.

"Human intuition" is shorthand for these observable choices, not a claim that a
model understands music as a person does. Beat placement alone is insufficient:
two maps can hit the same onsets and differ greatly in flow, emphasis and playability.

This is an active prototype. The repository contains a working generation pipeline,
training code, bundled checkpoints, analysis tools and tests. Some experiment
artifacts and training datasets remain local; their presence in the development log
does not make a reported result independently reproducible from this repository.

## What exists today

| Component | Evidence in the repository | Research role |
| --- | --- | --- |
| Learned rhythm | `beatmap_ai/model.py`, `train.py`, `rhythm.py` | Predict note starts, slider use, sustain and spacing from audio and conditions |
| Learned placement | `beatmap_ai/sequence_model.py`, `placement_model.py` | Predict object movement and patterns conditioned on rhythm |
| Candidate scoring | `beatmap_ai/critic.py`, `songfit.py` | Investigate whether candidate ranking improves maps; classifier accuracy alone is insufficient |
| Section planning | `beatmap_ai/planner.py`, `structure.py` | Connect local patterns to song structure and repeated passages |
| Rhythm evaluation | `beatmap_ai/evaluate.py` | Compare heuristic and learned rhythm on held-out songs |
| Movement and pattern analysis | `scripts/pattern_stats.py`, `compare_maps.py`, `jump_geometry.py` | Examine distributions, flow proxies and extreme movement |
| Vocal analysis prototype | `beatmap_ai/vocals.py`, `docs/gesang-rhythmus.md` | Study when human mapping follows vocals; not integrated into generation |
| Validity checks | `tests/` | Catch timing, serialization and geometry regressions |

The default application uses sequence v2. The opt-in "v4 Formen" configuration
combines a newer sequence model with planning and explicit shape/density controls.
The system is hybrid; improvements cannot automatically be credited to the learned
model rather than those controls.

## Intended ecosystem contribution

The intended beneficiaries are osu! mappers and researchers studying audio-conditioned
generation and playable content. The proposed contribution is an inspectable pipeline
and evaluation tools that expose *why* a generated map differs from human mapping,
including failure cases and experiments that do not improve quality.

This is intended value, not established ecosystem dependence. As of 2026-10-06 the
project has no substantial adoption metrics to report. No claim is made that other
projects depend on it, that it is critical infrastructure, or that it is the first
neural beatmap generator.

Related projects already explore learned beatmap generation:
[osu-diffusion](https://github.com/OliBomby/osu-diffusion) and
[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator).
Beatmap-AI's research focus is the joint evaluation of musical choice, movement and
playability. Comparative superiority has not been demonstrated.

## Evaluation protocol to develop

1. **Fix the experiment identity.** Record commit, checkpoint hashes, source map IDs,
   song grouping, difficulty targets, flags, seeds, device and dependency versions.
   Keep songs used for development separate from an untouched final test set.
2. **Compare explicit baselines.** Generate heuristic rhythm + rule placement,
   learned rhythm + rule placement, the default learned pipeline, and opt-in v4
   variants. Change one component at a time and compare at similar achieved difficulty.
3. **Measure the entire pipeline.** Count rhythm figures after tick selection,
   slider conversion, placement, repeated-section processing and star adjustment.
   This separates model failures from post-processing losses.
4. **Use multiple outcomes.** Report timing/rhythm F1, slider share, rhythm figures,
   density, achieved star error, movement extremes, pattern variety and runtime.
   A single human reference is not the only valid musical interpretation.
5. **Assess musical quality and playability.** A future evaluation should use
   anonymized, order-randomized paired playtests with experienced players/mappers.
   Report evaluator count, songs, uncertainty and failure cases. Automated metrics
   alone cannot establish enjoyable play or human-like musical judgment.

Existing diagnostic command (requires a user-supplied local collection):

```bash
pip install -e ".[train]"
beatmap-ai evaluate /path/to/osu/Songs -m beatmap_ai/models/rhythm.pt --max-songs 40
```

This command evaluates timing/rhythm with reference-map timing and density; it is
not a complete end-to-end playability benchmark. Historical validation songs have
been used repeatedly during development. The untouched final test set and the
broader protocol above remain planned work, not completed validation.

## Next milestones and acceptance evidence

| Milestone | Completion evidence |
| --- | --- |
| Reproduce rhythm losses | A run manifest and stage-by-stage counts for the same songs, seeds and settings |
| Improve musical variation | Ablations showing where doubles, triples, bursts, sliders and pauses improve without worse flow or difficulty errors |
| Validate vocal signals | A manually checked sample and failure analysis before generator integration |
| Make experiments reproducible | Versioned configuration, checkpoint identification, environment instructions and shareable result summaries |
| Evaluate actual play | Paired player/mapper assessments alongside automated measurements, with limitations reported |

Claude is intended to help inspect interactions across the training and generation
pipeline, debug models and data processing, design ablations, improve tests and turn
local experiments into documented contributions. Training runs on the maintainer's
own compute; the requested subscription would support research and development.

## Licensing and data release

No project license is currently present. Before describing the repository as
licensed open source, the maintainer must select an OSI-approved license for material
they can license and review third-party code, map assets and bundled model weights.
Music and maps retain their respective owners' rights. Prefer map identifiers and
locally reproduced features over redistributing third-party audio or datasets.
