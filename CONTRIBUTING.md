# Contributing to Beatmap-AI

Beatmap-AI is a research prototype. Useful contributions include reproducible
generation failures, evaluation improvements and small, measured changes to musical
choice or playability. See [the research plan](docs/RESEARCH.md) and
[the current project status](docs/STATUS.md) before proposing a large experiment.

## Report a failure

Open an issue with:

- Commit, Python/dependency versions, operating system and compute device.
- Command or GUI settings, random seed, model names and checkpoint hashes.
- Song/map identifier, target difficulty and achieved star rating.
- Expected musical or movement behavior and the observed failure.
- A minimal example you have permission to share; do not upload third-party songs.

Distinguish a format/geometry bug from a subjective mapping preference. If possible,
identify the affected timestamps or object numbers.

## Propose a change

Explain the hypothesis, which part of the pipeline changes and how it will be
compared with the previous behavior. For model or generation changes, use the same
songs, seeds and settings, report multiple quality measures and include regressions.
Keep default behavior and experimental configurations clearly documented.

From an existing checkout:

```bash
pip install -e ".[dev]"
pytest
```

Run checks appropriate to the change. Documentation-only contributions do not require
GPU training. Do not commit songs, local datasets or new checkpoints as part of an
ordinary contribution. Changes to bundled checkpoints need evidence and maintainer
review.

AI coding assistants should also follow [AGENTS.md](AGENTS.md), including updating
the status and work log. AI-assisted contributions need human review; generated
analysis is not evidence until checked against code or experiment results.

## Licensing status

There is currently no project license. Please discuss substantial code contributions
with the maintainer while licensing is unresolved; a public repository alone does not
establish open-source reuse rights. Do not add material whose rights you cannot grant.
