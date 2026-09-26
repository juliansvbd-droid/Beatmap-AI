# Instructions for every AI agent working on BeatMap-AI

Several agents (Claude Code, Antigravity/Gemini, others) take turns on this repo. The
user switches between them when usage limits run out. To keep everyone in sync:

## Before you start
1. Read `docs/STATUS.md` (where the project stands right now) and the newest entries at
   the top of `docs/WORKLOG.md` (what the previous agents did).
2. Check "Currently running / in progress" in `docs/STATUS.md`. If another agent's job
   (training, dataset generation) is listed as running, do **not** start GPU jobs or edit
   the files listed there — ask the user first.
3. Re-read any file right before editing it (another agent may have changed it). Make
   targeted edits; never rewrite whole files from an old copy.

## While you work
- Update "Currently running / in progress" in `docs/STATUS.md` when you start a long job
  (what, which files, which log, when started) and clear it when done.
- Environment rules (they caused real crashes): see "Environment" in `docs/STATUS.md`.
  Most important: at most **one** PyTorch (ROCm) process at a time; worker processes must
  not import torch; large files go to `D:\BeatMap-AI-Dataset\`, C: is almost full.

## Before you stop (also when the user says the limit is almost used up)
1. Add an entry at the **top** of `docs/WORKLOG.md`:
   date/time, agent name, what you changed (files), results with numbers, anything left
   half-done and how to continue, and what you promised the user.
2. Update `docs/STATUS.md` so it is true again (models in use, what works, open
   problems, next steps, running jobs).
3. Keep both in the user's language for user-facing notes (German) where it helps; code
   comments stay English.

## Git / GitHub
- The project is on GitHub: `juliansvbd-droid/Beatmap-AI`, branch
  `claude/osu-beatmap-mp3-generation-f2quz1` (local branch `main` tracks it). Cloud agents
  work from this repo; they have no GPU and no dataset on D:, so GPU/data tasks stay local.
- The user has approved commits and pushes. After a finished step: `git pull` first (a cloud
  agent may have pushed), then commit your own files with a clear message and push with
  `git push origin main:claude/osu-beatmap-mp3-generation-f2quz1`.
- Never commit songs, datasets or checkpoints (`.gitignore` covers `data/`, `best_maps/`,
  `Music/`, `checkpoints/`, audio, `.osz`). Only the bundled `beatmap_ai/models/*.pt`
  are tracked; replace one only when the prompt says the new model is better.

Bulk tasks meant for another agent are written as prompts in `prompts/NN-*.md`; their
status is tracked in `docs/STATUS.md` ("Prompts").
