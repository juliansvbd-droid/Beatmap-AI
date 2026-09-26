# Projektstand BeatMap-AI

_Zuletzt aktualisiert: 2026-09-26 02:36, von Codex._
Jeder Agent aktualisiert diese Datei, bevor er aufhört (siehe `AGENTS.md`).

## Currently running / in progress
- **26.09.2026 02:36 – Codex, Prompt 01b:** Nachtrag im Generator, gepaarten Loader,
  Critic-Diagnostik und A/B-Skript umgesetzt; Syntaxprüfung erfolgreich. Die ROCm-UI mit
  PyTorch wurde auf Nutzerfreigabe geschlossen; der Prozesscheck ist frei. Jetzt läuft die
  Neuerzeugung von 3.000 Paaren (je 750 pro Sternbereich, möglichst mindestens ein Drittel
  Kiai-Fenster) nach `D:\BeatMap-AI-Dataset\critic_negatives_paired\`. Log
  `D:\BeatMap-AI-Dataset\critic_paired_generation.log`. Danach folgen die 200-Paar-Prüfung,
  Critic-Training und 40-Song-Messung. Trainings-Worker laden `critic_data.py` ohne PyTorch.

## Was die App gerade benutzt (`Start BeatMap AI.bat` → `.venv-rocm`)
| Aufgabe | Datei | Parameter | Stand |
|---|---|---|---|
| Rhythmus (wann Noten kommen) | `beatmap_ai/models/rhythm.pt` | 6,06 Mio. | Conformer v4, 10.387 Songs, Schwellen je Sternbereich |
| Platzierung (wo, Muster, Slider-Richtung, Hitsounds) | `beatmap_ai/models/sequence.pt` | 10,33 Mio. | Sequenzmodell v2 (fertig trainiert), wird per `SequencePlacer.follow` entlang des Rhythmus benutzt |
| Rückfall-Platzierung | `beatmap_ai/models/placement.pt` | 4,90 Mio. | v1, nur wenn sequence.pt fehlt |
| Bewerter (Critic) | `beatmap_ai/models/critic.pt` | 4,88 Mio. | aktiv (Pre-LN Transformer v1, Backup in `checkpoints/critic/critic_v1.pt`) |
| Tempo-Wahl | `beatmap_ai/timing.py` (`TEMPO_WEIGHTS`) | 8 Gewichte | aktiv |

Generator-Ablauf (`beatmap_ai/generator.py` → `generate_beatmap`): Rhythmus aus Frame-KI
(`rhythm.plan_objects`) → Platzierung mit v2 (`SequencePlacer.follow`) mit optionaler Best-of-N Critic-Bewertung (`critic.score_chunk`) → Sterne erreichen:
Modelle um höchstens −20 %/+25 % härter/leichter anfragen (`steer`), dann Notenmenge, dann
Abstände nur ±15 % → Refrain-Kopien (`structure.copy_sections`) + Kiai (mit schnelleren
Slidern im Kiai, `sv_at`).

## Messwerte (auf Songs, die die Modelle nie gesehen haben)
- Rhythmus-F1 (±30 ms gegen menschliche Map): gesamt ~0,75; 0–3★ 0,67 · 3–4,5★ 0,75 ·
  4,5–6★ 0,80 · 6★+ 0,82. Zwei Menschen untereinander: ~0,53–0,84 je nach Sternen
  (nur 54 Paare gemessen).
- Bewegung mit v2 (Mensch → KI): gleiche Wendung wie davor 26,4 → 26,0 %, scharfe
  Wendungen 30 → 31 %, hin und zurück 7,7 → 6,7 %, Überdeckungen 2,1 → 0,7 %.
- Slider (nach den Korrekturen vom 25.09. abends): gebogen 3,5★ 11 % (Mensch 7 %),
  Kreisslider 0–2 %, wiederholende Slider vorhanden.
- **Bewerter-KI (Critic v1 Stand)**:
  - Val Accuracy: 98,24 % (Shortcut-Problem: Positiv/Negativ-Diskrepanz bei Songs & Sternen, Frame-Rhythmus).
  - A/B Best-of-4: Critic Score stieg von 1,30 % auf 4,36 %, Bewegungsfluss verbessert.
  - Nachbesserung läuft (Prompt 01b) für echten Platzierungs-Fokus.

## Rückmeldungen des Nutzers (was noch stört)
- Gut: Stil-Auswahl funktioniert, oft besser als „Auto“.
- Offen: noch nicht perfekt; Maps fühlen sich noch nicht individuell für den Song an;
  bei hohen Sternen inkonsistenter; Jump-Muster (Zickzack, Vielecke) fehlen bzw. zu
  zufällig; Doubles zu selten (1,0 statt 2,4 pro 100 Noten).

## Offene Punkte / nächste Schritte
1. **Prompt 01b (in Arbeit)**:
   - Gepaarte Daten: Positiv = Quell-Difficulties der Negativ-Maps (gleicher Song, gleiche Sterne, gleiches Timing).
   - Gleicher Rhythmus, andere Platzierung: 3.000 Negativ-Maps mit menschlichem Rhythmus (75 % Sequenz, 25 % Regeln).
   - Abkürzungen abstellen: Ausgewogene Verteilung (4x 750 je Sternbereich), Jitter-Test + Audio-Shuffle-Test.
   - Re-Training & A/B-Messung auf 40 Validierungssongs (alle Diff-Stufen).
2. Prompt 02: Song-Passungs-KI (`prompts/02-songpassung-und-mehrere-durchgaenge.md`).
3. 5,5★+ nach den Slider-Korrekturen vom 25.09. noch nicht nachgemessen (GPU frei nötig).
4. Angeboten, noch nicht entschieden: **„Auto“ wählt einen Stil per einfacher
   Songanalyse** (viele schnelle Noten → Stream, klare Beats → Jump, ruhig → Flow), bis
   die Vorplanungs-KI (Prompt 04) existiert. Hintergrund: Nutzer bekommt mit gewähltem
   Stil oft bessere Maps als mit Auto. Nutzer fragen, ob gewünscht.
5. Beim Feinschliff der Sterne (Abstände ±15 %) werden noch ~15–25 % der Sprünge
   nachträglich gedreht → besser neu würfeln statt skalieren.
6. Angeboten, noch nicht entschieden: **Git im Projekt einrichten** (nur lokal), damit
   Änderungen der verschiedenen Agenten nachvollziehbar/rückgängig machbar sind.
7. `README.md` beschreibt an mehreren Stellen noch den älteren Stand (u. a. regelbasierte
   Platzierung als Standard und fehlende Hitsounds/Kiai), während der aktuelle Generator
   bereits das Sequenzmodell und Kiai nutzt. Nach Abschluss von Prompt 01b aktualisieren.

## Prompts für Antigravity (Reihenfolge, nie parallel)
| Nr. | Datei | Inhalt | Status |
|---|---|---|---|
| 01 | `prompts/01-bewerter-ki.md` | Bewerter-KI (Mensch vs. KI), Best-of-N | **fertig** |
| 01b | `prompts/01b-bewerter-ki-nachbessern.md` | Critic nachbessern: gepaarte Positiv-Maps (gleicher Song/Sterne), Negativ-Maps mit menschlichem Rhythmus, A/B auf 40 Songs | **läuft** |
| sol-01 | `prompts/sol-01-muster-messungen.md` | Für GPT-6 Sol, ohne GPU, parallel zu 01b möglich: Muster-Messwerkzeug (Jump-Muster, Doubles, Wiederholung, zu schwere Muster je Sternbereich) | bereit |
| 02 | `prompts/02-songpassung-und-mehrere-durchgaenge.md` | Song-Passungs-KI (Idee des Nutzers) + mehrere Durchgänge | wartet auf 01b |
| 03 | `prompts/03-v3-sliderformen-und-muster.md` | v3: echte Slider-Formen, Auto-Tagging + ausgewogene Daten (Sterne × Stil), Abschnitts-Vorgaben, Muster-Messungen, AR/OD/HP/CS | wartet auf 02 |
| 04 | `prompts/04-vorplanungs-ki.md` | Vorplanungs-KI (Idee des Nutzers): Songteile erkennen (Hauptteil/Höhepunkt, gelernt aus Kiai + Intensitätswechseln), Sterne/Stil empfehlen, Plan pro Teil | wartet auf 03 |

## Environment (wichtig, hat echte Abstürze verursacht)
- Python: `.venv-rocm\Scripts\python.exe` (PyTorch 2.9 + ROCm 7.2.1, AMD RX 7700 XT).
  MIOpen läuft unter Windows nicht → Gerät immer über `beatmap_ai.train.resolve_device("cuda")`.
  `.venv` = alte DirectML-Umgebung (Rückfall).
- **Höchstens ein PyTorch-Prozess gleichzeitig.** Jeder reserviert 3–4 GB Commit; Limit
  32 GB RAM + 32 GB Auslagerungsdatei (D:) war mehrfach voll. Die Oberfläche beim Testen
  zählt mit. Hilfsprozesse ohne PyTorch (Muster: `sequence_data.batch_stream`).
- Speicher: C: ~4 GB frei (!), D: ~65 GB frei. Datensatz: `D:\BeatMap-AI-Dataset`
  (~13.400 Sets mit Audio + `mel.npy`, `tags.json` Community-Tags, `catalog.json` aller
  38.048 ranked std-Sets) plus `data/`, `best_maps/` im Repo.
- Tests: `.venv-rocm\Scripts\python.exe -m pytest -q` (38 Tests, alle grün am 25.09. abends).
- Validierungs-Aufteilung immer nach `beatmap_ai.dataset.is_validation(song_key(bm))`.
- Messwerkzeuge: `beatmap-ai evaluate`, `scripts/tune_threshold.py`, `scripts/map_stats.py`,
  `scripts/eval_sequence.py` (`--follow` = so wie die App), `scripts/human_agreement.py`.
