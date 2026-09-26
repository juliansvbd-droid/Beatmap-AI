# Aufgabe: Gesangs-Rhythmus – Silben erkennen und messen, wann Mapper dem Gesang folgen

**Nur Prozessor, kein PyTorch, keine GPU** – nachts läuft der Nachtlauf (Prompt 05) auf der
Grafikkarte und darf nicht gestört werden. Zuerst `AGENTS.md`, `docs/STATUS.md`,
`docs/WORKLOG.md` lesen.

## Hintergrund
Nutzerbefund aus dem Blindtest: Unsere Rhythmus-KI (v4) ist gleich gut wie der beste
bekannte KI-Mapper, erkennt aber **Noten auf Gesang/Lyrics** kaum – ihre einzige
Rhythmus-Schwäche. Schwierig ist nicht nur Silben zu finden, sondern zu entscheiden,
**wann** man dem Gesang und wann dem Beat folgt.

Der Nutzer hat für ein eigenes Guitar-Hero-ähnliches Spiel (rein algorithmisch, ohne KI)
eine **Vocal Rhythm Engine** geschrieben. Sie ist die Vorlage (eigener Code des Nutzers,
darf verwendet werden):
- Original (maßgeblich): `C:\Users\Julian\Documents\nigga\RythmGame\systems\VocalRhythmEngine.js`
  und `systems\vocal\analysis.js`, `systems\vocal\chart.js`; die Eingaben (Bandfilter, HPSS,
  Flatness, Novelty, MFCC, Abschnitte mit Gesangsanteil „VIR“) kommen aus
  `systems\AudioProcessor.js`.
- Die Python-Dateien unter `audioprocessor_python\` sind ein **kaputter automatischer Port**
  (enthalten noch JavaScript-Syntax, laufen nicht) – nicht benutzen.
- Den RythmGame-Ordner **nicht verändern**.

## Schritte
1. **`beatmap_ai/vocals.py`** (numpy/librosa, kein torch, Repo-Stil): den Kern der Engine
   sauber neu umsetzen – nicht alles:
   - Gesangsband (2–6 kHz) + harmonisch/perkussiv-Trennung + Flatness → Stimm-Erkennung
     pro Frame (stimmhaft / stimmlos / still), mit Glättung wie im Original;
   - Silben-Anfänge (Energie-Anstieg, MFCC-Delta-Fluss, spektrale Neuheit, adaptive
     Schwelle, Mindestabstand) mit Stärke;
   - gehaltene Vokale (Dauer, Tonhöhen-Stabilität) als Slider-Kandidaten;
   - Gesangsanteil pro Abschnitt (Gesang im Vordergrund / instrumental / gemischt).
   Schwellen wie im Original übernehmen, Frame-Raster wie `beatmap_ai.audio` (FPS), damit
   es zu unseren Merkmalen passt. Laufzeit pro Song messen (Ziel: wenige Sekunden).
   Kleine Tests in `tests/test_vocals.py` mit synthetischen Signalen (z. B. Sinus-Silben
   mit Pausen → richtige Anzahl Silben; Rauschen → stimmlos; Stille → still).
2. **Messen an echten Maps** (Validierungssongs nach `dataset.is_validation(song_key)`,
   ausgewogen über Sternstufen <3 / 3–4,5 / 4,5–6 / 6+, einige hundert Maps; Audio aus
   `D:\BeatMap-AI-Dataset` bzw. `data`, `best_maps`):
   - pro Abschnitt (4–8 Takte) Label: Noten liegen eher auf **Silben** oder eher auf
     **Schlägen** (Onsets der perkussiven Spur / Beat-Raster), oder gemischt – mit einer
     Zufalls-Grundlinie (verschobene Silben), damit „Treffer über Zufall“ sichtbar ist;
   - Anteil „folgt Gesang“ je Sternstufe, je Songteil (Kiai ja/nein) und nach
     Gesangsanteil/Drum-Dichte des Abschnitts;
   - Treffsicherheit der Silben-Erkennung: Wie viele menschliche Noten in
     Gesangs-Abschnitten liegen auf einer erkannten Silbe (±30 ms), wie viele Silben haben
     eine Note?
   - Wie gut sagt eine **einfache Regel** (z. B. Gesangsanteil hoch + Drum-Dichte niedrig
     → folgt Gesang) das Label voraus (Genauigkeit auf getrennten Songs)?
3. **Bericht** `docs/gesang-rhythmus.md`: Zahlen, 5–10 Beispiele (Map, Zeit in ms), die der
   Nutzer im osu!-Editor nachschauen kann, Empfehlung für die Übergangsregel im Generator
   und welche Merkmale später in die Vorplanungs-KI (Prompt 04, „Anker: Gesang/Beat/
   gemischt“ pro Songteil) und in Rhythmus v5 sollen.

## Grenzen
- **Nichts am Generator, an der App oder an Dateien ändern, die der Nachtlauf benutzt**
  (`sequence_model.py`, `sequence_data.py`, `placement_*`, `rhythm.py`, `generator.py`,
  `ui.py`, `cli.py`, `scripts/night_*`). Nur neue Dateien: `beatmap_ai/vocals.py`,
  `tests/test_vocals.py`, `scripts/vocal_stats.py`, `docs/gesang-rhythmus.md`.
- Höchstens 4 Prozesse, große Zwischenergebnisse nach `D:\BeatMap-AI-Dataset\vocals\`.
- Tests nur für die neuen Dateien laufen lassen (die volle Suite lädt PyTorch).
- Am Ende nur die neuen Dateien committen und pushen; Eintrag in `docs/WORKLOG.md`.
