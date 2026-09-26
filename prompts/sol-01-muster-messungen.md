# Aufgabe: Muster-Messungen für BeatMap-AI (ohne GPU)

## Worum es geht
BeatMap-AI erzeugt osu!standard-Maps aus MP3s. Der Nutzer bemängelt beim Spielen:
- Jump-Muster (Zickzack, Dreiecke, Vierecke, Fünfecke, Sterne, Rückwärts-Jumps) fehlen oder
  wirken zufällig; Muster sollen wiederkehren, aber nicht stumpf (außer es passt zum Song).
- Muster aus 6★+-Maps tauchen in 4★-Maps auf; kurze, stark gebogene Slider.
- Zu wenige Doubles/Triples (KI 1,0 statt Mensch 2,4 pro 100 Noten).
- „Fühlt sich an, als hätte sich ein Mensch 5 Maps angeguckt und die Muster reingeforced.“

Bisher gibt es dafür keine Zahlen. Baue ein **Messwerkzeug**, das diese Punkte pro
Sternbereich für menschliche Maps und KI-Maps vergleichbar macht. Spätere Modelle
(Prompts 02–04) werden damit bewertet.

## Umgebung und Regeln (genau beachten)
- Repo: `C:\Users\Julian\Documents\BeatMap-AI`. Zuerst `AGENTS.md`, `docs/STATUS.md`,
  `docs/WORKLOG.md` lesen; vor dem Aufhören beide Dateien aktualisieren.
- **Kein GPU, kein PyTorch.** Ein anderer Agent (Antigravity, Prompt 01b) trainiert gerade auf
  der GPU; RAM/Auslagerungsdatei sind knapp. Dein Code darf `torch` weder direkt noch indirekt
  importieren (also nicht `beatmap_ai.generator`, `*_model.py`, `critic.py`, `train.py`).
  Erlaubt: `beatmap_ai.osu`, `beatmap_ai.dataset` (nur die torch-freien Funktionen),
  `beatmap_ai.placement_data`, `beatmap_ai.style`, numpy. Höchstens 4 Prozesse.
- Python: `.venv-rocm\Scripts\python.exe`. Keine neuen Abhängigkeiten.
- **Nicht anfassen:** `beatmap_ai/critic*.py`, `beatmap_ai/sequence_model.py`,
  `scripts/make_paired_negatives.py`, `scripts/eval_ab_critic.py` (daran arbeitet Antigravity),
  `beatmap_ai/models/`, `checkpoints/`, alle Daten auf D:. Keine Maps erzeugen.
- Code-Stil wie im Repo (englische Docstrings/Kommentare, gleiche Dichte/Benennung).
- Tests: `.venv-rocm\Scripts\python.exe -m pytest -q tests/test_patterns.py` (nur deine neuen
  Tests laufen lassen, die volle Suite lädt PyTorch).

## Daten
- Menschlich: `data/`, `best_maps/`, `D:/BeatMap-AI-Dataset/` über
  `beatmap_ai.dataset.iter_beatmap_texts`; Sterne über `beatmap_ai.style.star_rating`.
- KI: `D:/BeatMap-AI-Dataset/critic_negatives/manifest.jsonl` – 3.000 bereits erzeugte KI-Maps
  (.osu) mit Verweis auf die menschliche Quell-Map (`human_name`) und Sternen. Damit geht der
  Vergleich ohne neue Erzeugung, auch gepaart (gleicher Song).
- Später auch einzelne Dateien: Kommandozeile soll beliebige .osu-Dateien/Ordner annehmen,
  damit der Nutzer seine eigenen erzeugten Maps messen kann.

## Umsetzen
1. `beatmap_ai/patterns.py` (torch-frei), Funktionen auf Basis von
   `placement_data.map_objects(bm)` bzw. den HitObjects:
   - **Rhythmus-Gruppen:** Doubles, Triples, Bursts (4–8), Streams (9+) pro 100 Noten,
     jeweils nach Notenabstand (1/4, 1/6, 1/8 Beat); Slider-Anteil, wiederholende Slider.
   - **Jump-Muster:** Folgen von ≥3 Sprüngen mit ähnlichem Abstand erkennen und klassifizieren:
     Zickzack/hin-und-zurück, Dreieck, Viereck, Fünfeck/Stern, Linie, Kreisbogen/Flow, gemischt
     (über Winkelfolgen und Schließen des Polygons). Anteil pro Typ, typische Größe (Abstand in
     Kreisradien), Länge der Muster.
   - **Wiederholung:** Wie oft kommt dieselbe Form (bis auf Drehung/Spiegelung/Verschiebung)
     wieder vor? Unterscheide „passt zum Song“ (Wiederholung an gleicher Taktposition bzw. in
     wiederholten Songteilen, grob über gleiche Objekt-Zeitmuster pro Takt) von „stumpf“
     (dieselbe Form sehr oft hintereinander).
   - **Schwierigkeits-Passung:** Merkmale, die typisch für hohe Sterne sind (Sprungabstand in
     Radien pro ms, scharfe Winkel bei hohem Tempo, lange Streams, Cross-Screen-Jumps). Lege für
     jeden Sternbereich (<3, 3–4,5, 4,5–6, 6+) aus den **menschlichen** Maps die
     5./50./95.-Perzentile fest und zähle bei einer Map, wie viele Objekte außerhalb des 95.
     Perzentils ihres Sternbereichs liegen („Muster zu schwer für die Sterne“).
   - **Slider-Formen:** Länge in Radien, Biegung (Sehne/Pfadlänge), Anteil kurzer stark
     gebogener Slider.
2. `scripts/pattern_stats.py`: Tabelle Mensch vs. KI pro Sternbereich (Markdown-Ausgabe, und
   `--json` für Maschinen). Optionen: `--human <ordner...>`, `--manifest <jsonl>`,
   `--files <.osu/Ordner...>`, `--max-maps`, `--workers` (≤4), `--write-reference` (speichert die
   menschlichen Perzentile nach `beatmap_ai/pattern_reference.json`, klein halten).
   Außerdem eine Gesamtnote pro Map „Muster-Abweichung vom Menschen“ (0 = wie Mensch), damit
   spätere Modelle mit einer Zahl verglichen werden können.
3. Plausibilität prüfen: Zeige für jeden Mustertyp 2–3 Beispiele (Map, Zeit in ms), die der
   Nutzer im osu!-Editor nachschauen kann. Wenn die Erkennung offensichtlich falsch liegt,
   Schwellen anpassen.
4. Tests `tests/test_patterns.py` mit synthetischen Maps: perfekte Dreiecke/Vierecke/Zickzack
   werden erkannt, Doubles/Triples richtig gezählt, Drehung/Spiegelung zählt als Wiederholung.
5. Lauf: 2.000 menschliche Maps (ausgewogen über die Sternbereiche, Validierungssongs nach
   `dataset.is_validation(song_key(bm))`) + alle 3.000 KI-Maps aus dem Manifest.

## Rückmeldung am Ende
Die Vergleichstabelle Mensch vs. KI pro Sternbereich, die 5 größten Unterschiede in einem
Satz erklärt, Beispiel-Zeitstempel, Laufzeit, geänderte/neue Dateien, offene Probleme.
Einen Eintrag in `docs/WORKLOG.md`, in `docs/STATUS.md` das neue Werkzeug unter
„Messwerkzeuge“ ergänzen.
