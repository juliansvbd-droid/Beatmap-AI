# Aufgabe: Bewerter-KI (Critic) für BeatMap-AI

## Ziel
Baue eine dritte KI, die osu!standard-Maps bewertet: Sie lernt, echte Mapper-Maps von
KI-generierten Maps zu unterscheiden. Beim Erzeugen soll sie aus mehreren gewürfelten
Platzierungs-Varianten jedes Abschnitts die menschlichste auswählen (Best-of-N). Das soll
das „zufällige/unnatürliche“ Gefühl der Platzierung reduzieren.

## Projekt und Umgebung (bitte genau beachten)
- Repo: `C:\Users\Julian\Documents\BeatMap-AI` (Paket `beatmap_ai`, Skripte in `scripts/`).
- Python-Umgebung: `.venv-rocm\Scripts\python.exe` (PyTorch 2.9 mit ROCm unter Windows,
  GPU AMD RX 7700 XT). Gerät immer über `beatmap_ai.train.resolve_device("cuda")` holen:
  MIOpen funktioniert unter Windows nicht, `resolve_device` schaltet es ab. Keine GRUs,
  keine Abhängigkeit von MIOpen-Kerneln; Conv/BatchNorm/Attention/Linear gehen.
- Tests: `.venv-rocm\Scripts\python.exe -m pytest -q` (müssen am Ende alle grün sein).
- **Speicher:** C: ist fast voll (~4 GB frei) → alle großen Dateien (Negativ-Maps,
  Caches, Zwischenstände) nach `D:\BeatMap-AI-Dataset\...` schreiben. RAM 32 GB, die
  Auslagerungsdatei ist knapp: nie zwei GPU-Trainings gleichzeitig, höchstens 4
  Hilfsprozesse. Hilfsprozesse dürfen **kein PyTorch importieren** (jeder ROCm-Import
  reserviert mehrere GB) – Muster dafür: `beatmap_ai/sequence_data.py` (`batch_stream`,
  `SharedMaps` mit memory-mapped .npy) und die Trennung Daten-/Modellmodul wie
  `placement_data.py` / `placement_model.py`. Unter Windows startet `multiprocessing`
  mit „spawn“: Worker-Code muss in einem importierbaren Modul liegen, nicht in stdin.
- Keine neuen Abhängigkeiten. Code-Stil wie im Repo: englische Docstrings/Kommentare,
  gleiche Dichte, gleiche Benennung.
- **Nicht verändern:** `beatmap_ai/models/*.pt`, `checkpoints/sequence/sequence-v2.pt`,
  die Trainingsdaten in `data/`, `best_maps/`, `D:\BeatMap-AI-Dataset\` (nur neue
  Unterordner/Dateien dort anlegen).

## Vorhandene Bausteine
- Daten: `beatmap_ai.dataset.iter_beatmap_texts(folder)` liefert (name, .osu-Text,
  Audioquelle, Stempel); `song_key(bm)` + `is_validation(key)` = Aufteilung
  Training/Validierung nach Song. **Diese Aufteilung unbedingt einhalten** (Critic auf
  Trainingssongs trainieren, auf Validierungssongs messen).
- Ordner: `data`, `best_maps`, `D:/BeatMap-AI-Dataset` (Sets mit Audio + `mel.npy`,
  Community-Tags in `D:/BeatMap-AI-Dataset/tags.json`).
- Rhythmus: `beatmap_ai/models/rhythm.pt` (`beatmap_ai.model.load_checkpoint`,
  `predict_all`, `threshold_for`), Planung mit `beatmap_ai.rhythm.plan_objects`.
- Platzierung: Sequenzmodell `beatmap_ai.sequence_model` (`load_sequence`,
  `SequencePlacer.follow(plan)` platziert einen gegebenen Rhythmus Objekt für Objekt,
  `render(plan, choices, scale)` erzeugt HitObjects). Beispiel für den kompletten Ablauf
  mit menschlichem Timing: `scripts/eval_sequence.py --follow`.
- Merkmale pro Objekt: `beatmap_ai.placement_data.map_objects(bm)` (Positionen, Zeiten,
  Slider, Richtungen) und `beatmap_ai.sequence_data.sequence_features(...)`.
- Statistik-Vergleich KI vs. Mensch: `scripts/map_stats.py` (Funktion `describe`).

## Schritte
1. **Negativ-Beispiele erzeugen** (`scripts/make_critic_negatives.py`): Für Difficulties
   von Trainingssongs (Mensch-Timing via `beatmap_ai.evaluate.constant_timing`, Dichte
   und Sterne der menschlichen Map als Vorgabe) KI-Maps erzeugen – Rhythmus aus
   `plan_objects`, Platzierung per `SequencePlacer.follow` (für Vielfalt ~25 % zusätzlich
   mit der alten Regel-Platzierung `beatmap_ai.placement.Placer`). Jede KI-Map als
   **.osu-Text** speichern (über `Beatmap.to_osu_string()`) unter
   `D:\BeatMap-AI-Dataset\critic_negatives\...`, zusammen mit dem Pfad zur Audio-/Mel-
   Quelle und der zugehörigen menschlichen Map. Aktualisiertes Ziel: 3.000 Maps, fortsetzbar
   (bereits vorhandene überspringen), GPU nutzen, Fortschritt ins Log.
2. **Abkürzungen verhindern:** Menschliche und KI-Maps müssen durch denselben Weg
   (.osu-Text → `parse_osu` → `map_objects`) laufen, damit Rundung und Format gleich
   sind. Dem Critic keine verräterischen Nebensachen geben (Kurventyp, Hitsounds,
   exakte Kommazahlen, Combo-Farben). Nur: Zeiten relativ zum Beat, Objektart,
   Slider-Länge/Wiederholungen, Positionen, Bewegungsvektoren, Musik an der Stelle
   (Mel wie in `sequence_features`), Sterne der Map. Prüfe nach dem Training mit einem
   kurzen Test, dass der Critic nicht nur an einem trivialen Merkmal hängt (z. B. alle
   Positionen ganzzahlig runden → Genauigkeit darf sich nicht stark ändern).
3. **Critic-Modell** (`beatmap_ai/critic_data.py` ohne PyTorch, `beatmap_ai/critic.py`
   mit Modell/Training): Fenster von 64 aufeinanderfolgenden Objekten → bidirektionaler
   Transformer-Encoder (Größenordnung 3–6 Mio. Parameter) → ein Logit „menschlich“.
   Ausgewogen 50/50 trainieren (BCE), Validierung auf Validierungssongs, bestes
   Checkpoint speichern. Berichte Genauigkeit und AUC auf Validierungssongs.
   CLI-Befehl `beatmap-ai train-critic` (in `beatmap_ai/cli.py`).
4. **Best-of-N beim Erzeugen:** `SequencePlacer.follow` bekommt optional `critic` und
   `candidates` (Standard 4): Abschnittsweise (z. B. 16–32 Objekte) vom selben Zustand aus
   `candidates` Varianten würfeln, jede mit dem Critic (inklusive der vorherigen ~32
   Objekte als Kontext) bewerten, die beste behalten, weiter mit dem nächsten Abschnitt.
   Deterministisch bei gleichem Seed. Einbau in `beatmap_ai/generator.py`
   (`generate_beatmap`/`generate`): Critic wird aus `beatmap_ai/models/critic.pt` geladen,
   wenn vorhanden (wie `bundled_placement`), CLI-Optionen `--critic` / `--no-critic` /
   `--candidates`. Die Erzeugungszeit pro Difficulty soll höchstens ~2× so lang werden.
5. **Messen:** `scripts/eval_sequence.py --follow` mit und ohne Critic auf denselben
   40 Validierungssongs laufen lassen (Option dafür ergänzen). Erwartung: Bewegungs-
   Statistiken (Wendungen, „same turn as before“, Stacks, Überdeckungen) mindestens so
   nah am Menschen wie ohne Critic, Rhythmus-F1 unverändert (≈ 0,74).
6. **Tests** in `tests/` ergänzen: Critic-Forward-Form, Datenfenster, Best-of-N wählt
   bei einem Dummy-Critic die Variante mit dem höchsten Score. Alle Tests grün.
7. README (Abschnitt „How it works“ / Training) kurz um den Critic ergänzen.

## Rückmeldung am Ende
Kurze Zusammenfassung: geänderte/neue Dateien, Anzahl Negativ-Maps, Critic-
Validierungsgenauigkeit und AUC, Vergleichstabelle mit/ohne Critic aus Schritt 5,
Erzeugungszeit pro Difficulty vorher/nachher, offene Probleme. Kopiere das finale
Critic-Checkpoint nur dann nach `beatmap_ai/models/critic.pt`, wenn Schritt 5 keine
Verschlechterung zeigt.
