# Aufgabe: Song-Passung + mehrere Durchgänge (baut auf Prompt 01 auf)

**Erst starten, wenn Prompt 01 (Bewerter-KI/Critic) fertig ist** und `beatmap_ai/critic.py`,
`beatmap_ai/critic_data.py` sowie die Negativ-Maps in `D:\BeatMap-AI-Dataset\critic_negatives\`
existieren.

## Ziel
1. Die KI soll erkennen, ob eine Map zu **genau diesem Song** passt („Zusammengehörigkeit
   von MP3 und Map“). Maps sollen sich individuell für den Song anfühlen, nicht wie
   auswendig gelernte Muster, die irgendwie hineingedrückt wurden.
2. Die Map wird in **mehreren Durchgängen** verbessert: erzeugen → Abschnitte bewerten →
   die schwächsten Abschnitte neu erzeugen → wiederholen.

## Projekt und Umgebung (unverändert wichtig)
- Repo `C:\Users\Julian\Documents\BeatMap-AI`, Umgebung `.venv-rocm\Scripts\python.exe`
  (PyTorch 2.9 + ROCm, Windows, RX 7700 XT). Gerät immer über
  `beatmap_ai.train.resolve_device("cuda")`.
- **Speicher ist der Engpass:** Jeder Prozess, der PyTorch (ROCm) lädt, reserviert 3–4 GB
  Commit. Das Limit (32 GB RAM + 32 GB Auslagerungsdatei) war heute schon zweimal voll
  und hat Läufe abstürzen lassen. Daher: höchstens **ein** PyTorch-Prozess gleichzeitig
  (keine parallelen Test-/Mess-Skripte neben einem Training), Hilfsprozesse ohne PyTorch
  (Muster: `sequence_data.batch_stream`), große Dateien nach `D:\BeatMap-AI-Dataset\...`
  (C: hat nur ~4 GB frei). Der Nutzer testet parallel über die Oberfläche – das ist
  ebenfalls ein PyTorch-Prozess.
- Tests: `.venv-rocm\Scripts\python.exe -m pytest -q`, am Ende alle grün.
- Code-Stil wie im Repo (englische Docstrings/Kommentare), keine neuen Abhängigkeiten.
- Nicht verändern: `beatmap_ai/models/rhythm.pt`, `beatmap_ai/models/sequence.pt`,
  Trainingsdaten. Ein zweiter Agent (Claude) arbeitet am selben Repo: vor jeder
  Änderung die Datei frisch lesen, gezielt ändern statt ganze Dateien neu schreiben.

## Aktueller Stand des Generators (nicht kaputt machen)
- `generate_beatmap` (generator.py): Rhythmus aus der Frame-KI (`plan_objects`),
  Platzierung per `SequencePlacer.follow` (sequence_model.py). Sterne werden über
  `steer["k"]` erreicht (die Modelle werden um höchstens −20 %/+25 % härter/leichter
  angefragt), dann über die Notenmenge, dann Abstände nur noch ±15 %.
- Slider-Biegung wird aus der echten Verteilung gezogen (`human_chord` in
  sequence_model.py), Kick-Slider und wiederholende Slider hängen von den Sternen ab
  (`rhythm.plan_objects`). Refrains werden kopiert (`structure.copy_sections`).

## Schritte
1. **Song-Passung lernen:** Den Critic um einen zweiten Kopf „passt zum Song“ erweitern
   (oder ein eigenes Modell `beatmap_ai/songfit.py`, falls sauberer). Trainingspaare:
   - positiv: menschliche Map mit ihrem eigenen Audio;
   - negativ A: dieselbe Map mit Audio eines **anderen** Songs ähnlicher BPM
     (Beat-Raster passt, Musik nicht);
   - negativ B: dieselbe Map, Audio um ½ oder 1 Beat verschoben;
   - negativ C: dieselbe Map, Audio desselben Songs aus einem **anderen Abschnitt**
     (z. B. Strophe statt Refrain).
   Eingaben wie im Critic (Fenster von ~64 Objekten, Mel an den Objekten und etwas
   Kontext), Validierung auf Validierungssongs (`is_validation(song_key)`), Genauigkeit
   und AUC je Negativ-Art berichten.
2. **Mehrere Durchgänge** in `SequencePlacer.follow` bzw. generator.py: Nach dem ersten
   vollständigen Durchgang jeden Abschnitt (16–32 Objekte) mit Critic + Song-Passung
   bewerten, die schwächsten ~25 % mit anderem Seed neu platzieren (ab dem Zustand vor
   dem Abschnitt, Übergang zum Folgeabschnitt beachten: Abstand/Richtung zum nächsten
   Objekt nicht zerstören), neu bewerten, nur Verbesserungen übernehmen. Anzahl der
   Durchgänge per Option (`--passes`, Standard 2), in der Oberfläche (beatmap_ai/ui.py)
   als Einstellung „Durchgänge“ unter „Stil“. Erzeugungszeit pro Difficulty messen und
   berichten; Ziel höchstens ~3× so lang wie ohne.
3. **Messen:** `scripts/eval_sequence.py --follow` auf denselben 40 Validierungssongs:
   ohne Critic / mit Critic (Best-of-N aus Prompt 01) / mit Critic + Song-Passung +
   2 Durchgängen. Tabelle mit Rhythmus-F1 und allen Bewegungs-Statistiken
   (Wendungen, „same turn as before“, Stacks, Überdeckungen, Doubles/Triples,
   Slider-Anteile) gegen Mensch. Zusätzlich die Song-Passungs-Bewertung der
   erzeugten Maps im Vergleich zu menschlichen Maps (Mittelwert).
4. **Tests** für Song-Passungs-Daten (Negativ-Paare korrekt erzeugt) und für die
   Durchgänge (nur bessere Abschnitte werden übernommen). README kurz ergänzen.

## Rückmeldung am Ende
Geänderte Dateien, Genauigkeit/AUC der Song-Passung je Negativ-Art, die Vergleichstabelle
aus Schritt 3, Erzeugungszeit mit/ohne Durchgänge, offene Probleme. Neue Checkpoints nur
dann nach `beatmap_ai/models/` kopieren, wenn Schritt 3 keine Verschlechterung zeigt.
