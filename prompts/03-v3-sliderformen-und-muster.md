# Aufgabe: Sequenzmodell v3 – echte Slider-Formen, Jump-Muster, mehr Musik

**Erst starten, wenn Prompt 02 fertig ist** (beide brauchen die GPU; nie zwei Trainings
gleichzeitig).

## Ziel
Das Sequenzmodell (`beatmap_ai/sequence_model.py`, aktuell v2 in
`beatmap_ai/models/sequence.pt`) neu trainieren als v3, das
1. **Slider-Formen** wirklich lernt (S-Kurven, Wellen, gerade Stücke mit Knick, passende
   Länge) statt nur eines Kreisbogens mit Richtung;
2. **Jump-Muster** (Zickzack, Dreiecke, Quadrate/Fünfecke, Back-and-forth, Sterne) sauber
   bildet – abwechslungsreich, aber mit Wiederholung, wenn die Musik sich wiederholt;
3. **sternzahlgerecht** mappt: keine 6★-Muster in 4★-Maps;
4. stärker **auf die Musik** hört, damit Maps individueller für den Song werden.

## Was der Nutzer beim Spielen bemängelt hat (Grundlage)
- „Je höher die Schwierigkeit, desto inkonsistenter.“
- Muster aus 6★+-Maps tauchen in 4★-Maps auf, viele kleine/kurze Slider und
  „Kreisslider“.
- „Fühlt sich an, als hätte ein Mensch 5 Maps angesehen, sich Muster gemerkt und die in
  den Song gedrückt.“
- Wiederholende (hin-und-her) Slider als Stream-/Double-Ersatz in niedrigen Sternen
  sind erwünscht.

## Messwerte echter Maps (Zielwerte, aus ~6.000 Maps)
| Sterne | Slider-Anteil | ≤¼-Beat-Slider | ½ Beat | ≥1 Beat | gebogen (Sehne/Länge < 0,9) | stark (< 0,6) | wiederholend |
|---|---|---|---|---|---|---|---|
| 1–2 | 58 % | 0 % | 11 % | 86 % | 11 % | 1 % | 22 % |
| 2–3 | 55 % | 3 % | 32 % | 60 % | 7 % | 1 % | 17 % |
| 3–4 | 52 % | 11 % | 52 % | 29 % | 7 % | 1 % | 15 % |
| 4–5 | 44 % | 21 % | 51 % | 20 % | 9 % | 2 % | 10 % |
| 5–6 | 41 % | 27 % | 48 % | 16 % | 10 % | 4 % | 9 % |
| 6–7 | 38 % | 35 % | 42 % | 13 % | 13 % | 5 % | 8 % |

## Projekt und Umgebung
Wie in Prompt 02 (bitte dort nachlesen): `.venv-rocm`, `resolve_device`, **höchstens ein
PyTorch-Prozess gleichzeitig**, Hilfsprozesse ohne PyTorch, große Dateien nach D:, Tests
grün, Repo-Stil, zweiter Agent (Claude) arbeitet mit – Dateien frisch lesen, gezielt
ändern. Daten: `data`, `best_maps`, `D:\BeatMap-AI-Dataset` (+ `tags.json`), Aufteilung
nach `is_validation(song_key)`. Vorbild für Training mit geteiltem memory-mapped Speicher:
`train_sequence` / `sequence_data.batch_stream` (v2 lief so ~2,5 h für 30 Epochen).

## Datenlage Stil-Tags (wichtig)
Maps pro Sternstufe gibt es genug (4.000–10.000), Community-Tags aber fast nur für
schwere Maps: 1–2★ 303 getaggt (20 als Jump), 2–3★ 904 (42 Jump), 3–4★ 2.309 (280),
4–5★ 5.098 (1.682), 5–6★ 6.606 (2.350), 6–7★ 3.921 (1.441). Darum hat v2 „Jump“ fast
nur an 5★+ gelernt und bringt bei 3–4★ Muster aus schweren Maps mit.

## Schritte
0. **Auto-Tagging und ausgewogene Daten:**
   - Einen kleinen Tagger trainieren (`beatmap_ai/tagger.py`, Daten ohne PyTorch wie
     üblich getrennt): Eingabe eine ganze Map (Objekt-Folge wie `map_objects`, dazu
     Sterne), Ausgabe die 30 Tags aus `sequence_data.TAGS` (Anteil an den Top-Stimmen).
     Trainiert auf den ~25.000 getaggten Difficulties, Validierung nach Song. Pro Tag
     AUC berichten, besonders getrennt für 1–4★ (dort wenige echte Tags).
   - Damit **alle** Maps ohne Tags mit vorhergesagten Tags versehen (nur Tags mit
     ausreichender Vorhersagegüte übernehmen; im Datensatz markieren, dass sie
     vorhergesagt sind, z. B. als eigenes „known“-Flag).
   - Das Training von v3 zieht die Fenster ausgewogen nach **(Sternstufe × Stil)**:
     z. B. je halbe Sternstufe gleich oft, und innerhalb der Stufe Jump-, Stream-,
     Tech-, Flow-/Simple- und gemischte Maps annähernd gleich oft (mindestens ~1.000
     Maps je Sternstufe und Hauptstil, soweit vorhanden; seltene Kombinationen
     entsprechend öfter ziehen).
0b. **Abschnitts-Vorgaben (für Prompt 04 nötig):** Zusätzlich zu den Map-weiten
   Bedingungen bekommt jedes Objekt die **lokalen** Werte seines Abschnitts (±4 Takte,
   aus der menschlichen Map berechnet): Notendichte, mittlere Sprungweite (distance
   snap), Stream-Anteil, Slider-Anteil, Anteil scharfer Wendungen. Im Training
   zufällig (z. B. 40 %) verstecken, damit „Auto“ ohne Plan weiter funktioniert. Beim
   Erzeugen können diese Werte später von der Vorplanungs-KI (Prompt 04) kommen.
1. **Slider-Form-Ziel:** In `placement_data.map_objects` zusätzlich den tatsächlichen
   Slider-Pfad speichern: osu!-Kurve (L/P/B inkl. Mehrsegment-Bézier, auf die
   Pixel-Länge gekürzt) in K = 8 gleichmäßigen Punkten, im lokalen Rahmen der
   Bewegungsrichtung, geteilt durch die Länge. Pfadberechnung wie osu!
   (lineare/Perfect-Circle/Bézier-Approximation) – mit Test gegen bekannte Slider.
   Den Objekt-Cache (`placement.pkl`) dafür neu erzeugen (neue Cache-Version).
2. **Modell v3** (neue Klasse bzw. Config-Version, v2-Checkpoints müssen weiter laden):
   - Slider-Form-Kopf, der die 8 Punkte vorhersagt (z. B. Mischung über eine kleine
     PCA-Basis der Formen oder ein Punkt-für-Punkt-Mischmodell), statt Sehne + Bogen.
     Die bisherige Sehne/Biegung bleibt als Rückfall.
   - Eingaben zusätzlich: AR, OD, HP, CS der Map; Musik ±4 Beats (nicht nur voraus);
     Schlagstärke/Onset je Viertelbeat.
   - Sterne- und Stil-Bedingungen seltener verstecken als in v2 (v2: 30 %), damit sie
     stärker wirken; Tags wie bisher.
   - Training auf allen Daten, gleiche Größenordnung wie v2 oder etwas größer, solange
     es in ~3 h passt.
3. **Erzeugen mit v3:** `SequencePlacer.follow` nutzt den Form-Kopf für den Slider-Pfad
   (als Bézier-Kontrollpunkte im .osu schreiben), die Biegungs-Tabelle `human_chord`
   nur noch als Rückfall für v2.
4. **Muster-Messungen** in `scripts/map_stats.py` (Funktion `describe`) ergänzen:
   Zickzack (abwechselnde Wendungen gleicher Größe), Dreiecke/Vielecke (≥3 gleiche
   Wendungen in Folge, Winkel ≈ 360°/n), Back-and-forth, Wiederholungsrate von
   Mustern innerhalb 8 Takten vs. über den Song, Slider-Form-Vielfalt – jeweils
   Mensch vs. KI.
5. **Messen:** `scripts/eval_sequence.py --follow` mit v2 und v3 auf denselben 40
   Validierungssongs, zusätzlich getrennt nach Sternstufen (2–3, 3–4, 4–5, 5–6, 6+):
   Rhythmus-F1, Bewegung, Muster, Slider-Tabelle wie oben. Außerdem je eine Map bei
   3★, 4,5★ und 6★ mit `beatmap-ai generate` für einen Testsong erzeugen und nach
   `Beispiel-Maps\` legen, damit der Nutzer anspielen kann.
6. Tests (Slider-Pfad-Berechnung, Form-Kopf-Ausgabeform, v2 lädt weiter) und README.

## Rückmeldung am Ende
Geänderte Dateien, Trainingsverlauf, Vergleichstabelle v2 vs. v3 je Sternstufe, die
Pfade der Beispiel-Maps, offene Probleme. v3 nur dann nach
`beatmap_ai/models/sequence.pt` kopieren (v2 vorher nach `checkpoints/sequence/`
sichern), wenn Schritt 5 insgesamt besser ist.
