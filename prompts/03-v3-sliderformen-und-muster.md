# Aufgabe: Platzierungsmodell v3 – bessere Platzierung, echte Slider-Formen, Jump-Muster

**Jetzt als nächster großer Schritt** (Reihenfolge seit 26.09.: 03 → 04 → 02). Nie zwei
GPU-Trainings gleichzeitig.

## Ausgangslage (26.09.2026, Blindtest des Nutzers)
- Der Nutzer hat unsere Maps blind gegen **Mapperatorinator** (bekannter KI-Mapper,
  219 Mio. Parameter) gespielt, Insane, 2 Songs: **Rhythmus gleich gut, Platzierung bei
  Mapperatorinator jedes Mal einen Ticken besser** („keine Welten“). Die Platzierung ist
  also die Lücke, die dieser Prompt schließen soll.
- **Regel des Nutzers: nichts kopieren.** Kein Code, keine Gewichte, keine von
  Mapperatorinator erzeugten Maps als Trainingsdaten. Es liegt nur als Messlatte unter
  `D:\Mapperatorinator` (eigene venv) für spätere Blindtests – nicht importieren.
- Der Generator hat inzwischen Regeln, die Schwächen von v2 abfangen (siehe unten
  „Krücken“). v3 soll das selbst können, sodass die Krücken gelockert werden können.

## Ziel
Das Sequenzmodell (`beatmap_ai/sequence_model.py`, aktuell v2 in
`beatmap_ai/models/sequence.pt`, 10,3 Mio. Parameter) als v3 neu trainieren, das
1. **besser platziert**: Abstände, Cursor-Flow und Muster so, wie Menschen es bei der
   jeweiligen Sternzahl tun;
2. **Slider-Formen** wirklich lernt (S-Kurven, Wellen, gerade Stücke mit Knick, passende
   Länge) statt nur Sehne + Bogen;
3. **Jump-Muster** (Zickzack, Dreiecke, Quadrate/Fünfecke, Back-and-forth, Sterne) sauber
   bildet – abwechslungsreich, mit Wiederholung, wenn die Musik sich wiederholt;
4. **sternzahlgerecht** mappt: keine 6★-Muster in 4★-Maps; leichte Maps fließen;
5. stärker **auf die Musik** hört, damit Maps individueller werden.

## Was der Nutzer beim Spielen bemängelt hat (Grundlage)
- Platzierung „einen Ticken“ schlechter als beim besten bekannten KI-Mapper; beim Spielen
  hat man einen Cursor-Flow, menschliche Maps haben Muster, die es leichter machen, statt
  zufälliger Platzierung.
- Früher: „Je höher die Schwierigkeit, desto inkonsistenter“, 6★-Muster in 4★-Maps,
  kurze Slider und „Kreisslider“, „wie in den Song gedrückt“.
- Erwünscht: wiederholende (hin-und-her) Slider als Stream-/Double-Ersatz in niedrigen
  Sternen. Doubles/Triples nur, wo die Musik sie hergibt.

## Zielwerte echter Maps
Messwerkzeuge (torch-frei, messen jeweils auch die menschliche Referenz):
`scripts/compare_maps.py` (Wendungen entlang des **Cursorwegs inkl. Slider**),
`scripts/slider_stats.py`, `scripts/pattern_stats.py` / `beatmap_ai/patterns.py` (Jump-
Muster, von GPT-6 Sol, Prompt sol-01 – nutzen, nicht neu bauen).

Cursor-Wendungen (Anteil der Bewegungen in eine Note):
| Sterne | scharf > 120° | gerade < 30° |
|---|---|---|
| < 2,5 | 6 % | 24 % |
| 2,5–3,5 | 9 % | 20 % |
| 3,5–4,5 | 23 % | 16 % |
| 4,5–5,5 | 41 % | 15 % |
| 5,5+ | 38 % | 17 % |

Slider:
| Sterne | Slider-Anteil | ≤¼-Beat-Slider | ½ Beat | ≥1 Beat | gebogen (Sehne/Länge < 0,9) | stark (< 0,6) | wiederholend | Note < ¼ Beat nach Slider-Ende |
|---|---|---|---|---|---|---|---|---|
| 1–2 | 58 % | 0 % | 11 % | 86 % | 11 % | 1 % | 22 % | 0 % |
| 2–3 | 55 % | 3 % | 32 % | 60 % | 7 % | 1 % | 17 % | 1 % |
| 3–4 | 52 % | 11 % | 52 % | 29 % | 7 % | 1 % | 15 % | 7 % |
| 4–5 | 44 % | 21 % | 51 % | 20 % | 9 % | 2 % | 10 % | 8 % |
| 5–6 | 41 % | 27 % | 48 % | 16 % | 10 % | 4 % | 9 % | 21 % |
| 6–7 | 38 % | 35 % | 42 % | 13 % | 13 % | 5 % | 8 % | 21 % |
Kurze Slider (< 70 px) biegen Menschen praktisch nie.

Bekannte Lücken von v2 (Messungen 26.09.): Abstand pro Beat bei 4,5–6★ 220 statt 350 px
(40-Song-A/B aus 01b); Insane-Sprünge auf 1 Beat 130 statt ~169 px; Doubles 1,5 statt 2,8
pro 100 Noten (4,5–6★); Stacks bei Expert oft ~20 % statt ~6 %; ohne Krücken bei
Normal/Hard bis 30–89 % scharfe Wendungen.
Aus dem Blindtest (`pattern_stats.py` auf `D:\Mapperatorinator\compare\`): unsere Maps haben
viel öfter Bewegungen über dem 95. Perzentil menschlicher Maps derselben Sternstufe –
v. a. **scharfe Wendungen bei hohem Tempo** und **Cross-Screen-Sprünge** (z. B. ALQUIMIA
Insane 24 bzw. 15 pro 100 Objekte, Mapperatorinator 0 bzw. 1; unsere Normals 10–42 bzw.
13–31). Genau bei ALQUIMIA Insane fand der Nutzer Mapperatorinator besser. Achtung: Sols
Tabelle „wiederholte Muster KI 3–5 %“ stammt aus den alten Critic-Negativ-Maps ohne
Refrain-Kopien; mit der App (`structure.copy_sections`) liegen wir bei 15–60 %.

## Krücken im Generator (Stand 26.09.; v3 soll sie überflüssig machen)
- `SequencePlacer.sharp_keep` (verwirft scharfe Wendungen bei niedrigen Sternen),
- `sequence_model.human_chord` + `BEND_SCALE`, `STRAIGHT_BELOW_PX` (Slider-Biegung aus
  Tabelle statt Modell), Biegung pro Slider fest (`_bend_rng`),
- in `rhythm.plan_objects`: `kick_confidence`, ½ Beat Pause nach Slidern
  (`quick_release`), Slider-Anteil-Deckel (`slider_cut`),
- `difficulty.DEFAULT_STARS` (Sternziele für benannte Diffs; später Prompt 04).
v3 zuerst **mit** den Krücken messen (so läuft die App), dann `sharp_keep`=1 und Form-Kopf
statt `human_chord` – wenn v3 ohne Krücken gleich gut oder besser ist, Krücken für v3
abschalten (für v2 als Rückfall behalten).

## Projekt und Umgebung
`.venv-rocm`, Gerät über `beatmap_ai.train.resolve_device("cuda")`, **höchstens ein
PyTorch-Prozess gleichzeitig** (auch die App zählt), Hilfsprozesse ohne PyTorch, große
Dateien nach D:, Tests grün (`.venv-rocm\Scripts\python.exe -m pytest -q`), Repo-Stil,
andere Agenten arbeiten mit – `AGENTS.md` lesen, Dateien frisch lesen, gezielt ändern,
nach fertigen Schritten committen und pushen (Regeln in `AGENTS.md`). Daten: `data`,
`best_maps`, `D:\BeatMap-AI-Dataset` (+ `tags.json`), Aufteilung nach
`is_validation(song_key)`. Vorbild für Training mit geteiltem memory-mapped Speicher:
`train_sequence` / `sequence_data.batch_stream` (v2 lief ~2,5 h für 30 Epochen).

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
     je halbe Sternstufe gleich oft, und innerhalb der Stufe Jump-, Stream-, Tech-,
     Flow-/Simple- und gemischte Maps annähernd gleich oft (mindestens ~1.000 Maps je
     Sternstufe und Hauptstil, soweit vorhanden; seltene Kombinationen öfter ziehen).
0b. **Abschnitts-Vorgaben (für Prompt 04 nötig):** Zusätzlich zu den Map-weiten
   Bedingungen bekommt jedes Objekt die **lokalen** Werte seines Abschnitts (±4 Takte,
   aus der menschlichen Map berechnet): Notendichte, mittlere Sprungweite (distance
   snap), Stream-Anteil, Slider-Anteil, Anteil scharfer Wendungen, Kiai ja/nein. Im
   Training zufällig (z. B. 40 %) verstecken, damit „Auto“ ohne Plan weiter funktioniert.
   Beim Erzeugen können diese Werte später von der Vorplanungs-KI (Prompt 04) kommen.
1. **Slider-Form-Ziel:** In `placement_data.map_objects` zusätzlich den tatsächlichen
   Slider-Pfad speichern: osu!-Kurve (L/P/B inkl. Mehrsegment-Bézier, auf die
   Pixel-Länge gekürzt) in K = 8 gleichmäßigen Punkten, im lokalen Rahmen der
   Bewegungsrichtung, geteilt durch die Länge. Pfadberechnung wie osu!
   (lineare/Perfect-Circle/Bézier-Approximation) – mit Test gegen bekannte Slider.
   Den Objekt-Cache (`placement.pkl`) dafür neu erzeugen (neue Cache-Version).
2. **Modell v3** (neue Klasse bzw. Config-Version, v2-Checkpoints müssen weiter laden):
   - **Platzierung ist die Hauptsache.** Das Modell darf deutlich größer werden als v2
     (Richtwert 25–50 Mio. Parameter, mehr Kontext – z. B. 256 Objekte – damit Muster
     über mehrere Takte gelernt werden) und länger trainieren (bis ~10–12 h sind ok, die
     GPU ist sonst frei; Zwischenstände speichern, fortsetzbar). Vorher an einem kleinen
     Lauf (z. B. 1 h) prüfen, dass größer wirklich besser wird.
   - Ideen, die sich lohnen können (eigene Umsetzung): Positionen als Mischung plus
     optional grobes Raster als Klassen (leichter für Muster/Symmetrie); Rhythmus des
     ganzen Fensters als Eingabe (nicht nur bis zum aktuellen Objekt), damit die
     Platzierung „weiß“, was kommt; Wiederholungs-Hinweis aus `structure.find_repeats`
     (gleiche Stelle im Refrain) als Eingabe.
   - Slider-Form-Kopf, der die 8 Punkte vorhersagt (z. B. Mischung über eine kleine
     PCA-Basis der Formen oder ein Punkt-für-Punkt-Mischmodell), statt Sehne + Bogen.
     Die bisherige Sehne/Biegung bleibt als Rückfall.
   - Eingaben zusätzlich: AR, OD, HP, CS der Map; Musik ±4 Beats (nicht nur voraus);
     Schlagstärke/Onset je Viertelbeat.
   - Sterne- und Stil-Bedingungen seltener verstecken als in v2 (v2: 30 %), damit sie
     stärker wirken; Tags wie bisher.
3. **Erzeugen mit v3:** `SequencePlacer.follow` nutzt den Form-Kopf für den Slider-Pfad
   (als Bézier-Kontrollpunkte im .osu schreiben), `human_chord` nur noch als Rückfall für
   v2. Die Bewerter-KI (`critic.pt` v1 / `critic-v2.pt`) muss mit v3 weiter funktionieren
   (ggf. nur neu messen, nicht neu trainieren).
4. **Messen:** `scripts/eval_sequence.py --follow` mit v2 und v3 auf denselben 40
   Validierungssongs, getrennt nach Sternstufen (2–3, 3–4, 4–5, 5–6, 6+): Rhythmus-F1,
   Abstände pro Beat, Cursor-Wendungen, Stacks, Slider-Tabelle, Jump-Muster
   (`pattern_stats.py`), jeweils gegen die menschlichen Maps derselben Songs. Mit und
   ohne Krücken (siehe oben).
5. **Blindtest-Material für den Nutzer:** Mit v3 die Insane (4,5★) von
   `Music/03 … MONTAGEM GUERREIRO` und `Music/02 … MONTAGEM ALQUIMIA` erzeugen (Stil
   Auto, Standard-Critic) und nach `Vergleich/v3/` legen. Claude Code baut daraus den
   Blindtest gegen die bisherigen Maps (`D:\Mapperatorinator\blindtest`, dort liegen v2
   und Mapperatorinator). Nicht selbst Mapperatorinator ausführen.
6. Tests (Slider-Pfad-Berechnung, Form-Kopf-Ausgabeform, v2 lädt weiter) und README.

## Rückmeldung am Ende
Geänderte Dateien, Parameterzahl, Trainingsverlauf, Vergleichstabelle v2 vs. v3 je
Sternstufe (mit/ohne Krücken), Pfade der Blindtest-Maps, offene Probleme. v3 nur dann nach
`beatmap_ai/models/sequence.pt` kopieren (v2 vorher nach `checkpoints/sequence/`
sichern), wenn Schritt 4 insgesamt besser ist. STATUS und WORKLOG aktualisieren, pushen.
