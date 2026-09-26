# Aufgabe: Vorplanungs-KI („Preloader“) – was für eine Map passt zu diesem Song?

**Erst starten, wenn Prompt 03 fertig ist** (v3 muss die Abschnitts-Vorgaben aus Schritt 0b
von Prompt 03 gelernt haben). Nie parallel zu einem anderen Training.

## Ziel
Eine KI, die sich den ganzen Song anhört und **vor** dem eigentlichen Mappen festlegt,
was für eine Map daraus werden sollte – so wie ein Mapper erst den Song hört und dann
plant. Die Haupt-KI (Sequenzmodell v3) setzt diesen Plan danach um. Ziel: Maps, die sich
individuell für den Song anfühlen, statt überall dieselben Muster.

Die Vorplanungs-KI liefert:
1. **Für den ganzen Song:** passende Sternbereiche (z. B. „bis ~5,5★ sinnvoll“, ein
   ruhiger Song taugt kaum für 7★) und passende Stile (Jump, Stream, Tech,
   Flow/Simple, Alt …) als Wahrscheinlichkeiten.
2. **Pro Abschnitt (je 8 Takte, bzw. entlang der gefundenen Songteile aus
   `beatmap_ai/structure.py`):** relative Werte für Notendichte, Sprungweite,
   Stream-Anteil, Slider-Anteil, Anteil scharfer Wendungen – also z. B. „Strophe ruhig
   mit Slidern, Refrain große Sprünge, Bridge Stream-Teil“. Relativ zur Ziel-Sternzahl,
   damit derselbe Plan für 3★ und 6★ funktioniert.

## Songteile erkennen (erster Schritt der Vorplanung)
Bevor geplant wird, teilt die KI den Song in **so viele Teile, wie musikalisch sinnvoll**
(nicht starr alle 8 Takte) und ordnet jeden ein: Intro, Strophe, Aufbau/Pre-Chorus,
**Hauptteil (Refrain/Drop)**, Bridge, Outro. Sie erkennt den **Höhepunkt**: dort wird die
Map etwas schwerer und dort kommen die großen Sprünge hin; Intro/Outro bleiben ruhiger.
Pro Teil entscheidet sie, was passt (z. B. Strophe Slider/Flow, Aufbau Streams,
Refrain Jumps).
- **Lernsignal aus den Maps:** Kiai-Zeiten (`TimingPoint.effects & 1` in allen ~52.000
  Maps) markieren fast immer den Hauptteil → Label „Hauptteil“. Teilgrenzen: Stellen,
  an denen Dichte/Sprungweite/Slider-Anteil menschlicher Maps deutlich umschlagen
  (Mehrheit über die Difficulties eines Sets), plus Wiederholungen aus
  `structure.find_repeats` (wiederkehrende Teile = oft Refrain).
- Ausgabe: Liste von Teilen (Start, Ende, Art, Intensität 0–1, Stil-Wahrscheinlichkeiten).
  Die Abschnittswerte unten werden je Teil geplant, nicht je 8 Takte.
- Die gelernte Hauptteil-Erkennung ersetzt die Regel in `structure.kiai_sections` (Kiai
  auf die erkannten Hauptteile), sofern sie auf Validierungssongs besser mit den
  menschlichen Kiai-Zeiten übereinstimmt (Überlappung in % berichten).
- Messen: Übereinstimmung der erkannten Grenzen mit den Intensitätswechseln menschlicher
  Maps (±1 Takt) und der Hauptteile mit deren Kiai-Zeiten.

## Umgebung
Wie in Prompt 02/03: `.venv-rocm`, `resolve_device`, höchstens ein PyTorch-Prozess
gleichzeitig (Speicher!), Hilfsprozesse ohne PyTorch, große Dateien nach D:, Tests grün,
Repo-Stil, zweiter Agent (Claude) arbeitet mit – Dateien frisch lesen, gezielt ändern.

## Daten
- Pro Song gibt es mehrere Difficulties (ranked Sets) mit Sternen, Stil-Tags (echt oder
  vom Tagger aus Prompt 03 vorhergesagt) und Objekten.
- Song-Ziele: höchste Sternzahl des Sets, Verteilung der Sterne im Set, Stil-Tags der
  schwersten 1–2 Difficulties (die zeigen, wofür der Song „gedacht“ ist).
- Abschnitts-Ziele: aus jeder menschlichen Difficulty je Abschnitt die lokalen Werte
  (wie Prompt 03, Schritt 0b), geteilt durch den Mittelwert der ganzen Map.
- Aufteilung nach `is_validation(song_key)`.

## Schritte
1. `beatmap_ai/planner_data.py` (ohne PyTorch) und `beatmap_ai/planner.py`: Modell über
   das ganze Lied (Mel pro Takt + Onset/Energie + Wiederholungsstruktur aus
   `structure.find_repeats`), bidirektional über die Takte. Köpfe: Song-Sterne
   (Verteilung), Song-Stile (Mehrfach-Label), Abschnittswerte (Regression, Eingabe die
   Ziel-Sterne). Training + Validierung, Fehler berichten (z. B. mittlere Abweichung der
   Maximal-Sterne, AUC je Stil, Korrelation der Abschnittswerte).
2. **Einbau:**
   - `generate` / Oberfläche: Option „Plan: Auto (Vorplanung) / aus“. Mit Vorplanung
     bekommt jeder Abschnitt seine Werte als Abschnitts-Vorgabe für v3; Stil „Auto“
     nimmt die vom Planer empfohlenen Stile (schwach gewichtet), ein vom Nutzer gewählter
     Stil hat Vorrang.
   - In der Oberfläche (beatmap_ai/ui.py) nach Auswahl eines Songs eine kurze Anzeige:
     „Empfohlen: bis ~X★, Stil: …“ und ein Knopf, die empfohlenen Sternzahlen ins
     Sternfeld zu übernehmen.
3. **Messen:** Auf 40 Validierungssongs mit `scripts/eval_sequence.py --follow` (Option
   für Planer ergänzen): ohne / mit Planer. Zusätzlich die Song-Passung aus Prompt 02
   (falls vorhanden) und je Song, wie stark sich Abschnitte unterscheiden
   (Dichte-/Sprungweiten-Kurve über den Song, Korrelation mit der menschlichen Map
   desselben Songs). Erwartung: Abschnitte der KI-Map folgen dem Songaufbau deutlich
   besser (höhere Korrelation), Rhythmus-F1 nicht schlechter.
4. Tests und README.

## Rückmeldung am Ende
Geänderte Dateien, Güte des Planers (Sterne, Stile, Abschnitte), Vergleich ohne/mit
Planer, drei Beispiel-Maps in `Beispiel-Maps\` (ein ruhiger, ein mittlerer, ein
intensiver Song, jeweils mit Planer), offene Probleme. Checkpoints nur bei Verbesserung
nach `beatmap_ai/models/` übernehmen.
