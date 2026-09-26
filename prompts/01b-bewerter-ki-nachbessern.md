# Aufgabe: Bewerter-KI (Critic) nachbessern – vor Prompt 02

Prompt 01 ist umgesetzt (`beatmap_ai/critic.py`, `critic_data.py`, `scripts/make_critic_negatives.py`,
`scripts/eval_ab_critic.py`, Modell `beatmap_ai/models/critic.pt`). Die Prüfung des Ergebnisses
zeigt ein **Abkürzungs-Problem**, das vor Prompt 02 (der auf dem Critic aufbaut) behoben
werden muss.

## Befund
- Validierungsgenauigkeit 98,2 %, AUC 0,997 – zu leicht. Im A/B-Test wird selbst die beste
  von 4 Varianten nur zu 4,4 % als menschlich bewertet (vorher 1,3 %): Der Critic urteilt vor
  allem nach Dingen, die Best-of-N gar nicht ändern kann.
- Ursache (in `critic_data.py`, Laden der menschlichen Maps): Die Positiv-Beispiele sind die
  **ersten N Maps aus den Ordnern**, also andere Songs und eine andere Sternverteilung als
  die Negativ-Maps (Negativ: 59 % unter 3★, 50 von 3.000 über 6★). Die Sterne sind ein
  Eingabemerkmal → „leicht ⇒ KI“ und Song/Audio sind Abkürzungen. Zudem unterscheidet sich
  der Rhythmus (Frame-KI statt Mensch), den Best-of-N ebenfalls nicht ändert.
- A/B-Messung nur auf 12 Maps (vorgesehen: 40 Validierungssongs).

## Ziel
Ein Critic, der **die Platzierung** beurteilt (Muster, Abstände, Flow, Slider-Formen), damit
Best-of-N und die späteren Durchgänge (Prompt 02) wirklich bessere Platzierungen auswählen.

## Umgebung
Wie in Prompt 01/02 (`docs/STATUS.md` → Environment): `.venv-rocm`, `resolve_device`, höchstens
ein PyTorch-Prozess, Hilfsprozesse ohne PyTorch, große Dateien nach D:, Tests grün, Repo-Stil,
vor dem Aufhören `docs/STATUS.md` und `docs/WORKLOG.md` aktualisieren (`AGENTS.md`).

## Schritte
1. **Gepaarte Daten:** Positiv-Beispiele = genau die menschlichen Difficulties, aus denen die
   Negativ-Maps erzeugt wurden (das Manifest kennt die Quell-Map). Gleicher Song, gleiche
   Sterne, gleiches Timing. Für jede Negativ-Map ihre Quell-Map als Positiv-Beispiel.
2. **Gleicher Rhythmus, andere Platzierung:** Zusätzlich (Hauptteil der neuen Daten, ~3.000)
   Negativ-Maps erzeugen, die den **menschlichen Rhythmus** der Quell-Map übernehmen
   (Objektzeiten, Arten, Slider-Längen/-Wiederholungen, neue Combos) und nur neu **platziert**
   sind (`SequencePlacer.follow` mit einem Plan aus den menschlichen Objekten; ~25 % mit der
   Regel-Platzierung). Dann unterscheidet sich ein Paar nur in der Platzierung.
3. **Abkürzungen weiter abstellen:** Sternzahl nicht mehr als direktes Merkmal geben, oder nur
   bei gepaarten Daten (gleiche Verteilung positiv/negativ). Verteilungen positiv vs. negativ
   (Sterne, Dichte, Länge) im Log ausgeben – sie müssen gleich sein. Zusätzlich zum Jitter-Test:
   Genauigkeit bei vertauschtem/zufälligem Audio (wenn sie gleich bleibt, nutzt der Critic die
   Musik nicht – dann berichten, nicht verstecken).
4. **Neu trainieren** (gleiche Architektur ist ok). Erwartung: Genauigkeit deutlich unter 98 %
   (z. B. 80–90 %), dafür reagiert der Critic auf Platzierung.
5. **Messen mit genug Daten:** `scripts/eval_ab_critic.py` auf **40 Validierungssongs** (alle
   ihre Difficulties), ohne Critic / alter Critic / neuer Critic: Critic-Score der gewählten
   Maps, Rhythmus-F1, alle Bewegungs-Statistiken aus `scripts/map_stats.py` (Wendungen,
   „same turn as before“, Stacks, Überdeckungen, Doubles/Triples, Slider) gegen Mensch, dazu
   getrennt nach Sternstufen (<3, 3–4,5, 4,5–6, 6+). Erzeugungszeit pro Difficulty.
6. Neues Modell nur dann nach `beatmap_ai/models/critic.pt` (altes nach `checkpoints/critic/`
   sichern), wenn die Bewegungs-Statistiken mit dem neuen Critic näher am Menschen liegen als
   mit dem alten und die Rhythmus-F1 gleich bleibt.

## Nachtrag (Prüfung von `scripts/make_paired_negatives.py`, 26.09.)
Die laufende Erzeugung hat zwei neue Abkürzungen – bitte stoppen, korrigieren, neu erzeugen:
1. **Slider-Länge verrät die KI:** `SequencePlacer._row` rechnet die Slider-Pixellänge mit
   `preset.slider_multiplier` und SV = 1, die Negativ-Map bekommt nur eine rote Linie und
   keine grünen. Die menschliche Map hat ihren eigenen Slider Multiplier und SV-Wechsel →
   `slider_len`/`slider_dx/dy` unterscheiden sich systematisch zwischen den Paaren. Lösung:
   Slider Multiplier und **alle** Timing-Punkte (rot + grün) der menschlichen Map übernehmen
   (Preset mit menschlichem `slider_multiplier`, `sv_at` aus den grünen Linien, in der .osu
   dieselben Timing-Punkte). Maps ohne konstantes Tempo (`constant_timing` = None) entweder
   überspringen oder ebenfalls alle roten Linien übernehmen – sonst stimmen Phase/Takt nicht.
2. **Nur der Anfang jeder Map:** `human_to_plan` nimmt `hit_objects[:max_objects]`, also nur
   die ersten 96 Objekte = meist das ruhige Intro. Kiai/Refrain/Höhepunkt sieht der Critic
   nie, beim Erzeugen bewertet er aber genau diese Stellen. Lösung: pro Map ein **zufälliges
   Fenster** (oder 2–3 Fenster) à 96–128 Objekte, Startindex im Manifest speichern und für
   die menschliche Map dasselbe Fenster verwenden. Mindestens ein Drittel der Fenster aus
   Kiai-Abschnitten.
3. **Prüfung danach:** Für 200 Paare die Verteilung von Slider-Pixellänge pro Beat, Abstand
   pro Beat und Beat-Phase positiv vs. negativ ausgeben (müssen bei Länge/Phase gleich sein).
   Zusätzlich zum Audio-Test ein **Merkmals-Test**: Genauigkeit mit genullten Slider-Merkmalen
   und mit genullten Positions-Merkmalen berichten – nur beim Nullen der Positionen darf sie
   stark fallen.

## Rückmeldung am Ende
Verteilungen positiv/negativ, Genauigkeit/AUC (normal, Jitter, falsches Audio), die
Vergleichstabelle aus Schritt 5 (40 Songs), Erzeugungszeit, geänderte Dateien, offene Probleme.
