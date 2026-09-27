# Projektstand BeatMap-AI

_Zuletzt aktualisiert: 2026-09-27 ~23:45 W. Europe Summer Time, von Claude Code (Cloud)._
Jeder Agent aktualisiert diese Datei, bevor er aufhört (siehe `AGENTS.md`).

## Currently running / in progress
- **Kein BeatMap-AI-Job läuft.** Nachtlauf am 27.09.2026 um 07:58:45 beendet; kein Python-/PyTorch-Prozess aktiv. Gesamtstatus und Berichte: `D:\BeatMap-AI-Dataset\night\2026-09-26_220013\`.
- Prompt 05 nur teilweise abgeschlossen: Phasen A/B/C liefen jeweils ins 45-Minuten-Limit, während der Datensatz eingelesen/vorbereitet wurde. Phase D trainierte v3 bis Epoche 97/1000; bester gespeicherter Checkpoint Epoche 93, Validierungsverlust 2,7839 (11,26 Mio. Parameter; 45,1 MB). `sequence-v3.pt` liegt ausschließlich im Nachtlaufordner; in der App bleibt v2 aktiv.
- Phase E erreichte nach 52,5 Minuten Datensatzaufbereitung noch kein Planner-Training. Der angeforderte Validierungslauf 128 vs. 256 und die Auswahl nach Hauptteil/Kiai/Stern-/Abschnittsfehlern fehlen. Phase F trainierte Song-Passung bis Epoche 78; bester AUC-Wert 0,9557 in Epoche 56; Checkpoint ebenfalls nur im Nachtlaufordner.
- Phase G erstellte zwei Blindtest-Maps und den P95-Bericht. ALQUIMIA Insane v3 liegt weiterhin bei 60,704 P95-Ausreißern/100 Objekte, davon 49,560 schnelle scharfe Wendungen; GUERREIRO Insane v3 bei 8,844 gesamt. Die v2/v3-Folgeauswertungen liefen in ihre Zeitlimits; Planner-Auswertung wurde mangels Checkpoint übersprungen. Kein belastbarer Gesamtsieg von v3 ist damit belegt. Details: `Phase-G-Muster-und-P95.md`, `Phase-G-Messung.md` im Nachtlaufordner.
- Der automatische Statusabschluss des Supervisors scheiterte an einem Windows-Pfad in einer Regex-Ersetzung. Behoben und mit Regressionstest geprüft; der Lauf und seine Checkpoints waren davon nicht betroffen.
- Prompt-05-Vorprüfung: `pytest -q` ergab 50 bestandene Tests. ROCm-Schrittzeit Planner: 128 =
  0,0106 s (0,59 Mio. Parameter), 256 = 0,0135 s (2,30 Mio.). Tagger-, Planner-,
  Song-Passung- und v3-Smokes bestanden. Windows Update ist bis 2026-09-27 19:50 UTC
  pausiert. Commit `2061372` ist gepusht; kein Modell wird automatisch in die App übernommen.
- Prompt sol-01 ist am 26.09.2026
  abgeschlossen: 2.000 menschliche Validierungs-Maps und 2.997 gültige Maps aus 3.000
  Manifestzeilen; 286 passende Validierungspaare wurden verbunden. Bericht:
  `D:\BeatMap-AI-Dataset\pattern_stats_2026-09-26.log`; menschliche Referenz:
  `beatmap_ai/pattern_reference.json`. Der Lauf war CPU-only und importierte kein PyTorch.
- Prompt 01b ist am 26.09.2026 abgeschlossen und ausgewertet.
  Die Dateinamenbereinigung greift jetzt auch **nach** dem Kürzen (`[:60]` und `[:40]`),
  wodurch der TUYU-Pfadfehler behoben ist. Neu erzeugt wurden 3.000 Paare, je 750 pro
  Sternbereich; Kiai-Fenster: `<3`: 252, `3-4.5`: 318, `4.5-6`: 368, `6+`: 341.
  Ausgabe/Manifest: `D:\BeatMap-AI-Dataset\critic_negatives_paired\`; Log:
  `D:\BeatMap-AI-Dataset\critic_paired_generation_resume2.log`. Die Vollprüfung über alle
  3.000 Paare meldet keine fehlenden Dateien/Quellen und keine Rhythmus-/Timingpunkt-Abweichungen.
  Slider-Pixellänge/Beat: Mittel 163.08821 in beiden Gruppen (mittlere Paardifferenz 0.00011);
  Abstand/Beat: 0.63094 (Paardifferenz 0.00000052); Beat-Phase: 0.68388 (Paardifferenz
  <0.00000001). Kiai-Fenster: 1.279/3.000. In der sauberen 200-Paar-Featureprüfung gab es
  0 Rhythmus-/Timingpunkt-Abweichungen; 139/200 Fenster sind Kiai.
  Single-Process-Training (15 Epochen, `--workers 0`): 90,33 % Accuracy / 0,9593 AUC.
  Jitter: 90,04 % / 0,9587; Audio-Shuffle: 89,45 % / 0,9607; Audio genullt: 90,63 % / 0,9644.
  Slider genullt: 87,70 % / 0,9489; Position genullt: 50,00 % / 0,5000. Checkpoint:
  `D:\BeatMap-AI-Dataset\critic_paired.pt`; Trainingslog:
  `D:\BeatMap-AI-Dataset\critic_paired_training_dropout.log`. Eine frühe Variante mit
  `--workers 4` wurde beendet, als Spawn-Prozesse PyTorch importierten; das abgeschlossene
  Training nutzte ausschließlich `--workers 0`.
  A/B auf 40 Validierungssongs und allen 247 Difficulties (Best-of-4) ist beendet.
  Rhythmus-F1 ist in allen drei Varianten gleich: 0,671 insgesamt; je Sternbereich `<3`: 0,561,
  `3-4.5`: 0,652, `4.5-6`: 0,736, `6+`: 0,763. Bewegungsfehler gegen Menschen:
  Baseline 0,2305; alter Critic 0,2104; neuer Critic 0,2146. Der neue Critic verbessert
  die Baseline, erreicht aber nicht den alten Critic; er wurde nicht
  nach `beatmap_ai/models/critic.pt` übernommen. A/B-Log mit allen Bewegungsmetriken je
  Sternbereich: `D:\BeatMap-AI-Dataset\critic_paired_ab.log`. Generation pro Difficulty:
  3,39 s Baseline, 13,79 s alter Critic, 13,80 s neuer Critic (4,1× Laufzeit).
  Es läuft kein Python-/PyTorch-Prozess.

## Letzter externer Auftrag (26.09.2026)
- Luna-Nachtauftrag in `D:\osu\Training-UnlockedBrick32` abgeschlossen. Öffentliche Profildaten ausgewertet; 6★-Track mit 24 installierten Maps und 14 Tempo-Maps erstellt.
- 120 neue `.osz`-Sets (883.317.333 Bytes) nach `D:\osu\downloads\neu-6sterne\` geladen und geprüft. 1/15 Sets erfüllt im gefundenen Pool den exakten hohen AR-Bereich; 14 Ergänzungen sind im Manifest gekennzeichnet.
- Trainingsplan, Morgenbericht und Importhilfe liegen im Trainingsordner. Noch nicht importiert: osu! lief nicht und wurde nicht gestartet. PC-Wachhalter ist zurückgesetzt; geschützte Konfigurationen blieben unverändert.

## Was die App gerade benutzt (`Start BeatMap AI.bat` → `.venv-rocm`)
| Aufgabe | Datei | Parameter | Stand |
|---|---|---|---|
| Rhythmus (wann Noten kommen) | `beatmap_ai/models/rhythm.pt` | 6,06 Mio. | Conformer v4, 10.387 Songs, Schwellen je Sternbereich |
| Platzierung (wo, Muster, Slider-Richtung, Hitsounds) | `beatmap_ai/models/sequence.pt` | 10,33 Mio. | Sequenzmodell v2 (fertig trainiert), wird per `SequencePlacer.follow` entlang des Rhythmus benutzt |
| Rückfall-Platzierung | `beatmap_ai/models/placement.pt` | 4,90 Mio. | v1, nur wenn sequence.pt fehlt |
| Bewerter (Critic) | `beatmap_ai/models/critic.pt` (v1) / `critic-v2.pt` (Luna) | je 4,88 Mio. | v1 Standard; in der UI wählbar Alt/Neu/Aus |
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
- **Bewerter-KI (Critic v1 bleibt aktiv; Prompt 01b ausgewertet)**:
  - 3.000 gepaarte Negativ-Maps, balanciert mit je 750 pro Sternbereich; Positiv/Negativ
    stimmen pro Paar bei Song, Sternen und Rhythmus überein. Sterne, Dichte und Objektzahl
    sind im Loader ausgeglichen (beide Gruppen: 4,47 ± 1,71★, 95,6 ± 3,3 Objekte,
    Dichte 3,684 ± 1,814; gepaarte Differenzen 0).
  - Bestes Training: 90,33 % Accuracy / 0,9593 AUC; Jitter 90,04 % / 0,9587.
    Audio-Shuffle 89,45 % / 0,9607 und genulltes Audio 90,63 % / 0,9644 zeigen weiterhin
    geringe Audionutzung. Genullte Slider-Merkmale: 87,70 % / 0,9489; genullte Positions-
    Merkmale: 50,00 % / 0,5000.
  - A/B 40 Songs / 247 Difficulties: Critic Probability 12,465 % Baseline → 20,447 % alter →
    50,807 % neuer Critic; Rhythmus-F1 bleibt in allen Varianten 0,671. Der normalisierte
    Bewegungsfehler sinkt mit dem alten Critic auf 0,2104, steigt mit dem neuen aber auf
    0,2146 (Baseline 0,2305). Deshalb bleibt `beatmap_ai/models/critic.pt` auf v1; das neue
    Modell liegt nur unter `D:\BeatMap-AI-Dataset\critic_paired.pt`.

## Muster-Messungen (Prompt sol-01, 26.09.2026)
- 2.000 menschliche Validierungs-Maps, gleichmäßig 500 je Sternbereich; 5.757 Kandidaten
  geprüft. 2.997/3.000 KI-Maps bestanden die osu!standard-/Objekt-/Timingpunkt-Prüfung
  (alle Quelldateien sind vorhanden). KI-Anzahl je Sternbereich: 1.778 · 854 · 315 · 50.
- Muster-Abweichung ist der mittlere robuste Abstand jeder Map zum menschlichen Medianprofil
  ihrer Sternklasse; 0 heißt exakt dieses Medianprofil. Mittelwert je Map Mensch/KI:
  `<3`: 0,492/0,648 · `3–4,5`: 0,641/0,634 · `4,5–6`: 0,708/0,668 · `6+`: 0,692/0,779.
- Größte Unterschiede: `6+` wiederholte Muster 32,28/5,06 %; `<3` Zickzack 0,20/1,15 je 100;
  `6+` Taktpassung 25,29/2,14 %; `<3` Slider-Anteil 57,40/42,53 %; `4,5–6` wiederholte Muster
  24,53/5,24 % (jeweils Mensch/KI). Bei `4,5–6` liegen Doubles auf 1/4 Beat bei 3,85/2,05
  Gruppen je 100 Objekte und Slider bei 42,37/31,56 %.
- Im Manifest als Validierung markierte direkte Paare: 286 (Sternbereiche: 170 · 86 · 29 · 1).
  Die `6+`-KI-Stichprobe umfasst nur 50 Maps, dort gibt es nur ein direktes Paar.
- Die 5./50./95. Perzentile zu Sprunggeschwindigkeit, scharfen Wendungen bei hohem Tempo,
  Streamlänge und Bildschirmabdeckung stehen je Sternbereich in `pattern_reference.json`.
  Der Bericht enthält pro Mustertyp bis zu drei Maps mit Zeitstempel für die Prüfung im Editor.
  Die Takt-/Songteilpassung ist ein Näherungswert; die Beispiele wurden noch nicht manuell
  im osu!-Editor kontrolliert.

## Rückmeldungen des Nutzers (was noch stört)
- Gut: Stil-Auswahl funktioniert, oft besser als „Auto“.
- 26.09. nach dem Stand mit Sternzielen + Slider-Korrekturen: **„deutlich über einem
  mittelmäßigen menschlichen Mapper“, extrem spielbar.** Rhythmus sehr gut, Qualität aber
  schwierigkeitsabhängig. Einziger Wunsch gerade: **Platzierung noch einen kleinen Tick
  besser.** Doubles/Triples nur, wo die Musik sie hergibt, nie „einfach so“ (Messung
  01b-A/B: Doubles 1,5 statt 2,8 pro 100 Noten bei 4,5–6★).
- 26.09.: Stand mit Critic v1 + Slider-Korrekturen wirkt „deutlich menschlicher“. Aber
  **Insane ist immer am besten, alle anderen Schwierigkeiten hinken hinterher.**
  Nachgemessen (`scripts/compare_maps.py`, Maps aus osu!lazer in `Vergleich/`, 2 Songs:
  SEM SAIDA 130 BPM, Cavalona 205 BPM; vor Critic vs. Critic v1 vs. 60 menschliche Maps
  je Sternbereich):
  1. **Expert ist nicht schwerer als Insane** (SAIDA 3,9–4,1★ vs. 4,0–4,3★; Cavalona
     4,5–4,8★ vs. 4,3–4,7★). Mit fester 5★-Vorgabe wird es über Streams erreicht (65 %
     1/4-Abstände, 29 % Stacks) statt über Sprünge.
  2. **Normal/Hard: Zickzack wie Insane.** Scharfe Wendungen >120° 36–92 % (Mensch 8–18 %),
     „gleiche Wendung wie davor“ bei SAIDA-Hard 59–71 % (Mensch 21 %), Sprünge größer als
     beim Menschen; SAIDA-Hard außerdem 5–6 Doubles/Triples pro 100 (Mensch <1).
  3. **Critic v1 erhöht Stacks stark** (in fast allen Diffs, z. B. 3 → 14 %, 5 → 16 %,
     12 → 26 %; Mensch 6–11 %). In der 01b-A/B-Messung lagen Stacks insgesamt bei 7,3 %
     Mensch, 10,0 % Baseline, 10,8 % alter und 9,1 % neuer Critic; der neue Critic liegt
     hier näher am Menschen, sein gesamter Bewegungsfehler ist aber trotzdem höher als beim alten.
  4. SAIDA (Funk): kaum Slider (3–17 %, Mensch 40–57 %), zu dicht, Combos zu lang (7–11
     statt 4–5 Noten). Bei Cavalona normal – songabhängig (Rhythmus-KI).
  Insane liegt in fast allen Merkmalen im menschlichen Bereich.
  Nutzer: Die Version vor dem Critic traf die Schwierigkeit besser als die mit Critic v1.
  Passt zu den Zahlen: gleicher Rhythmus, aber Critic v1 verschiebt die Sterne ohne
  Kontrolle (SAIDA: Hard 3,68 → 3,87★ mit größeren Sprüngen, Insane 4,27 → 4,00★ und
  Expert 4,10 → 3,86★ mit viel mehr Stacks), weil benannte Schwierigkeiten kein Sternziel
  haben. UI hat jetzt einen Schalter „Bewerter-KI“ (`--no-critic`).
  **Erledigt 26.09. nachmittags (Claude Code):** (a) feste Sternziele für die Namen
  (`difficulty.DEFAULT_STARS`: Easy 1,5 / Normal 2,0 / Hard 3,0 / Insane 4,5 / Expert 5,6;
  nur Standard, bis die Vorplanungs-KI aus Prompt 04 die Sterne pro Song festlegt) – auf
  3 Songs × 4 Diffs jetzt alle innerhalb ±0,26★. Sternsuche ohne Critic, nur das Ergebnis
  wird mit Critic gerankt (Rückfall auf das Suchergebnis, wenn der Critic die Sterne um
  >0,2 verschiebt): ~100 s statt 180–360 s pro Song mit 4 Diffs. (b) Slider: Kick-Slider
  seltener (`rhythm.kick_confidence` 0,92/0,8/0,65 bei 4/6/7★; vorher 87 % 1/4-Slider bei
  Expert, jetzt 0–3 %, Mensch 11–15 %); Slider < 70 px gerade, bis 110 px weniger gebogen,
  Biegung insgesamt halbiert (`sequence_model.BEND_SCALE`; gebogen jetzt 9 %, Mensch 7 %;
  fast Kreis 2–3 %, Mensch ~1 %); Biegung pro Slider fest gewürfelt, damit der Critic sie
  nicht aussuchen kann. UI: Auswahl Bewerter-KI Alt/Neu/Aus
  (`beatmap_ai/models/critic-v2.pt` = Lunas Critic), Mausrad verstellt keine Auswahlfelder
  mehr (hat vorher Stil/Critic beim Scrollen zurückgesetzt).
  **Cursor-Flow (26.09. abends):** `SequencePlacer.sharp_keep` – bei niedrigen Sternen wird
  ein Teil der scharfen Umkehrungen (> 120°) neu gewürfelt (behalten: 10 % bei 2★, 25 % bei
  3★, 60 % bei 4★, alle ab 5★). Gemessen entlang des Cursorwegs (inkl. Slider, so misst
  `scripts/compare_maps.py` jetzt – die alte Messung über Objekt-Startpunkte hat das Zickzack
  bei sliderreichen Maps stark übertrieben): Normal 1–30 % → 8–18 %, Hard 9–89 % → 5–10 %
  (Mensch 3–9 %). Noch offen: zu wenig gerade Weiterführungen bei Normal/Hard (5–15 % statt
  ~21 %), bei Expert eher zu wenige scharfe Wendungen (28–32 % statt ~50 %).
  Nutzer: Rhythmus bei Songs mit Gesang im Vordergrund noch schwach (in osu! selten);
  wichtiger: KI soll wissen, wann der Hauptteil kommt (→ Prompt 04, Songteile).
  **Slider spielbarer (26.09. abends, Nutzer: „zu viele Slider, rhythmisch ok, spielerisch
  unangenehm“):** Ursache war eine Note nur 1/4 Beat nach dem Slider-Ende (Insane/Expert
  52–87 %, Mensch 7–21 %). `rhythm.plan_objects`: nach Slidern meist ½ Beat Pause, kurze
  Pause nur mit `quick_release` (8/15/25 % bei 4/5,5/6,5★) → jetzt 8–16 %. Slider-Anteil
  gedeckelt auf das obere Quartil menschlicher Maps (`slider_cut`, 68/66/55/48 % bei
  2/3,5/5/6★): Normal 85–90 % → 49–55 %. Nebenwirkung: bei dichten Songs Insane/Expert
  teils nur 8–18 % Slider (Mensch 38–45 %) – beobachten.
  **Noch offen:** Sternziel über „was zur Musik passt“ statt Notenmenge (langsame Songs:
  Expert 6,2 Noten/s statt ~3,9, 3 % Slider, 13 % Stacks); Normal/Hard noch zu viel
  Zickzack (scharfe Wendungen 42–81 % statt 8–18 %) und zu viele neue Combos bei langsamen
  Songs (1,5–2,4 Noten pro Combo statt ~5).
- Offen: noch nicht perfekt; Maps fühlen sich noch nicht individuell für den Song an;
  bei hohen Sternen inkonsistenter; Jump-Muster (Zickzack, Vielecke) fehlen bzw. zu
  zufällig; Doubles zu selten (1,0 statt 2,4 pro 100 Noten).

## Vergleich mit Mapperatorinator (26.09. abends, Nutzerwunsch)
- Nutzerziel: „die bestmöglichen KI-Maps zum Spielen“. Mapperatorinator (OliBomby, 219 Mio.
  Parameter, ~5.700 GPU-h) ist der bekannte Stand der Technik. Lokal installiert unter
  `D:\Mapperatorinator` (eigene venv, nutzt unser ROCm-torch per .pth, `torchaudio` aus dem
  AMD-Repo, `sitecustomize.py` schaltet MIOpen ab; Modelle in `D:\Mapperatorinator\hf`).
  Läuft auf der RX 7700 XT, ~65–95 s pro Difficulty. Offiziell nur Linux für AMD.
- 3 Songs × 2,0/4,5/5,6★ mit beiden erzeugt; Messung `D:\Mapperatorinator\compare\compare.md`.
  S0N6F0RMYD34TH: Mapperatorinator-Timing kaputt (Start erst nach 33–57 s, Noten bei 0 ms,
  3,65★/8,4★ statt 4,5/5,6; fp32-Neuversuch: „No timing points“) – evtl. unsere Umgebung.
- Blindtest für den Nutzer: `D:/Mapperatorinator/blindtest/` (GUERREIRO, ALQUIMIA; A/B
  zufällig, Auflösung in `AUFLOESUNG_erst_nach_dem_Spielen_oeffnen.json`). **Ergebnis (Nutzer, nur
  Insane gespielt): beide gut, Mapperatorinator („B“) bei beiden Songs einen Ticken besser,
  „keine Welten“. Rhythmus gleich gut, Platzierung bei Mapperatorinator besser.**
  Entscheidung des Nutzers: **eigenes Platzierungsmodell v3 (Prompt 03), nichts kopieren**
  (Hybrid mit Mapperatorinators Platzierung abgelehnt). Mapperatorinator bleibt nur
  Messlatte für Blindtests. Prompt 03 entsprechend überarbeitet (größeres Modell erlaubt,
  Zielwerte, Krücken, Blindtest-Material).

**Nutzerbeobachtung aus Mapperatorinator-Videos (27.09.):** dort oft (1) inkonsistent – derselbe
Songteil bekommt plötzlich ein anderes Muster – und (2) unangenehme Platzierung. Konsistenz ist
also unser Unterscheidungsmerkmal; im nächsten Blindtest mitbewerten (Refrain beim zweiten Mal),
Kennzahlen `music_aligned_repeat_pct`/`stale_repeat_pct` aus `pattern_stats.py`.
**Stärke laut Nutzer (Blindtest):** BeatMap AI ist pro Song deutlich **konsistenter** – es
wählt eine Musterart (z. B. Fünfeck-Jumps) und zieht sie durch die Map; Mapperatorinator
wechselt das Jump-Muster in jedem Jump-Teil. Muss in v3/04 erhalten bleiben (Wiederholung
mit Variation nach Songteilen, nicht stumpf – `pattern_stats.py`: `stale_repeat_pct`,
`repeated_windows_pct`).
**Nächster Blindtest (nach dem Nachtlauf, Claude Code):** fair nach Sternen – erst
Mapperatorinator erzeugen, dessen tatsächliche Sterne messen, dann BeatMap AI (heute + v3)
mit genau diesen Sternen erzeugen (`-d <Sterne>`). Nutzer: GUERREIRO Normal 2,65★ gegen
2,02★ war nicht vergleichbar (auch Insane/Expert-Paare lagen 0,2–1,1★ auseinander).

## Tageslauf 27.09. (19:15–22:31): v3 groß + Vorplanung – Ergebnis
- **v3 groß** (26,14 Mio., hidden 512, 8 Schichten, Kontext 256, Tagger-Tags für 36.236 Maps):
  168 Epochen × 500 Schritte × 8, bestes Val **1,871** (kleines v3 im Nachtlauf: 2,78).
  Checkpoint `D:/BeatMap-AI-Dataset/day/2026-09-27/sequence-v3.pt` (fortsetzbar: `.last.pt`).
  Training ~0 am Ende → starkes Auswendiglernen; mehr Daten wären der nächste Hebel.
- **Vorplanung** (0,6 Mio.): 8 Epochen, Hauptteil P/R ≈ 0,68/0,69, Kiai P/R ≈ 0,72/0,62,
  Sterne-MAE ≈ 0,5★, Abschnittswerte-MAE 0,11. Checkpoint `…/day/2026-09-27/planner.pt`.
  Val-Verlust ab Epoche 3 steigend → früh stoppen/mehr Regularisierung.
- **v3 gegen v2** (5 Songs × Normal/Insane/Expert, App-Einstellungen, v2 = heutiger App-Stand):
  Insane P95-Ausreißer 39,8 → **18,9**/100, Musterabweichung 0,94 → **0,53**; Expert 16,0 →
  9,7 bzw. 0,66 → 0,48 – **die gemessene Lücke zu Mapperatorinator halbiert**. Aber **zu
  zahm**: Sprünge ½ Beat Insane 108 → 66 px (Mensch ~157), Expert 129 → 78 (~200); scharfe
  Wendungen Insane 38 → 18 % (~33 %), Expert 36 → 15 % (~51 %). Normal etwas schlechter
  (Musterabweichung 1,20 → 1,59). Vermutung: Krücken (`sharp_keep`, Abstands-Feinanpassung)
  sind auf v2 abgestimmt; v3 sah im Training Abschnitts-Vorgaben, die beim Erzeugen fehlen.
  **Nicht übernommen.**
- **Blindtest v2 vs. v3 (Nutzer, 27.09. spät, GUERREIRO + ALQUIMIA, Insane + Expert): v2 gewinnt
  3 von 4** (v3 nur GUERREIRO Insane, das 0,36★ leichter war). v3-Expert: „sehr viele sehr
  knappe Doubles“, „Schlangen-Hüpfer durch die ganze Map“; v2 „deutlich konsistenter“, bessere
  Platzierung. Bei **beiden**: Schwierigkeit über **mehr Noten statt besserer Sprünge**, zu eng
  am Beat, Noten in zwei geteilt. Lehre: P95-Ausreißer allein reichen nicht als Qualitätsmaß
  (v3 hatte weniger Ausreißer, aber zu kleine Sprünge → Sternsuche über mehr Noten).
  **Nächster Schritt:** Sternsuche umbauen – zuerst größere Sprünge (über `follow(scale)`,
  gewürfelt statt gestreckt), erst dann mehr Noten und nur wo die Musik sie hergibt; danach
  v3 erneut testen.
- **Sprünge zuerst (27.09. nachts, `--jumps-first`, `generator.JUMP_SCALE_MAX`; Standard aus):**
  Expert, 5 Songs: v2 Noten/s 5,20 → **3,81** (Mensch 3,94), Triples 2,75 → 0,24/100,
  Sprung ½ Beat 129 → **187 px** (Mensch 200) – genau „weniger Noten, bessere Sprünge“. Aber
  P95-Ausreißer 16 → 53/100, scharfe Wendungen 36 → 62 % (Mensch 51 %), Musterabweichung
  0,66 → 0,99. v3: Noten/s 5,10 → 4,42, Sprünge 78 → 117 px, Ausreißer 9,7 → 28,
  Musterabweichung 0,48 → 0,59. Insane ähnlich, schwächer; Normal unverändert. Sterne
  treffen etwas schlechter (Expert teils 5,1–5,4 statt 5,6). Zielkonflikt: menschliche
  Dichte/Sprunggröße gegen mehr unsaubere Bewegungen → **Blindtest** (Expert: v2 alt / v2
  Sprünge zuerst / v3 Sprünge zuerst) entscheidet. App unverändert (geprüft: bitgleich).
- **Befund Abschnittswerte (27.09. spät, Claude Code Cloud, nur gelesen) – wahrscheinlicher
  Grund für „v3 zu zahm“:** v3 lernt mit 8 Abschnittswerten (Dichte, Sprunggröße, Streams,
  Slider, scharfe/schnelle Wendungen, Quer-Sprünge, Kiai), berechnet aus der **menschlichen Map
  selbst** über ±16 Beats um jede Note – also inklusive der Sprünge, die v3 vorhersagen soll
  (`sequence_data.section_controls`; in 60 % der Trainingsfenster vorhanden, sonst 0). In der
  App sind sie ohne `--planner` immer 0 (`SequencePlacer._model_features`). Der Val-Verlust
  (1,871 groß / 2,78 klein) läuft mit `hide=False`, also **immer mit** den menschlichen Werten:
  er misst nicht die Lage in der App, und der „beste“ Checkpoint wurde danach gewählt. v3 +
  Vorplanung wurde nie zusammen gemessen; die UI sucht die Vorplanung nur unter
  `night/*/planner.best.pt`, der Tageslauf-Planner liegt unter `day/2026-09-27/`.
  Vorschlag (noch nicht mit dem Nutzer abgestimmt): erst ohne Training messen (Val-Verlust mit
  menschlichen Werten / Nullen / Planner-Werten; 5 Songs mit `--planner`), dann v3 vom
  vorhandenen Checkpoint mit App-ähnlichen Werten nachtrainieren (4-Takt-Mittel + Rauschen oder
  Planner-Vorhersagen, Auswahl nach App-ähnlichem Val) und die Sternsuche über den Sprung-Wert
  steuern statt über Strecken/mehr Noten.

## Datenvorbereitung für den nächsten Nachtlauf (27.09., Claude Code) – erledigt
- **Warum der Nachtlauf hing:** (1) `sequence_data.section_controls` rechnete pro Objekt ein
  ±16-Beat-Fenster in Python – **3 s pro v3-Trainingsbatch** (32 Fenster, Kontext 192), v3 hat
  also meist auf Daten gewartet; auch die Vorplanungs-Daten (Phase E) kamen dadurch nicht
  durch. Jetzt vektorisiert: **7 ms pro Batch**, Werte identisch (max. Abweichung 6e-8,
  Test `tests/test_section_controls.py` gegen die alte Version `_section_controls_reference`).
  (2) Der Objekt-Cache für D: (`D:\BeatMap-AI-Dataset\.beatmap_ai_cache\placement-v3.pkl`,
  5 GB) war erst um 01:38 fertig; er ist jetzt vorhanden, Laden ~2 min.
- Vorplanungs-Daten: jede Map einmal (`planner_data._chunk_controls`), Kiai als Maximum über
  die Difficulties. Kompletter Aufbau echter Daten: **6,5 min** (59.699 Difficulties,
  13.375 Songs; vorher > 50 min ohne Ende).
- **Tagger trainiert:** `D:\BeatMap-AI-Dataset\prepared\tagger.pt` (+ `.best.pt`), val AUC
  0,78, bei 1–4★ 0,63; Training 5 s + 2 min Laden. Vorher lernte er nichts (Verlust NaN):
  `tagger_features` bildete Mittelwerte leerer Abschnitte und reichte fehlende Sterne als NaN
  durch – behoben (`_mean`, `nan_to_num`).
- Nächster Nachtlauf kann Phasen A–C fast überspringen (Cache + Tagger liegen bereit) und
  v3 groß (~26 Mio.) mit der vollen Zeit trainieren.

## Nachmessung der Review-Fixes (27.09. vormittags, Claude Code) – wichtig
5 Songs (GUERREIRO, ALQUIMIA, S0N6F0RMYD34TH, LOUCURA LETAL, AKAI) × Normal/Insane/Expert,
Code von gestern (d4d0968) gegen heute, gemessen mit `pattern_stats.py` und
`compare_maps.py` (Wendungen entlang des Cursorwegs):
- **d) Combo-Kopf verschlechtert:** Expert 2,6 statt 6,0 Objekte pro Combo (ranked ~4,4),
  und über die Combo-Eingabe mehr scharfe Wendungen. → abgeschaltet
  (`SequencePlacer.learned_combos = False`), Taktregel wie vorher.
- **a) Richtungseingabe verschlechtert Insane/Expert** (auch ohne d): scharfe Wendungen
  Insane 38 → 60 %, Expert 36 → 50 % (ranked ~23–41 %), gerade Wege 16 → 7 %,
  Musterabweichung Insane 0,94 → 1,48, P95-Ausreißer Expert 16 → 29 pro 100; nur Normal
  leicht besser. Technisch korrekt (Training = Erzeugen), aber die trainierten Modelle
  platzieren ohne eigene Richtung vorsichtiger und menschlicher. → abgeschaltet
  (`placement_model.FEED_HEADING = False`); erst für ein neu trainiertes/gemessenes Modell.
- Mit beiden Schaltern aus erzeugt die App **bitgenau dieselben Maps wie gestern** (geprüft).
  b), c), e), f), KV-Cache bleiben aktiv (betreffen v3, Song-Passung, Vorplanung, nicht die
  App mit v2).
- **v3 (Nachtlauf, 11,3 Mio., Epoche 93, Val 2,78):** die beiden Insane-Maps sind schlechter
  als v2 (ALQUIMIA 60,7 statt 31,8 P95-Ausreißer/100, GUERREIRO Musterabweichung 1,92 statt
  0,41). Nicht übernommen. Vorplanung (Phase E) nicht trainiert (Daten laden > Zeitlimit),
  Song-Passung trainiert (AUC 0,956, mit Platzierungs-Negativen).

## Code-Review einer Cloud-Sitzung (27.09., nur lesend) – Befunde
**Behoben von Claude Code am 27.09. nachts (während Phase D lief; der Nutzer hat mit Luna
abgestimmt, dass ein Eingriff unwahrscheinlich ist; wirkt ab Phase E/F/G und in der App):**
a) Richtung vor dem Modellaufruf eintragen (`_Walker.heading_into`, in `follow`, `sample` und
`LearnedPlacer.sample`; Test `tests/test_placement_inputs.py`). b) v3-Pfad als glatte Kurve
**durch** die vorhergesagten Punkte (`placement_model.through_points`, Catmull-Rom als
Mehrsegment-Bézier) + Drehen/Spiegeln statt Aufgeben (`_Walker._fit_path`). c) Sehne/Biegung
(kalibriert) wird bei v3 immer mitgezogen (Rückfall); die App (Krücken an) nutzt sie statt des
gemittelten Formkopfs, nur „ohne Krücken“ (Messung) nutzt den Formkopf – richtige Lösung:
Formklassen trainieren. d) Combo-Kopf entscheidet neue Combos (`SequencePlacer._decide_combo`,
lange Pause oder 16 Objekte erzwingen eine). e) Song-Passung lernt zusätzlich „richtige Musik,
Platzierung aus einem anderen Teil der Map“ (`songfit.NEGATIVE_TYPES`/`PLACEMENT_COLUMNS`).
f) Vorplanung: ein Beispiel pro Difficulty mit deren Sternen als Bedingung, Abschnitte ab
Taktanfang (`planner_data.audio_section_features(offset_ms)`, `plan_song(offset_ms)`).
KV-Cache: ab vollem Fenster volle Neuberechnung (`SequencePlacer.sample`).
Offen aus dem Review: g) echte Überarbeitung schwacher Abschnitte (`--passes`), Eingabe der
Vorplanung (nur Mel-Statistik), Best-of-N mit P5–P95-Strafe, und die großen Hebel (v4).
**Achtung Messung:** Phase D startete vor den Code-Review-Fixes; Phase G nutzte die Fixes.
Die formalen v2/v3-Auswertungen in G liefen ins Zeitlimit, daher bleibt eine vollständige
Messung mit dem aktuellen Code offen.
- **a) Richtungseingabe beim Platzieren:** Claude Code hat den bestätigten Fehler am 27.09.
  behoben (`_Walker.heading_into` wird vor dem Modellaufruf eingetragen); gezielte Tests
  einschließlich `tests/test_placement_inputs.py` waren grün. v2 und v3 mit diesem Fix
  müssen noch vollständig neu verglichen werden.
- b) v3-Slider: 8 Punkte auf der Kurve werden als Kontrollpunkte einer Bézier geschrieben →
  Kurve kürzer/glatter, Ende stimmt nicht; passt der Pfad nicht, wird still ein Kreis draus.
- c) v3-Formkopf ist Regression ohne Würfeln → Links/Rechts mitteln sich, S-Kurven fehlen.
  Vorschlag: Formklassen (64–256) vorhersagen und würfeln (auch gut für Konsistenz).
- d) Trainierter Combo-Kopf wird in `follow` nicht benutzt; Combos per Taktregel
  (`generator.assign_combos`) – dort die gemessenen Combo-Probleme.
- e) Song-Passungs-KI lernt „gleiche Map, falsches Audio“ → ignoriert Positionen, taugt
  nicht fürs Best-of-N der Platzierung (nur für Rhythmus-Varianten).
- f) Vorplanungs-KI: trainiert mit Set-Höchststernen, abgefragt mit Ziel-Sternen;
  4-Takt-Blöcke ab 0 ms statt Downbeat; Eingabe nur Mel-Mittel/Streuung.
- g) `--passes` = nur mehr Kandidaten, keine echte Überarbeitung schwacher Abschnitte.
- Große Hebel: Platzierung sieht den **kommenden Rhythmus** (bidirektionaler Leser +
  autoregressive Positionen; Mapperatorinator kennt den künftigen Rhythmus nicht);
  Classifier-Free Guidance für Stil/Sterne (ohne Training); Raster-Klassen statt
  Mischverteilung + Neuversuche; Ära/Jahr als Bedingung; Best-of-N mit Modell-
  Wahrscheinlichkeit + P5–P95-Strafe; KV-Cache/RoPE für Tempo (Cache falsch, sobald das
  Fenster voll ist); 1/3-, 1/6-, 1/8-Raster und 3/4-Takt; Timing-Genauigkeit messen.
- Messaufbau: größerer Blindtest (10 Songs × 3 Stufen, Gewinnquote); eigenes Testset nur
  für Endentscheidungen (die 40 Validierungssongs werden überbenutzt); Messskript über den
  echten App-Weg (`generate_beatmap`); Mapperatorinator auf 20+ Songs als Messlatte.
- Aufräumen: Repo öffentlich, 124 MB Modelle in der Git-Historie (Git LFS/Releases);
  Prompts enthalten lokale Pfade und den Vornamen; Testanzahl in STATUS widersprüchlich.

## Offene Punkte / nächste Schritte (Plan vom 26.09. abends, mit Nutzer abgestimmt)
**Phase A – Feinschliff im Generator (Claude Code, GPU lokal, je 1–3 h):**
1. Hauptteil betonen (Zwischenlösung bis Prompt 04): in Kiai-/Refrain-Abschnitten größere
   Sprünge, außerhalb kleinere, gleiche Gesamtsterne.
2. Sterne über passende Mittel statt Notenmenge (Expert bei langsamen Songs: heute 5–7
   Noten/s, 0–14 % Slider, ~20 % Stacks statt ~3,9 / 40 % / 6 %).
3. Flow bei leichten Maps: mehr gerade Weiterführungen (5–15 % statt ~21 %).
4. Standard-Bewerter-KI festlegen (v1/v2/aus) nach Nutzertests; 5,5★+ nachmessen.
**Parallel ohne GPU:** `prompts/sol-01-muster-messungen.md` (GPT-6 Sol); README in einer
Cloud-Sitzung aktualisieren. Prompt sol-01 ist abgeschlossen; das Werkzeug steht für Folge-
messungen an eigenen Dateien und künftigen Modellen bereit.
**Phase B – große Prompts, neue Reihenfolge 03 → 04 → 02** (nie parallel, GPU-Trainings):
- 03 zuerst ergänzen mit den Messungen vom 26.09. (Abstände 220 vs. 350 px/Beat bei
  4,5–6★, Doubles 1,5 vs. 2,8, Slider-Grenzen, Cursor-Flow-Messung aus compare_maps) und
  den neuen Vergleich aus `scripts/pattern_stats.py`.
- 04 Vorplanungs-KI inkl. Songteile/Hauptteil (Nutzerwunsch), ersetzt `DEFAULT_STARS`
  und Auto-Stil.
- 02 Song-Passung + mehrere Durchgänge.
**Später (Nutzerwunsch, erst nach allem anderen): Rhythmus-KI v5.** Ideen: leichte Maps
(0–3★ F1 nur 0,67), gesangslastige Songs, Doubles/Triples nur wo die Musik sie hergibt,
ausgewogene Daten nach Sternen × Stil (Tagger aus 03), Abschnitts-Vorgaben der
Vorplanungs-KI (04) als Eingabe, und die heutigen Regeln (Kick-Slider, Pausen nach
Slidern, Slider-Anteil) mitlernen statt nachträglich anwenden.

## Prompts für Antigravity (Reihenfolge, nie parallel)
| Nr. | Datei | Inhalt | Status |
|---|---|---|---|
| 01 | `prompts/01-bewerter-ki.md` | Bewerter-KI (Mensch vs. KI), Best-of-N | **fertig** |
| 01b | `prompts/01b-bewerter-ki-nachbessern.md` | Critic nachbessern: gepaarte Positiv-Maps (gleicher Song/Sterne), Negativ-Maps mit menschlichem Rhythmus, A/B auf 40 Songs | **fertig; neues Modell nach A/B nicht aktiviert** |
| sol-01 | `prompts/sol-01-muster-messungen.md` | Torch-freie Muster-Messungen; Validierungspaare, Referenzperzentile und Abweichungsscore | **fertig** |
| 02 | `prompts/02-songpassung-und-mehrere-durchgaenge.md` | Song-Passungs-KI (Idee des Nutzers) + mehrere Durchgänge | **Checkpoint trainiert (78 Epochen); Auswertung offen** |
| 03 | `prompts/03-v3-sliderformen-und-muster.md` | **Platzierungsmodell v3** (größer, bessere Platzierung, Slider-Formen, Auto-Tagging + ausgewogene Daten, Abschnitts-Vorgaben); überarbeitet 26.09. nach Blindtest | **teilweise: 97 Epochen; Vergleich offen** |
| 05 | `prompts/05-nachtlauf.md` | Nachtlauf ~10 h: 03 → 04 → 02 nacheinander trainieren (Zeitbudget, Absicherung, nichts automatisch in die App) | **beendet; Phasen teilweise, siehe Nachtlaufbericht** |
| 06 | `prompts/06-gesang-rhythmus.md` | Vocal Rhythm Engine des Nutzers nachbauen (CPU), messen wann Mapper dem Gesang folgen | bereit (Luna, parallel zum Nachtlauf) |
| 04 | `prompts/04-vorplanungs-ki.md` | Vorplanungs-KI (Idee des Nutzers): Songteile erkennen (Hauptteil/Höhepunkt, gelernt aus Kiai + Intensitätswechseln), Sterne/Stil empfehlen, Plan pro Teil | **nicht trainiert; Phase E vor dem Training abgelaufen** |

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
  `scripts/eval_sequence.py` (`--follow` = so wie die App), `scripts/human_agreement.py`,
  `scripts/pattern_stats.py` (torch-frei; Markdown oder `--json`, Referenz in `beatmap_ai/pattern_reference.json`).

