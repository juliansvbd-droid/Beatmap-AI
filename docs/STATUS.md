# Projektstand BeatMap-AI

_Zuletzt aktualisiert: 2026-09-26 17:52, von Codex._
Jeder Agent aktualisiert diese Datei, bevor er aufhört (siehe `AGENTS.md`).

## Currently running / in progress
- Keine laufenden BeatMap-AI-Mess- oder Trainingsjobs. Prompt sol-01 ist am 26.09.2026
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
**Bekannt, niedrige Priorität:** Rhythmus bei gesangslastigen Songs schwächer.

## Prompts für Antigravity (Reihenfolge, nie parallel)
| Nr. | Datei | Inhalt | Status |
|---|---|---|---|
| 01 | `prompts/01-bewerter-ki.md` | Bewerter-KI (Mensch vs. KI), Best-of-N | **fertig** |
| 01b | `prompts/01b-bewerter-ki-nachbessern.md` | Critic nachbessern: gepaarte Positiv-Maps (gleicher Song/Sterne), Negativ-Maps mit menschlichem Rhythmus, A/B auf 40 Songs | **fertig; neues Modell nach A/B nicht aktiviert** |
| sol-01 | `prompts/sol-01-muster-messungen.md` | Torch-freie Muster-Messungen; Validierungspaare, Referenzperzentile und Abweichungsscore | **fertig** |
| 02 | `prompts/02-songpassung-und-mehrere-durchgaenge.md` | Song-Passungs-KI (Idee des Nutzers) + mehrere Durchgänge | wartet auf 04 |
| 03 | `prompts/03-v3-sliderformen-und-muster.md` | **Platzierungsmodell v3** (größer, bessere Platzierung, Slider-Formen, Auto-Tagging + ausgewogene Daten, Abschnitts-Vorgaben); überarbeitet 26.09. nach Blindtest | **als Nächstes** |
| 05 | `prompts/05-nachtlauf.md` | Nachtlauf ~10 h: 03 → 04 → 02 nacheinander trainieren (Zeitbudget, Absicherung, nichts automatisch in die App) | **bereit** |
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
  `scripts/eval_sequence.py` (`--follow` = so wie die App), `scripts/human_agreement.py`,
  `scripts/pattern_stats.py` (torch-frei; Markdown oder `--json`, Referenz in `beatmap_ai/pattern_reference.json`).

