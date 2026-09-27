# Gesangs-Rhythmus: Messung und Empfehlung

**Stand: 27.09.2026** · CPU-only · keine Änderungen am Generator oder am Nachtlauf

## Ergebnis in Kürze

Unter dem hier verwendeten strengen Vergleich lag der Notenstart fast immer näher an einem Beat-/Transientenereignis als an einem Silbenanfang. Zugleich findet der Silben-Detektor deutlich mehr Treffer als eine zeitlich verschobene Zufallsgrundlinie. Das spricht dafür, Gesang künftig **abschnittsweise als zusätzlichen Anker** anzubieten und seine Qualität separat zu bewerten. Eine einfache globale Regel „viel Hochton, wenig Drums“ ist mit diesen Daten nicht verlässlich.

Das ist **kein direkter Generationsvergleich mit Mapperatorinator**. Der bisherige Blindtest des Nutzers berichtet Rhythmus-Parität und eine Schwäche bei Gesangsnoten; diese Messung prüft vorhandene menschliche Maps und den Gesangssignal-Detektor.

## Stichprobe und Verfahren

- Gelesen wurden data und best_maps im Repository, nicht der Datensatzordner des Nachtlaufs. Der Nachtlauf war währenddessen in Phase D aktiv (PID 9932).
- Aus 3.052 gelesenen Beatmap-Dateien blieben 208 Maps mit mindestens 20 Objekten übrig, deren song_key laut dataset.is_validation() in der Validierungsmenge liegt.
- Nach Sternklasse ausgewählt wurden 171 Maps aus 30 Songgruppen: <3★ 50 Maps, 3–4,5★ 45, 4,5–6★ 50 und 6★+ 26. Für die höchste Klasse gab es nur 11 unterschiedliche Songgruppen.
- Die Maps umfassen 3.890 vollständige Vier-Takt-Abschnitte. 32 bereits vorhandene Mel-Feature-Caches wurden ausgewertet; es wurden keine Audio-Dateien neu erzeugt.
- Ein Abschnitt zählt als **Gesang**, wenn der Anteil menschlicher Notenstarts innerhalb von ±30 ms zu einem erkannten Silbenanfang den Anteil zu Beat-/Transientenereignissen um mehr als fünf Prozentpunkte übertrifft. Umgekehrt zählt er als **Beat**; dazwischen als **gemischt**. Beat-Ereignisse sind Viertel-Beats plus Peaks des spektralen Flusses.
- „Stimme erkannt“ bedeutet hier, dass mindestens 40 % der Frames im Abschnitt als stimmhaft eingestuft wurden. Die Silben-Trefferquote betrachtet menschliche Notenstarts innerhalb dieser Abschnitte. Die Zufallsgrundlinie verschiebt die erkannte Silbenfolge innerhalb jedes Abschnitts 100-mal zyklisch.
- Für die einfache Regel wurden Songgruppen deterministisch in Trainings- und Testgruppe getrennt. Im Test lagen 8 Songgruppen mit 669 Abschnitten; nur vier Abschnitte hatten das Label „Gesang“.

Die Stimmenergie wird im 2–6-kHz-Band gemessen. Wie in der JS-Referenz werden HPSS, spektrale Flachheit, MFCC und Neuheit über das Gesamtspektrum berechnet; im Datensatzlauf geschieht das auf dem vorhandenen Mel-Raster von etwa 86 Frames/s. Ein VAD-Label „stimmhaft“ ist keine manuelle Bestätigung von Gesang: harmonische Instrumente können ebenfalls so eingestuft werden. Mel-Caches erlauben keine verlässliche F0-Stabilität; die separate Rohsignal-Funktion schätzt F0 und erzeugt Slider-Kandidaten aus langen, stabilen Vokalen.

## Messwerte

### Welche Songteile folgen Gesang oder Beat?

| Sternklasse | Abschnitte | Folgen Gesang |
|---|---:|---:|
| <3★ | 1.089 | 5 (0,46 %) |
| 3–4,5★ | 1.008 | 1 (0,10 %) |
| 4,5–6★ | 1.127 | 1 (0,09 %) |
| 6★+ | 666 | 0 (0,00 %) |
| **Gesamt** | **3.890** | **7 (0,18 %)** |

In allen Abschnitten zusammen lauten die abgeleiteten Labels: 7 Gesang (0,18 %), 3.787 Beat (97,35 %) und 96 gemischt (2,47 %).

| Songteil | Abschnitte | Folgen Gesang |
|---|---:|---:|
| Ohne Kiai | 2.824 | 6 (0,21 %) |
| Mit Kiai | 1.066 | 1 (0,09 %) |

Die seltenen Gesang-Labels sind eine Folge der absichtlich strengen Regel: Der Silben-Treffer muss den Beat-/Transienten-Treffer um mehr als fünf Punkte schlagen. Sie bedeuten **nicht**, dass diese Maps keine Stimmen enthalten. Das VAD stufte 3.556 Abschnitte (91,4 %) in mindestens 40 % der Frames als stimmhaft ein; das ist ein Signal-Proxy, kein manuell bestätigtes Gesang-Label.

### Silben-Erkennung

| Messung in VAD-stimmhaft eingestuften Abschnitten | Ergebnis |
|---|---:|
| Abschnitte mit mindestens 40 % VAD-stimmhaften Frames | 3.556 |
| Menschliche Notenstarts in diesen Abschnitten | 60.557 |
| Notenstart innerhalb ±30 ms einer erkannten Silbe | 16.321 (26,95 %) |
| Erkannte Silben in diesen Abschnitten | 39.555 |
| Erkannte Silben mit menschlicher Note innerhalb ±30 ms | 16.321 (41,26 %) |

Über alle Abschnitte, in denen mindestens eine Silbe gefunden wurde, lagen im Mittel 27,44 % der Notenstarts innerhalb ±30 ms einer Silbe. Bei 100 zufälligen zyklischen Verschiebungen lag der Mittelwert bei 11,79 % (Differenz: **+15,65 Prozentpunkte**, etwa **2,33×**). Das zeigt verwertbares Timing-Signal, aber noch keine vollständige Silben-Erkennung.

### Gesangsanteil, Kiai und Transienten

| Abschnittsmerkmal | Abschnitte | Folgen Gesang |
|---|---:|---:|
| Hochtonanteil <20 % | 3.885 | 12 (0,31 %) |
| Hochtonanteil 20–40 % | 0 | – |
| Hochtonanteil ≥40 % | 5 | 0 (0 %) |
| Transienten <1/s | 0 | – |
| Transienten 1–3/s | 269 | 0 (0 %) |
| Transienten ≥3/s | 3.621 | 7 (0,19 %) |

Der Hochtonanteil ist die Energie im Band 2–6 kHz geteilt durch die Energie aller übrigen Bänder. Er ist ein Stimm-Proxy, kein Gesangs-Stem; Hi-Hats und Becken tragen ebenfalls dazu bei. Die Transientendichte kommt aus spektralem Fluss und ist keine isolierte Drum-Spur. Die fünf Abschnitte mit mindestens 40 % Hochtonanteil folgten in dieser Stichprobe alle dem Beat statt den Silben.

### Einfache Übergangsregel

Der unveränderte Vorschlag „hoher Gesangsanteil und wenige Drums“ wurde als Hochtonanteil ≥20 % und Transientendichte <1/s geprüft. Er sagte in den acht Test-Songs keinen der vier Gesangs-Abschnitte voraus. Seine Accuracy von 99,4 % entspricht nur der trivialen Regel „nie Gesang vorhersagen“; Recall und balancierte Accuracy liegen bei 0 % bzw. 50 %.

Zusätzlich wurden Schwellen nur auf der Trainingsgruppe so gewählt, dass die balancierte Accuracy möglichst hoch ist. Das ergab im Test:

| Regel | Accuracy | Balancierte Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Trainierte Schwellen (7,17 % Hochton, höchstens 3,28 Transienten/s) | 98,5 % | 49,5 % | 0 % | 0 % |
| Immer „kein Gesang“ (Mehrheits-Baseline) | 99,4 % | 50,0 % | – | 0 % |

Die trainierte Regel fand keinen der vier Gesangs-Abschnitte und markierte sechs Beat-/Misch-Abschnitte fälschlich als Gesang. Die Testmenge ist mit nur vier positiven Abschnitten zu klein, um die Schwellen zu übernehmen. Der feste Hochton-/Drum-Übergang wird daher nicht empfohlen.

### Laufzeit

Die 32 vorhandenen Mel-Caches benötigten zusammen 33,49 Sekunden Analysezeit auf einem CPU-Prozess: Median **0,87 s pro Audioquelle**, 95. Perzentil **2,21 s**. Der Lauf importierte kein PyTorch, startete keine GPU-Arbeit und änderte keine vom Nachtlauf genutzte Datei.

## Beispiele im osu!-Editor

Die Zeiten sind Beatmap-Zeitstempel in Millisekunden. Bei den Gesangsbeispielen steht der menschliche Notenstart links und der erkannte Silbenanfang rechts; bei den Beatbeispielen steht rechts der nächste Beat-/Transienten-Treffer. Die Prozentwerte sind Abschnittsanteile und können bei kurzen Abschnitten stark springen.

| Map / Schwierigkeit | ★ | Abschnitt ab | Notenstart → Referenz (ms) | Silbe / Beat-Treffer |
|---|---:|---:|---|---:|
| Linked Horizon – Guren no Yumiya (TV Size) / Saten's Easy | 1,91 | 53.251 | 54.578 → Silbe 54.579; 56.073 → 56.099 | Gesang 50 %, Beat 25 % |
| Syaro – Caffeine Fighter / Little's Normal | 2,28 | 51.531 | 52.300 → Silbe 52.326; 53.261 → 53.290; 53.838 → 53.859 | Gesang 72,7 %, Beat 54,5 % |
| UVERworld – ODD FUTURE (short ver.) / Easy | 1,62 | 60.430 | 61.367 → Silbe 61.370; 62.305 → 62.334; 63.476 → 63.472 | Gesang 100 %, Beat 83,3 % |
| Syaro – Caffeine Fighter / Easy | 2,00 | 63.838 | 65.184 → Silbe 65.213; 66.146 → 66.165; 66.915 → 66.920 | Gesang 85,7 %, Beat 71,4 % |
| WEAVER – Kuchizuke Diamond / Fast's Normal | 2,24 | 12.567 | 13.116 → Silbe 13.096; 13.299 → 13.317; 14.030 → 14.060 | Gesang 88,9 %, Beat 77,8 % |
| AKI AKANE – FIRST / Chris' Normal | 2,22 | 15.371 | 17.897 → Beat 17.898; 18.371 → 18.367; 19.160 → 19.161 | Gesang 0 %, Beat 100 % |
| AKI AKANE – FIRST / pkhg's Insane | 4,64 | 146.740 | 151.792 → Beat 151.792 | Gesang 0 %, Beat 100 % |
| cYsmix – Moonlight Sonata / Easy | 2,03 | 39.429 | 39.429 → Beat 39.429; 40.366 → 40.367; 41.304 → 41.304 | Gesang 0 %, Beat 100 % |

## Empfehlung für Vorplanung und Rhythmus v5

1. **Anker pro Songabschnitt wählen**, nicht pauschal den ganzen Song auf Gesang umstellen. Zuerst eine Vertrauenszahl aus stimmhaften Frames, Silbenstärke und Silben-Trefferquote bilden. Gesang nur wählen, wenn der Silben-Anker den Beat-Anker klar übertrifft; andernfalls Beat oder „gemischt“.
2. **Kontinuierliche Werte behalten:** Anteil stimmhafter/stimmloser/stiller Frames, Silbenzeiten und -stärke, 2–6-kHz-Energieverhältnis, spektrale Transientenrate, Beat-Abstand und Unsicherheit pro Vier- bis Acht-Takt-Abschnitt.
3. **Lange Vokale als Slider-Kandidaten** markieren, wenn sie mindestens 150 ms dauern und die F0-MAD höchstens etwa 80 Cent beträgt. Das bleibt ein Vorschlag für die Platzierung, keine Änderung an Rhythmus v4.
4. Für Prompt 04 den Anker Gesang / Beat / gemischt samt Vertrauenswert pro Songteil vorsehen. Rhythmus v5 kann die Silben-Onsets, stimmhafte Frames, F0-Stabilität, Transienten und Beat-Abstände als getrennte Eingänge lernen.

Eine spätere Aussage „unsere KI ist besser/schlechter als Mapperatorinator“ braucht einen echten A/B-Lauf: dieselben Audio-Dateien, gleiche Sternziele und dieselben menschlichen Referenzen. Diese Auswertung zeigt vorerst, **wo** ein Gesangs-Anker in den menschlichen Maps messbar auftaucht und wie viel besser er als die Zufallsverschiebung abschneidet.

## Grenzen der Aussage

- Die Messung umfasst nur die lokalen Ordner data und best_maps. Der große Ordner D:/BeatMap-AI-Dataset wurde während des aktiven Nachtlaufs bewusst nicht durchsucht.
- Es gibt nur 30 unterschiedliche song_key in dieser Stichprobe und acht in der getrennten Regel-Testgruppe; in 6★+ nur 26 Maps aus elf Songs.
- Die Labels „Gesang/Beat/gemischt“ werden aus Map-Timing und Signalmerkmalen abgeleitet, nicht von Menschen annotiert. Viertel-Beat plus Transienten ist ein großzügiger Beat-Vergleich und begünstigt Beat-Treffer.
- Das VAD bezeichnet harmonische Frames als stimmhaft; ein harmonisches Instrument kann deshalb fälschlich als Stimme erscheinen. Die 3.556 VAD-Abschnitte sind nicht als 3.556 echte Gesangsabschnitte zu lesen.
- Das vorhandene Mel-Raster glättet Frequenzdetails. Für die Datensatzmessung standen keine hochauflösenden Rohsignale oder manuellen Silben-Labels zur Verfügung. Die Resultate sind ein Diagnose-Pilot, keine endgültige Qualitätsrangliste.
- Es wurde nichts in docs/STATUS.md oder docs/WORKLOG.md geschrieben: Der Nutzer hat für diese Aufgabe ausschließlich neue Dateien freigegeben; WORKLOG.md wurde außerdem parallel vom Nachtlauf aktualisiert.
