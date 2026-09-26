# Auftrag: Nachtlauf (~10 h GPU) – Prompts 03, 04 und 02 nacheinander

Der Nutzer lässt den PC über Nacht ~10 Stunden laufen und möchte, dass in dieser Zeit
möglichst alle geplanten Modelle trainiert werden. Dieser Auftrag regelt **Reihenfolge,
Zeitbudget und Absicherung**. Was jeweils gebaut wird, steht in den einzelnen Prompts:

1. `prompts/03-v3-sliderformen-und-muster.md` – Platzierungsmodell v3 (**Priorität 1**)
2. `prompts/04-vorplanungs-ki.md` – Vorplanungs-KI mit Songteilen/Hauptteil
3. `prompts/02-songpassung-und-mehrere-durchgaenge.md` – Song-Passungs-KI + Durchgänge

Zuerst `AGENTS.md`, `docs/STATUS.md`, `docs/WORKLOG.md` lesen. Regel des Nutzers: nichts
von anderen KI-Mappern übernehmen (siehe Prompt 03, Ausgangslage).

## Grundsätze
- **Nur ein PyTorch-Prozess gleichzeitig.** Alle Trainings streng nacheinander.
- **Erst bauen, dann schlafen lassen:** Solange du aktiv bist, den Code für alle drei
  Teile schreiben und mit Mini-Läufen prüfen (je ein paar Minuten, wenige Daten: läuft,
  Loss sinkt, Checkpoint wird geschrieben, Fortsetzen klappt). Danach **einen einzigen
  Nachtlauf** starten, der alles Weitere ohne dich schafft – falls deine Sitzung endet oder
  ein Limit erreicht wird, muss der Lauf trotzdem weiterlaufen.
- Dafür ein Skript `scripts/night_run.py` (+ `night_run.bat` zum Starten), das die Phasen
  unten nacheinander als **eigene Unterprozesse** startet, mit hartem Zeitlimit je Phase
  (Prozess nach Ablauf beenden, letzter Checkpoint zählt), alles nach
  `D:\BeatMap-AI-Dataset\night\<Datum>\` loggt (Start/Ende/Exit-Code je Phase in
  `night_run.log`, dazu Log je Phase) und bei einem Absturz **die nächste Phase trotzdem
  startet**. Jede Phase ist fortsetzbar (Checkpoints mindestens alle 30 min).
- **Nichts automatisch in die App übernehmen.** Kein Kopieren nach `beatmap_ai/models/`
  während der Nacht; das entscheiden wir morgens nach den Messungen und dem Blindtest.
  Die App muss morgens genauso funktionieren wie heute.
- Nur committen/pushen, was getestet ist (`pytest -q` grün). Halbfertiges bleibt lokal.
- Zeitbudget ist wichtiger als Vollständigkeit: lieber v3 gut als alle drei halb.

## Zeitplan (Summe ~10 h; Budgets sind Obergrenzen)
| Phase | Inhalt | Budget |
|---|---|---|
| A | Datenvorbereitung ohne GPU, wo möglich vorher: Slider-Pfade/Cache (03 Schritt 1), Abschnittswerte (03 Schritt 0b), Kiai-/Songteil-Labels (04), Paare für die Song-Passung (02) | vor dem Nachtlauf oder ≤ 45 min |
| B | Tagger trainieren + alle Maps taggen (03 Schritt 0) | ≤ 45 min |
| C | v3-Probelauf klein vs. groß (je ~20 min), größere Variante nur wenn sie im Probelauf klar besser ist | ≤ 45 min |
| D | **v3-Haupttraining** | ≤ 5 h |
| E | Vorplanungs-KI trainieren (04) | ≤ 1 h 15 min |
| F | Song-Passungs-KI trainieren (02) | ≤ 1 h 15 min |
| G | Messen: 40-Song-Vergleich v2 vs. v3 (03 Schritt 4, mit/ohne Krücken), Blindtest-Maps nach `Vergleich/v3/` (03 Schritt 5), kurze Auswertung von E und F auf Validierungssongs | ≤ 1 h |

Falls eine frühere Phase schneller fertig ist, verfällt die Restzeit nicht: D darf die
gesparte Zeit nutzen, solange G sicher noch vor Ablauf der 10 h fertig wird.

## Was morgens da sein muss
- `D:\BeatMap-AI-Dataset\night\<Datum>\night_run.log` mit allen Phasen, Zeiten, Exit-Codes.
- Checkpoints: v3, Tagger, Vorplanungs-KI, Song-Passungs-KI (je bestes + letztes) auf D:.
- Messtabellen aus G als Markdown in derselben Mappe, dazu `Vergleich/v3/` mit den
  Insane-Maps von GUERREIRO und ALQUIMIA.
- Einträge in `docs/STATUS.md` und `docs/WORKLOG.md`: was fertig ist, was abgebrochen
  wurde (mit Grund), was noch fehlt (z. B. Einbau von 04/02 in den Generator, falls die
  Zeit nur fürs Training gereicht hat).

## Hinweise für den Nutzer (vor dem Start prüfen)
- BeatMap-AI-Oberfläche schließen (sie belegt sonst PyTorch-Speicher).
- Windows darf nicht in den Standby gehen und nicht für Updates neu starten
  (Energieoptionen: Standby „Nie“; Windows Update für einen Tag pausieren).
- D: braucht Platz: ~20–30 GB frei reichen.
