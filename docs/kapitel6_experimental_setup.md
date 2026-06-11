# 6. Experimental Setup

## 6.1 Simulationsdesign (Monte-Carlo, OSRM-Routen)

### 6.1.1 Überblick

Die Evaluation der Wartungsrichtlinien (Myopic, MyopicPlus, CFA, VFA, DB, CFA-DB) erfolgt über eine ereignisdiskrete Tagessimulation, implementiert in `src/models/simulator.py` (Klasse `MaintenanceSimulator`). Die Simulation bildet den operativen Ablauf eines Wartungsbetriebs für E-Ladesäulen in Würzburg über einen Zeitraum von bis zu 365 Tagen ab und wird wiederholt im Rahmen von Monte-Carlo-Experimenten ausgeführt, um die stochastische Natur der Störungsereignisse und Fahrzeiten statistisch abzusichern.

### 6.1.2 Simulationsablauf

Die Simulation läuft tageweise (`sim.run(mal_df, max_days)`):

1. **Tagesstart (8:00 Uhr)**: Der `DailyZoneSelector` (`src/planning/selector.py`) wählt für jedes der zwei Wartungsteams eine Startzone und expandiert diese über Nearest-Neighbor zu einer Liste von bis zu 20 Routinestationen pro Team (`max_stations_per_team: 20`). Übrig gebliebene Störungsaufgaben des Vortags (`carryover_tasks`) werden priorisiert in den Tagesplan integriert.
2. **Initialplanung**: Die jeweilige Policy (`policy.create_initial_plan(tasks)`) erzeugt für jedes Team eine Route mittels Greedy-Cheapest-Insertion. Modellabhängig werden dabei Soft-Deadlines, Drop-Entscheidungen (V̂-basiert) und Penalty-Terme berücksichtigt (siehe Kapitel zu den Policy-Tiers).
3. **Stündliche Simulation (8:00–16:00 Uhr)**: Pro Stunde
   - werden neue Störungsereignisse generiert (`_generate_day_disruptions`),
   - wird der Fortschritt der Teams entlang ihrer Route fortgeschrieben (Fahrzeiten aus den OSRM-Matrizen, Servicezeiten),
   - ruft die Policy bei Bedarf `handle_disruptions(disruptions, sim_routes, time_min, hour, log)` auf, um die laufende Route neu zu planen (Replan),
   - werden Type-1- und Type-2-Störungen entweder noch am selben Tag behandelt oder als `carryover` in den Folgetag übertragen.
4. **Tagesabschluss**: Betriebskosten (Lohn-, Kraftstoff-, Ausfallkosten) werden berechnet und je Tag protokolliert (`DayResult`).
5. **Abbruchbedingung**: Die Simulation terminiert automatisch, sobald alle 397 Stationen mindestens einmal routinemäßig gewartet wurden **und** keine offenen (carryover) Störungsaufgaben mehr existieren, spätestens jedoch nach `max_days` Tagen (Standard: 365).

### 6.1.3 Monte-Carlo-Methodik

Da sowohl die Störungsgenerierung (stochastischer Modus) als auch die Fahrzeiten (Stochastic-Travel-Time-Modul) Zufallskomponenten enthalten, wird jede Policy über mehrere unabhängige Simulationsläufe (Monte-Carlo-Runs) evaluiert:

- **Run-Skripte**: `scripts/monte_carlo/run_mc_<modell>.py` (z. B. `run_mc_myopic.py`, `run_mc_cfa.py`, `run_mc_db.py`, …)
- **Parameter**:
  - `--runs N` — Anzahl der Monte-Carlo-Wiederholungen (Standard: 30)
  - `--start-run` — Startwert für die Seed-Sequenz (Standard: 1)
  - `--max-days` — maximale Simulationsdauer pro Run (Standard: 365)
  - `--verbose` — aktiviert detailliertes Logging (z. B. OR-Tools-Ausgaben, sofern relevant)
- **Seeding**: Jeder Run erhält einen eigenen, fortlaufenden Seed (1, 2, 3, …, N), der den globalen Projekt-Seed (`config.yaml`, `project.seed: 42`) für diesen Lauf überschreibt. Damit ist jede Wiederholung reproduzierbar, gleichzeitig aber stochastisch unabhängig von den übrigen Runs.
- **Getrennte Zufallsströme**: Innerhalb eines Runs verwendet der Simulator zwei unabhängige Zufallszahlengeneratoren (`numpy.random.default_rng`):
  - einen für die Störungsgenerierung (Ausfallereignisse je Stunde/Station),
  - einen separaten für die stochastischen Fahrzeiten,
  um eine ungewollte Kopplung der beiden Zufallsquellen zu vermeiden.
- **Aggregation**: Nach Abschluss aller Runs werden die Kennzahlen pro Policy mittels `src/utils/mc_analyse.py` aggregiert (Mittelwert, Standardabweichung, Minimum, Maximum) und in einer Übersichtsdatei (`logs/<modell>/log/<modell>_overview.log`) abgelegt.

### 6.1.4 OSRM-basierte Routenberechnung

Für realistische Fahrzeiten zwischen den 397 Ladestationen sowie dem Depot wird das Open Source Routing Machine (OSRM)-Projekt lokal als Docker-Container betrieben:

- **Setup** (`scripts/setup/setup_osrm.sh`): Lädt den OpenStreetMap-Datenextrakt für Bayern (`bayern-latest.osm.pbf`, ca. 800 MB, Quelle: Geofabrik) und führt die OSRM-Vorverarbeitungspipeline aus (Extract → Partition → Customize) mit dem **MLD**-Algorithmus (Multi-Level Dijkstra), `max-table-size 500`.
- **Betrieb** (`scripts/setup/start_osrm.sh`): Startet den OSRM-Server lokal unter `http://localhost:5000`.
- **Matrixaufbau** (`scripts/setup/build_travel_matrix.py`, `src/api/osrm.py`): Über die OSRM-`/table`-API wird eine vollständige (398 × 398)-Distanz-/Fahrzeitmatrix berechnet (397 Stationen + 1 Depot, Depot = Index 0). Das Ergebnis wird als `.npy`-Datei (`data/distance_matrices/travel_times_duration.npy`) zwischengespeichert, um wiederholte API-Aufrufe zu vermeiden.
- **Stündliche Verkehrsmatrizen**: Mit `scripts/setup/build_traffic_matrix.py` werden zusätzlich zehn stundenspezifische Fahrzeitmatrizen (`traffic_matrix_8uhr.npy` bis `traffic_matrix_17uhr.npy`) erzeugt, die zeitabhängiges Verkehrsaufkommen während des Arbeitstags (8–17 Uhr) abbilden. Jede Matrix hat die Dimension 398 × 398 (float32, ca. 619 KB).
- **Timeout**: HTTP-Anfragen an den OSRM-Server sind auf 120 Sekunden begrenzt (`config.yaml`, `osrm.timeout: 120`).

### 6.1.5 Stochastische Fahrzeiten

Zur Abbildung von Verkehrsschwankungen innerhalb einer Stunde wird jede deterministische Fahrzeit aus den Verkehrsmatrizen zusätzlich mit einer log-normalverteilten Zufallskomponente überlagert (`config.yaml`, Abschnitt `stochastic_travel_times`, `src/utils/stochastic_travel.py`):

```
T_ij,h ~ Lognormal(μ = log(m_ij,h), σ = sqrt(log(1 + cv_h²)))
```

mit `m_ij,h` als deterministischer Fahrzeit (in Sekunden) zwischen Knoten i und j zur Stunde h aus der jeweiligen Verkehrsmatrix, und `cv_h` als stundenabhängigem Variationskoeffizienten:

| Stunde | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| `cv_h` | 0,20 | 0,18 | 0,10 | 0,10 | 0,10 | 0,10 | 0,10 | 0,15 | 0,15 |

Zur Bewertung der Robustheit einer geplanten Route gegenüber dieser Streuung führt der Simulator pro Route eine interne Mini-Monte-Carlo-Simulation mit `n_mc_runs: 200` Stichproben durch und berechnet:

- `mean_end_min`, `p95_end_min` — mittlere bzw. 95 %-Quantil-Rückkehrzeit zum Depot,
- `overtime_prob = P(Endzeit > Arbeitstagsende)`,
- `mean_overtime_min = E[max(0, Endzeit − 480 min)]`.

Ein Schwellenwert `feasibility_alpha: 0.95` legt fest, dass eine Route nur dann als realisierbar gilt, wenn sie mit mindestens 95 % Wahrscheinlichkeit innerhalb des Arbeitstags abgeschlossen werden kann.

---

## 6.2 Datengrundlage (Würzburg-Grid, Störungsverteilung)

### 6.2.1 Stationsdaten

Grundlage der Simulation ist ein realer Datensatz öffentlich zugänglicher Ladeinfrastruktur im Stadtgebiet Würzburg:

- **Quelle**: `data/raw/charging_stations_wue.csv`
- **Umfang nach Filterung**: 397 Stationen mit Status „In Betrieb" (`src/data/loader.py`)
- **Relevante Spalten**:
  - `Ladeeinrichtungs-ID` — eindeutige Stations-ID
  - `Art der Ladeeinrichtung` — Typ: Normalladeeinrichtung (AC, 279 Stationen) oder Schnellladeeinrichtung (DC, 118 Stationen)
  - `Anzahl Ladepunkte` — Anzahl Ladepunkte je Station (typischerweise 2)
  - `Nennleistung Ladeeinrichtung [kW]` — Nennleistung (Minimum 6 kW, Maximum 400 kW, Mittelwert ≈ 85,0 kW)
  - `Inbetriebnahmedatum` — Inbetriebnahmedatum, daraus abgeleitetes Alter in Jahren (0–24 Jahre, Mittelwert ≈ 10,0 Jahre)
  - `Breitengrad`, `Laengengrad` — Geokoordinaten (WGS84)

Beim Laden (`src/data/loader.py`) werden deutsche Dezimalkommata geparst, nicht-operative Stationen gefiltert und das Depot als Index 0 vor die Stationsliste gestellt (`node_idx = station_index + 1`).

### 6.2.2 Depot

- **Name**: WVV Betriebshof Sanderau
- **Adresse**: Friedrich-Spee-Straße 58–64, 97072 Würzburg
- **Koordinaten**: 49,776414° N, 9,938582° O

### 6.2.3 Geografische Zoneneinteilung

Zur Strukturierung der täglichen Tourenplanung wird das Stadtgebiet mittels K-Means-Clustering (`src/planning/clustering.py`) in **178 Zonen** unterteilt (`config.yaml`, `planning.n_zones: 178`). Jede Zone besitzt:

- einen Zentroid (geografischer Mittelpunkt der zugeordneten Stationen),
- eine konvexe Hülle (Fläche als Maß für die räumliche Ausdehnung),
- eine mittlere Distanz zum Depot.

Die tägliche Zonenauswahl (`DailyZoneSelector`) wählt pro Team eine Startzone aus den `n_top_candidates: 20` bestbewerteten offenen Zonen (Mindestabstand zwischen den Startzonen der beiden Teams: `min_team_separation_km: 0.0`, d. h. keine zwingende Trennung) und expandiert sie über Nearest-Neighbor zu maximal 20 Stationen pro Team und Tag (`max_stations_per_team: 20`).

### 6.2.4 Störungsmodell (Failure Simulation)

Das Projekt unterstützt zwei Modi zur Erzeugung von Störungsereignissen, gesteuert über `config.yaml` (`failure_simulation.mode`):

#### Modus „csv" (historisch/Validierung)

- **Quelle**: `data/malfunction.csv`, generiert über `scripts/setup/malfunction_poisson.py`
- **Umfang**: 100 Simulationstage, 371 Störungsereignisse
- **Format je Eintrag**: Tag (1–100), Uhrzeit (8–16 Uhr), Typ (Typ 1 / Typ 2), Stations-ID
- **Erzeugung**: Poisson-Prozess pro Stunde des Arbeitstags (9 Stunden je Tag), mit
  - λ₁ ≈ 3/9 ≈ 0,333 Type-1-Ereignisse pro Stunde (≈ 1,5 / Tag)
  - λ₂ ≈ 1/9 ≈ 0,111 Type-2-Ereignisse pro Stunde (≈ 0,5 / Tag)

Hinweis: `value_based_zone_selection` hat im CSV-Modus keine Wirkung, da hierfür `_days_since_maintenance` benötigt wird, welches in diesem Modus nicht gepflegt wird (klassische Zonenscoring-Logik wird verwendet).

#### Modus „stochastic" (Hauptmodus für die Evaluation)

In diesem Modus, der für alle aktiven Policy-Tiers (Myopic bis CFA-DB) als Standard verwendet wird (`config.yaml`, `failure_simulation.mode: "stochastic"`), wird für jede Station und jede Stunde des Arbeitstags individuell und probabilistisch entschieden, ob eine Störung auftritt (`MaintenanceSimulator._generate_day_disruptions`, `simulator.py`):

**Basisformel (Erholungskurve):**

```
p(t) = p_base × [initial_factor + (1 − initial_factor) × t / recovery_days]
```

mit:
- `p_base ∈ {p1_per_hour, p2_per_hour}` — Basiswahrscheinlichkeit je Störungstyp:
  - `p1_per_hour = 0,00150` (Type 1 — Vor-Ort-Reparatur)
  - `p2_per_hour = 0,00075` (Type 2 — Modulwechsel mit Werkstattaufenthalt)
- `t = days_since_maintenance` (`dsm`) — Tage seit der letzten Wartung der Station, je Knoten individuell verfolgt (`_days_since_maintenance[node_idx]`)
- `recovery_days = 365` — Anzahl Tage bis zur vollständigen „Erholung" auf 100 % der Basiswahrscheinlichkeit
- `initial_factor = 0,1` — Restwahrscheinlichkeit unmittelbar nach einer Wartung (10 %)

Direkt nach einer Wartung (t = 0) liegt die Ausfallwahrscheinlichkeit somit bei 10 % des Basiswerts und steigt linear über ein Jahr auf 100 % an, was eine alterungsbedingte Degradation seit der letzten Wartung modelliert.

**Stationsspezifischer Faktor:**

Zusätzlich wird `p(t)` mit einem stationsspezifischen Faktor skaliert, der sich aus zwei Komponenten zusammensetzt (`src/data/loader.py`):

1. **Ladertyp-Komponente**: Schnellladestationen (DC) weisen eine doppelt so hohe Ausfallrate auf wie Normalladestationen (AC) (`α_DC = 2 × α_AC`), wobei die Faktoren so normiert sind, dass der flottenweite Mittelwert 1,0 beträgt (Verhältnis 279 AC : 118 DC).
2. **Alters-Komponente**: linearer Skalierungsfaktor `1,0 + β × (Alter − mittleres Alter)` mit `β = 0,03`, begrenzt auf das Intervall [0,5; 2,0]. Eine neue Station (Alter 0) erhält damit einen Faktor von ≈ 0,7, eine 20 Jahre alte Station ≈ 1,3.

Die Gesamtverteilung des kombinierten Faktors über alle 397 Stationen reicht von ≈ 0,54 bis ≈ 2,19 bei einem Mittelwert von 1,0.

**Restriktion**: Pro Station und Tag kann höchstens eine Störung auftreten.

**Erwartete Ereignisrate** (bei 397 Stationen, 9 Betriebsstunden, mittlerem dsm und Faktor 1,0):
- Type 1: 0,00150 × 397 × 9 ≈ 5,4 Ereignisse/Tag (im eingeschwungenen Zustand; unmittelbar nach flächendeckender Wartung deutlich niedriger wegen `initial_factor`)
- Type 2: 0,00075 × 397 × 9 ≈ 2,7 Ereignisse/Tag

Optional kann über `randomize_initial_dsm: false/true` gesteuert werden, ob alle Stationen zu Simulationsbeginn mit `dsm = 0` starten (Standard) oder jede Station mit einem zufälligen, aber über alle Runs identischen `dsm ~ Uniform(0, recovery_days)` initialisiert wird (fixer Seed).

### 6.2.5 Servicezeiten

- **Routinewartung**: feste Servicezeit `mean_service_time: 45` Minuten (`service_time_mode: "fixed"`); alternativ `"per_charging_point"` mit `minutes_per_charging_point: 15`.
- **Type-1-Störung**: Vor-Ort-Reparatur, ca. 60 Minuten Servicezeit.
- **Type-2-Störung**: Demontage (≈ 30 Min) + Hin-/Rückfahrt zur Werkstatt + Remontage (≈ 30 Min) + Handling (≈ 5 Min).

---

## 6.3 Evaluierungsmetriken (Kosten, Verfügbarkeit, Rückstände)

Die Auswertung jeder Policy basiert auf den von `MaintenanceSimulator.write_json()` (`src/models/simulator.py`) erzeugten JSON-Logs (`meta`, `summary`, `days`, `hourly`) sowie den daraus abgeleiteten, über `src/utils/mc_analyse.py` aggregierten Übersichtsdateien (`logs/<modell>/log/<modell>_overview.log`).

### 6.3.1 Kostenmetriken

Pro Tag und Team werden folgende Kostenkomponenten erfasst und zu einer Tagesgesamtkosten-Größe `total_cost_eur` summiert:

1. **Betriebskosten (`operational_cost_eur`)** = Lohnkosten + Kraftstoffkosten
   - **Lohnkosten** (`wage_cost_eur`): Arbeitsstunden × 35,00 €/h
     - An regulären Tagen wird ein voller Arbeitstag (480 Min) je aktivem Team angesetzt; am letzten Simulationstag werden die tatsächlich geleisteten Stunden (Fahrt- + Servicezeit) verwendet.
   - **Kraftstoffkosten** (`fuel_cost_eur`): gefahrene Distanz [km] × 0,30 €/km
2. **Ausfallkosten (`downtime_cost_eur`)**: monetarisierte Kosten der Nichtverfügbarkeit einer Ladestation während einer Störung
   - Type-1-Störung: Ausfallzeit [h] × Nennleistung [kW] × 0,50 €/kWh
   - Type-2-Störung: analog, jedoch über die längere Gesamtausfalldauer (inkl. Werkstattzeit)
   - Carryover-Störungen: Ausfallkosten werden ab 8:00 Uhr bis zur tatsächlichen Behebung berechnet
   - Unbehandelte Störungen: Ausfallkosten ab Meldezeitpunkt bis Tagesende

**Kostenparameter** (`src/models/cost_params.py`):

| Parameter | Wert |
|---|---|
| Stundenlohn | 35,00 €/h |
| Kraftstoffkosten | 0,30 €/km |
| Ausfallkosten | 0,50 €/kWh |

Auf Monte-Carlo-Ebene werden je Policy Mittelwert, Standardabweichung, Minimum und Maximum von `total_cost_eur`, `operational_cost_eur`, `wage_cost_eur`, `fuel_cost_eur` und `downtime_cost_eur` über alle Runs ausgewiesen.

### 6.3.2 Verfügbarkeitsmetriken

1. **`days_to_complete`**: Anzahl Tage, bis erstmals alle 397 Stationen mindestens einmal routinemäßig gewartet wurden (Maß für die Geschwindigkeit der Erstabdeckung des Stationsnetzes).
2. **`remaining_stations_at_end`**: Anzahl Stationen, die bei Erreichen von `max_days` noch nicht routinemäßig gewartet wurden (sollte im Normalfall 0 sein, sofern `max_days` ausreichend groß gewählt ist).
3. **`total_disruptions`**: Gesamtzahl generierter Störungsereignisse (Type 1 + Type 2) bis zum Abschlusstag.
4. **`same_day_handled`**: Anzahl der Störungen, die noch am Tag ihres Auftretens behoben wurden.
5. **`same_day_rate`** = `same_day_handled / total_disruptions`: Anteil der Störungen mit Behebung am selben Tag — zentrale Kennzahl für die Reaktionsfähigkeit/Verfügbarkeit des Wartungssystems.

### 6.3.3 Rückstandsmetriken (Carryover)

1. **`disruptions_carryover`** (pro Tag): Anzahl der am Tagesende noch unbehobenen Störungsaufgaben, die als Carryover-Tasks in den Folgetag übernommen werden und dort priorisiert eingeplant werden (insbesondere bei MyopicPlus, CFA und CFA-DB über `soft_deadline_min = 0` und Penalty proportional zur Leistung `power_kW`).
2. **`total_carryover`**: Summe der täglichen Carryover-Werte über die gesamte Simulationsdauer — Indikator für die Akkumulation von Rückständen und damit für die strukturelle Stabilität der Politik (eine dauerhaft steigende Carryover-Zahl deutet auf eine Überlastung des Systems hin).

### 6.3.4 Stochastische Routenmetriken

Ergänzend zu den oben genannten Kennzahlen werden je geplanter Route (sofern `stochastic_travel_times.enabled: true`) Robustheitsmetriken aus der internen 200-fachen Mini-Monte-Carlo-Simulation (`src/utils/stochastic_travel.py`) erfasst:

- `det_end_min` — deterministisch geplante Rückkehrzeit zum Depot,
- `mean_end_min`, `p95_end_min` — mittlere bzw. 95 %-Quantil-Rückkehrzeit unter Fahrzeitstreuung,
- `overtime_prob` — Wahrscheinlichkeit einer Überschreitung des Arbeitstagsendes (480 Min),
- `mean_overtime_min` — erwartete Überstundenminuten `E[max(0, Endzeit − 480)]`.

### 6.3.5 Beispiel: Aggregierte Ergebnisse (Myopic, 40 Runs)

Zur Illustration des Aggregationsformats (`logs/myopic/log/myopic_overview.log`):

| Kennzahl | Mittelwert | Std.-Abw. | Min | Max |
|---|---|---|---|---|
| Gesamtkosten [€] | 33.228,71 | 3.206,72 | 26.572 | 40.177 |
| Same-Day-Rate | 68,27 % | – | 58,28 % | 80,45 % |
| Gesamtcarryover | 48,02 | – | 26 | 70 |

Diese Tabellenstruktur (Mittelwert/Std.-Abw./Min/Max je Kennzahl) wird für alle sechs Policy-Tiers identisch erzeugt und bildet die Grundlage für die vergleichende Auswertung in den nachfolgenden Kapiteln.

---

## 6.4 Zusammenfassung der zentralen Experimentparameter

| Parameter | Wert | Quelle |
|---|---|---|
| Stationen | 397 (279 AC, 118 DC) | `data/raw/charging_stations_wue.csv` |
| Wartungsteams | 2 | `config.yaml: maintenance.n_teams` |
| Arbeitszeit | 8:00–16:00 Uhr (480 Min) | `config.yaml: maintenance.workday_*` |
| Zonen | 178 | `config.yaml: planning.n_zones` |
| Max. Stationen/Team/Tag | 20 | `config.yaml: planning.max_stations_per_team` |
| Simulationshorizont | bis zu 365 Tage | `--max-days` |
| Monte-Carlo-Runs | 30 (Standard, teils 40) | `--runs` |
| Seed-Schema | sequenziell 1…N | Run-Skripte |
| Störungsmodus | „stochastic" | `config.yaml: failure_simulation.mode` |
| p₁ (Type 1) | 0,00150 / Std. / Station | `config.yaml: failure_simulation.p1_per_hour` |
| p₂ (Type 2) | 0,00075 / Std. / Station | `config.yaml: failure_simulation.p2_per_hour` |
| Erholungsdauer | 365 Tage | `config.yaml: failure_simulation.recovery_days` |
| Anfangsfaktor | 0,1 | `config.yaml: failure_simulation.initial_factor` |
| Routine-Servicezeit | 45 Min | `config.yaml: maintenance.mean_service_time` |
| Stundenlohn | 35,00 €/h | `cost_params.py` |
| Kraftstoffkosten | 0,30 €/km | `cost_params.py` |
| Ausfallkosten | 0,50 €/kWh | `cost_params.py` |
| CV Fahrzeiten | 0,10–0,20 (stundenabhängig) | `config.yaml: stochastic_travel_times.cv_by_hour` |
| MC-Runs (Fahrzeit) | 200 | `config.yaml: stochastic_travel_times.n_mc_runs` |
| Feasibility-Schwelle | α = 0,95 | `config.yaml: stochastic_travel_times.feasibility_alpha` |
