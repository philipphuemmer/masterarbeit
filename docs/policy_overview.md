# Übersicht der fünf Wartungsplanungs-Policies

Dieses Dokument beschreibt die fünf implementierten Planungsstrategien für die
Wartungsoptimierung der 397 EV-Ladesäulen in Würzburg. Die Policies bauen
hierarchisch aufeinander auf und stellen unterschiedliche Kompromisse zwischen
Rechenaufwand, Planungsqualität und Zukunftsorientierung dar.

**Gemeinsame Rahmenbedingungen:**

| Parameter | Wert |
|---|---|
| Stationen | 397 |
| Teams | 2 |
| Arbeitstag | 08:00–16:00 (480 min) |
| Zonen (K-Means) | 40 (konfigurierbar) |
| Max. Stops/Team/Tag | 20 |
| Lohnkosten | 40 €/h |
| Kraftstoffkosten | 0,30 €/km |
| Ausfallkosten | 0,50 €/kWh |
| Servicezeit Typ-1-Störung | 60 min |
| Servicezeit Typ-2-Störung | 30 + Depot-Rundfahrt + 5 + 30 min |

---

## 1. Myopic

**Datei:** [src/models/myopic.py](../src/models/myopic.py)

### Konzept

Die Myopic-Policy ist die einfachste und schnellste Strategie. Sie trifft alle
Entscheidungen rein reaktiv – ohne Modell zukünftiger Kosten. Der Name
„myopic" (kurzsichtig) beschreibt das Kernprinzip: Jede Entscheidung minimiert
nur die unmittelbaren Zusatzkosten der aktuellen Stunde, ohne Langzeitfolgen
zu berücksichtigen.

Der **Initialplan** wird vollständig von OR-Tools erstellt (über `VRPSolver`).
Das eigentliche Alleinstellungsmerkmal der Myopic-Policy liegt im
**Störungshandling**: Bei jeder stündlichen Störungsmeldung wird greedy
per *Cheapest Insertion* entschieden.

### Initialplan

Delegiert unverändert an `VRPSolver.create_initial_plan()`. Es werden keine
stations- oder prioritätsspezifischen Modifikationen vorgenommen. OR-Tools
erhält alle Routine-Tasks als gleichwertige Pflichthalte.

### Störungshandling: Greedy Cheapest Insertion

**Schritt 1 – Sortierung der Störungswarteschlange:**  
Für jede eingehende Störung `d` wird zunächst die kostengünstigste Einfügeposition
über alle Teamrouten berechnet. Die Störungen werden aufsteigend nach diesen
Insertionskosten sortiert – die billigste Störung wird zuerst eingearbeitet.

**Schritt 2 – Direkte Einfügung:**  
Für Position `pos` in der Restroute werden die Zusatzkosten berechnet als:

```
extra_travel  = t(prev → d) + t(d → next) − t(prev → next)
extra_time_h  = (extra_travel + service_min) / 60
extra_km      = km(prev→d) + km(d→next) − km(prev→next)
downtime_h    = max(0, arrival_at_d − report_min) / 60

cost = extra_time_h × 40 €/h
     + max(0, extra_km) × 0,30 €/km
     + downtime_h × power_kW × 0,50 €/kWh
```

Die Machbarkeit (`feasible`) ist gegeben, wenn das Team nach der Einfügung
spätestens um 16:00 Uhr (480 min nach 08:00) am Depot ankommt.

**Schritt 3 – Drop-and-Insert (Fallback):**  
Falls keine direkte Einfügung machbar ist, wird versucht, Routine-Stops
zu „opfern". Der Fallback arbeitet **von hinten**: Die letzten Routine-Stops
werden iterativ entfernt (1, 2, 3, …), bis die Störung eingepasst werden kann.
Kriterium ist weiterhin minimale Gesamtkosten. Falls auch dieser Ansatz
scheitert, wird die Störung als **Carryover** auf den nächsten Tag verschoben.

### Stärken und Schwächen

| + | − |
|---|---|
| Einfach, schnell, deterministisch | Ignoriert zukünftige Ausfallwahrscheinlichkeiten |
| Kein Training erforderlich | Drop-Reihenfolge positionsbasiert, nicht wertbasiert |
| Explizit nachvollziehbare Entscheidungen | Keine globale Neuplanung bei Störungen |

---

## 2. MyopicPlus

**Datei:** [src/models/myopic_plus.py](../src/models/myopic_plus.py)

### Konzept

MyopicPlus erweitert Myopic in zwei Dimensionen:

1. **Initialplan**: OR-Tools erhält stationsindividuelle *Soft-Deadlines* mit
   leistungsgewichteten Strafkosten → Hochleistungs-Stationen werden bevorzugt
   früh besucht.
2. **Störungshandling**: Statt greedy Insertion wird der gesamte Resttag vollständig
   neu geplant (*OR-Tools Replan*). Störungsknoten erhalten Pflicht-Soft-Deadlines,
   Routine-Knoten können übersprungen werden (`AddDisjunction`).

### Soft-Deadline-Formel

Für den Initialplan werden alle Routine-Tasks nach **Depot-Distanz** aufsteigend
sortiert (nahe Stationen → frühe Einplanung). Die Deadline ist linear über den
Arbeitstag verteilt:

```
deadline(k) = rank(k) / n_tasks × 480 min
```

Der **Deadline-Penalty** (Kosten pro Minute Überschreitung) hängt von der
Nennleistung ab:

```
penalty_per_min = α × power_kW × p_h × downtime_eur_per_kwh / wage_per_min

mit:
  α           = 10,0  (Skalierungsfaktor aus config.yaml → cfa.alpha)
  p_h         = p1_per_hour + p2_per_hour ≈ 0,00112  (Gesamtausfallrate)
  downtime    = 0,50 €/kWh
  wage_per_min = 40/60 €/min
```

Der Penalty ist mindestens 1 (Integer, da OR-Tools nur ganzzahlige Kosten
akzeptiert).

### Skip-Penalty (Routine-Drops im Replan)

Damit OR-Tools Routine-Stops nur dann fallen lässt, wenn wirklich nötig, erhält
jede Routine-Station einen `AddDisjunction`-Penalty:

```
skip_penalty = service_time + max(0, α × power_kW × p_h × remaining_hours × downtime / wage_per_min)
```

`service_time = 45 min` als Basis verhindert, dass leistungsschwache Stationen
ohne echten Zeitdruck gedroppt werden.

### Störungshandling: OR-Tools Vollreplan

Beim Eingang von Störungen wird der gesamte Resttag neu gelöst:

- **Störungsknoten**: mandatory, `soft_deadline_min = time_min` (sofortige Bedienung
  angestrebt), `deadline_penalty ∝ power_kW`
- **Routine-Knoten**: optional via `AddDisjunction`, Skip-Penalty ∝ kW × remaining_hours

Falls der erste Solver-Aufruf INFEASIBLE liefert, werden iterativ die
schlechtesten Routine-Stops (von hinten) entfernt und ein Retry gestartet.

### Stärken und Schwächen

| + | − |
|---|---|
| Leistungsgewichtete Priorisierung im Initialplan | Skip-Penalty ist heuristisch, nicht gelernt |
| OR-Tools optimiert global beim Störungsreplan | Deadline-Sortierung ignoriert `days_since_maintenance` |
| Kein Training erforderlich | Drop-Reihenfolge beim Retry positionsbasiert |

---

## 3. CFA Light

**Datei:** [src/models/cfa_light.py](../src/models/cfa_light.py)

### Konzept

CFA Light ist eine **hybride Policy**: Der Initialplan wird identisch zu
MyopicPlus erstellt (OR-Tools mit Soft-Deadlines). Das Störungshandling ist
jedoch eine Mischung aus erlernter Wertschätzung (für die Drop-Entscheidung)
und schnellem Greedy-Routing (Cheapest Insertion statt OR-Tools Replan).

**Kernidee**: Wenn eine Störung nicht direkt einzuplanen ist, wird der zu
„opfernde" Routine-Stop nicht nach Position, sondern nach seinem **gelernten
V̂-Wert** ausgewählt. Stationen mit niedrigstem V̂ werden zuerst gedroppt.

### V̂-Approximation (Station-Level)

Der Stationswert wird als **Skip-Penalty** approximiert:

```
V̂(stop) = service_time + max(0, α × power_kW × p_h × remaining_hours × downtime / wage_per_min)
```

Dieser Wert steigt mit:
- **Nennleistung**: Hochleistungs-Stationen verursachen höhere Ausfallkosten
- **Verbleibende Zeit**: Je früher im Tag, desto höher der zeitgewichtete Wert

Der Wert ist technisch identisch mit dem `_skip_penalty` aus MyopicPlus, wird
hier aber explizit als **V̂-Approximation** interpretiert und zur Sortierung der
Drop-Kandidaten genutzt.

### Ablauf bei Störungen

1. **Direkte Insertion** (wie Myopic): Cheapest Insertion über alle Teams
2. **V̂-basierter Drop + Insertion** (Fallback):
   - Routine-Stops werden nach aufsteigendem V̂ sortiert (niedrigster Wert → erste Kandidaten)
   - Iterativ werden Stops mit niedrigstem V̂ entfernt (1, 2, 3, …)
   - Nach jedem Drop wird eine neue Cheapest-Insertion-Suche durchgeführt
   - Bei Erfolg: Einfügung an günstigster verbleibender Position
3. **Carryover**: Falls kein Drop hilft

### Unterschied zu Myopic

| Aspekt | Myopic | CFA Light |
|---|---|---|
| Drop-Auswahl | letzte Stops (positionsbasiert) | niedrigstes V̂ (wertbasiert) |
| Routing nach Drop | Cheapest Insertion | Cheapest Insertion |
| Initialplan | OR-Tools plain | OR-Tools + Soft-Deadlines |

### Stärken und Schwächen

| + | − |
|---|---|
| V̂-basiertes Droppen schützt wertvolle Stationen | V̂ ist Heuristik, kein gelernter Skalar θ |
| Schnell (kein OR-Tools Replan bei Störungen) | Kein globales Replanning → suboptimale Routen |
| Kein separates Training erforderlich | Approximation ignoriert `days_since_maintenance` |

---

## 4. CFA

**Datei:** [src/models/cfa.py](../src/models/cfa.py)  
**Training:** [scripts/train_cfa.py](../scripts/train_cfa.py)  
**Parameter:** `data/cfa/theta.json`

### Konzept

Die CFA-Policy (*Cost Function Approximation*) ist die erste Strategie mit
**gelernten** Parametern. Sie approximiert die Zukunftskosten durch eine
skalare Wertfunktion:

```
V̂(k) = θ × power_kW[k] × days_since_maintenance[k]
```

`θ` (in €/(kW·Tag)) ist der einzige gelernte Parameter und kodiert, wie stark
eine Kombination aus Nennleistung und Wartungsüberfälligkeit zukünftige Kosten
beeinflusst.

### Training

**Methode**: Offline Monte-Carlo-Regression auf Myopic-Rollouts.

1. N Simulationen (Standard: 20) mit `failure_simulation.mode = stochastic`
   und verschiedenen Seeds werden mit der Myopic-Policy durchgeführt.
2. Pro Tag `t` wird erfasst:
   - **Feature**: `Σ_{k ∈ remaining} power_kW[k] × dsm[k]` (Gesamtdringlichkeit)
   - **Target**: `G_t = Σ_{t' ≥ t} cost(t')` (tatsächliche Restkosten ab Tag t)
3. OLS-Regression:
   ```
   G_t ≈ θ × feature_t + intercept
   ```
4. `θ` wird in `data/cfa/theta.json` gespeichert. Nur `θ` wird in der Policy
   verwendet; `intercept` ist statistisches Artefakt.

### Initialplan: V̂-basierte Soft-Deadlines

Alle Routine-Tasks erhalten Soft-Deadlines nach **V̂-gewichteter Depot-Distanz**:

```
urgency[k]  = V̂(k) = θ × power_kW[k] × dsm[k]
score[k]    = urgency[k] / max(0.1, km(station[k], depot))
```

Stationen mit hohem `score` (dringend UND nah) erhalten frühere Deadlines.
Der Penalty pro Minute Überschreitung skaliert mit der absoluten Dringlichkeit:

```
deadline_penalty[k] = max(1, round(V̂(k) / wage_per_min))
```

### Störungshandling: OR-Tools Replan + V̂-Drop

Wie MyopicPlus wird der gesamte Resttag neu geplant:

1. Alle verbleibenden Stops + neue Störungen → OR-Tools
2. Falls INFEASIBLE: Routine-Stops werden nach **aufsteigendem V̂** sortiert
   (`power_kW × dsm`), und die niedrigsten werden iterativ gedroppt
3. Bei gelöstem Plan: neuen Plan in SimRoutes übernehmen

**Kritischer Unterschied zu MyopicPlus**: Die Drop-Reihenfolge basiert auf dem
gelernten V̂, nicht auf der Position in der Route. Stationen mit hohem `dsm`
und hoher Leistung werden bis zuletzt geschützt.

### Zonenauswahl (value_based_zone_selection)

Wenn `planning.value_based_zone_selection: true` in der Konfiguration gesetzt
ist, übergibt der Simulator `selector.value_fn = policy._value`. Der
`DailyZoneSelector` berechnet dann den Zonenwert als:

```
zone_score = w_value × Σ V̂(station) + w_depot × depot_dist
```

### Stärken und Schwächen

| + | − |
|---|---|
| Gelerntes θ: evidenzbasierte Wertschätzung | Eindimensionale Wertfunktion (nur kW × dsm) |
| Schützt hohe `dsm`-Stationen beim Drop | Kein Ausfallrisiko-Modell (keine Exponentialterme) |
| OR-Tools Globaloptimierung beim Replan | Training erfordert stochastischen Modus |

---

## 5. VFA

**Datei:** [src/models/vfa.py](../src/models/vfa.py)  
**Training:** [scripts/train_vfa.py](../scripts/train_vfa.py)  
**Parameter:** `data/vfa/theta.json`

### Konzept

Die VFA-Policy (*Value Function Approximation*) ist die ausgereifteste Strategie.
Sie approximiert den **globalen Systemzustand** durch einen 6-dimensionalen
Feature-Vektor und lernt einen entsprechenden Gewichtsvektor:

```
V̂(s) = θᵀ φ(s) + intercept
```

Statt wie CFA eine stationsindividuelle Wertfunktion zu verwenden, bewertet VFA
den **gesamten Restzustand** – inklusive Flottenlast, Überfälligkeitsverteilung
und offener Störungsrückstände.

### Feature-Vektor φ(s)

| Index | Name | Formel | Beschreibung |
|---|---|---|---|
| f0 | `total_urgency` | `Σ_k power_kW[k] × dsm[k]` | Gesamtdringlichkeit (wie CFA) |
| f1 | `expected_damage` | `Σ_k (1 − e^{−λ·dsm[k]}) × power_kW[k]` | Erwarteter Schadenwert |
| f2 | `mean_dsm` | `mean(dsm[k])` | Mittlere Wartungsüberfälligkeit |
| f3 | `max_urgency` | `max(power_kW[k] × dsm[k])` | Größte Einzeldringlichkeit |
| f4 | `frac_remaining` | `n_remaining / n_stations` | Auslastungsgrad (normiert) |
| f5 | `n_carryover` | Anzahl offener Störungsrückstände | Störungslast |

Die Ausfallrate `λ = (p1_per_hour + p2_per_hour) × 24` ist die Tagesrate.
f1 modelliert explizit nicht-lineares Ausfallrisiko (Sättigungseffekt).

### Training

Analog zu CFA, aber mit 6-dimensionalem Feature-Vektor:

1. N Myopic-Simulationen mit verschiedenen Seeds
2. Pro Tag: φ(s) vor der Tagesplanung + `G_t = Σ_{t' ≥ t} cost(t')`
3. OLS-Regression:
   ```
   G_t ≈ θᵀ φ(s_t) + intercept
   ```
4. θ-Vektor und intercept in `data/vfa/theta.json` gespeichert

### ΔV̂ als Extra-Kosten für OR-Tools

Die zentrale Idee der VFA ist die **marginale Wertberechnung**:

```
ΔV̂(k) = V̂(s) − V̂(s \ {k})
```

`ΔV̂(k)` misst, wie viel der Gesamtzustandswert sinkt, wenn Station `k` aus der
geplanten Menge entfernt wird. Hohe `ΔV̂` bedeutet: Besuch dieser Station spart
viele zukünftige Kosten.

Da OR-Tools nur nicht-negative Arc-Kosten akzeptiert, werden die ΔV̂-Werte
umgekehrt und auf 0 verschoben:

```
extra_cost[k] = max_bonus − ΔV̂(k)   (in Minuten-Einheiten)
max_bonus     = max_k(ΔV̂(k))
```

Stationen mit hohem ΔV̂ erhalten `extra_cost = 0` (kein Zuschlag), während
Stationen mit niedrigem ΔV̂ einen Kostenzuschlag bekommen, der sie im
OR-Tools-Plan nach hinten schiebt.

**Wichtig**: Für ΔV̂ werden nur die stationsindividuellen Features f0–f3
verwendet (via `_station_feature_mask`). f4 (`frac_remaining`) und f5
(`n_carryover`) ändern sich bei Entnahme einer einzelnen Station für alle
identisch und liefern kein Differenzierungssignal.

### Initialplan

```python
extra_costs = self._compute_extra_costs(tasks)
return self.solver.create_initial_plan(tasks, extra_costs=extra_costs)
```

OR-Tools bekommt keine Soft-Deadlines (im Unterschied zu CFA), sondern
stationsspezifische Kostenzuschläge, die die ΔV̂-basierte Präferenzordnung
kodieren.

### Störungshandling

Wie CFA: vollständiger OR-Tools Replan mit neu berechneten `extra_costs`.
Im Infeasibility-Fallback werden Routine-Stops nach aufsteigendem ΔV̂ gedroppt:

```python
def delta_v(task):
    phi_prime = self._phi(all_tasks, exclude_node=task.node_idx)
    return v_s - self._value(phi_prime)

routine_tasks.sort(key=delta_v)  # niedrigstes ΔV̂ zuerst
```

### Stationsindividuelle Näherung für Zonenauswahl

Für `value_based_zone_selection` ist eine stations-lokale Approximation von ΔV̂
nötig, da der globale Zustand beim Zonenscoring nicht vollständig bekannt ist.
`_station_value()` approximiert mit nur f0 und f1:

```python
def _station_value(node_idx, dsm):
    power = node_to_power[node_idx]
    urgency = power * dsm
    failure_risk = 1 − exp(−λ_day × dsm)
    return θ[0] × urgency + θ[1] × failure_risk × power
```

### Stärken und Schwächen

| + | − |
|---|---|
| Modelliert globalen Systemzustand | 6 Features: höhere Overfitting-Gefahr |
| Explizites Ausfallrisiko-Modell (f1) | ΔV̂-Berechnung ist O(n) pro Task |
| ΔV̂ differenziert marginal, nicht absolut | Training erfordert stochastischen Modus |
| Schützt kritische Stationen beim Drop | OR-Tools reagiert nicht immer sensitiv auf extra_costs |

---

## Vergleich aller fünf Policies

| Merkmal | Myopic | MyopicPlus | CFA Light | CFA | VFA |
|---|---|---|---|---|---|
| **Initialplan** | OR-Tools plain | OR-Tools + Soft-Deadlines | OR-Tools + Soft-Deadlines | OR-Tools + V̂-Deadlines | OR-Tools + ΔV̂-Kosten |
| **Störungshandling** | Cheapest Insertion | OR-Tools Replan | Cheapest Insertion | OR-Tools Replan | OR-Tools Replan |
| **Drop-Kriterium** | letzte Position | letzter Stop (Retry) | niedrigstes V̂ | niedrigstes V̂(θ × kW × dsm) | niedrigstes ΔV̂ |
| **Gelernte Parameter** | – | – | – | θ ∈ ℝ (1 Skalar) | θ ∈ ℝ⁶ (Vektor) |
| **Zustandsdarstellung** | keine | keine | stationslokal (heuristisch) | stationslokal (kW × dsm) | global (6 Features) |
| **Ausfallrisiko-Modell** | nein | nein | nein | nein | ja (exp-Sättigung) |
| **Training erforderlich** | nein | nein | nein | ja | ja |
| **Trainings-Modus** | – | – | – | stochastic | stochastic |
| **Zonenauswahl (value-based)** | nein | nein | nein | `_value(k)` | `_station_value(k)` |

### Hierarchie der Entscheidungsqualität

```
Myopic
  └─ + leistungsgewichtete Reihenfolge + OR-Tools Replan
        → MyopicPlus
             └─ + V̂-basiertes Droppen (ohne Replan)
                   → CFA Light
                        └─ + gelerntes θ, dsm-Bewusstsein, OR-Tools Replan
                              → CFA
                                   └─ + globaler Zustandsvektor, ΔV̂-Differenzierung,
                                         Ausfallrisiko-Modell
                                              → VFA
```

---

## Kostenmodell (gemeinsam für alle Policies)

### Operative Kosten

```
C_op = Fahrzeit_h × 40 €/h + Fahrstrecke_km × 0,30 €/km
```

### Ausfallkosten (Downtime)

```
C_down = Wartezeit_h × Nennleistung_kW × 0,50 €/kWh
```

Dabei ist `Wartezeit_h = max(0, Ankunftszeit − Meldezeit)` in Stunden.

### Servicezeiten

- **Typ-1-Störung** (Vor-Ort-Reparatur): 60 min
- **Typ-2-Störung** (Austausch mit Depot-Fahrt):
  30 min Demontage + Depot-Rundfahrt + 5 min Lagerhandling + 30 min Montage

---

## Trainingsworkflow (CFA und VFA)

Beide lernenden Policies verwenden denselben Trainingsansatz:

```
1. Konfiguration: failure_simulation.mode = stochastic  (Pflicht)
2. N Myopic-Simulationen (Standard: 20 Läufe, je bis zu 365 Tage)
3. Pro Tag: Feature(s) vor Planung + G_t (Restkosten ab diesem Tag)
4. OLS-Regression: G_t ≈ θᵀ φ(s) + intercept
5. θ gespeichert in data/cfa/theta.json bzw. data/vfa/theta.json
```

**Wichtig**: Das Training läuft auf Myopic-Rollouts. Die gelernten θ-Werte
modellieren also die **Kostenstruktur der Myopic-Policy**, nicht die Optimalkosten.
CFA und VFA nutzen dieses Wissen, um bessere Entscheidungen zu treffen, als
Myopic es selbst täte – nach dem Prinzip der *approximate dynamic programming*.
