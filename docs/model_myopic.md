# Myopic-Policy — Vollständige Modellbeschreibung (Greedy)

## Überblick

Die Myopic-Policy ist die einfachste der implementierten Policies. Sie trifft alle Planungsentscheidungen rein auf Basis der aktuell vorliegenden Information, ohne jegliche Zukunftsschätzung. „Myopic" bedeutet dabei: keine gelernten Parameter, keine Wertfunktion, kein Lookahead.

**Rolle im System:** Basislinie (Benchmark) für alle anderen Policies.

---

## Systemparameter & Kosten

### Kostenparameter (`CostParams`)

| Symbol | Bedeutung | Wert |
|---|---|---|
| $w$ | Stundenlohn pro Team | 35 €/h |
| $c_\text{km}$ | Kraftstoffkosten | 0,30 €/km |
| $c_\text{dt}$ | Ausfallkosten | 0,50 €/kWh |
| $s_1$ | Servicezeit Typ-1-Störung | 60 min |
| $s_2^\text{ab}$ | Demontage Typ-2-Störung | 30 min |
| $s_2^\text{depot}$ | Lagerhandling Typ-2 | 5 min |
| $s_2^\text{auf}$ | Montage Typ-2-Störung | 30 min |

### Arbeitstag

- Beginn: 08:00 (Minute 0 im internen Zeitstrahl)
- Ende: 16:00 (Minute 480)
- Arbeitstag in Minuten: $T = (16 - 8) \times 60 = 480$ min
- Mittagspause: konfigurierbar (`lunch_duration_min`, default 0 min); frühestens ab Minute 240 (12:00)

### Stationsindexierung

- Depot = Index 0 in allen Matrizen und Koordinatenlisten
- Stationen: $\text{node\_idx} = \text{station\_index} + 1$ (1-basiert)

---

## Fahrzeit- und Distanzmodell

### Stündliche Verkehrsmatrizen

Die Fahrzeit zwischen zwei Knoten $i, j$ ist tageszeit-abhängig:

$$\tau_{ij}(h) = \text{traffic\_matrices}[h][i, j] \quad \text{[Sekunden]}$$

Zur Umrechnung in Minuten: $\tau_{ij}^{\min}(h) = \tau_{ij}(h) / 60$.

Die passende Matrix für einen Zeitpunkt $t$ (Minuten ab 08:00) wird bestimmt über:

$$h = 8 + \lfloor t / 60 \rfloor, \quad h \in [\min H, \max H]$$

### Approximative Luftlinienentfernung

Für die Routing-Heuristik wird die Luftlinienentfernung $d_{ij}$ in km genutzt:

$$d_{ij} = \text{\_approx\_km}(\text{coords}[i], \text{coords}[j])$$

Diese Funktion berechnet die Euklidische Näherung über Lat/Lon-Differenzen (keine Haversine, da kleine Gebiete).

---

## Stationsauswahl (Tagesbeginn)

### Zonen-Clustering

Alle 397 Stationen sind via K-Means in $N_z = 178$ Zonen geclustert. Pro Zone werden vorberechnet:
- Zentroid (Mittelpunkt)
- Mittlere Depot-Distanz $\bar{d}_z$
- Konvexe-Hülle-Fläche $A_z$
- Mittlere Distanz zu allen anderen Zonenzentroids

### Zonenauswahl (Zonenpriorisierung)

Im Myopic-Modell wird je nach Konfiguration `zone_selection_mode` entweder der klassische Score oder ein wertfunktionsbasierter Score genutzt.

**Classic-Modus** (`zone_selection_mode: "classic"`):

$$\text{score}(z) = w_\text{depot} \cdot \hat{d}_z + w_\text{area} \cdot \hat{A}_z$$

mit normierten Größen $\hat{\cdot}$ (Min-Max-Normierung über alle offenen Zonen).

**Value-based-Modus** (`zone_selection_mode: "value_based"`):

$$\text{score}(z) = \sum_{k \in z,\, k \in \text{verbleibend}} \hat{V}(k, \text{dsm}_k)$$

wobei $\hat{V}$ die stationsindividuelle `_zone_value`-Funktion ist (siehe unten).

### `_zone_value(node_idx, dsm)`

Die heuristische Stationswertfunktion schätzt erwartete Ausfallkosten:

$$\hat{V}(k, t) = P_k \cdot \rho(t)$$

$$\rho(t) = f_0 + (1 - f_0) \cdot \frac{\min(t, T_r)}{T_r}$$

| Symbol | Bedeutung | Konfigurationsschlüssel | Wert |
|---|---|---|---|
| $P_k$ | Nennleistung Station $k$ [kW] | — | aus Stationsdaten |
| $f_0$ | Ausfallwahrscheinlichkeit direkt nach Wartung | `failure_simulation.initial_factor` | 0,1 |
| $T_r$ | Erholungszeit [Tage] | `failure_simulation.recovery_days` | 365 |
| $t$ | Tage seit letzter Wartung (dsm) | — | simulationsabhängig |

### Teamzuweisung & Expansion

1. **Top-$N$ Kandidatenzonen** nach Score ermitteln ($N = 20$).
2. **Startzone pro Team:** Team 0 erhält die nächste Zone zu seiner aktuellen Position; Team 1 erhält die nächste Zone mit Mindestabstand $\geq 0$ km zur Startzone von Team 0.
3. **Expansion:** Startzone vollständig laden; anschließend Nearest-Neighbor aus allen verfügbaren Stationen bis zur Kapazitätsgrenze.
4. **Kapazität:** $\leq 20$ Routine-Stops pro Team; begrenzt durch verbleibende Zeitbudget abzüglich Carryover-Servicezeit.
5. **Carryover-Verteilung:** Vortags-Störungen werden priorisiert dem Team mit geringster akkumulierter Carryover-Servicezeit zugeteilt (Tie-Breaker: geografische Nähe).

---

## Initialplan (Greedy Cheapest Insertion)

### Algorithmus

Der Tagesplan wird greedy aufgebaut — kein OR-Tools, kein globaler Optimierer.

```
Für jedes Team:
  1. Carryover-Tasks: Nearest-Neighbor (Reihenfolge nach kürzester Fahrzeit)
  2. Routine-Tasks: greedy nach route_score_fn
```

### `route_score_fn` (Myopic)

$$\text{score}(k, \text{dsm}_k, \text{cur}) = \frac{1}{\max(0.1,\; d_{\text{cur},k})}$$

Die Station mit dem **höchsten** Score wird als nächste ausgewählt. Das entspricht dem Nearest-Neighbor-Algorithmus: die nächste unbesuchte Station (kleinste Distanz = größter Score).

### Feasibility-Check

Vor jedem Stop wird geprüft, ob nach der Bedienung rechtzeitig zum Depot zurückgekehrt werden kann:

$$t_\text{abfahrt}(k) + \tau_{k,0}^{\min}(h) \leq T$$

Ist die Bedingung verletzt, wird $k$ nicht eingeplant (und verbleibt für den nächsten Tag).

### Mittagspause

Falls `lunch_duration_min > 0`: Überquert eine Fahrt den Pausenbeginn, wird die Pausendauer zur Ankunftszeit addiert.

$$t_\text{ankunft} = t_\text{abfahrt}^\text{eff} + \tau + \Delta_\text{pause}$$

$$\Delta_\text{pause} = \begin{cases} l_\text{dauer} & \text{wenn } t_\text{abfahrt}^\text{eff} < l_\text{start} < t_\text{abfahrt}^\text{eff} + \tau \\ 0 & \text{sonst} \end{cases}$$

---

## Störungs-Replan (Greedy Cheapest Insertion)

### Auslöser

Stündlich (08:00–15:00) werden neue Störungen gemeldet. Diese müssen in die laufenden Tagesrouten eingeplant werden.

### Phasen des Replans

#### Phase 1: Sortierung der Störungswarteschlange

Alle eingehenden Störungen werden nach ihren initialen Insertionskosten aufsteigend sortiert. Die günstigste Störung wird zuerst eingebaut.

#### Phase 2: Direkteinfügung (Cheapest Insertion)

Für jede Störung $d$ und jeden Team-Index $t$ wird jede Einfügeposition $p \in \{0, \ldots, |R_t|\}$ bewertet:

$$\text{pos}^* = \arg\min_{t, p} C_\text{insert}(d, t, p)$$

**Einfügekosten** (Myopic: `travel_time_only=True`):

$$C_\text{insert}(d, t, p) = \tau_{\text{prev}, d}^{\min} + \tau_{d, \text{next}}^{\min} - \tau_{\text{prev}, \text{next}}^{\min}$$

Das sind reine Extra-Fahrzeiten in Minuten (keine Wirtschaftskosten im Myopic-Replan).

**Feasibility-Check:**

$$t_\text{end,neu} = t_\text{letzter Stop,neu} + \tau_{\text{letzter},0}^{\min} \leq T$$

#### Phase 3: Drop & Insert (falls Phase 2 fehlschlägt)

Falls keine direkte Insertion möglich ist, werden Routine-Stops iterativ entfernt:

**Drop-Score (Myopic):**

$$\text{drop\_score}(k, \text{dsm}_k, r_h, \text{cur}, \delta) = d_{\text{cur}, k}$$

Der Routine-Stop mit dem **niedrigsten** Drop-Score (kleinste Distanz zum aktuellen Knoten = am ungünstigsten gelegen) wird zuerst gedroppt.

Vorgehen:
1. Alle Routine-Stops aufsteigend nach Drop-Score sortieren.
2. Iterativ droppen (1, 2, … Stops) bis eine feasible Insertion von $d$ möglich ist.
3. Gedropte Stops werden als Carryover auf den nächsten Tag verschoben.

#### Phase 4: Carryover

Falls auch nach maximalem Drop keine Insertion möglich ist, wird $d$ als Carryover eingetragen.

### Detour-Vorabberechnung

Für jeden Routine-Stop $k$ in der verbleibenden Route wird vorab geschätzt, wie viel Zeit das Entfernen von $k$ einspart:

$$\delta_k = \max\!\left(0,\; \frac{\tau_{\text{prev}(k), k} + \tau_{k, \text{next}(k)} - \tau_{\text{prev}(k), \text{next}(k)}}{60}\right) \quad [\text{min}]$$

Dies ist eine Approximation (exakt nur vor dem ersten Drop).

---

## Downtime-Kostenberechnung

Nach erfolgreicher Einfügung einer Störung $d$ werden die Ausfallkosten berechnet:

$$C_\text{dt}(d) = \max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right) \cdot P_d \cdot c_\text{dt}$$

| Symbol | Bedeutung |
|---|---|
| $t_\text{ankunft}(d)$ | Ankunftszeit des Teams bei Störung $d$ [min ab 08:00] |
| $t_\text{meldung}$ | Meldezeitpunkt $(h - 8) \times 60$ [min ab 08:00] |
| $P_d$ | Nennleistung der gestörten Station [kW] |
| $c_\text{dt}$ | Ausfallkostensatz [€/kWh] |

---

## Gesamtkostenstruktur

Die Gesamtkosten eines Tages setzen sich zusammen aus:

$$C_\text{gesamt} = C_\text{operativ} + C_\text{dt}$$

**Operative Kosten** (werden im Replan nicht explizit minimiert, entstehen implizit):

$$C_\text{operativ} = \underbrace{\frac{t_\text{Fahrzeit, gesamt}}{60} \cdot w}_{\text{Lohnkosten Fahren}} + \underbrace{\frac{t_\text{Service, gesamt}}{60} \cdot w}_{\text{Lohnkosten Service}} + \underbrace{d_\text{km, gesamt} \cdot c_\text{km}}_{\text{Kraftstoff}}$$

**Downtime-Kosten:**

$$C_\text{dt} = \sum_{d \in \text{Störungen}} \max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}(d)}{60}\right) \cdot P_d \cdot c_\text{dt}$$

---

## Algorithmus-Zusammenfassung

```
=== TAGESBEGINN ===
1. Zonenscoring → Startzonen je Team → Stationsauswahl (NN-Expansion)
2. Initialplan: Carryover (NN) → Routine (NN = Myopic route_score_fn)

=== STÜNDLICHE SCHLEIFE (08:00–15:00) ===
3. Störungen melden
4. Störungen nach Insertionskosten sortieren (günstigste zuerst)
5. Für jede Störung:
   a) Direkteinfügung: Cheapest Insertion (Δfahrzeit, travel_time_only)
   b) Falls nicht feasible: Drop (drop_score = dist(cur, k), niedrigster zuerst) + Insert
   c) Falls immer noch nicht feasible: Carryover

=== TAGESENDE ===
6. Gesamtkosten berechnen (operativ + downtime)
7. Carryovers in nächsten Tag übergeben
```

---

## Konfigurationsschlüssel (Auszug)

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `solver.use_or_tools` | `false` | Greedy-Routing aktiv |
| `maintenance.n_teams` | 2 | Anzahl Teams |
| `maintenance.workday_start_hour` | 8 | Arbeitsbeginn |
| `maintenance.workday_end_hour` | 16 | Arbeitsende |
| `maintenance.mean_service_time` | 45 min | Servicezeit je Station |
| `planning.n_zones` | 178 | Anzahl K-Means-Zonen |
| `planning.max_stations_per_team` | 20 | Kapazitätslimit |
| `planning.zone_selection_mode` | `value_based` | Zonenauswahlmodus |
| `planning.zone_expansion_mode` | `score_rank` | Expansionsmodus |
| `failure_simulation.recovery_days` | 365 | Erholungszeit [Tage] |
| `failure_simulation.initial_factor` | 0,1 | Ausfallrate direkt nach Wartung |
