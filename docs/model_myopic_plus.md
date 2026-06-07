# Myopic Plus — Vollständige Modellbeschreibung (Greedy)

## Überblick

Myopic Plus erweitert die Myopic-Policy um **leistungsgewichtete Priorisierung**. Stationen mit hoher Nennleistung werden bevorzugt früh bedient, da ihr Ausfall mehr Kosten verursacht. Es gibt keine gelernten Parameter — alle Prioritäten ergeben sich direkt aus den Stationseigenschaften (Nennleistung, Depot-Distanz, Ausfallwahrscheinlichkeit).

**Kernidee:** Ersetze den reinen Nearest-Neighbor durch einen wirtschaftlich motivierten Score, der Nennleistung $P_k$ und Distanz kombiniert.

---

## Systemparameter & Kosten

Identisch zur Myopic-Policy (siehe `model_myopic.md`). Zusätzlich:

| Symbol | Bedeutung | Konfigurationsschlüssel | Wert |
|---|---|---|---|
| $\alpha$ | Skalierungsfaktor für Penalty-Terme | `cfa.alpha` | 10,0 |
| $p_h$ | Gesamte Ausfallwahrscheinlichkeit pro Stunde | `p1_per_hour + p2_per_hour` | 0,00225 |
| $w_\text{min}$ | Stundenlohn in €/min | — | $w / 60$ |

### Ausfallwahrscheinlichkeit

$$p_h = p_\text{Typ1/h} + p_\text{Typ2/h} = 0{,}00150 + 0{,}00075 = 0{,}00225 \text{ pro Stunde}$$

---

## Stationsauswahl (Tagesbeginn)

Identisch zur Myopic-Policy: Zonenscoring, Teamzuweisung, Expansion.

Im Value-based-Modus verwendet Myopic Plus dieselbe `_zone_value`-Funktion wie Myopic:

$$\hat{V}(k, t) = P_k \cdot \rho(t), \quad \rho(t) = f_0 + (1 - f_0) \cdot \frac{\min(t, T_r)}{T_r}$$

Damit werden Zonen mit hoher kumulierter Ausfallrisiko-gewichteter Leistung bevorzugt.

---

## Initialplan (Greedy — Leistungsgewichteter Nearest-Neighbor)

### `route_score_fn` (Myopic Plus)

$$\text{score}(k, \text{dsm}_k, \text{cur}) = \frac{P_k}{\max(0.1,\; d_{\text{cur},k})}$$

Der Nenner verhindert Division durch null. Der Zähler priorisiert leistungsstarke Stationen. Die Station mit dem **höchsten** Score wird als nächste gewählt.

Im Vergleich zu Myopic ($\text{score} = 1 / d$): Myopic Plus bevorzugt hohe Nennleistung gegenüber geringer Distanz. Eine weit entfernte 300-kW-Station kann einen Score erhalten, der höher liegt als eine nahe 22-kW-Station.

### Carryover-Tasks

Identisch zu Myopic: Nearest-Neighbor (Reihenfolge nach Fahrzeit), mandatory.

Für Carryover-Tasks (Vortags-Störungen) gilt im Greedy-Pfad:

- `soft_deadline_min = 0` (frühestmöglich einplanen)
- kein expliziter Penalty im Greedy-Pfad (Sortierung durch Nearest-Neighbor)

### Feasibility-Check

Identisch zu Myopic:

$$t_\text{abfahrt}(k) + \tau_{k,0}^{\min}(h) \leq T$$

---

## Störungs-Replan (Greedy Cheapest Insertion mit wirtschaftlichem Drop-Score)

### Kostenfunktion der Insertion

Im Gegensatz zu Myopic verwendet Myopic Plus **vollständige wirtschaftliche Kosten** (`travel_time_only=False`):

$$C_\text{insert}(d, t, p) = \underbrace{\frac{\Delta\tau + s_d}{60} \cdot w}_{\text{Lohnkosten}} + \underbrace{\max(0, \Delta d_\text{km}) \cdot c_\text{km}}_{\text{Kraftstoff}} + \underbrace{\max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right) \cdot P_d \cdot c_\text{dt}}_{\text{Ausfallkosten}}$$

mit:

$$\Delta\tau = \tau_{\text{prev}, d}^{\min} + \tau_{d, \text{next}}^{\min} - \tau_{\text{prev}, \text{next}}^{\min} \quad \text{[Extra-Fahrzeit in min]}$$

$$\Delta d_\text{km} = d_{\text{prev}, d} + d_{d, \text{next}} - d_{\text{prev}, \text{next}} \quad \text{[Extra-Kilometer]}$$

$$s_d = \text{Servicezeit der Störung [min]}$$

### Drop-Score (Myopic Plus)

Wenn keine direkte Insertion möglich ist, werden Routine-Stops nach folgendem Score gedroppt:

$$\text{drop\_score}(k, \text{dsm}_k, r_h, \text{cur}, \delta) = \frac{P_k}{\max(0.1,\; d_{\text{cur},k})}$$

Der Routine-Stop mit dem **niedrigsten** Drop-Score wird zuerst gedroppt. Das bedeutet:

- Stationen mit geringer Nennleistung werden bevorzugt gedroppt
- Stationen, die weit vom aktuellen Teamknoten entfernt sind, werden bevorzugt gedroppt

Im Gegensatz zur Myopic-Policy (die nur die Distanz beachtet) schützt Myopic Plus also leistungsstarke Stationen stärker vor dem Drop.

### Drop-Algorithmus

Identisch zu Myopic (iteratives Droppen, niedrigster Score zuerst, bis Insertion feasible):

1. Routine-Stops aufsteigend nach `drop_score` sortieren.
2. Ersten Stop droppen, Route neu berechnen, Insertion versuchen.
3. Falls nicht feasible: zweiten Stop droppen, usw.
4. Falls nach allen möglichen Drops immer noch nicht feasible: Carryover.

---

## Detaillierte Formelübersicht

### Insertion-Kosten-Berechnung (vollständig)

Sei $p$ die Einfügeposition zwischen Stops $\text{prev}$ und $\text{next}$:

**Zeitgrößen:**

$$t_\text{ankunft}(d) = t_\text{abfahrt}(\text{prev}) + \tau_{\text{prev}, d}^{\min}$$

$$\Delta\tau = \tau_{\text{prev}, d}^{\min} + \tau_{d, \text{next}}^{\min} - \tau_{\text{prev}, \text{next}}^{\min}$$

$$t_\text{ende,neu} = t_\text{abfahrt}(\text{letzter Stop, neu}) + \tau_{\text{letzter}, 0}^{\min}$$

**Feasibility:**

$$t_\text{ende,neu} \leq T = 480 \text{ min}$$

**Kosten:**

$$C_\text{insert} = \underbrace{\frac{\Delta\tau + s_d}{60}}_{\Delta t_h} \cdot w + \underbrace{\max(0, \Delta d)}_{\Delta d_\text{km}} \cdot c_\text{km} + \underbrace{\max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right)}_{\text{Wartezeit}_h} \cdot P_d \cdot c_\text{dt}$$

---

## Downtime-Kostenberechnung

Identisch zu Myopic:

$$C_\text{dt}(d) = \max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right) \cdot P_d \cdot c_\text{dt}$$

---

## Unterschiede zu Myopic auf einen Blick

| Aspekt | Myopic | Myopic Plus |
|---|---|---|
| `route_score_fn` | $1/d$ | $P_k / d$ |
| Replan-Kostenfunktion | nur Δfahrzeit | vollständige Wirtschaftskosten |
| `drop_score_fn` | $d_{\text{cur},k}$ | $P_k / d_{\text{cur},k}$ |
| Gelernte Parameter | keine | keine |
| Zonenauswahl | `_zone_value` = $P_k \cdot \rho(t)$ | identisch |

---

## Algorithmus-Zusammenfassung

```
=== TAGESBEGINN ===
1. Zonenscoring → Startzonen je Team → Stationsauswahl (NN-Expansion)
2. Initialplan:
   - Carryover (NN nach Fahrzeit, mandatory)
   - Routine: greedy nach P_k / dist(cur, k)  ← Leistungsgewichteter NN

=== STÜNDLICHE SCHLEIFE (08:00–15:00) ===
3. Störungen melden
4. Störungen nach wirtschaftlichen Insertionskosten sortieren (günstigste zuerst)
5. Für jede Störung d:
   a) Direkteinfügung: Cheapest Insertion (vollständige Wirtschaftskosten)
   b) Falls nicht feasible:
      - Drop-Score = P_k / dist(cur, k)  [niedrigster Score = schwächste Station = zuerst droppen]
      - Iterativ droppen bis feasible
   c) Falls immer noch nicht feasible: Carryover

=== TAGESENDE ===
6. Gesamtkosten (operativ + downtime)
7. Carryovers weitergeben
```

---

## Konfigurationsschlüssel (Auszug)

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `solver.use_or_tools` | `false` | Greedy-Routing aktiv |
| `cfa.alpha` | 10,0 | Skalierungsfaktor (nur OR-Tools-Pfad relevant) |
| `failure_simulation.p1_per_hour` | 0,00150 | Typ-1-Ausfallrate |
| `failure_simulation.p2_per_hour` | 0,00075 | Typ-2-Ausfallrate |
| `failure_simulation.recovery_days` | 365 | Erholungszeit |
| `failure_simulation.initial_factor` | 0,1 | Ausfallrate direkt nach Wartung |
| `planning.zone_selection_mode` | `value_based` | Zonenauswahlmodus |
