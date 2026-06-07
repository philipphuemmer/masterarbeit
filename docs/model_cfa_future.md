# CFA Future — Vollständige Modellbeschreibung (Greedy)

## Überblick

CFA Future (Cost Function Approximation Future) approximiert die Kosten des **Weglassens** einer einzelnen Station $k$ durch einen parametrischen Funktionsapproximator:

$$\tilde{C}(\text{drop } k) \approx \theta^\top \phi(k)$$

Dieser Wert $\tilde{C}(k)$ — kurz: der „Drop-Score" der Station — steuert alle Planungsentscheidungen: Initialplan-Reihenfolge und Störungs-Replan-Drops. Je höher $\tilde{C}(k)$, desto teurer ist es, $k$ auszulassen, desto früher soll $k$ bedient werden.

**Unterschied zur CFA (train_cfa.py):** CFA lernt $\theta$ über OLS auf Myopic-Rollout-Zustandswerte. CFA Future lernt $\theta$ **kontrastiv**: Labels entstehen aus gepaarten Suffix-Simulationen, die direkt den Kostenunterschied zwischen Bedienen und Weglassen einer Station messen.

---

## Feature-Vektor $\phi(k)$

### Definition

$$\phi(k) = \begin{bmatrix} P_k \\ a_k \\ \rho(t_k) \\ \bar{d}_k \end{bmatrix} \in \mathbb{R}^4$$

| Index | Feature | Symbol | Bedeutung |
|---|---|---|---|
| 0 | Nennleistung | $P_k$ | [kW], aus Stationsdaten |
| 1 | Stationsalter | $a_k$ | [Jahre] ab Inbetriebnahme bis 01.01.2026; Fallback: 5 Jahre |
| 2 | Erholungskurve | $\rho(t_k)$ | aktueller Ausfallrisiko-Faktor (dimensionslos, $\in [f_0, 1]$) |
| 3 | Mittlere Distanz | $\bar{d}_k$ | mittlere Luftlinienentfernung zu allen anderen Stationen [km] |

### Erholungskurve $\rho(t)$

$$\rho(t) = f_0 + (1 - f_0) \cdot \frac{\min(t, T_r)}{T_r}$$

| Symbol | Konfigurationsschlüssel | Wert |
|---|---|---|
| $f_0$ | `failure_simulation.initial_factor` | 0,1 |
| $T_r$ | `failure_simulation.recovery_days` | 365 |
| $t$ | — | Tage seit letzter Wartung (dsm) |

$\rho(0) = f_0 = 0{,}1$ (direkt nach Wartung), $\rho(T_r) = 1{,}0$ (vollständig erholt).

### Feature-Standardisierung

Während der Nutzung (Inference) werden die Features z-standardisiert mit Trainings-Statistiken:

$$\phi_\text{scaled}(k) = \frac{\phi(k) - \mu}{\max(\sigma, 10^{-8})}$$

$\mu, \sigma \in \mathbb{R}^4$ werden aus `data/training/cfa_future/theta.json` geladen (Felder `feature_means`, `feature_stds`).

---

## Wertfunktion $\tilde{C}(k)$

$$\tilde{C}(k, t) = \theta^\top \phi_\text{scaled}(k, t)$$

- $\theta \in \mathbb{R}^4$: gelernter Gewichtsvektor
- Einheit: interpretiert als approximierte Zusatzkosten bei Weglassen von $k$ [€]
- Hoher $\tilde{C}(k)$: Weglassen von $k$ ist teuer → $k$ priorisieren
- Negativer $\tilde{C}(k)$: Weglassen von $k$ ist unbedenklich oder günstig

---

## Training von $\theta$ (Kontrastive Suffix-Simulation)

### Ziel

$$\theta^* = \arg\min_\theta \sum_{(k, t)} \left( \theta^\top \phi_\text{scaled}(k, t) - y_{k,t} \right)^2 + \lambda \|\theta\|^2$$

mit Ridge-Regularisierung $\lambda = 0{,}1$.

### Labels

Das Label $y_{k,t}$ misst den Kostenunterschied zwischen zwei Zukunftspfaden ab Tag $t$, in denen Station $k$ einmal bedient (serve) und einmal nicht bedient (drop) wird:

$$y_{k,t} = \underbrace{\sum_{h=1}^{H} \gamma^h \cdot c_h^{\text{drop},k}}_{\text{Gesamtkosten ohne } k} - \underbrace{\sum_{h=1}^{H} \gamma^h \cdot c_h^{\text{serve},k}}_{\text{Gesamtkosten mit } k}$$

| Symbol | Bedeutung | Wert |
|---|---|---|
| $H$ | Suffix-Horizont [Tage] | 30 |
| $\gamma$ | Diskontierungsfaktor pro Tag | 0,995 |
| $c_h^{\text{drop/serve}}$ | Gesamtkosten (operativ + downtime) an Tag $h$ | simuliert |

### Common Random Numbers (CRN)

Für alle Stationen $j \neq k$ werden dieselben Störungsereignisse in beiden Pfaden verwendet. Nur für Station $k$ selbst entwickelt sich der Störungspfad endogen (da ihr `dsm`-Wert pfadabhängig ist):

- **serve-Pfad:** $k$ wird an Tag $t$ gewartet → $\text{dsm}[k] = 0$ ab Tag $t+1$
- **drop-Pfad:** $k$ wird nicht gewartet → $\text{dsm}[k]$ steigt weiter

### Stratifiziertes Sampling

Pro Trainings-Run werden alle $N_s = 5$ Tage ein Snapshot gezogen. An jedem Snapshot-Tag werden $K = 3$ Stationen stratifiziert ausgewählt (je eine aus hohem, mittlerem und niedrigem Prioritätsbereich).

### Replikationen zur Varianzreduktion

Jeder Trainingspunkt (Station $k$ an Tag $t$) wird $R = 3$ mal mit unterschiedlichen Zufalls-Seeds repliziert. Der Label-Mittelwert wird für OLS verwendet.

### Iteratives Policy Improvement

```
Runde 1: Myopic-Policy als Bootstrap (initiale Rollouts)
Runde r: CFA-Future(θ_{r-1})-Policy
```

Jede Trainingsrunde liefert ein neues $\theta$, das die Folgepolitik für Runde $r+1$ bestimmt.

### Ausgabe

`data/training/cfa_future/theta.json`:

```json
{
  "theta": [θ_0, θ_1, θ_2, θ_3],
  "feature_means": [μ_0, μ_1, μ_2, μ_3],
  "feature_stds":  [σ_0, σ_1, σ_2, σ_3],
  "r2": 0.xxxx,
  "n_runs": N
}
```

---

## Stationsauswahl (Tagesbeginn)

Im Value-based-Modus wird `_value(node_idx, dsm) = ̃C(k, dsm)` als stationsindividueller Zonenwert verwendet:

$$\text{score}(z) = \sum_{k \in z,\, k \in \text{verbleibend}} \tilde{C}(k, \text{dsm}_k)$$

Zonen mit höherem kumuliertem Drop-Score werden bevorzugt zuerst bedient.

---

## Initialplan (Greedy — $\tilde{C}$/Distanz-Priorisierung)

### `route_score_fn` (CFA Future)

$$\text{score}(k, \text{dsm}_k, \text{cur}) = \frac{\tilde{C}(k, \text{dsm}_k) + s}{\max(0.1,\; d_{\text{cur},k})}$$

Dabei ist $s$ ein Shift-Term, der verhindert, dass negative $\tilde{C}$-Werte den Score invertieren:

$$s = \max\!\left(0,\; -\min_{k \in \text{Tasks}} \tilde{C}(k, \text{dsm}_k)\right) + 1.0$$

Dadurch ist $\tilde{C}(k) + s \geq 1.0$ für alle $k$. Der Nearest-Neighbor-Charakter (höherer Score für näher liegende Stationen) bleibt erhalten, wird aber durch das Verhältnis zu $\tilde{C}$ moduliert.

**Interpretation:** Eine Station mit hohem $\tilde{C}$ (teures Weglassen) und geringer Distanz erhält den höchsten Score und wird zuerst gewählt. Eine Station mit niedrigem $\tilde{C}$ und großer Distanz wird spät oder gar nicht bedient.

### Carryover-Tasks

Identisch zu Myopic/Myopic Plus: Nearest-Neighbor (Reihenfolge nach Fahrzeit), mandatory, immer zuerst.

### Feasibility-Check

Identisch zu Myopic:

$$t_\text{abfahrt}(k) + \tau_{k,0}^{\min}(h) \leq T$$

---

## Störungs-Replan (Greedy Cheapest Insertion mit $\tilde{C}$-Drop-Score)

### Kostenfunktion der Insertion

Identisch zu Myopic Plus — vollständige wirtschaftliche Kosten (`travel_time_only=False`):

$$C_\text{insert}(d, t, p) = \frac{\Delta\tau + s_d}{60} \cdot w + \max(0, \Delta d_\text{km}) \cdot c_\text{km} + \max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right) \cdot P_d \cdot c_\text{dt}$$

### Drop-Score (CFA Future)

$$\text{drop\_score}(k, \text{dsm}_k, r_h, \text{cur}, \delta_k) = \tilde{C}(k, \text{dsm}_k) - w_\text{min} \cdot \delta_k$$

| Term | Bedeutung |
|---|---|
| $\tilde{C}(k, \text{dsm}_k)$ | approximierter Zukunftswert von $k$ (Kosten des Weglassens) |
| $w_\text{min} = w / 60$ | Stundenlohn in €/min |
| $\delta_k$ | Detour-Zeit [min]: Zeit die das Entfernen von $k$ aus der Route einspart |

Der Subtraktionsterm $- w_\text{min} \cdot \delta_k$ berücksichtigt, dass das Droppen einer Route mit großem Umweg einen zeitlichen Gewinn bringt, der den Wert von $k$ aus Sicht des Tagesplans reduziert. Stationen, deren Entfernung viel Zeit einspart und die ohnehin niedrigen $\tilde{C}$-Wert haben, werden bevorzugt gedroppt.

**Niedrigster Drop-Score = zuerst droppen.**

### Drop-Algorithmus (identisch zu Myopic/Myopic Plus)

1. Routine-Stops aufsteigend nach `drop_score` sortieren.
2. Ersten Stop droppen, Route neu berechnen, Insertion versuchen.
3. Falls nicht feasible: nächsten Stop droppen, usw.
4. Falls nach allen möglichen Drops nicht feasible: Carryover.

---

## Stationsindividuelle Vorberechnungen

Beim Modellinitialisierung werden drei stationsindividuelle Wörterbücher befüllt:

### `_node_to_failure_factor`

Stationsindividuelle Ausfallrate-Skalierung aus `get_failure_rate_factors(stations_df)`:

$$\lambda_k = \text{failure\_rate\_factor}[k]$$

Wird aktuell im Feature-Vektor nicht direkt als Feature verwendet, aber für den Failure-Simulator genutzt.

### `_node_to_age`

Alter der Station $k$ in Jahren (Referenz: 01.01.2026):

$$a_k = \frac{(t_\text{ref} - t_\text{IBN}[k]).\text{days}}{365{,}25}$$

Fallback: 5 Jahre falls `Inbetriebnahmedatum` fehlt.

### `_node_to_mean_dist`

Mittlere Luftlinienentfernung zu allen anderen Stationen:

$$\bar{d}_k = \frac{1}{n-1} \sum_{j \neq k,\; j \geq 1} d_{k,j} \quad [\text{km}]$$

---

## Detour-Vorabberechnung

Für jeden Routine-Stop $k$ in der verbleibenden Route:

$$\delta_k = \max\!\left(0,\; \frac{\tau_{\text{prev}(k), k} + \tau_{k, \text{next}(k)} - \tau_{\text{prev}(k), \text{next}(k)}}{60}\right) \quad [\text{min}]$$

Approximation (exakt nur vor dem ersten Drop; Nachbarn verändern sich danach).

---

## Downtime-Kostenberechnung

Identisch zu Myopic/Myopic Plus:

$$C_\text{dt}(d) = \max\!\left(0,\; \frac{t_\text{ankunft}(d) - t_\text{meldung}}{60}\right) \cdot P_d \cdot c_\text{dt}$$

---

## Vergleich der drei Policies

| Aspekt | Myopic | Myopic Plus | CFA Future |
|---|---|---|---|
| `route_score_fn` | $1/d$ | $P_k/d$ | $(\tilde{C}(k)+s)/d$ |
| Replan-Kosten | Δfahrzeit | vollst. Wirtschaft | vollst. Wirtschaft |
| `drop_score_fn` | $d_{\text{cur},k}$ | $P_k/d_{\text{cur},k}$ | $\tilde{C}(k) - w_\text{min}\cdot\delta_k$ |
| Gelernte Parameter | keine | keine | $\theta \in \mathbb{R}^4$ |
| Feature-Vektor | — | — | $[P_k, a_k, \rho(t), \bar{d}_k]$ |
| Trainingsmethode | — | — | OLS (Ridge) auf kontrastive Labels |
| Trainings-Label | — | — | $y_{k,t} = \text{cost\_drop} - \text{cost\_serve}$ |

---

## Algorithmus-Zusammenfassung

```
=== TRAINING (einmalig, vor dem Einsatz) ===
1. Myopic-Bootstrap-Rollout: N Runs à max 200 Tage
2. Alle N_s=5 Tage: K=3 Stationen stratifiziert sampeln
3. Pro Snapshot (k, t): R=3 Suffix-Paare simulieren (H=30, γ=0.995)
   - serve-Pfad: k jetzt warten → dsm[k]=0 ab Tag t+1
   - drop-Pfad:  k nicht warten → dsm[k] steigt weiter
   - Kosten CRN-korreliert (j≠k identische Störungen)
4. Label y_{k,t} = mean_r(cost_drop_r - cost_serve_r)
5. Ridge-OLS: θ = (Φ^T Φ + λI)^{-1} Φ^T y
6. Speichern: theta.json mit θ, μ, σ, R²
7. Policy Iteration: Wiederhole mit CFA-Future(θ)-Policy

=== TAGESBEGINN ===
8. Zonenscoring via sum(C̃(k)) → Startzonen → Expansion
9. Initialplan:
   - Carryover (NN nach Fahrzeit, mandatory)
   - Routine: greedy nach (C̃(k) + shift) / dist(cur, k)

=== STÜNDLICHE SCHLEIFE (08:00–15:00) ===
10. Störungen melden
11. Sortieren nach wirtschaftlichen Insertionskosten
12. Für jede Störung d:
    a) Direkteinfügung: Cheapest Insertion (vollst. Wirtschaftskosten)
    b) Falls nicht feasible:
       - Drop-Score = C̃(k) - w_min × δ_k  [niedrigster Score → zuerst droppen]
       - Iterativ droppen bis feasible
    c) Falls immer noch nicht feasible: Carryover

=== TAGESENDE ===
13. Gesamtkosten (operativ + downtime)
14. Carryovers weitergeben
```

---

## Konfigurationsschlüssel (Auszug)

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `solver.use_or_tools` | `false` | Greedy-Routing aktiv |
| `failure_simulation.recovery_days` | 365 | $T_r$ |
| `failure_simulation.initial_factor` | 0,1 | $f_0$ |
| `failure_simulation.p1_per_hour` | 0,00150 | Typ-1-Rate |
| `failure_simulation.p2_per_hour` | 0,00075 | Typ-2-Rate |
| `planning.zone_selection_mode` | `value_based` | Zonenauswahl via $\tilde{C}$ |
| `planning.zone_expansion_mode` | `score_rank` | Expansion nach Score-Rang |

## Modell-Dateipfade

| Datei | Inhalt |
|---|---|
| `src/models/cfa_future.py` | Modell-Klasse `CFAFutureModel` |
| `scripts/train/train_cfa_future.py` | Trainings-Skript |
| `data/training/cfa_future/theta.json` | Gelernter Gewichtsvektor + Statistiken |
