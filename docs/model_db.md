# DB — Vollständige Modellbeschreibung (Greedy)

## Überblick

DB baut vollständig auf **CFA Future** auf und erweitert es um einen einzigen, zustandsabhängigen Steuerparameter $\delta \in (0, 1)$. Dieser Parameter moduliert, wie stark Fahrdistanz gegenüber dem approximierten Stationswert $\tilde{C}(k)$ gewichtet wird.

**Kernidee:** CFA Future verwendet immer `dist^1` als Nenner im Routing-Score. DB ersetzt diesen durch `dist^(2δ)`:

- $\delta = 0{,}5$: `dist^1` → exakt identisches Verhalten wie CFA Future
- $\delta < 0{,}5$: Distanz wird schwächer bestraft → $\tilde{C}(k)$ dominiert stärker
- $\delta > 0{,}5$: Distanz wird stärker bestraft → Routing-Effizienz dominiert stärker

$\delta$ wird am **Tagesbeginn** aus dem aktuellen Systemzustand bestimmt und bleibt für den gesamten Tag konstant.

**Verhältnis zu CFA Future:** DB ist eine direkte Subklasse von `CFAFutureModel`. Alle θ-Lade-Logik, Feature-Berechnung `_phi()`, Wertfunktion `_value()` und Zonenauswahl sind 1:1 geerbt. Nur die Greedy-Routingfunktionen `create_initial_plan()` und `handle_disruptions()` werden überschrieben.

---

## Ererbte Komponenten aus CFA Future

Vollständig identisch übernommen (keine Änderungen):

- **Feature-Vektor** $\phi(k) = [P_k,\; a_k,\; \rho(t_k),\; \bar{d}_k] \in \mathbb{R}^4$
- **Wertfunktion** $\tilde{C}(k, t) = \theta^\top \phi_\text{scaled}(k, t)$
- **θ-Laden** aus `data/training/cfa_future/theta.json`
- **Z-Standardisierung** mit $\mu, \sigma$ aus demselben File
- **Zonenauswahl** via $\text{score}(z) = \sum_{k \in z} \tilde{C}(k, \text{dsm}_k)$
- **Carryover-Handling** (Nearest-Neighbor, mandatory)
- **Störungs-Insertionskosten** (vollständige Wirtschaftskosten)
- **Downtime-Kosten** $C_\text{dt}(d)$

Für die Definitionen dieser Komponenten siehe [model_cfa_future.md](model_cfa_future.md).

---

## Der Parameter $\delta$

### Wertebereiche

$\delta$ wird aus einem diskreten Gitter gezogen:

$$\delta \in \{0{,}1,\; 0{,}3,\; 0{,}5,\; 0{,}7,\; 0{,}9\}$$

### Drei Betriebsmodi (`delta_mode`)

#### Modus `"rule"` (aktuell aktiv)

Regelbasierte Stufenfunktion auf Basis der **Anzahl noch offener Stationen** $n_r$:

| $n_r$ (verbleibende Stationen) | $\delta$ |
|---|---|
| $n_r \geq 370$ | 0,5 |
| $330 \leq n_r < 370$ | 0,5 |
| $180 \leq n_r < 330$ | 0,5 |
| $130 \leq n_r < 180$ | 0,5 |
| $n_r < 130$ | 0,9 |

In der aktuellen Konfiguration schaltet das Modell also erst im **Endspiel** (< 130 verbleibende Stationen, ca. Tag 38+) auf $\delta = 0{,}9$ um, was Routing-Effizienz stark priorisiert. Für den Rest der Simulation gilt $\delta = 0{,}5$ (= CFA Future).

#### Modus `"rf"` (trainierbar)

Ein `RandomForestClassifier` sagt $\delta^*$ aus einem 2-dimensionalen Zustandsvektor vorher. Fallback: `default_delta = 0.5`.

#### Modus `"fixed"` (Konstante)

$\delta = \text{default\_delta}$ für alle Tage (kein Zustandsbezug).

---

## Zustandsvektor für $\delta$-Schätzung

Obwohl im Code-Docstring 3 Features erwähnt werden, berechnet `extract_balance_features()` tatsächlich **2 Features**:

$$\psi(S_t) = \begin{bmatrix} f_0 \\ f_1 \end{bmatrix} \in \mathbb{R}^2$$

| Index | Feature | Formel | Wertebereich |
|---|---|---|---|
| 0 | Tagesfortschritt | $f_0 = \frac{\text{day} - 1}{364}$ | $[0, 1]$ |
| 1 | Offene Stationen | $f_1 = \lvert \text{remaining} \rvert$ | $[0, 397]$ |

Im Modus `"rule"` wird $f_1$ direkt in `rule_delta()` verwendet; $f_0$ ist als Platzhalter übergeben (wird in `rule_delta()` ignoriert, da nur $n_r$ entscheidend ist).

---

## Initialplan — Modifizierte Score-Funktion

### `route_score_fn` (DB)

$$\text{score}(k, \text{dsm}_k, \text{cur}) = \frac{\tilde{C}(k, \text{dsm}_k) + s}{\max\!\left(0.1,\; d_{\text{cur},k}\right)^{2\delta}}$$

Der Shift-Term $s$ ist identisch zu CFA Future:

$$s = \max\!\left(0,\; -\min_{k \in \text{Tasks}} \tilde{C}(k, \text{dsm}_k)\right) + 1.0$$

### Einfluss von $\delta$ auf den Nenner

| $\delta$ | Distanzexponent $2\delta$ | Effekt |
|---|---|---|
| 0,1 | 0,2 | Distanz fast irrelevant; $\tilde{C}$ dominiert vollständig |
| 0,3 | 0,6 | Distanz schwach bestraft |
| **0,5** | **1,0** | **Identisch mit CFA Future** |
| 0,7 | 1,4 | Distanz stärker bestraft |
| 0,9 | 1,8 | Distanz fast quadratisch; Routing-Effizienz dominiert |

---

## Störungs-Replan — Modifizierter Drop-Score

### `drop_score_fn` (DB)

$$\text{drop\_score}(k, \text{dsm}_k, r_h, \text{cur}, \delta_k) = \tilde{C}(k, \text{dsm}_k) - 2\delta \cdot w_\text{min} \cdot \delta_k$$

| Term | Bedeutung |
|---|---|
| $\tilde{C}(k, \text{dsm}_k)$ | Approximierter Zukunftswert (Kosten des Weglassens) |
| $2\delta$ | Skalierungsfaktor (= $\beta$); bei $\delta=0{,}5$: identisch zu CFA Future |
| $w_\text{min} = w/60$ | Stundenlohn in €/min |
| $\delta_k$ | Detour-Zeit [min]: Zeit, die das Entfernen von $k$ aus der Route einspart |

Der Faktor $\beta = 2\delta$ skaliert, wie stark die Zeitersparnis eines Drops bei der Drop-Entscheidung gewichtet wird:

- $\delta < 0{,}5$: Zeitersparnis weniger relevant → Stationen mit niedrigem $\tilde{C}$ werden bevorzugt gedroppt
- $\delta > 0{,}5$: Zeitersparnis wichtiger → Stationen mit großem Umweg werden bevorzugt gedroppt, unabhängig von $\tilde{C}$

**Niedrigster Drop-Score = zuerst droppen** (identisch zur allgemeinen Logik).

---

## δ-Bestimmung vor Tagesbeginn

Der `DBSimpleMaintenanceSimulator` überschreibt `_run_day()` und setzt $\delta$ **vor** jedem Tagesstart:

```
Tagesbeginn (vor Initialplan):
1. Zustandsvektor ψ(S_t) = [f0, f1] berechnen
2. δ bestimmen:
   - "rule":  δ = rule_delta(f0, f1)
   - "rf":    δ = rf_classifier.predict(ψ)
   - "fixed": δ = default_delta
3. policy.set_precomputed_delta(δ)
4. Tagesplan und Replan nutzen dieses δ
```

$\delta$ bleibt den gesamten Tag konstant (kein Refit innerhalb eines Tages).

---

## Training des RF-Modells (Modus `"rf"`)

Das Training ist zweiphasig. Es wird nur benötigt wenn `delta_mode: "rf"`.

### Phase 1: Label-Sammlung (`--collect`)

**Ziel:** Für jeden Tages-Snapshot den optimalen $\delta^*$ bestimmen.

**Vorgehen:**

1. Pilot-Simulation mit $\delta = 0{,}5$ (CFA Future) über `max_days` Tage.
2. Alle $N_s = 5$ Tage einen Snapshot speichern: $(\text{remaining}, \text{dsm\_array}, \text{carryover}, \text{day})$.
3. Für jeden Snapshot: **CRN-Rollout** über $H = 20$ Tage mit jedem $\delta \in \{0{,}1, 0{,}3, 0{,}5, 0{,}7, 0{,}9\}$.
4. Label: $\delta^* = \arg\min_\delta \sum_{h=1}^{H} C_h^\delta$
5. Ausgabe: $(\psi_t, \delta^*)$-Paare → `data/training/db_simple/labels.pkl`

**CRN (Common Random Numbers):** Alle $\delta$-Varianten eines Snapshots erhalten dieselben vorgenerierten Störungsereignisse. Dadurch werden Stichprobenfehler eliminiert und $\delta$-Unterschiede treten klarer hervor.

**Stratifizierung der Pilot-Simulationen:** `n_outer` unabhängige Pilot-Runs mit unterschiedlichen Seeds.

### Phase 2: Modelltraining (`--train`)

RandomForestClassifier auf den gesammelten $(\psi, \delta^*)$-Labels:

```
Eingabe: X ∈ ℝ^{N×2}  (N Samples, 2 Features: day_progress, n_remaining)
         y ∈ {0.1, 0.3, 0.5, 0.7, 0.9}^N  (optimales δ je Snapshot)

Modell:  RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5)
Scaling: StandardScaler (z-Normierung der Features)
Output:  data/training/db_simple/model.pkl  (clf + scaler)
```

### Sanity-Check

`DBSimplePolicy(δ=0.5)` und `CFAFutureModel` müssen auf identischem Seed exakt dieselben Gesamtkosten liefern (Regressionstest).

---

## Dateipfade

| Datei | Inhalt |
|---|---|
| `src/models/db_simple.py` | Policy-Klasse `DBSimplePolicy`, Balance-Modell, Simulator-Subklasse |
| `scripts/train/train_db_simple.py` | Training (`--collect`, `--train`), Benchmark, Sanity-Check |
| `data/training/cfa_future/theta.json` | Geerbt: θ, μ, σ aus CFA Future Training |
| `data/training/db_simple/labels.pkl` | (RF-Modus) Gesammelte $(\psi, \delta^*)$-Trainingspaare |
| `data/training/db_simple/model.pkl` | (RF-Modus) Trainierter RandomForestClassifier + Scaler |

---

## Algorithmus-Zusammenfassung

```
=== VOR DEM EINSATZ (nur Modus "rf") ===
Phase 1 (--collect):
  Für n_outer Pilot-Runs (δ=0.5):
    Alle 5 Tage Snapshot speichern
    Für jeden Snapshot: H=20-Tage-CRN-Rollout mit δ ∈ {0.1,0.3,0.5,0.7,0.9}
    Label δ* = bestes δ
Phase 2 (--train):
  RandomForestClassifier auf (ψ, δ*)-Labels

=== TAGESBEGINN ===
1. ψ(S_t) = [day_progress, n_remaining] berechnen
2. δ bestimmen (rule / rf / fixed)
3. policy._precomputed_delta = δ

4. Zonenscoring via sum(C̃(k)) → Startzonen → Expansion  [geerbt von CFA Future]
5. Initialplan:
   - Carryover (NN nach Fahrzeit, mandatory)                [geerbt]
   - Routine: greedy nach (C̃(k) + shift) / dist^(2δ)       ← DB-spezifisch

=== STÜNDLICHE SCHLEIFE (08:00–15:00) ===
6. Störungen melden
7. Sortieren nach wirtschaftlichen Insertionskosten         [geerbt]
8. Für jede Störung d:
   a) Direkteinfügung: Cheapest Insertion                   [geerbt]
   b) Falls nicht feasible:
      Drop-Score = C̃(k) − (2δ) × wage_min × δ_k            ← DB-spezifisch
      Iterativ droppen bis feasible
   c) Immer noch nicht feasible: Carryover                  [geerbt]

=== TAGESENDE ===
9. Gesamtkosten (operativ + downtime)                       [geerbt]
10. Carryovers weitergeben
11. δ im delta_log protokollieren
```

---

## Konfigurationsschlüssel (Auszug)

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `solver.use_or_tools` | `false` | Greedy-Routing aktiv |
| `db_simple.delta_mode` | `"rule"` | δ-Bestimmungsmodus |
| `planning.zone_selection_mode` | `value_based` | Zonenauswahl via $\tilde{C}$ (geerbt) |
| `failure_simulation.mode` | `stochastic` | Pflicht für dsm-abhängige Features |
| `failure_simulation.recovery_days` | 365 | $T_r$ (geerbt) |
| `failure_simulation.initial_factor` | 0,1 | $f_0$ (geerbt) |

---

## Vergleich aller Greedy-Policies

| Aspekt | Myopic | Myopic Plus | CFA Future | DB |
|---|---|---|---|---|
| `route_score_fn` | $1/d$ | $P_k/d$ | $(\tilde{C}+s)/d$ | $(\tilde{C}+s)/d^{2\delta}$ |
| `drop_score_fn` | $d_{\text{cur},k}$ | $P_k/d_{\text{cur},k}$ | $\tilde{C}(k) - w_\text{min}\delta_k$ | $\tilde{C}(k) - 2\delta \cdot w_\text{min}\delta_k$ |
| Gelernte Parameter | keine | keine | $\theta \in \mathbb{R}^4$ | $\theta \in \mathbb{R}^4$ + opt. RF |
| Zustandsabhängige Steuerung | nein | nein | nein | $\delta(S_t)$ |
| δ bei `"rule"`, $n_r \geq 130$ | — | — | — | 0,5 (= CFA Future) |
| δ bei `"rule"`, $n_r < 130$ | — | — | — | 0,9 (Routing dominiert) |
