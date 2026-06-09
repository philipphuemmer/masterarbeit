# VFA — Vollständige Modellbeschreibung (Online-Rollout)

## Überblick

VFA bezeichnet im Rahmen dieser Arbeit **keinen eigenen Planungsalgorithmus**, sondern einen **Online-Rollout-Layer** (Rolling Horizon), der über eine Basisheuristik gelegt wird. Die Basisheuristik bleibt vollständig erhalten; VFA greift ausschließlich an kritischen Entscheidungspunkten ein und ersetzt die Policy-Entscheidung, wenn ein Horizont-Rollout eine bessere Alternative identifiziert.

**Basis-Policies (Offline-Heuristiken):**

| Bezeichnung | Modellklasse |
|---|---|
| Myopic | `MyopicPolicy` |
| Myopic+ | `MyopicPlusModel` |
| CFA | `CFAFutureModel` |
| DB | `DBSimplePolicy` |

VFA ist damit eine **Offline/Online-Hybridstrategie** im Sinne von Powell (2011):
- **Offline:** Die Basisheuristik (θ, δ, Scoring-Funktion) ist vortrainiert und bleibt unverändert.
- **Online:** Rollout-Simulationen am Entscheidungszeitpunkt schätzen den Zukunftswert alternativer Aktionen.

---

## Konzept: Rollout als Wertfunktionsapproximation

Sei $s$ der aktuelle Systemzustand und $x$ eine mögliche Aktion. Der Rollout-Wert einer Aktion ist:

$$Q^\pi(s, x) = c(s, x) + V^\pi(s^x)$$

- $c(s, x)$: Sofortkosten der Aktion (Downtime-Wartezeit bis Service)
- $V^\pi(s^x)$: Zukunftskosten ab dem Post-Decision-Zustand $s^x$, approximiert durch $H$-Tage-Rollout mit Basis-Policy $\pi$

**Entscheidungsregel:**

$$x^* = \arg\min_{x \in \mathcal{X}(s)} \hat{Q}^\pi(s, x)$$

$\hat{Q}^\pi$ ist der Erwartungswert über $M$ stochastische Szenarien (Common-Random-Numbers).

---

## Eingriffspunkte

VFA greift an **zwei optionalen** Entscheidungspunkten ein, konfigurierbar über `rolling_horizon`:

### 1. Störungs-Replan (Drop-Entscheidung) — `enable_replan: true`

Wenn eine Störung nicht direkt eingefügt werden kann und ein Routine-Stop gedroppt werden muss, bewertet VFA die Top-$k$ Einzel-Drop-Kandidaten und wählt den mit dem niedrigsten erwarteten Horizont-Gesamtkosten.

**VFA greift AUSSCHLIESSLICH bei Einzel-Drop-Fällen ein.** Bei Multi-Drops (≥ 2 Stops müssen weichen) und bei direkter Insertion (kein Drop nötig) übernimmt die Basis-Policy unverändert.

### 2. Initialplan (Seed-Rollout) — `enable_initial: false` (aktuell deaktiviert)

Am Tagesbeginn werden für jedes Team $k$ alternative Startstationen (Seeds) per Rollout bewertet. Das beste Seed wird als Anfang der Tagesroute fixiert; der Rest wird greedy vervollständigt.

---

## Systemzustand

Der persistente Systemzustand $S_t$ wird von Tag zu Tag fortgeschrieben:

$$S_t = (\text{day},\; \text{dsm}[1{:}N],\; \text{remaining},\; \text{carryover},\; \text{rng})$$

| Komponente | Typ | Bedeutung |
|---|---|---|
| `day` | int | Aktueller Simulationstag (1-basiert) |
| `days_since_maintenance` | `np.ndarray[N+1]` | dsm pro node_idx; Index 0 = Depot (immer 0) |
| `remaining` | `set[int]` | 0-basierte Stationsindizes noch nicht jährlich gewartet |
| `carryover_tasks` | `list[MaintenanceTask]` | Unerledigte Störungen vom Vortag |
| `rng` | `np.random.Generator` | Zufallszahlengenerator (reproduzierbar per Seed) |

---

## Post-Decision-Zustand

Der Post-Decision-Zustand $s^x$ entspricht dem Systemzustand **nach** Anwendung einer deterministischen Aktion $x$, aber **vor** künftiger Zufälligkeit:

$$s^x = (S_t,\; \text{sim\_routes mit angewendetem Drop+Insert},\; \text{hour})$$

**Berechnung von $V^\pi(s^x)$ (Post-Decision-Rollout):**

1. Ab Stunde `hour+1`: Resttag stochastisch simulieren (neue Störungen, Basis-Policy für Replan).
2. Ab Tag `day+1`: $H-1$ vollständige Rollout-Tage mit Basis-Policy.
3. Terminal-Kosten: $w_\text{terminal} \cdot |\text{remaining}| \cdot s_\text{min} \cdot w / 60$.

---

## Horizon-Evaluierung (HorizonEvaluator)

### Parameter

| Symbol | Konfigurationsschlüssel | Wert |
|---|---|---|
| $H$ | `rolling_horizon.horizon_days` | 40 |
| $M$ | `rolling_horizon.n_scenarios` | 25 |
| $k$ | `rolling_horizon.top_k_candidates` | 3 |
| $w_\text{terminal}$ | `rolling_horizon.terminal_remaining_weight` | 1,0 |
| $T_\text{budget}$ | `rolling_horizon.time_budget_sec` | 100,0 s |

### Common-Random-Numbers (CRN)

Alle $k$ Kandidaten einer Entscheidung werden auf **denselben** $M$ Störungsszenarien evaluiert:

```
scenario_seeds = draw_scenario_seeds(M)   # einmal pro Störungsereignis
für jeden Kandidaten x_i:
    costs_i = evaluate_post_decision_scenarios(s^{x_i}, H, scenario_seeds)
    Q̂(s, x_i) = c(s, x_i) + mean(costs_i)
```

CRN eliminiert Stichproben-Varianz und macht Kandidatenvergleiche statistisch belastbarer.

### Szenario-Simulation (_run_day_fast)

Ein schneller Simulationstag ohne Logging:

```
1. Stationsauswahl via DailyZoneSelector
2. Initialplan via Basis-Policy.create_initial_plan()
3. Stochastische Störungsgenerierung (dsm-abhängig):
   p(t) = (p1 + p2) × [f0 + (1−f0) × min(t, Tr)/Tr] × station_factor
4. Störungs-Replan via Basis-Policy.handle_disruptions() (KEIN RH-Eingriff)
5. Operative Kosten + Downtime-Kosten akkumulieren
```

Im Rollout wird **immer die Basis-Policy** verwendet — kein rekursiver RH-Eingriff.

### Terminal-Kostenpauschale

Am Horizont-Ende noch ungewartete Stationen erhalten eine Strafkosten-Pauschale:

$$C_\text{terminal} = w_\text{terminal} \cdot |\text{remaining}| \cdot s_\text{min} \cdot \frac{w}{60}$$

| Symbol | Wert |
|---|---|
| $w_\text{terminal}$ | 1,0 (vollständige Wartungskosten) |
| $s_\text{min}$ | 45 min (mittlere Servicezeit) |
| $w$ | 35 €/h |

---

## Störungs-Replan mit VFA (enable_replan)

### Ablauf für jede Störung $d$

```
1. Direkteinfügung möglich (kein Drop)?
   → Basis-Policy: Cheapest Insertion, kein VFA-Eingriff.

2. Basis-Policy bestimmt via _find_best_drop_and_insert:
   2a. Kein Drop möglich → Carryover (kein VFA-Eingriff).
   2b. Multi-Drop (≥ 2 Stops) → Basis-Policy direkt (kein VFA-Eingriff).
   2c. Einzel-Drop → VFA evaluiert top-k Kandidaten.

3. Kandidatenliste:
   Kandidat 0: Basis-Policy-Wahl (aus _find_best_drop_and_insert)
   Kandidaten 1..k-1: feasibility-geprüfte Alternativen (aufsteigend nach drop_score_fn)

4. CRN-Rollout für jeden Kandidaten:
   Q̂(s, x_i) = c(s, x_i) + V̂(s^{x_i})

5. Kandidatenauswahl (via candidate_selection_mode):
   "win_rate":      Kandidat muss Basis in ≥ 70% der Szenarien schlagen UND größten Ø-Gewinn haben.
   "expected_value": Kandidat mit niedrigstem Erwartungswert.

6. Ausführung des gewählten Drops + Insertion.
```

### Feasibility-Prüfung der Kandidaten

Jeder alternative Kandidat wird **vorab** auf Machbarkeit geprüft:

- Basis-Kandidat: kommt aus `_find_best_drop_and_insert` (immer feasible).
- Alternativen: `_get_feasible_drop_candidates` simuliert für jeden Routine-Stop, ob sein Entfernen allein ausreicht, um Störung $d$ einzufügen. Nur feasible Drops kommen in die Kandidatenliste.

Damit ist garantiert, dass kein RH-Override zu einem Multi-Drop-Fallback führt.

### Kandidatenauswahlmodi

#### `"win_rate"` (aktuell aktiv)

$$x^* = \arg\max_{x_i \neq x_0:\; \text{WR}(x_i, x_0) \geq 0{,}7} \left[\mathbb{E}[Q̂(s, x_0)] - \mathbb{E}[Q̂(s, x_i)]\right]$$

- $\text{WR}(x_i, x_0) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[Q_m(s, x_i) < Q_m(s, x_0)]$
- Win-Rate-Schwelle: 0,70 (Kandidat muss in 70 % der Szenarien besser sein)
- Kein Wechsel wenn kein Kandidat die Schwelle erfüllt (Basis-Policy bleibt)

#### `"expected_value"`

$$x^* = \arg\min_{x_i} \mathbb{E}_M[Q̂(s, x_i)]$$

---

## Initialplan-RH (enable_initial, aktuell deaktiviert)

### Prinzip

Statt die Tagesroute vollständig greedy aufzubauen, wird der **Startblock** (1–2 erste Stationen) per Rollout optimiert. Der Rest der Route wird greedy vervollständigt.

### Ablauf

```
Für jedes Team (sequenziell, Team 0 zuerst):
  1. Basis-Seed aus base_plan extrahieren (längster Zonenblock, ≤ block_size Stops)
  2. Top-(top_k-1) alternative Seeds nach Prescore ranken
     Prescore = C̃(k) - wage_per_min × travel(depot→k)
  3. Jeden Seed per evaluate_with_forced_day0_plan bewerten (CRN, H Tage)
  4. Bestes Seed wählen (win_rate oder expected_value)
  5. Route mit fixem Seed-Präfix greedy vervollständigen
```

### Seed-Prescore

Heuristik zur Vorfilterung vor dem teuren Rollout:

**1-Stop-Block:**
$$\text{prescore}(k) = \tilde{C}(k, \text{dsm}_k) - w_\text{min} \cdot \tau_{0,k}^{\min}$$

**2-Stop-Block $(k_1, k_2)$:**
$$\text{prescore}(k_1, k_2) = \tilde{C}(k_1) + \tilde{C}(k_2) - w_\text{min} \cdot (\tau_{0,k_1}^{\min} + \tau_{k_1,k_2}^{\min})$$

Diversitätsfilter: Kein zwei Kandidaten mit identischer erster Station.

---

## Konfigurationsschlüssel

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `rolling_horizon.enabled` | `true` | VFA-Layer aktiv |
| `rolling_horizon.horizon_days` | 40 | Rollout-Horizont $H$ |
| `rolling_horizon.n_scenarios` | 25 | Szenario-Anzahl $M$ |
| `rolling_horizon.top_k_candidates` | 3 | Top-$k$ Drop-Kandidaten |
| `rolling_horizon.enable_replan` | `true` | RH bei Drop-Entscheidungen |
| `rolling_horizon.enable_initial` | `false` | RH bei Initialplanung |
| `rolling_horizon.use_post_decision_rollout` | `true` | Echtes Post-Decision (sonst: nächsten Tag) |
| `rolling_horizon.candidate_selection_mode` | `"win_rate"` | `"win_rate"` oder `"expected_value"` |
| `rolling_horizon.win_rate_threshold_replan` | 0,7 | Mindest-Gewinnquote Replan |
| `rolling_horizon.win_rate_threshold_initial` | 0,7 | Mindest-Gewinnquote Initialplan |
| `rolling_horizon.terminal_remaining_weight` | 1,0 | Gewicht der Terminalkosten |
| `rolling_horizon.time_budget_sec` | 100,0 | Max. Evaluierungszeit pro Störung [s] |
| `rolling_horizon.fallback_to_legacy_on_timeout` | `true` | Basis-Policy bei Timeout |
| `rolling_horizon.top_k_initial` | 3 | Seed-Kandidaten pro Team (Initial-RH) |
| `rolling_horizon.initial_seed_block_size` | 1 | Seed-Blockgröße (1 oder 2 Stops) |

---

## Gesamtkosten im Rollout-Tag

Operative Kosten (Lohn + Kraftstoff) für einen Rollout-Tag:

$$C_\text{op} = \sum_{\text{Teams}} \left( \frac{T}{60} \cdot w + d_\text{km} \cdot c_\text{km} \right)$$

Downtime-Kosten für Carryover-Störungen (konservative Schätzung: restliche Restzeit des Tages):

$$C_\text{dt,co} = \sum_{d \in \text{carryover}} \frac{T - t_\text{meldung}}{60} \cdot P_d \cdot c_\text{dt}$$

---

## Algorithmus-Zusammenfassung

```
=== VOR DEM LAUF ===
Basis-Policy laden/trainieren (Myopic / Myopic+ / CFA / DB)
PolicyAdapter wrappen → einheitliche Schnittstelle

=== TAGESBEGINN ===
1. pre_day_hook: modellspezifische Vorbereitung (z.B. δ bei DB)
2. Stationsauswahl via DailyZoneSelector
3a. [enable_initial=false] Initialplan via Basis-Policy (kein RH)
3b. [enable_initial=true]  Initial-RH:
    - Basis-Seed + top-(k-1) Alternativen nach Prescore
    - CRN-Rollout pro Seed über H Tage
    - Bestes Seed (win_rate oder expected_value) wählen
    - Route greedy vervollständigen

=== STÜNDLICHE SCHLEIFE (08:00–15:00) ===
4. Störungen melden
5. Sortieren nach Insertionskosten (günstigste zuerst)
6. Für jede Störung d:
   a) Kein Drop nötig → Direkteinfügung (Basis-Policy, kein RH)
   b) Multi-Drop (≥2) → Basis-Policy direkt (kein RH)
   c) Kein Platz → Carryover
   d) Einzel-Drop → [enable_replan=true]:
      - Kandidat 0: Basis-Policy-Drop
      - Kandidaten 1..k-1: feasibility-geprüfte Alternativen
      - CRN-Rollout: Q̂(s, x_i) = c(s, x_i) + V̂(s^{x_i}) über M Szenarien
      - Win-Rate ≥ 70% UND größter Ø-Gewinn → Override
      - Sonst: Basis-Policy-Drop bestätigt

=== TAGESENDE ===
7. dsm +1 für alle; dsm=0 für heute bediente Stationen
8. Carryovers in nächsten Tag übergeben
9. Replan-Overrides und Initial-Overrides protokollieren
```

---

## Dateipfade

| Datei | Inhalt |
|---|---|
| `src/models/rolling_horizon.py` | `RollingHorizonRunner`, `HorizonEvaluator`, `PolicyAdapter`, `DailyTaskGenerator`, `SystemState`, `PostDecisionState` |
| `scripts/run/run_cfa_future_rollout.py` | Ausführen von VFA mit CFA als Basis-Policy |
| `scripts/run/run_db_simple_rollout.py` | Ausführen von VFA mit DB als Basis-Policy |
| `scripts/monte_carlo/run_mc_cfa_future_rollout.py` | Monte-Carlo-Auswertung VFA+CFA |
| `scripts/monte_carlo/run_mc_db_simple_rollout.py` | Monte-Carlo-Auswertung VFA+DB |

---

## Vergleich: Basis-Policy vs. VFA

| Aspekt | Basis-Policy allein | VFA (+ Basis-Policy) |
|---|---|---|
| Initialplan | Greedy (CFA/DB-Score) | Identisch oder Seed-Override |
| Drop-Entscheidung | `drop_score_fn` der Basis-Policy | Rollout-basiert (H=40, M=25) |
| Laufzeitoverhead | keiner | ≤ 100 s/Störungsereignis |
| Override-Rate | — | 0–30 % (je nach Policy und Störungsrate) |
| Reproduzierbarkeit | vollständig | CRN garantiert faire Kandidatenvergleiche |
| Zusätzliche Parameter | keine | H, M, k, Win-Rate-Schwelle |
