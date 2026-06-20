# Übersicht der fünf aktiven Wartungsplanungs-Policies

> **Namenskonvention:** CFA (`cfa_future.py`), DB (`db_simple.py`)

Dieses Dokument beschreibt die fünf implementierten Planungsstrategien für die
Wartungsoptimierung der 397 EV-Ladesäulen in Würzburg. Die ersten vier sind
**Basispolicies** — sie verwenden ausschließlich **Greedy Cheapest-Insertion**
für Initialplan und Störungshandling. Das fünfte Modell, **VFA**, ist eine
**Online-Rollout-Policy**: Sie wrапpt jede der vier Basispolicies und ersetzt
deren greedy Drop-Entscheidung durch eine kurzfristige Monte-Carlo-Vorausschau.
OR-Tools wird von keiner aktiven Policy verwendet.

**Gemeinsame Rahmenbedingungen:**

| Parameter | Wert |
|---|---|
| Stationen | 397 |
| Teams | 2 |
| Arbeitstag | 08:00–16:00 (480 min, effektiv bis 17:00 durch Mittagspause) |
| Zonen (K-Means) | 40 (konfigurierbar) |
| Max. Stops/Team/Tag | 20 |
| Lohnkosten | 40 €/h |
| Kraftstoffkosten | 0,30 €/km |
| Ausfallkosten | 0,50 €/kWh |
| Servicezeit Typ-1-Störung | 60 min |
| Servicezeit Typ-2-Störung | 30 min + Depot-Rundfahrt + 5 min + 30 min |

---

## 1. Myopic

**Datei:** [src/models/myopic.py](../src/models/myopic.py)

### Konzept

Die Myopic-Policy ist die einfachste und schnellste Strategie. Sie trifft alle
Entscheidungen rein reaktiv — ohne Modell zukünftiger Kosten. Der Name
„myopic" (kurzsichtig) beschreibt das Kernprinzip: Jede Entscheidung minimiert
nur die unmittelbaren Zusatzkosten, ohne Langzeitfolgen zu berücksichtigen.
Sowohl Initialplan als auch Störungshandling basieren vollständig auf
Greedy Cheapest-Insertion.

### Initialplan: Greedy Cheapest-Insertion

Die Routinen-Tasks werden iterativ in die Teamrouten eingefügt. Bei jeder
Iteration wird die (Station, Einfügeposition, Team)-Kombination mit den
geringsten Zusatzkosten gewählt:

```
extra_travel  = t(prev → k) + t(k → next) − t(prev → next)
extra_time_h  = (extra_travel + service_min) / 60
extra_km      = km(prev→k) + km(k→next) − km(prev→next)

cost = extra_time_h × 40 €/h + max(0, extra_km) × 0,30 €/km
```

Carryover-Tasks aus Vortagen werden ohne besondere Priorisierung wie normale
Routine-Tasks behandelt.

### Störungshandling: Greedy Cheapest-Insertion

**Schritt 1 – Direkte Einfügung:**
Für jede Störung wird die günstigste Einfügeposition über alle Teamrouten
gesucht. Zusätzlich zur Fahrtzeit fallen Ausfallkosten an:

```
downtime_h = max(0, arrival_at_d − report_min) / 60
cost       = extra_time_h × 40 €/h
           + max(0, extra_km) × 0,30 €/km
           + downtime_h × power_kW × 0,50 €/kWh
```

Machbarkeit: Das Team muss nach Einfügung spätestens um 16:00 (480 min)
am Depot ankommen.

**Schritt 2 – Drop-and-Insert (Fallback):**
Falls keine direkte Einfügung machbar ist, werden Routine-Stops
**positionsbasiert von hinten** iterativ entfernt (1, 2, 3, …),
bis die Störung eingefügt werden kann. Gedropte Stops werden als
Carryover auf den nächsten Tag verschoben.

**Schritt 3 – Carryover:**
Scheitern beide Schritte, wird die Störung selbst auf den nächsten Tag
verschoben.

### Stärken und Schwächen

| + | − |
|---|---|
| Einfach, schnell, deterministisch | Ignoriert zukünftige Ausfallwahrscheinlichkeiten |
| Kein Training erforderlich | Drop-Reihenfolge positionsbasiert, nicht wertbasiert |
| Explizit nachvollziehbare Entscheidungen | Carryover-Tasks nicht priorisiert |

---

## 2. Myopic+

**Datei:** [src/models/myopic_plus.py](../src/models/myopic_plus.py)

### Konzept

Myopic+ erweitert Myopic um **leistungsgewichtete Soft-Deadlines** im Initialplan.
Die Grundidee: Hochleistungs-Stationen sollen früher im Tag besucht werden,
um Ausfallkosten bei Störungen zu minimieren. Das Routing selbst bleibt
vollständig greedy — kein OR-Tools, kein gelerntes Modell.

Carryover-Tasks aus Vortagen erhalten `soft_deadline_min = 0` sowie einen
leistungsgewichteten Penalty, sodass sie bei der Greedy-Insertion bevorzugt
früh eingeplantt werden.

### Initialplan: Depot-Distanz-Sortierung mit Soft-Deadlines

Routine-Tasks werden aufsteigend nach **Depot-Distanz** sortiert (nahe
Stationen → frühe Einplanung). Die Deadline ist gleichmäßig über den Arbeitstag
verteilt:

```
deadline(k) = rank(k) / n_tasks × 480 min
```

Der **Deadline-Penalty** skaliert mit der Nennleistung:

```
penalty_per_min = α × power_kW × p_h × downtime_eur_per_kwh / wage_per_min

mit:
  α            = 10,0  (Skalierungsfaktor aus config.yaml → cfa.alpha)
  p_h          = p1_per_hour + p2_per_hour  (Gesamtausfallrate)
  downtime     = 0,50 €/kWh
  wage_per_min = 40/60 €/min
```

Der Penalty ist mindestens 1. Die Soft-Deadlines fließen als zusätzliche
Strafkosten in die Cheapest-Insertion-Bewertung ein: Stationen, die nach
ihrer Deadline besucht werden, erzeugen Penalty-Kosten proportional zur
Überschreitungsdauer.

### Störungshandling: Greedy Replan mit leistungsbasiertem Drop

Bei Störungen wird der gesamte Resttag neu geplant — vollständig greedy,
ohne OR-Tools. Die noch nicht besuchten Routine-Tasks und die neuen
Störungsknoten werden gemeinsam per Cheapest-Insertion neu eingeplant.

**Drop-Reihenfolge:** Routine-Stops werden nach aufsteigendem
`power_kW` gedroppt — Niedrigleistungsstationen zuerst. Gestoppte Stops
werden als Carryover verschoben.

### Stärken und Schwächen

| + | − |
|---|---|
| Leistungsgewichtete Priorisierung im Initialplan | Deadline-Sortierung ignoriert `days_since_maintenance` |
| Kein Training erforderlich | Drop-Reihenfolge basiert auf Leistung, nicht auf gelerntem Wert |
| Carryover-Tasks werden priorisiert | Kein Ausfallrisiko-Modell |

---

## 3. CFA

**Datei:** [src/models/cfa_future.py](../src/models/cfa_future.py)  
**Training:** [scripts/train/train_cfa_future.py](../scripts/train/train_cfa_future.py)  
**Parameter:** `data/training/cfa_future/theta.json`

### Konzept

CFA (*Cost Function Approximation*) ist die erste Policy mit **gelernten**
Parametern. Sie approximiert die Zukunftskosten jeder Station durch eine
lineare Wertfunktion C̃ über vier Features:

```
C̃(k) = θᵀ × φ_scaled(k)
```

C̃ steuert sowohl die Reihenfolge im Initialplan als auch die Drop-Entscheidung
beim Störungshandling. Alles bleibt greedy — C̃ ist eine bessere Priorität,
kein Optimizer.

### Feature-Vektor φ(k)

```
φ(k) = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others]

recovery_curve(dsm) = initial_factor + (1 − initial_factor) × dsm / recovery_days
```

Die Features werden mit den Trainings-Mittelwerten und -Standardabweichungen
z-skaliert:

```
φ_scaled(k) = (φ(k) − μ) / σ
```

μ und σ werden zusammen mit θ in `data/training/cfa_future/theta.json`
gespeichert und immer gemeinsam geladen.

### Training: Kontrastive Suffix-Simulations-Regression

θ ∈ ℝ⁴ wird **nicht** per OLS auf Rollout-Kosten gelernt, sondern durch
kontrastive Regression:

```
Label: cost_drop_k − cost_serve_k   (diskontiert, H = 30 Tage Horizont)
```

Für jede Station k wird simuliert, was es kostet, sie jetzt zu droppen vs.
zu bedienen. θ wird so angepasst, dass C̃(k) diesen Differenz-Wert approximiert.
Das macht C̃ direkt zu einer Schätzung des Mehrwerts der Bedienung.

### Initialplan: C̃/Depot-Distanz-Sortierung

Tasks werden nach dem **Verhältnis C̃ / depot_dist** sortiert — Stationen
mit hohem Wert UND kurzer Depot-Distanz kommen früh:

```
priority(k) = C̃(k) / max(0.1, km(k, depot))
```

Soft-Deadlines werden analog zu Myopic+ aus dem Rang abgeleitet; der
Deadline-Penalty skaliert proportional zu C̃(k).

Carryover-Tasks erhalten `soft_deadline_min = 0` und einen Penalty ∝ power_kW,
sodass sie in der Greedy-Insertion prioritär früh eingeplant werden.

### Störungshandling: Manueller C̃-Drop-Loop

Bei Störungen wird der Resttag greedy neu geplant. Wenn der Platz nicht reicht,
werden Routine-Stops nach **aufsteigendem C̃** iterativ gedroppt
(niedrigster C̃ → erster Kandidat):

```
1. Sortiere verbleibende Routine-Stops aufsteigend nach C̃(k)
2. Entferne Stops iterativ, bis die Störung eingepasst werden kann
3. Gedropte Stops → Carryover
```

### Zonenauswahl (value_based_zone_selection)

Wenn `planning.value_based_zone_selection: true`, wird `selector.value_fn = policy._value`
gesetzt. Der `DailyZoneSelector` berechnet den Zonenwert als:

```
zone_score = w_value × Σ C̃(station) + w_depot × depot_dist
```

### Stärken und Schwächen

| + | − |
|---|---|
| Gelerntes θ: evidenzbasierter Stationswert | Training erfordert stochastischen Simulationsmodus |
| C̃ berücksichtigt Alter, Ausfallrisiko, Distanz | 4-dimensionaler Featurevektor, kein globaler Zustand |
| Schützt hohe C̃-Stationen beim Drop | C̃-Drop-Loop ist greedy, nicht global optimal |

---

## 4. DB

**Datei:** [src/models/db_simple.py](../src/models/db_simple.py)  
**Balancemodell:** `data/training/db_simple/model.pkl` (optional)

### Konzept

DB (*Distance-Balanced Simple*) erweitert CFA um einen
**zustandsabhängigen Balance-Parameter δ ∈ [0,1]**, der das Verhältnis
zwischen Zukunftswert (C̃) und Routingeffizienz (Distanz) dynamisch steuert.
Bei δ = 0,5 ist DB identisch zu CFA.

DB erbt alle Methoden von CFA (`_phi()`, `_value()`, θ-Laden,
Feature-Skalierung) und überschreibt nur die Scoring-Funktionen.

### Balance-Parameter δ

```
δ ∈ [0,1]:
  δ < 0.5  → C̃ (Zukunftswert) stärker gewichtet
  δ = 0.5  → identisch zu CFA
  δ > 0.5  → Routingeffizienz (Distanz) stärker gewichtet
```

δ wird von `DBSimpleBalanceModel` geliefert:
- **Gelernt** (RF-Modell aus `data/training/db_simple/model.pkl`): zustandsabhängig
- **Regelbasiert** (`rule_delta()`): heuristisch aus Systemzustand
- **Fallback**: `default_delta = 0.5` (entspricht CFA)

### Greedy-Scoring im Initialplan

```
score(k) = (C̃(k) + shift) / dist(cur, k)^(2δ)
```

- `shift`: Konstante, um negative C̃-Werte auszugleichen (alle Scores positiv)
- `2δ`: Distanzexponent — bei δ = 0,5 ist der Exponent 1 (linear, wie CFA);
  bei δ > 0,5 wird die Distanz stärker bestraft (effizientere Routen)

### Drop-Score beim Störungshandling

```
drop_score(k) = C̃(k) − (2δ) × wage_per_min × detour(k)
```

- Stops mit niedrigem Drop-Score werden zuerst gedroppt
- Bei δ = 0,5: `drop_score = C̃(k) − wage_per_min × detour(k)` (wie CFA)
- `detour(k)`: Zeit-Umweg, der durch den Drop eingespart wird

### Zonenauswahl

Identisch zu CFA — DB erbt `_value()` unverändert.

### Stärken und Schwächen

| + | − |
|---|---|
| Flexibel: kontinuierliches Spektrum zwischen Zukunftswert und Effizienz | δ-Modell benötigt stochastischen Simulationsmodus |
| δ = 0,5 als sicherer Fallback (= CFA) | Komplexer als CFA, Gewinn hängt von δ-Qualität ab |
| Erbt gesamte CFA-Infrastruktur | Mehr Hyperparameter (δ-Modell, shift) |

---

## 5. VFA (Rollout-Policy)

**Datei:** [src/models/rolling_horizon.py](../src/models/rolling_horizon.py)  
**Konfiguration:** `rolling_horizon.enabled: true` in `configs/config.yaml`

### Konzept

VFA (*Value Function Approximation via Rollout*) ist das fünfte Modell und
stellt die **Online-Erweiterung** der vier Basispolicies dar. Die Grundidee
folgt dem klassischen Rollout-Prinzip aus der approximativen dynamischen
Programmierung:

```
Offline-Teil  →  Basispolicy (Myopic / Myopic+ / CFA / DB)
                 liefert Heuristik für Drop-Entscheidungen

Online-Teil   →  VFA-Rollout bewertet Drop-Kandidaten durch
                 kurzfristige Monte-Carlo-Vorausschau und
                 überschreibt die greedy Entscheidung der Basispolicy
                 wenn eine bessere Alternative gefunden wird
```

Die Basispolicy bleibt unverändert — VFA setzt ausschließlich am
**Drop-Entscheidungspunkt** beim Störungshandling an.
Initialplan und Zonenauswahl werden von der jeweiligen Basispolicy übernommen.

### Architektur: RollingHorizonRunner + PolicyAdapter

`RollingHorizonRunner` treibt die Simulation Tag für Tag. `PolicyAdapter`
kapselt jede der vier Basispolicies hinter einer einheitlichen Schnittstelle:
- `get_drop_score_fn()` — Drop-Ranking der Basispolicy
- `get_value_fn()` — stationslokal Wertschätzung (für Zonenauswahl)
- `get_route_score_fn_for_tasks()` — Routing-Bewertung

### Online-Rollout beim Störungshandling

Bei jeder Störung, bei der die Basispolicy einen Drop vornehmen würde:

```
1. Basispolicy rankt alle Routine-Drops → Top-k Kandidaten
   (k = rolling_horizon.top_k_candidates)

2. Für jeden Kandidaten c ∈ Top-k:
   - Simuliere horizon_days Tage weiter
   - Wiederhole n_scenarios mal mit verschiedenen Störungsszenarien
   - Schätze erwartete Horizont-Kosten E[H-Kosten | drop c]

3. Wähle c* = argmin E[H-Kosten]
   → überschreibt greedy Wahl der Basispolicy falls c* ≠ greedy-Wahl
```

Der Rollout selbst verwendet intern die Basispolicy für alle
Folge-Entscheidungen im Horizont — konsistent mit dem Rollout-Prinzip.

### Konfiguration

| Parameter | Bedeutung |
|---|---|
| `enabled` | `true` → VFA aktiv; `false` → identisch zur Basispolicy |
| `horizon_days` | Vorausschauhorizont in Tagen |
| `n_scenarios` | Anzahl stochastischer Szenarien pro Kandidat |
| `top_k_candidates` | Anzahl evaluierter Drop-Kandidaten |
| `enable_replan` | Rollout beim Störungshandling aktiv |
| `enable_initial` | Rollout auch beim Initialplan aktiv |
| `time_budget_sec` | Zeitlimit pro Rollout (Fallback auf Basispolicy) |

### Einordnung: Offline vs. Online

| Aspekt | Offline (Basispolicy) | Online (VFA-Rollout) |
|---|---|---|
| **Wann** | vor dem Tag (Planung) | während des Tages (Störung) |
| **Entscheidung** | Initialplan + Prioritätsregeln | Drop-Auswahl bei Engpass |
| **Methode** | Greedy CI + gelernte Heuristik | Monte-Carlo-Vorausschau |
| **Kosten** | O(n) pro Insertion | O(k × n_scenarios × horizon) |
| **Qualitätsgarantie** | keine | ≥ Basispolicy (per Konstruktion) |

---

## Vergleich aller fünf Policies

| Merkmal | Myopic | Myopic+ | CFA | DB | VFA |
|---|---|---|---|---|---|
| **Initialplan** | Greedy CI | Greedy CI + Soft-Deadlines (depot_dist) | Greedy CI + Soft-Deadlines (C̃/dist) | Greedy CI mit δ-Exponent | von Basispolicy |
| **Störungshandling** | Greedy CI | Greedy Replan | Greedy Replan + C̃-Drop-Loop | Greedy Replan + δ-Drop-Score | Rollout über Top-k |
| **Drop-Kriterium** | letzte Position | aufsteigend power_kW | aufsteigend C̃ | aufsteigend C̃ − 2δ × detour | min. E[Horizont-Kosten] |
| **Gelernte Parameter** | – | – | θ ∈ ℝ⁴ (kontrastiv) | θ ∈ ℝ⁴ + δ-Modell | von Basispolicy |
| **Zustandsdarstellung** | keine | power_kW | C̃ = θᵀφ | C̃ = θᵀφ + δ | stochastische Szenarien |
| **Carryover-Priorisierung** | nein | ja | ja | ja | von Basispolicy |
| **Training erforderlich** | nein | nein | ja | ja (+ opt. δ) | nein (Basispolicy nötig) |
| **Rechenaufwand** | gering | gering | gering | gering | hoch (Online-Rollout) |
| **Zonenauswahl (value-based)** | `power × recovery_curve` | `power × recovery_curve` | `θᵀφ_scaled` | `θᵀφ_scaled` | local C̃ der Basispolicy |

### Hierarchie der Entscheidungsqualität

```
Myopic
  └─ + leistungsgewichtete Reihenfolge + power-basiertes Drop
        → Myopic+
             └─ + gelerntes θ, C̃-Bewusstsein, C̃-basiertes Drop
                   → CFA
                        └─ + zustandsabhängige Balance δ zwischen C̃ und Distanz
                              → DB
                                   └─ + Online-Rollout: Monte-Carlo-Vorausschau
                                         ersetzt greedy Drop durch optimierten Drop
                                              → VFA (wraps any of the above)
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

`Wartezeit_h = max(0, Ankunftszeit − Meldezeit)` in Stunden.

### Servicezeiten

- **Typ-1-Störung** (Vor-Ort-Reparatur): 60 min
- **Typ-2-Störung** (Austausch mit Depot-Fahrt):
  30 min Demontage + Depot-Rundfahrt + 5 min Lagerhandling + 30 min Montage
