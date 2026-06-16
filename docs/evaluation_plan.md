# Evaluationsplan – Masterarbeit Wartungsroutenoptimierung

Stand: 2026-06-16 | 500 Monte-Carlo-Läufe (Seeds 1–500) pro Variante | stochastisches Störungsmodell

---

## 1  Datenbasis

### 1.1  Verfügbare Ergebnisse (logs/ergebnisse/)

| Variante | Modell | Zone-Selection | Läufe | MW Gesamtkosten |
|---|---|---|---|---|
| myopic_centrality | Myopic | centrality | 500 | 48.023 € |
| myopic_plus_centrality | Myopic+ | centrality | 500 | 40.144 € |
| cfa_future_centrality | CFA-Future | centrality | 500 | 40.187 € |
| db_simple_centrality | DB-Simple | centrality | 500 | 40.183 € |Basierend auf dem Overview-Log habt ihr schon die zentralen KPIs pro Seed (Gesamtkosten, Lohn/Fahrt/Ausfallkosten, Same-Day-Rate, Störungen/Carryover, Rollout-Overrides). Für die Evaluation würde ich folgende Struktur vorschlagen:

1. Primärmetriken (über alle Modelle/Varianten vergleichbar)

Gesamtkosten (MW ± SD, Min/Max) — Hauptzielgröße
Aufschlüsselung: Lohn-, Fahrt-, Ausfallkosten (zeigt warum sich Kosten unterscheiden)
Same-Day-Rate (%) — Servicequalität
Carryover-Anzahl — "Backlog"-Indikator
Simulationstage (Konsistenzcheck, sollte über Modelle ähnlich sein)
2. VergleichsachsenBasierend auf dem Overview-Log habt ihr schon die zentralen KPIs pro Seed (Gesamtkosten, Lohn/Fahrt/Ausfallkosten, Same-Day-Rate, Störungen/Carryover, Rollout-Overrides). Für die Evaluation würde ich folgende Struktur vorschlagen:

1. Primärmetriken (über alle Modelle/Varianten vergleichbar)

Gesamtkosten (MW ± SD, Min/Max) — Hauptzielgröße
Aufschlüsselung: Lohn-, Fahrt-, Ausfallkosten (zeigt warum sich Kosten unterscheiden)
Same-Day-Rate (%) — Servicequalität
Carryover-Anzahl — "Backlog"-Indikator
Simulationstage (Konsistenzcheck, sollte über Modelle ähnlich sein)
2. Vergleichsachsen

Basismodelle: Myopic vs Myopic+ vs CFA-Future vs DB-Simple (jeweils classical zone selection)
Zone-Selection: classical vs centrality vs value_based — pro Modell, um zu zeigen ob value-based wirklich was bringt
Rollout/VFA: jeweils Basismodell vs Basismodell+Rollout — Δ Gesamtkosten, plus Override-Quote (Replan-Rollout-Overrides als Anteil an Carryover) als Indikator, wie oft der Rollout überhaupt eingreift
3. Statistische Signifikanz

Da ihr 221 gepaarte Seeds habt: gepaarter t-Test oder Wilcoxon-Signed-Rank auf Gesamtkosten-Differenzen (Modell A vs B, gleicher Seed = gleiche Störungsrealisierung)
Effektgröße (Cohen's d) zusätzlich zu p-Werten, da MW-Unterschiede bei 221 Läufen schnell signifikant aber klein sein können
4. Visualisierungen

Boxplots/Violinplots der Gesamtkosten pro Modell/Variante (zeigt Verteilung + Ausreißer, aussagekräftiger als nur MW±SD)
Stacked Bar: Kostenkomponenten (Lohn/Fahrt/Ausfall) pro Modell
Scatter: Same-Day-Rate vs Gesamtkosten (Trade-off Servicequalität↔Kosten)
Für Rollout: Histogramm/Bar der Override-Häufigkeit + Boxplot Δ-Kosten (mit vs ohne Rollout, gleicher Seed)
Ggf. Lernkurve/Konvergenz für CFA-Future θ-Training (falls relevant für Methodikteil)
5. Sensitivitätsanalyse (falls Zeit)

δ-Variation bei DB-Simple (z.B. 0.3/0.5/0.7) — eure db_simple_rollout_0_8 / _h40_w1_0_7 Ordner deuten darauf hin, dass ihr das schon testet
n_zones, horizon_days/n_scenarios für Rollout

Basismodelle: Myopic vs Myopic+ vs CFA-Future vs DB-Simple (jeweils classical zone selection)
Zone-Selection: classical vs centrality vs value_based — pro Modell, um zu zeigen ob value-based wirklich was bringt
Rollout/VFA: jeweils Basismodell vs Basismodell+Rollout — Δ Gesamtkosten, plus Override-Quote (Replan-Rollout-Overrides als Anteil an Carryover) als Indikator, wie oft der Rollout überhaupt eingreift
3. Statistische Signifikanz

Da ihr 221 gepaarte Seeds habt: gepaarter t-Test oder Wilcoxon-Signed-Rank auf Gesamtkosten-Differenzen (Modell A vs B, gleicher Seed = gleiche Störungsrealisierung)
Effektgröße (Cohen's d) zusätzlich zu p-Werten, da MW-Unterschiede bei 221 Läufen schnell signifikant aber klein sein können
4. Visualisierungen

Boxplots/Violinplots der Gesamtkosten pro Modell/Variante (zeigt Verteilung + Ausreißer, aussagekräftiger als nur MW±SD)
Stacked Bar: Kostenkomponenten (Lohn/Fahrt/Ausfall) pro Modell
Scatter: Same-Day-Rate vs Gesamtkosten (Trade-off Servicequalität↔Kosten)
Für Rollout: Histogramm/Bar der Override-Häufigkeit + Boxplot Δ-Kosten (mit vs ohne Rollout, gleicher Seed)
Ggf. Lernkurve/Konvergenz für CFA-Future θ-Training (falls relevant für Methodikteil)
5. Sensitivitätsanalyse (falls Zeit)

δ-Variation bei DB-Simple (z.B. 0.3/0.5/0.7) — eure db_simple_rollout_0_8 / _h40_w1_0_7 Ordner deuten darauf hin, dass ihr das schon testet
n_zones, horizon_days/n_scenarios für Rollout
| myopic_value_based | Myopic | value_based | 500 | 40.596 € |
| myopic_plus_value_based | Myopic+ | value_based | 500 | 34.313 € |
| cfa_future_value_based | CFA-Future | value_based | 500 | 33.793 € |
| db_simple_value_based | DB-Simple | value_based | 500 | 33.671 € |

**logs/ergebnisse/alt/** (VFA = Rollout auf value_based-Basis):

| Variante | Modell | Zone-Selection | Läufe | MW Gesamtkosten | Ø Replan-Overrides |
|---|---|---|---|---|---|
| cfa_future_rollout | CFA-Future + Rollout | value_based | 500 | 33.749 € | 1,30 |
| db_simple_rollout | DB-Simple + Rollout | value_based | 500 | 33.734 € | 1,35 |

> **Alle 4 Rollout-Varianten laufen aktuell neu durch** (Myopic, Myopic+, CFA-Future, DB-Simple). Die Ergebnisse in `alt/` sind Vorgängerläufe; finale Auswertung erfolgt nach Abschluss der neuen Läufe.

### 1.2  Primäre Metriken pro Lauf (aus run_*.json)

| Feld | Bedeutung |
|---|---|
| `total_cost_eur` | Gesamtkosten = Lohn + Fahrt + Ausfall |
| `wage_cost_eur` | Lohnkosten (dominante Komponente) |
| `fuel_cost_eur` | Fahrtkosten |
| `downtime_cost_eur` | Ausfallkosten (Haupthebel der Policy) |
| `same_day_rate` | Anteil Störungen, die noch am selben Tag behoben wurden |
| `total_carryover` | Summe Carryover-Aufgaben über alle Tage |
| `total_disruptions` | Gesamte Störungsereignisse |
| `days_simulated` | Simulationstage (Konsistenzcheck) |
| `replan_overrides` | Replan-Entscheidungen, bei denen Rollout die Base-Policy überstimmt hat |
| `initial_overrides` | Analog für Initial-Plan (erwartet: 0) |

---

## 2  Vergleichsachsen

### 2.1  Achse A – Basismodelle (Centrality, gleiche Zone-Selection)

**Frage:** Wie gut sind die Policy-Mechanismen relativ zueinander, bei neutraler Zone-Selection?

**Varianten:** Myopic · Myopic+ · CFA-Future · DB-Simple (alle centrality)

**Erwartung:** Myopic schlechter (kein Prioritätsmechanismus); Myopic+, CFA-Future, DB-Simple ähnlich, da ihr Vorteil erst durch value_based Zone-Selection aktiviert wird.

**Ergebnis (bekannt):**
- Myopic: 48.023 € — klar schlechtester
- Myopic+, CFA-Future, DB-Simple: ~40.150–40.190 € — kaum Unterschied (Δ < 0,1 %)

### 2.2  Achse B – Zone-Selection (centrality vs. value_based, pro Modell)

**Frage:** Bringt value_based Zone-Selection einen messbaren Vorteil gegenüber centrality?

**Vergleiche (gepaart, gleiche Seeds):**

| Modell | Centrality | Value-based | Δ absolut | Δ relativ |
|---|---|---|---|---|
| Myopic | 48.023 € | 40.596 € | −7.427 € | −15,5 % |
| Myopic+ | 40.144 € | 34.313 € | −5.831 € | −14,5 % |
| CFA-Future | 40.187 € | 33.793 € | −6.394 € | −15,9 % |
| DB-Simple | 40.183 € | 33.671 € | −6.512 € | −16,2 % |

**Befund:** Value-based Zone-Selection reduziert Gesamtkosten bei allen 4 Modellen um ~15 % — getrieben fast ausschließlich durch niedrigere Ausfallkosten (Prioritätsmechanismus wählt die richtigen Zonen). Statistische Signifikanz mit gepaarten Tests zu prüfen (n=500).

**Wichtige Beobachtung – Myopic Value-based:** Myopic kann den value_based-Mechanismus theoretisch weniger gut ausnutzen, da es keine C̃-Bewertung hat — trotzdem ist die Verbesserung vergleichbar mit den anderen Modellen. Mögliche Erklärung: der Vorteil kommt primär aus der Zone-Selection (welche Zone wird besucht), nicht aus der detaillierten Policy-Entscheidung innerhalb der Zone.

### 2.3  Achse C – Basismodelle auf value_based Basis

**Frage:** Macht es einen Unterschied, welches Basismodell man bei gleicher Zone-Selection verwendet?

**Varianten:** Myopic · Myopic+ · CFA-Future · DB-Simple (alle value_based)

| Modell | MW | Δ zu Myopic |
|---|---|---|
| Myopic | 40.596 € | Referenz |
| Myopic+ | 34.313 € | −6.283 € (−15,5 %) |
| CFA-Future | 33.793 € | −6.803 € (−16,7 %) |
| DB-Simple | 33.671 € | −6.925 € (−17,1 %) |

Myopic+ vs CFA-Future vs DB-Simple sehr ähnlich (Δ < 1 %). Signifikanztest entscheidet ob diese Unterschiede real oder Rauschen sind.

### 2.4  Achse D – VFA-Rollout (base vs. base+Rollout, value_based)

**Frage:** Bringt der Rolling-Horizon-Rollout eine Verbesserung gegenüber der Basis-Policy?

**Vergleiche (gepaart, gleiche Seeds):**

| Modell | Ohne Rollout | Mit Rollout | Δ | Ø Overrides / Lauf |
|---|---|---|---|---|
| CFA-Future | 33.793 € | 33.749 € | −44 € (−0,13 %) | 1,30 |
| DB-Simple | 33.671 € | 33.734 € | +63 € (+0,19 %) | 1,35 |
| Myopic+ | – | – | ausstehend | – |
| Myopic | – | – | ausstehend | – |

**Erwarteter Befund:** Der Rollout greift selten ein (~1,3× pro Lauf bei ~150 Störungen/Lauf = < 1 % der Carryover-Entscheidungen). Der erwartete Kosteneffekt ist damit gering. Interessant: DB-Simple+Rollout ist minimal teurer als ohne — mögliche Ursache: Rollout-Kosten (Rechenzeit) führen zu leicht verzerrter Auswahl bei stochastischen Szenarien.

**Zusatzauswertung:** Override-Quote = Replan-Overrides / total_carryover als relativer Eingreifindikator.

---

## 3  Statistische Auswertung

### 3.1  Methodik

Alle Vergleiche mit **gepaarten Tests** (gleicher Seed = gleiche Störungsrealisierung = natürliches Matching):

- **Gepaarter t-Test** (parametrisch, Normalverteilung für n=500 gut erfüllt)
- **Wilcoxon-Vorzeichen-Rang-Test** (nicht-parametrisch, robuster gegenüber Ausreißern)
- **Cohen's d** als Effektgröße (wichtig: bei n=500 werden auch triviale Δ signifikant)

Schwellenwerte für Interpretation:
| Cohen's d | Interpretation |
|---|---|
| < 0,20 | vernachlässigbar |
| 0,20–0,50 | klein |
| 0,50–0,80 | mittel |
| > 0,80 | groß |

### 3.2  Erwartete Signifikanzen

| Vergleich | Erwartetes d | Erwartung |
|---|---|---|
| Myopic centrality vs. value_based | ~1,5 | hochsignifikant, großer Effekt |
| Myopic+ centrality vs. value_based | ~1,2 | hochsignifikant, großer Effekt |
| CFA-Future vs. DB-Simple (value_based) | < 0,05 | nicht signifikant |
| CFA-Future value_based vs. CFA-Future+Rollout | ~0,01 | nicht signifikant |

### 3.3  Automatisierung

`scripts/analysis/compare_models.py --dirs "Label=Pfad/json" … --out results/analysis/` erzeugt:
- `summary.csv` — MW, SD, Min, Max pro Variante
- `paired_tests.csv` — t-Test-p, Wilcoxon-p, Cohen's d für alle Paare mit gemeinsamen Seeds
- Alle Plots (siehe Abschnitt 4)

---

## 4  Visualisierungen

### 4.1  Hauptvergleichsplots

**Plot 1 – Boxplot Gesamtkosten (alle 10+ Varianten)**
- X-Achse: Varianten, gruppiert nach Achse (Centrality | Value-based | Rollout)
- Y-Achse: Gesamtkosten (€)
- Zeigt Median, IQR, Ausreißer; bei n=500 sehr stabile Verteilung

**Plot 2 – Stacked Bar: Kostenkomponenten**
- Pro Variante: MW Lohnkosten (blau) + Fahrtkosten (grün) + Ausfallkosten (rot)
- Verdeutlicht: Lohnkosten fast konstant (~22 k€); alle Gewinne aus Ausfallkosten

**Plot 3 – Boxplot Same-Day-Rate**
- Servicequalität analog zu Plot 1
- Erwartung: korreliert stark negativ mit Ausfallkosten

**Plot 4 – Scatter: Same-Day-Rate vs. Gesamtkosten**
- Jeder Punkt = ein Seed, eingefärbt nach Variante
- Zeigt Trade-off-Frontier; ideal: links-unten (hohe SDR, niedrige Kosten)

### 4.2  Zone-Selection fokussierte Plots

**Plot 5 – Δ-Kosten Centrality→Value-based, pro Modell**
- Bar-Chart: Kosteneinsparung in € und % pro Modell
- Zeigt: Alle 4 Modelle profitieren ähnlich stark

**Plot 6 – Ausfallkosten-Verteilung (Boxplot)**
- Nur Ausfallkosten; verdeutlicht woher die Einsparung kommt

### 4.3  Rollout / VFA fokussierte Plots

**Plot 7 – Δ-Kosten (base vs. base+Rollout)**
- Pro Seed: Kostendifferenz (gepaart), als Histogramm oder Boxplot
- Zeigt Verteilung der Rollout-Wirkung: wie oft hilft er, wie oft schadet er, wie oft neutral?

**Plot 8 – Override-Häufigkeit**
- Histogramm der Replan-Overrides pro Lauf (erwartet: viele 0–2, wenige > 4)
- Override-Quote = Overrides / total_carryover pro Lauf

**Plot 9 – Boxplot Δ-Kosten nach Override-Häufigkeit**
- Teile Läufe in Gruppen: 0 Overrides / 1–2 / 3+ Overrides
- Frage: Hilft Rollout mehr, wenn er öfter eingreift?

### 4.4  Modellmechanismus-Plots (optional, für Methodikteil)

**Plot 10 – CFA-Future θ-Konvergenz (Training)**
- θ-Werte über Trainingsiterationen; zeigt Konvergenz des kontrastiven Regressions-Trainings

**Plot 11 – DB-Simple δ-Sensitivität**
- Falls Ergebnisse für δ ∈ {0,3; 0,5; 0,7; 0,8} vorliegen (logs/db_simple_rollout_0_8 etc.)
- MW Gesamtkosten über δ; zeigt ob δ = 0,5 (= CFA-Future) optimal ist oder nicht

---

## 5  Ergebnisstruktur (Zusammenfassung)

### 5.1  Hauptergebnistabelle (für Thesis)

| Modell | Zone-Selection | MW Gesamt (€) | SD | MW Ausfall (€) | Same-Day-Rate | Cohen's d vs. Myopic-Centrality |
|---|---|---|---|---|---|---|
| Myopic | centrality | 48.023 | 5.397 | 24.819 | 63,4 % | Referenz |
| Myopic+ | centrality | 40.144 | 5.197 | 16.642 | 67,5 % | – |
| CFA-Future | centrality | 40.187 | 5.156 | 16.695 | 67,6 % | – |
| DB-Simple | centrality | 40.183 | 5.128 | 16.702 | 67,4 % | – |
| Myopic | value_based | 40.596 | 5.284 | 17.795 | 64,1 % | – |
| Myopic+ | value_based | 34.313 | 4.727 | 11.556 | 67,8 % | – |
| CFA-Future | value_based | 33.793 | 4.525 | 11.111 | 68,3 % | – |
| DB-Simple | value_based | 33.671 | 4.546 | 11.022 | 68,2 % | – |
| Myopic | value_based + Rollout | ausstehend | – | – | – | – |
| Myopic+ | value_based + Rollout | ausstehend | – | – | – | – |
| CFA-Future | value_based + Rollout | ausstehend (alt: 33.749) | – | – | – | – |
| DB-Simple | value_based + Rollout | ausstehend (alt: 33.734) | – | – | – | – |

### 5.2  Gepaarte Test-Matrix (Δ Gesamtkosten, p-Wert)

Wird durch `scripts/analysis/compare_models.py` erzeugt. Kernvergleiche:

| A | B | Erwartetes Δ (€) | Signifikanz |
|---|---|---|---|
| Myopic centrality | Myopic value_based | −7.427 | *** |
| Myopic+ centrality | Myopic+ value_based | −5.831 | *** |
| CFA-Future centrality | CFA-Future value_based | −6.394 | *** |
| DB-Simple centrality | DB-Simple value_based | −6.512 | *** |
| CFA-Future value_based | DB-Simple value_based | −122 | n.s. (erwartet) |
| CFA-Future value_based | CFA-Future+Rollout | −44 | n.s. (erwartet) |
| DB-Simple value_based | DB-Simple+Rollout | +63 | n.s. (erwartet) |

---

## 6  Offene Punkte / Ausstehend

- [ ] Rollout-Ergebnisse für alle 4 Modelle abwarten (Myopic, Myopic+, CFA-Future, DB-Simple laufen neu durch)
- [ ] δ-Sensitivitätsanalyse DB-Simple: Ergebnisse aus `logs/db_simple_rollout_0_8/` und `logs/db_simple_rollout_h40_w1_0_7/` einordnen (welches δ war das?)
- [ ] Prüfen ob `logs/ergebnisse/alt/` (cfa_future_rollout, db_simple_rollout) die endgültigen Rollout-Ergebnisse sind oder ob neue Läufe die alten ersetzen sollen
- [ ] Für θ-Konvergenzplot: Training-Log auswerten (`data/training/cfa_future/`)
- [ ] Entscheiden ob classic Zone-Selection (ohne centrality/value_based) als Baseline benötigt wird

---

## 7  Ausführungsplan

```bash
# Schritt 1: Zone-Selection-Vergleich (Centrality vs. Value-based, alle Modelle)
.venv/bin/python3 scripts/analysis/compare_models.py \
  --dirs "Myopic (centrality)=logs/ergebnisse/myopic_centrality/json" \
         "Myopic+ (centrality)=logs/ergebnisse/myopic_plus_centrality/json" \
         "CFA-Future (centrality)=logs/ergebnisse/cfa_future_centrality/json" \
         "DB-Simple (centrality)=logs/ergebnisse/db_simple_centrality/json" \
         "Myopic (value_based)=logs/ergebnisse/myopic_value_based/json" \
         "Myopic+ (value_based)=logs/ergebnisse/myopic_plus_value_based/json" \
         "CFA-Future (value_based)=logs/ergebnisse/cfa_future_value_based/json" \
         "DB-Simple (value_based)=logs/ergebnisse/db_simple_value_based/json" \
  --out results/analysis/zone_selection

# Schritt 2: VFA-Rollout-Vergleich (base vs. base+Rollout, nach Abschluss der 4 neuen Läufe)
# Pfade ggf. anpassen je nachdem wo die neuen Rollout-Läufe gespeichert werden
.venv/bin/python3 scripts/analysis/compare_models.py \
  --dirs "Myopic (value_based)=logs/ergebnisse/myopic_value_based/json" \
         "Myopic+Rollout=logs/ergebnisse/myopic_rollout/json" \
         "Myopic+ (value_based)=logs/ergebnisse/myopic_plus_value_based/json" \
         "Myopic++Rollout=logs/ergebnisse/myopic_plus_rollout/json" \
         "CFA-Future (value_based)=logs/ergebnisse/cfa_future_value_based/json" \
         "CFA-Future+Rollout=logs/ergebnisse/cfa_future_rollout/json" \
         "DB-Simple (value_based)=logs/ergebnisse/db_simple_value_based/json" \
         "DB-Simple+Rollout=logs/ergebnisse/db_simple_rollout/json" \
  --out results/analysis/rollout_comparison
```
