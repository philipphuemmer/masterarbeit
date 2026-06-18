# Evaluation Chapters — Metrics, Tables, Figures

Konkrete Inhalte für Kapitel 7.1–7.4 | Stand: 2026-06-17

Nummerierung in dieser Datei: Abschnitte nach Gliederung (7.1–7.4),
Tabellen als Tab. 1–5, Abbildungen als Abb. 1–8 (LaTeX nummeriert automatisch).

Reihenfolge: erst Modellvergleich (7.1), dann Zone-Selection als Erklärung (7.2) —
narrative Logik: Policies entwickeln → zeigen was sie können → Zone-Selection als überraschender Haupthebel.

---

## 7.1 Policy Mechanism Comparison (2 S.)

**Kernaussage:** Myopic+ ≈ CFA-Future ≈ DB-Simple, Myopic klar schlechter. Die Rangfolge gilt bei beiden Zone-Selection-Modi — nur das absolute Niveau verschiebt sich.

### Tab. 1 — Modellvergleich bei Centrality

| Modell | MW Gesamt (€) | SD (€) | MW Ausfall (€) | Same-Day-Rate | Δ zu Myopic |
|---|---|---|---|---|---|
| Myopic | 48.023 | 5.397 | 24.819 | 63,4 % | — |
| Myopic+ | 40.144 | 5.197 | 16.642 | 67,5 % | −7.879 € (−16,4 %) |
| CFA-Future | 40.187 | 5.156 | 16.695 | 67,6 % | −7.836 € (−16,3 %) |
| DB-Simple | 40.183 | 5.128 | 16.702 | 67,4 % | −7.840 € (−16,3 %) |

### Tab. 2 — Modellvergleich bei Value-based

| Modell | MW Gesamt (€) | SD (€) | MW Ausfall (€) | Same-Day-Rate | Δ zu Myopic |
|---|---|---|---|---|---|
| Myopic | 40.596 | 5.284 | 17.795 | 64,1 % | — |
| Myopic+ | 34.313 | 4.727 | 11.556 | 67,8 % | −6.283 € (−15,5 %) |
| CFA-Future | 33.793 | 4.525 | 11.111 | 68,3 % | −6.803 € (−16,7 %) |
| DB-Simple | 33.671 | 4.546 | 11.022 | 68,2 % | −6.925 € (−17,1 %) |

### Abb. 1 — Grouped Boxplot: Alle 8 Varianten nach Modell gruppiert

- 4 Gruppen (Modelle), je 2 Boxen (centrality dunkel, value_based hell)
- Zeigt gleichzeitig: Rangfolge der Modelle (Policy-Effekt) und vertikaler Abstand zwischen den beiden Boxen pro Modell (Zone-Selection-Effekt als Vorschau auf 7.2)
- Kernbefund: Myopic+ / CFA-Future / DB-Simple liegen bei beiden Modi auf einer Linie; Myopic fällt ab

### Abb. 2 — Scatter: Same-Day-Rate vs. Gesamtkosten

- Jeder Punkt = ein Seed (500 pro Variante), Farbe = Modell, Form = Zone-Selection-Modus
- Zeigt Trade-off-Frontier: Myopic+ / CFA-Future / DB-Simple bei value_based erreichen dieselbe Frontier
- Myopic liegt bei beiden Modi klar schlechter (links-oben im Scatter)

**Text erklärt:** Warum Myopic+ ≈ CFA ≈ DB trotz unterschiedlicher Scoring-Funktionen — und warum Myopic auch bei value_based schlechter bleibt (kein Prioritätsmechanismus beim Routing innerhalb der Zone).

**Script-Aufruf:**
```bash
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
```

---

## 7.2 Impact of Zone Selection on Cost (2 S.)

**Kernaussage:** Value-based Zone-Selection reduziert Gesamtkosten bei allen 4 Modellen um ~15 %, unabhängig vom Modell — getrieben ausschließlich durch Ausfallkosten. Zone-Selection ist der dominante Hebel, nicht der Policy-Mechanismus.

### Tab. 3 — Kostenvergleich Centrality vs. Value-based

Alle 8 Varianten. Pro Modell zwei Zeilen (centrality, value_based), Δ-Spalten zeigen die Einsparung.

| Modell | Zone-Selection | MW Gesamt (€) | MW Ausfall (€) | MW Lohn (€) | Same-Day-Rate | Δ Gesamt (€) | Δ (%) |
|---|---|---|---|---|---|---|---|
| Myopic | centrality | 48.023 | 24.819 | 22.060 | 63,4 % | — | — |
| Myopic | value_based | 40.596 | 17.795 | 21.602 | 64,1 % | −7.427 | −15,5 % |
| Myopic+ | centrality | 40.144 | 16.642 | 22.286 | 67,5 % | — | — |
| Myopic+ | value_based | 34.313 | 11.556 | 21.546 | 67,8 % | −5.831 | −14,5 % |
| CFA-Future | centrality | 40.187 | 16.695 | 22.271 | 67,6 % | — | — |
| CFA-Future | value_based | 33.793 | 11.111 | 21.520 | 68,3 % | −6.394 | −15,9 % |
| DB-Simple | centrality | 40.183 | 16.702 | 22.267 | 67,4 % | — | — |
| DB-Simple | value_based | 33.671 | 11.022 | 21.491 | 68,2 % | −6.512 | −16,2 % |

### Abb. 3 — Stacked Bar: Kostenkomponenten

- X-Achse: 8 Varianten, gruppiert nach Modell
- Balken gestapelt: Lohnkosten (blau) / Fahrtkosten (grün) / Ausfallkosten (rot)
- Zeigt: Lohnkosten konstant ~22 k€ über alle Varianten; Ausfallkosten halbieren sich bei value_based

### Abb. 4 — Boxplot Gesamtkosten (centrality vs. value_based)

- 8 Boxen nebeneinander, gruppiert: linke Hälfte centrality, rechte Hälfte value_based
- Zeigt Verteilung über 500 Seeds — Effekt ist konsistent, kein Artefakt des Mittelwerts
- Nebenaussage: Carryover sinkt bei value_based von ~53 auf ~48 (im Text erwähnen)

**Text erklärt:** Die Einsparung von ~6 k€ Ausfallkosten entsteht weil value_based die Zonen nach erwartetem Schaden priorisiert, nicht nach geografischer Zentralität. Lohn- und Fahrtkosten bleiben konstant — die Teams fahren gleich viel, nur andere Zonen zuerst.

**Script-Aufruf:** Identisch mit 7.1 (gleiche `compare_models.py`-Ausgabe).

---

## 7.3 VFA Rollout: Performance and Limitations (2 S.)

**Kernaussage:** Der Rollout greift bei ~1 % der Carryover-Entscheidungen ein; der Kosteneffekt ist statistisch nicht signifikant. Die Override-Quote ist zu gering für einen messbaren Gesamteffekt.

### Tab. 4 — Rollout-Effekt pro Modell

Vorläuferwerte für CFA-Future und DB-Simple aus `logs/ergebnisse/alt/`; alle 4 finale Läufe ausstehend.

| Modell | MW ohne Rollout (€) | MW mit Rollout (€) | Δ (€) | Δ (%) | Ø Replan-Overrides | Override-Quote¹ |
|---|---|---|---|---|---|---|
| Myopic | — | — | — | — | — | — |
| Myopic+ | — | — | — | — | — | — |
| CFA-Future | 33.793 | 33.749 | −44 | −0,13 % | 1,30 | ~2,7 % |
| DB-Simple | 33.671 | 33.734 | +63 | +0,19 % | 1,35 | ~2,8 % |

¹ Override-Quote = Ø Replan-Overrides / Ø total_carryover (~48)

### Abb. 5 — Histogramm Replan-Overrides pro Lauf

- Separate Histogramme für alle 4 Rollout-Varianten (oder überlagert)
- X-Achse: Anzahl Overrides (0, 1, 2, 3, …), Y-Achse: Häufigkeit über 500 Läufe
- Zeigt: Mehrheit der Läufe hat 0–2 Overrides — erklärt direkt warum Gesamteffekt gering ist

### Abb. 6 — Boxplot Δ-Kosten (gepaart, base vs. base+Rollout)

- Pro Seed: Δ = Kosten_base − Kosten_rollout (positiv = Rollout hat geholfen)
- X-Achse: 4 Modelle, Y-Achse: Δ in €
- Referenzlinie bei 0; Verteilung zeigt: Rollout hilft manchmal, schadet manchmal, im Mittel neutral

### Abb. 7 — Δ-Kosten nach Override-Häufigkeit (optional)

- Läufe in 3 Gruppen: 0 Overrides / 1–2 / 3+
- Boxplot Δ-Kosten pro Gruppe: prüft ob Rollout systematisch besser abschneidet wenn er öfter eingreift

**Script-Aufruf:**
```bash
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

> Pfade anpassen sobald die 4 neuen Rollout-Läufe abgeschlossen sind.

---

## 7.4 Statistical Validation (2 S.)

**Kernaussage:** Zone-Selection-Effekte sind hochsignifikant mit großer Effektstärke (Cohen's d > 1,2). Policy- und Rollout-Unterschiede sind nicht signifikant (d < 0,05).

### Methodik (kurz im Text)

- **Gepaarte Tests:** gleicher Seed = gleiche Störungsrealisierung = natürliches Matching
- **Gepaarter t-Test** (parametrisch) + **Wilcoxon-Vorzeichen-Rang-Test** (nicht-parametrisch, robust gegen Ausreißer)
- **Cohen's d** als Effektgröße — bei n=500 werden auch triviale Unterschiede signifikant, d ist das eigentliche Entscheidungskriterium

| Cohen's d | Interpretation |
|---|---|
| < 0,20 | vernachlässigbar |
| 0,20–0,50 | klein |
| 0,50–0,80 | mittel |
| > 0,80 | groß |

### Tab. 5 — Gepaarte Test-Matrix (Gesamtkosten)

| Vergleich | n Seeds | MW Δ (€) | Cohen's d | t-Test p | Wilcoxon p | Befund |
|---|---|---|---|---|---|---|
| Myopic: centrality → value_based | 500 | −7.427 | ~1,40 | < 0,001 | < 0,001 | *** groß |
| Myopic+: centrality → value_based | 500 | −5.831 | ~1,23 | < 0,001 | < 0,001 | *** groß |
| CFA-Future: centrality → value_based | 500 | −6.394 | ~1,41 | < 0,001 | < 0,001 | *** groß |
| DB-Simple: centrality → value_based | 500 | −6.512 | ~1,43 | < 0,001 | < 0,001 | *** groß |
| CFA-Future vs. DB-Simple (value_based) | 500 | −122 | ~0,03 | n.s. | n.s. | nicht sig. |
| Myopic+ vs. CFA-Future (value_based) | 500 | −520 | ~0,11 | — | — | nicht sig. |
| CFA-Future: value_based vs. +Rollout | 500 | −44 | ~0,01 | n.s. | n.s. | nicht sig. |
| DB-Simple: value_based vs. +Rollout | 500 | +63 | ~0,01 | n.s. | n.s. | nicht sig. |

> Cohen's d-Werte sind Schätzungen; exakte Werte kommen aus `paired_tests.csv`.

### Abb. 8 — Cohen's d Übersicht (horizontales Bar-Chart)

- Alle Vergleiche aus Tab. 5 als horizontale Balken, sortiert nach Effektstärke absteigend
- Vertikale Referenzlinien bei d = 0,20 / 0,50 / 0,80
- Zeigt auf einen Blick: Zone-Selection-Vergleiche klar im "groß"-Bereich, alle anderen unter 0,20

**Quelle:** manuell aus `paired_tests.csv` (noch nicht in `compare_models.py` — bei Bedarf ergänzen).

---

## Übersicht: Woher kommt was

| Inhalt | Quelle |
|---|---|
| Tab. 1–2 | `summary.csv` aus `compare_models.py` |
| Tab. 3 | `summary.csv` aus `compare_models.py` |
| Tab. 4 | `summary.csv` + `runs_combined.csv` (Override-Spalten) |
| Tab. 5 | `paired_tests.csv` aus `compare_models.py` |
| Abb. 1 | `boxplot_gesamtkosten.png` (Gruppierung nach Modell) |
| Abb. 2 | `scatter_same_day_vs_kosten.png` |
| Abb. 3 | `stacked_bar_kostenkomponenten.png` |
| Abb. 4 | `boxplot_gesamtkosten.png` (Gruppierung nach Zone-Selection) |
| Abb. 5 | `bar_rollout_overrides.png` |
| Abb. 6, 7 | manuell aus `runs_combined.csv` (gepaarte Δ-Spalten) |
| Abb. 8 | manuell aus `paired_tests.csv` |
