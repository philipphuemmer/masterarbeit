# Analyse-Befunde: Mechanismen und Beispiele für Kapitel 7

Erstellt: 2026-07-07  
Zweck: Detaillierte Aufbereitung der Kausalanalyse hinter den MC-Ergebnissen, als Grundlage für die Erweiterung von Kapitel 7.

---

## 1. Warum performt CFA besser als Myopic? (unter value-based Zone Selection)

### 1.1 Statistische Übersicht (500 Runs, value-based)

| Kennzahl | Wert |
|---|---|
| CFA besser als Myopic | 306/500 Runs (61,2 %) |
| Myopic besser als CFA | 194/500 Runs (38,8 %) |
| Mittlerer CFA-Vorteil | 1.092 € |
| Median CFA-Vorteil (wenn CFA besser) | 2.515 € |
| Median Myopic-Vorteil (wenn Myopic besser) | 1.667 € |

Der Unterschied ist nicht deterministisch: CFA gewinnt in 61 % der Runs, verliert aber in 39 %. Der Vorteil kommt nur zum Tragen, wenn die stochastischen Ausfallmuster mit dem Wert-Signal der CFA-Policy zusammenpassen.

### 1.2 Die drei Wirkmechanismen

**Mechanismus 1: Unterschiedliches Wert-Signal bei der Zone Selection**

Obwohl beide Policies „value-based zone selection" nutzen, verwenden sie unterschiedliche Wertfunktionen:

- **Myopic**: `zone_value(station) = power_kW × recovery_curve(dsm)`
  - Einfaches Produkt aus Leistung und Recovery-Kurve
- **CFA**: `zone_value(station) = θᵀ × φ_scaled(k)` mit φ = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others]
  - Gelernte Gewichtung über 4 Features (inkl. Alter der Station und geografische Isoliertheit)

Das führt dazu, dass CFA und Myopic auf denselben Daten **verschiedene Zonen auswählen**. CFA bevorzugt Zonen mit Stationen, die gleichzeitig hohe Leistung, hohes Alter und isolierte Lage haben (schwerere Ausfälle, schlechter durch andere Touren abdeckbar).

Konkretes Beispiel aus Seed 497, Tag 1:
- Myopic Team 1: Nodes 3, 21, 22, 1, 2 (geografischer Cluster A)
- CFA Team 1: Nodes 66, 56, 329, 330, 331 (geografischer Cluster B — komplett anderer Bereich)

Die Zone-Selection-Ebene ist die wirksamste Differenzierungsquelle zwischen beiden Policies.

**Mechanismus 2: Unterschiedliche Reihenfolge innerhalb der Zone**

Innerhalb einer zugewiesenen Zone bestimmt die `route_score_fn` die Reihenfolge der Routine-Stops:

- **Myopic**: `score = 1 / dist(current, node)` — rein entfernungsbasiert
- **CFA**: `score = (C̃(node) + shift) / dist(current, node)` — wertgewichtet

CFA besucht innerhalb einer Zone zuerst Stationen mit hohem approximiertem Kostenwert `C̃`, auch wenn sie etwas weiter weg liegen. Myopic besucht immer die nächste Station zuerst.

Beispiel (Seed 497, Tag 1, Team 0):
- Myopic: 214 → 215 → 216 → 218 → 217 (aufsteigend nach Distanz)
- CFA: 218 → 217 → 216 → 327 → 328 (218 hat höheres C̃, wird vorgezogen; dann wechselt CFA in eine andere Unterzone)

**Mechanismus 3: Unterschiedliche Drop-Entscheidungen bei Replanning**

Wenn eine neue Störung eintritt und ein Routine-Stop fallengelassen werden muss:

- **Myopic**: `drop_score = 1 / detour(k)` — der Stop mit dem kleinsten Umweg wird fallengelassen
- **CFA**: `drop_score = C̃(k) - wage_per_min × detour(k)` — der Stop mit dem niedrigsten Nettowert (Wert minus Umwegkosten) wird fallengelassen

CFA schützt Stationen mit hohem `C̃` auch beim Replanning. Myopic kann unbeabsichtigt hochwertige Stationen fallenlassen, wenn sie günstig auf dem Weg lagen (kleiner Umweg), aber trotzdem wirtschaftlich wichtig sind.

### 1.3 Der Kaskadeneffekt: Konkretes Beispiel (Seed 497)

Dies ist der wichtigste mechanistische Befund: die Unterschiede akkumulieren sich über Tage.

**Ausgangslage:**
- Seed 497 ist ein extremer Fall: CFA spart 15.300 € gegenüber Myopic
- Myopic Total: 52.491 € (davon 27.871 € Downtime)
- CFA Total: 37.191 € (davon 13.775 € Downtime)
- Differenz kommt fast ausschließlich aus Downtime-Kosten

**Tag-für-Tag-Vergleich (erste 15 Tage):**

| Tag | M-Disrupt | C-Disrupt | M-Downtime | C-Downtime | M-Total | C-Total | Δ (M−C) |
|-----|-----------|-----------|------------|------------|---------|---------|---------|
| 1   | 8         | 8         | 804 €      | 1.037 €    | 1.395 € | 1.637 € | −242 €  |
| 2   | 4         | 4         | 710 €      | 819 €      | 1.298 € | 1.399 € | −101 €  |
| 3   | 4         | 5         | 2.219 €    | 1.245 €    | 2.813 € | 1.841 € | +972 €  |
| 4   | 6         | 9         | 2.392 €    | 1.302 €    | 2.989 € | 1.904 € | +1.085 € |
| 5   | 5         | 4         | 1.516 €    | 1.426 €    | 2.123 € | 2.033 € | +90 €   |
| 6   | 2         | 2         | 2.465 €    | 1.775 €    | 3.064 € | 2.380 € | +684 €  |
| 7   | 4         | 4         | 1.885 €    | 1.061 €    | 2.475 € | 1.647 € | +827 €  |
| 8   | 4         | 6         | 599 €      | 1.123 €    | 1.185 € | 1.704 € | −519 €  |
| 9   | **9**     | **4**     | **2.895 €**| **694 €**  | 3.479 € | 1.275 € | **+2.204 €** |
| 10  | 4         | 3         | **4.481 €**| **241 €**  | 5.067 € | 835 €   | **+4.232 €** |
| 11  | 3         | 2         | 2.316 €    | 147 €      | 2.917 € | 737 €   | +2.181 € |
| 12  | 6         | 4         | 1.759 €    | 205 €      | 2.372 € | 810 €   | +1.562 € |
| 13  | 1         | 0         | 175 €      | 16 €       | 755 €   | 604 €   | +151 €  |

**Kumulierte Störungen Tage 1–12:**
- Myopic: 59 Störungen
- CFA: 55 Störungen

**Kumulierte Downtime-Kosten Tage 1–12:**
- Myopic: 24.041 €
- CFA: 11.076 €

**Interpretation:** Die Anzahl der Störungen ist fast identisch (59 vs. 55), aber die **Downtime-Kosten** unterscheiden sich massiv. Das bedeutet: CFA besucht dieselbe Anzahl von Stationen, aber die mit CFA besuchten Stationen haben — im Durchschnitt — eine **niedrigere Leistung oder kürzere Downtime-Dauer** wenn sie ausfallen. Oder: CFA hat die teuren Hochleistungsstationen bereits in Tagen 1–8 besucht und deren DSM reduziert, sodass diese auf Tag 9 eine niedrigere Ausfallwahrscheinlichkeit haben.

**Das eigentliche Erklärungsmuster:** Durch die CFA-Zone Selection (θᵀφ) werden ab Tag 1 gezielt Hochwertstationen besucht. Deren DSM sinkt. Da `p(failure) ∝ recovery_curve(dsm)`, steigt die Robustheit dieser Stationen. Wenn an Tag 9 die gleiche Poisson-Ausfallrate wie für Myopic gilt, fallen bei CFA weniger (und günstigere) Stationen aus — weil die teuren 350+-kW-Stationen already abgearbeitet wurden.

Bei Myopic bleiben diese Hochwertstationen länger im Carryover-Backlog. Wenn sie ausfallen, generieren sie massive Downtime-Kosten (z.B. 4.481 € an einem einzigen Tag 10 für Myopic vs. 241 € bei CFA).

---

## 2. Warum kann der Rollout das Ergebnis verschlechtern?

### 2.1 Häufigkeit und Ausmaß der Verschlechterung (alle 500 Runs)

| Policy | Rollout schlechter | Max. Verschlechterung | Median Verschlechterung | Rollout besser | Max. Verbesserung | Mittlerer Δ |
|--------|-------------------|----------------------|------------------------|---------------|------------------|-------------|
| Myopic | 109/500 (21,8 %)  | +5.139 €             | +268 €                 | 156/500 (31,2 %) | −3.928 €       | +5 €        |
| Myopic+| 122/500 (24,4 %)  | +4.132 €             | +213 €                 | 135/500 (27,0 %) | −5.960 €       | −19 €       |
| CFA    | 121/500 (24,2 %)  | +4.274 €             | +226 €                 | 231/500 (46,2 %) | −5.396 €       | −64 €       |
| DB     | 119/500 (23,8 %)  | +6.260 €             | +282 €                 | 218/500 (43,6 %) | −4.279 €       | −44 €       |

**Schlüsselbeobachtungen:**
- In **22–24 % aller Runs** ist der Rollout schlechter als die Basis-Policy — unabhängig von der Policy
- Die Standardabweichung des Rollout-Deltas liegt bei ±786–891 €, was zeigt, dass einzelne Runs sehr stark in beide Richtungen abweichen können
- Der **Mittelwert** der Verbesserung ist fast null (+5 € bis −64 €), aber die **Streuung** ist erheblich
- Die Lage ist **symmetrisch**: was der Rollout auf der Verlustseite riskiert (bis +6.260 €) ist ähnlich groß wie was er auf der Gewinnseite bringt (bis −5.960 €) — aber die mittlere Verbesserung reicht nicht, um den Overhead zu rechtfertigen

**Schlechteste Einzelfälle pro Policy:**
- Myopic: Seeds 6 (+5.139 €), 166 (+5.114 €), 18 (+4.311 €)
- Myopic+: Seeds 476 (+4.132 €), 147 (+3.428 €), 405 (+3.187 €)
- CFA: Seeds 164 (+4.274 €), 284 (+3.838 €), 118 (+3.461 €)
- DB: Seeds 87 (+6.260 €), 305 (+5.618 €), 184 (+4.827 €)

### 2.2 Konkretes Beispiel: Myopic Seed 6 (+5.139 €)

Dies ist der deutlichste Einzelfall für Myopic (zweitschlechtester ist Seed 166 mit fast identischem Ausmaß).

**Gesamtbild:**

| Kennzahl | Basis (Myopic) | Rollout | Δ |
|---|---|---|---|
| Total Cost | 40.433 € | 45.572 € | +5.139 € |
| Downtime Cost | 15.724 € | 20.376 € | +4.652 € |
| Carryover Tasks (kumuliert) | 64 | 80 | +16 |
| Replan Overrides im gesamten Run | — | **1** | — |

**Der einzige Override:** Tag 11, Stunde 9:

```
Replan-Rollout-Override: drop 318 statt 66
Horizont-Erwartungskosten: 20.875 € (für drop-318)
Win-Rate: 76 %
```

Das bedeutet: Das Monte Carlo hat über 25 Szenarien × 40 Tage evaluiert. In 19 von 25 Szenarien war „drop Station 318" besser als „drop Station 66". Die Basis-Policy (Myopic, 1/detour) hätte Station 66 fallengelassen.

**Kosten-Kaskade nach dem Override:**

| Tag | Basis-Kosten | Rollout-Kosten | Δ |
|-----|-------------|----------------|---|
| 10  | 980 €       | 980 €          | ±0 € (vor Override — identisch) |
| 11  | 1.802 €     | 1.877 €        | +75 € |
| 12  | 1.541 €     | 2.448 €        | **+907 €** |
| 13  | 1.405 €     | 2.388 €        | **+983 €** |
| 14  | 658 €       | 792 €          | +134 € |
| 15  | 650 €       | 742 €          | +92 € |
| 16  | 780 €       | 820 €          | +39 € |
| 17  | 771 €       | 816 €          | +44 € |

**Node-Tracking:**

Station 318 ist eine Hochleistungsstation. Sie erscheint in beiden Runs ab Tag 3 regelmäßig im Initial-Plan, wird aber durch Störungen immer wieder in den Carryover geschoben (Tage 3–10: wiederholt geplant und gedropt). Im BASE-Run wird Station 66 auf Tag 11 Stunde 9 gedropt (weil eine neue Störung eintrifft und Myopic 66 mit dem kleinsten 1/detour-Score auswählt). Im ROLLOUT-Run wird stattdessen Station 318 gedropt (Override). 

Beide Entscheidungen landen letztlich wieder im gleichen Problem — beide Stationen kommen in die nächsten Tagespläne und werden wieder gedropt. Aber die **Routing-Konsequenzen** sind verschieden: Die Reihenfolge, in der Carryover-Tasks in Folgetagen bearbeitet werden, verändert sich, und damit auch, welche Hochleistungsstation an Tag 12 und 13 noch im Rückstand ist und ihre Ausfallwahrscheinlichkeit akkumuliert.

**Die Erklärung:** Der Rollout lieferte mit 25 Szenarien eine Win-Rate von 76 % für „drop 318". Das klingt überzeugend — aber: 
- 25 Szenarien bedeuten, dass jedes Szenario eine zufällig gezogene Zukunft simuliert
- Die **echte** Zukunft (der tatsächliche Seed) war eine Realisierung aus dem 24%-Bereich, wo „drop 66" besser wäre
- Monte Carlo mit n=25 hat eine erhebliche Schätzunsicherheit — ein „76 % Win-Rate"-Signal kann im Einzelfall falsch liegen

**Quantifizierung der Schätzunsicherheit:** Bei einer wahren Win-Rate von 50:50 würde man mit 25 Szenarien in ~12 % der Fälle eine gemessene Win-Rate ≥ 76 % sehen (Binomialverteilung). Das heißt: selbst wenn „drop 318" und „drop 66" gleichwertig wären, würde der Rollout in ~12 % aller Fälle eine scheinbar klare Empfehlung von ≥ 76 % ausgeben — und sich trotzdem irren.

### 2.3 Der strukturelle Mechanismus: Warum 40-Tage-Rollout trotzdem versagen kann

1. **Finite-Sample-Rauschen:** 25 Szenarien sind zu wenig, um die Erwartungswertsschätzung für `C(drop-318)` vs. `C(drop-66)` zuverlässig zu trennen, wenn die wahre Differenz klein ist. Das zeigt sich daran, dass der mittlere Rollout-Effekt nahe null ist: Die Gewinne und Verluste heben sich fast auf.

2. **Winner's Curse:** Der Kandidat, der in den 25 Szenarien „gewinnt", kann das getan haben, weil er zufällig in günstigen Szenarien überrepräsentiert war (Sampling-Zufall), nicht weil er strukturell besser ist.

3. **Planungshorizont trifft Stochastik:** Über 40 Tage summiert sich die Stochastizität. Kleine Anfangsvorteile einer Entscheidung können durch spätere Zufallsereignisse (anders verteilte Störungen) kompensiert oder überkompensiert werden. Der Rollout kann nur die 25 gesampelten Zukünfte sehen — der reale Lauf kann in einer anderen Zukunft landen.

4. **Kaskadenwirkung durch Carryover:** Ein falscher Drop erhöht den DSM der fallengelassenen Station. Wenn diese Station hohe Leistung hat, steigt ihre Ausfallwahrscheinlichkeit. In den Folgetagen kommen mehr Störungen → mehr Carryover → weitere Kosten. Eine einzige schlechte Entscheidung kann sich über 10–15 Tage fortpflanzen (wie in Seed 6 sichtbar: Hauptmehrkosten an Tagen 12 und 13).

### 2.4 Warum CFA dennoch häufiger profitiert

Obwohl ~24 % der CFA-Runs schlechter werden, **profitieren 46 % der Runs** (vs. nur 31 % bei Myopic). Das liegt nicht daran, dass der Rollout bei CFA weniger Schaden anrichtet — er schadet genauso oft (~24 %). Aber:

- CFA's Drop-Score (`C̃(k) − wage × detour`) ist **differenzierter** als Myopic's `1/detour`. Die Kandidaten haben unterschiedlichere Scores, d.h. der Rollout findet häufiger eine echte Verbesserung gegenüber dem Greedy-Vorschlag.
- Bei Myopic sind die Drop-Kandidaten oft nahezu gleichwertig nach `1/detour` — der Rollout überschreibt den Greedy selten (0,78 Overrides/Run vs. 1,27 bei CFA), aber wenn er es tut, ist das Risiko, falsch zu liegen, genauso groß.
- **Netto-Effekt:** Mehr Overrides bei CFA = mehr Chancen auf Verbesserung, aber auch mehr Chancen auf Verschlechterung. Wegen des höherwertigen Ausgangssignals überwiegen bei CFA die Verbesserungen.

---

## 3. Faktischer Fehler in Kapitel 7 (Abschnitt Limitations, Zeile 931)

**Aktueller Text (falsch):**
> In the current setting ($k = 3$, 30-day horizon, 5 scenarios per candidate)

**Korrekte Parameter** (aus `configs/config.yaml` und `rolling_horizon_meta` in den JSON-Runs):

```yaml
rolling_horizon:
  horizon_days: 40
  n_scenarios: 25
  top_k_candidates: 3
```

**Korrekte Formulierung:**
> In the current setting ($k = 3$, 40-day horizon, 25 scenarios per candidate)

---

## 4. Einordnung: Was in Kapitel 7 fehlt

### Aktuell (Ist-Stand)

Die Kapitel 7.1–7.4 beschreiben die Ergebnisse korrekt und statistisch valide. Was fehlt, ist die **kausale Tiefe**:

| Fragestellung | Behandelt? |
|---|---|
| Was sind die Kosten-Unterschiede? | ✅ vollständig (Tab. 1–4, Abb. 1–5) |
| Warum ist value-based Zone Selection so viel besser? | ✅ qualitativ (Abschnitt 7.2 Discussion) |
| Warum genau performt CFA besser als Myopic unter value-based? | ⚠️ nur abstrakt erwähnt |
| Welche konkreten Mechanismen unterscheiden die Policies? | ❌ fehlt |
| Wie oft und wie stark verschlechtert Rollout ein Ergebnis? | ❌ fehlt (nur Durchschnitt, keine Verteilung) |
| Warum kann Rollout das Ergebnis verschlechtern? | ⚠️ nur strukturell erklärt, kein Beispiel |
| Wie viele Overrides führen zu Verschlechterung? | ❌ fehlt |

### Zu ergänzen (Soll)

**In 7.2 Discussion:**
- Paragraph: Unterschiedliche Wert-Signale der Policies führen zu unterschiedlicher Zone Selection, auch wenn alle „value-based" nutzen. Der DSM-Reduktions-Kaskadeneffekt als eigentlicher Mechanismus.

**In 7.3 Discussion:**
- Paragraph mit Häufigkeitsverteilung: 21–24 % der Läufe schlechter, Std. ≈ ±800–890 €, max. Einzelverlust +6.260 €
- Konkretes Beispiel Seed 6: Ein Override (Tag 11, Win-Rate 76 %) → +5.139 € Kaskade über Tage 12–26
- Erklärung der Kausalität: Monte Carlo-Schätzrauschen + Kaskadeneffekt des Carryovers

**In 7.1 Discussion (optional):**
- Kurzer Hinweis, dass unter Centrality die unterschiedlichen Routing-Scores kaum Effekt haben, weil das Stationspool keine DSM-Differenzierung erlaubt

**Fehler beheben:**
- Zeile 931: `30-day horizon, 5 scenarios` → `40-day horizon, 25 scenarios`
