# Signifikanzanalyse: Centrality vs. Value-Based Zonenwahl

Monte-Carlo-Analyse mit **500 Läufen (Seeds 1–500)** je Variante.  
Test: **Wilcoxon Signed-Rank** (gepaart per Seed).

---

## Übersicht: Mittlere Gesamtkosten

### Centrality

| Modell | MW (€) | SD (€) | Median (€) |
|---|---:|---:|---:|
| Myopic | 40.410 | 5.139 | 39.996 |
| Myopic+ | 40.144 | 5.202 | 39.753 |
| CFA-Future | 40.187 | 5.161 | 39.400 |
| DB-Simple | 40.183 | 5.133 | 39.674 |

### Value-Based

| Modell | MW (€) | SD (€) | Median (€) |
|---|---:|---:|---:|
| Myopic | 34.885 | 4.829 | 34.242 |
| Myopic+ | 34.313 | 4.732 | 33.838 |
| CFA-Future | 33.793 | 4.529 | 33.043 |
| DB-Simple | 33.671 | 4.550 | 32.930 |

### Rollout (Value-Based + Rolling-Horizon)

| Modell | MW (€) | SD (€) | Median (€) |
|---|---:|---:|---:|
| Myopic | — | — | — |
| Myopic+ | — | — | — |
| CFA-Future | 33.729 | 4.489 | 32.997 |
| DB-Simple | 33.627 | 4.573 | 32.860 |

*Myopic / Myopic+ Rollout ausstehend.*

---

## Effekt der Zonenwahl: Centrality vs. Value-Based

Value-Based schlägt Centrality bei **allen vier Modellen hochsignifikant** (~15–16 % Kostenreduktion).

| Modell | MW Cent (€) | MW VB (€) | Δ (€) | Δ relativ | p-Wert | |
|---|---:|---:|---:|---:|---|---|
| Myopic | 40.410 | 34.885 | −5.525 | −13,7 % | < 0,001 | *** |
| Myopic+ | 40.144 | 34.313 | −5.831 | −14,5 % | < 0,001 | *** |
| CFA-Future | 40.187 | 33.793 | −6.394 | −15,9 % | < 0,001 | *** |
| DB-Simple | 40.183 | 33.671 | −6.512 | −16,2 % | < 0,001 | *** |

---

## Modellvergleiche innerhalb Centrality

Alle Modelle sind bei Centrality-Zonenwahl **statistisch ununterscheidbar** — die Zonenwahl dominiert das Ergebnis vollständig.

| Vergleich | Δ (€) | p-Wert | |
|---|---:|---|---|
| Myopic vs. Myopic+ | +266 | 0,137 | ns |
| Myopic vs. CFA-Future | +223 | 0,157 | ns |
| Myopic vs. DB-Simple | +227 | 0,369 | ns |
| Myopic+ vs. CFA-Future | −43 | 0,739 | ns |
| Myopic+ vs. DB-Simple | −39 | 0,488 | ns |
| CFA-Future vs. DB-Simple | +4 | 0,732 | ns |

---

## Modellvergleiche innerhalb Value-Based

Die Modelle unterscheiden sich signifikant — **außer CFA-Future und DB-Simple** (p = 0,57).

| Vergleich | Δ (€) | p-Wert | |
|---|---:|---|---|
| Myopic vs. Myopic+ | +573 | < 0,001 | *** |
| Myopic vs. CFA-Future | +1.092 | < 0,001 | *** |
| Myopic vs. DB-Simple | +1.214 | < 0,001 | *** |
| Myopic+ vs. CFA-Future | +520 | 0,002 | ** |
| Myopic+ vs. DB-Simple | +642 | < 0,001 | *** |
| CFA-Future vs. DB-Simple | +122 | 0,568 | ns |

---

## Rollout vs. Value-Based (Effekt des Rolling-Horizon)

Der Rollout bringt eine kleine, aber **hochsignifikante** Verbesserung gegenüber dem reinen Value-Based-Lauf.

| Modell | MW VB (€) | MW Rollout (€) | Δ (€) | Δ relativ | p-Wert | |
|---|---:|---:|---:|---:|---|---|
| Myopic | 34.885 | — | — | — | — | — |
| Myopic+ | 34.313 | — | — | — | — | — |
| CFA-Future | 33.793 | 33.729 | −64 | −0,19 % | < 0,001 | *** |
| DB-Simple | 33.671 | 33.627 | −44 | −0,13 % | < 0,001 | *** |

### Modellvergleich innerhalb Rollout

| Vergleich | Δ (€) | p-Wert | |
|---|---:|---|---|
| CFA-Future vs. DB-Simple | +102 | 0,478 | ns |

---

## Fazit

- **Zonenwahl ist der dominante Effekt**: Value-Based spart ~15–16 % gegenüber Centrality (p < 0,001).
- **Centrality**: Kein Modell unterscheidet sich signifikant — die Modellwahl ist irrelevant.
- **Value-Based**: Myopic ist signifikant schlechter als alle anderen; CFA-Future und DB-Simple liegen gleichauf.
- **Rollout**: Bringt eine statistisch signifikante, aber sehr kleine Verbesserung (~0,1–0,2 %) gegenüber Value-Based; CFA-Future und DB-Simple Rollout sind ebenfalls gleichauf.

---

*Signifikanzniveaus: \*\*\* p < 0,001 · \*\* p < 0,01 · \* p < 0,05 · ns nicht signifikant*
