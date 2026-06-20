# Route-based MDP – Wartungsroutenoptimierung EV-Ladesäulen

## Mengen und Indizes

- Menge der Ladesäulen: $I = \{1, \ldots, 397\}$
- Fahrzeuge/Teams: $K = \{1, 2\}$
- Depot/Startpunkt (WVV-Startpunkt): Knoten $0$
- Knotenmenge im Graphen: $V = \{0\} \cup I$
- $d_{ij} \geq 0$: Fahrzeit von Knoten $i$ nach Knoten $j$, $i, j \in V$ (stunden­abhängig)
- Zonen: $Z = \{1, \ldots, N_Z\}$, K-Means-Cluster der Stationen ($N_Z = 178$)

---

## Entscheidungspunkte

Die Simulation läuft auf einem **stündlichen Gitter** $H = \{8, 9, \ldots, 16\}$ Uhr. Es gibt zwei Typen von Entscheidungspunkten:

| Typ | Zeitpunkt | Auslöser | Aktion |
|---|---|---|---|
| **Initialplan** | $h = 8$ (Tagesbeginn) | immer | Zonenauswahl + vollständiger Tagesplan |
| **Replan** | $h \in \{8, \ldots, 16\}$ | neue Störung(en) in Stunde $h$ | Greedy-Einfügung der Störung(en) in bestehende Route |

Die **Mittagspause** (12:00–13:00) ist kein Entscheidungspunkt, sondern wird deterministisch nach dem ersten Stop mit Abfahrt $\geq 12{:}00$ eingefügt und verschiebt alle nachfolgenden Ankunftszeiten um 60 min.

---

## Zustandsvektor $s \in S$

Ein Zustand zum Entscheidungspunkt $h$ sei:
$$s_h = (\tau_h,\; X_h,\; Y_h,\; J_h,\; \Pi_h)$$

### 1. Zeit

$$\tau_h = (h - 8) \cdot 60 \in [0, 480] \quad \text{[Minuten ab 8:00]}$$

### 2. Fahrzeugzustände $X_h$

Für jedes Fahrzeug $k \in K$:
$$X_h^k = (v_h^k,\; \sigma_h^k,\; b_h^k)$$

- $v_h^k \in V$: aktueller Standort (letzter abgeschlossener Knoten oder Depot)
- $\sigma_h^k \in \{\text{wartung},\; \text{fahrend},\; \text{rückfahrt},\; \text{inaktiv}\}$: Aktivitätsstatus

  | Status | Bedeutung |
  |---|---|
  | `wartung` | Team arbeitet an einer Station (Routine, Typ 1 oder Typ 2) |
  | `fahrend` | Team fährt zum nächsten geplanten Stop |
  | `rückfahrt` | Team fährt nach letztem Stop zurück zum Depot |
  | `inaktiv` | Keine Stops geplant für heute |

- $b_h^k \geq 0$: verbleibende Planungszeit bis Depotpflicht­rückkehr  
  $$b_h^k = T_{\max} - \tau_h - d_{v_h^k,\, 0} \quad \text{[Minuten]}$$
  wobei $T_{\max} = 480$ min (16:00 Uhr).

### 3. Stationszustände $Y_h$

Für jede Säule $i \in I$:
$$Y_h^i = (m_h^i,\; \text{dsm}_h^i)$$

- $m_h^i \in \{0, 1\}$: $1$ = Jahreswartung in dieser Periode bereits abgeschlossen, $0$ = noch offen
- $\text{dsm}_h^i \geq 0$: Tage seit letzter Wartung (*days since maintenance*); wird täglich um $1$ erhöht und bei Serviceabschluss auf $0$ zurückgesetzt. Bestimmt die stochastische Ausfallwahrscheinlichkeit.

> **Hinweis:** Der technische Zustand einer Säule (ok / gestört) wird nicht als persistente Zustandsgröße modelliert. Störungen werden als Ereignisse generiert und sofort als Wartungsauftrag in $J_h$ eingestellt.

### 4. Offene Wartungsaufträge $J_h$

$$J_h = (W_h^{reg},\; W_h^{fault},\; W_h^{carry})$$

- $W_h^{reg} \subseteq I$: offene Jahreswartungen ($i \in W_h^{reg}$ genau dann wenn $m_h^i = 0$)
- $W_h^{fault} \subseteq I \times \{1, 2\}$: neue Störungsaufträge der aktuellen Stunde $h$
- $W_h^{carry} \subseteq I \times \{1, 2\}$: Carryover-Störungen vom Vortag (nicht erledigte Störungen); werden zu Tagesbeginn priorisiert eingeplant

### 5. Routenpläne $\Pi_h$

Für jedes Fahrzeug $k \in K$:
$$\Pi_h^k = (r_h^k,\; \theta_h^k)$$

- $r_h^k = (v_h^k,\; i_1^k,\; i_2^k,\; \ldots,\; i_{n_k}^k,\; 0)$: geplante Knotensequenz für den Rest des Tages inkl. Depotabschluss
- $\theta_h^k$: Ankunfts- und Abfahrtszeiten je Knoten (deterministisch auf Basis der Fahrzeitmatrizen $d_{ij}^h$ berechnet)

$$\Pi_h = (\Pi_h^1,\; \Pi_h^2)$$

---

## Aktionsraum $A(s)$

### Entscheidungspunkt Initialplan ($h = 8$)

Die Aktion umfasst zwei Stufen:

**Stufe 1 – Zonenauswahl (Vorfilter):**

1. Bewerte alle Zonen $z \in Z$ mit offenen Stationen nach Score:
   $$\text{score}(z) = w_{depot} \cdot \overline{d}_{z,0} + w_{area} \cdot A_z \quad \text{(klassisch)}$$
   oder wertbasiert: $\text{score}(z) = \sum_{i \in z,\, m^i = 0} V(i,\, \text{dsm}^i)$

2. Weise jedem Team $k$ eine Startzone zu (Top-20-Kandidaten, mind. 1 km Abstand zwischen Zonen)

3. Erweitere via Nearest-Neighbor-Expansion bis max. $M = 20$ Stationen pro Team

Das Ergebnis ist eine Kandidatenmenge $C^k \subseteq W_h^{reg}$ mit $|C^k| \leq M$ pro Team.

**Stufe 2 – Routenplanung:**

Die Aktion ist ein vollständiger Initialplan für beide Teams:
$$a_h^{init} = \Pi_h^{new} = (\Pi_h^{1,new},\; \Pi_h^{2,new})$$

Erstellt via greedy Cheapest-Insertion über $C^k \cup W_h^{carry}$. Stationen, die nicht mehr in den Arbeitstag passen, werden fallengelassen (Drops).

### Entscheidungspunkt Replan ($h \in \{9, \ldots, 16\}$)

Bei Eingang neuer Störungen $W_h^{fault}$:
$$a_h^{replan} = \Pi_h^{new}$$

Greedy-Einfügung jeder neuen Störung $j \in W_h^{fault}$ in die Route des Teams mit geringsten Zusatzkosten. Ist keine Einplanung mehr möglich ($b_h^k < s_j + d_{v_h^k,j} + d_{j,0}$ für alle $k$), wird $j$ als Carryover $W_{h+1}^{carry}$ vorgemerkt.

### Durchführbarkeitsbedingungen

1. **Zeitliche Machbarkeit**: Für jedes $k$ gilt $\theta_h^k[\text{Depot}] \leq T_{\max}$

2. **Depotpflicht**: Jede Route endet in Knoten $0$: $r_h^{k,new}$ endet in $0$, $\forall k$

3. **Exklusivität**: Für jede Station $i$ und jeden Zeitpunkt $\tau$ arbeitet höchstens ein Team:
   $$\sum_{k \in K} \mathbf{1}\{\text{Team } k \text{ arbeitet an } i \text{ zur Zeit } \tau\} \leq 1$$

---

## Übergangsfunktion $P(s_{h+1} \mid s_h, a_h)$

### Routenfortschritt

Zwischen $\tau_h$ und $\tau_{h+1}$ werden Stops abgearbeitet:
- Ankunfts- und Abfahrtszeiten gemäß $\theta_h^k$
- Bei stochastischen Fahrtzeiten: Realisierung via Lognormal($\mu_{ij}^h$, $\sigma_{ij}^h$)

### Serviceabschlussregeln

**Jahreswartung (Routine):**
$$m_{h+1}^i = 1, \quad \text{dsm}_{h+1}^i = 0$$

**Typ-1-Störung** (Standardreparatur, $s_{typ1} = 60$ min):
$$\text{dsm}_{h+1}^i = 0$$

**Typ-2-Störung** (Schaden mit Teiletausch):

Die Servicezeit wird zu Störungsbeginn als Gesamtpaket berechnet:
$$s_{typ2}(i) = s_{ab} + \frac{d_{i,0}^h + d_{0,i}^h}{60} + s_{handling} + s_{an}$$

mit $s_{ab} = 30\;\text{min}$, $s_{handling} = 5\;\text{min}$, $s_{an} = 30\;\text{min}$.

> **Modellvereinfachung**: Die physische Depotfahrt (Beschaffung des Ersatzteils) wird nicht als separates Routen-Leg modelliert, sondern als Zeitanteil in $s_{typ2}(i)$ aufgeschlagen. Das Team bleibt konzeptuell an der Säule; der Routenplan enthält keinen Zwischenstop am Depot.

Nach Abschluss:
$$\text{dsm}_{h+1}^i = 0$$

### Days-since-maintenance Update (täglich, Tagesende)

$$\text{dsm}_{d+1}^i = \begin{cases} 0 & \text{falls Station } i \text{ heute gewartet} \\ \text{dsm}_d^i + 1 & \text{sonst} \end{cases}$$

### Stochastische Störungsgenerierung

Für jede Stunde $h \in H$ und Säule $i \in I$ (max. eine Störung pro Säule und Tag):

$$p_k^i(h) = p_{k,base} \cdot f(\text{dsm}^i) \cdot \eta^i, \quad k \in \{1, 2\}$$

mit Recovery-Kurve:
$$f(t) = \lambda + (1 - \lambda) \cdot \frac{\min(t,\; \tau_{rec})}{\tau_{rec}}$$

wobei $\lambda$ = initialer Faktor nach Wartung, $\tau_{rec}$ = Erholungszeit [Tage], $\eta^i$ = stationsindividueller Ausfallskalierungsfaktor.

### Carryover

Nicht eingeplante Störungen werden als $W_{d+1}^{carry}$ in den nächsten Tag übertragen.

---

## Belohnungsfunktion $R$

$$R = -\left(C_{op} + C_{dt}\right)$$

**Operative Kosten** (Lohn + Kraftstoff):
$$C_{op} = \underbrace{n_{active} \cdot T_{shift} \cdot c_L}_{\text{Lohnkosten}} + \underbrace{\sum_{\text{Fahrten } (i,j)} \text{km}_{ij} \cdot c_F}_{\text{Fahrtkosten}}$$

- $n_{active}$: Anzahl Teams mit mind. einem Stop
- $T_{shift} = 8\;\text{h}$ (voller Arbeitstag; am letzten Tag tatsächliche Arbeitszeit)
- $c_L = 35\;\text{€/h}$, $c_F = 0{,}30\;\text{€/km}$

**Ausfallkosten** (nicht bediente Störungen):
$$C_{dt} = \sum_{j \in \text{carried}} \Delta\tau_j^{wait} \cdot P_j \cdot c_{dt}$$

- $\Delta\tau_j^{wait}$: Wartezeit [h] bis Serviceabschluss (bzw. Restarbeitstag für Carryover)
- $P_j$: Nennleistung der gestörten Säule [kW]
- $c_{dt} = 0{,}50\;\text{€/kWh}$

---

## Alt (frühere Formulierung, nicht aktiv)

**Zustandsraum (*S*)**

$D$ Tage:
$$d \in \{1,\ldots,D\}$$
8 Arbeitsstunden:
$$t \in \{0,\ldots,8\}$$
Zustand $s_{d,t}$ am Tag $d$ zur Stunde $t$
$$s_{d,t} = \left( \mathbf{V}_{d,t}, \mathbf{W}_{d,t}, \mathbf{F}_{d,t}, \tau_t \right)$$
Wartungsvektor ($1 =$ Säule $j$ wurde in dieser Periode bereits gewartet, $0 =$ noch offen)
$$\mathbf{W}_{d,t} \in \{0, 1\}^{|V|}$$
Fehlerstatus-Vektor ($0=$ OK, $1=$ Teildefekt, $2=$ Totalausfall)
$$\mathbf{F}_{d,t} \in \{0, 1, 2\}^{|V|}$$

**Aktionsraum (*A*)**

Aktion wählt die nächsten Zielknoten
$$a_{d,t} = (a_1, a_2)$$

Zeit-Constraint: Die Aktion muss vor 16:00 Uhr abschließbar sein
$$\text{dist}(v_i, a_i) + \text{service\_time}(a_i) \leq \tau_t$$

**Übergangsfunktion (*P*)**

Fahrzeuge starten am nächsten Morgen wieder am Depot
$$\mathbf{V}_{d+1,0} = (\text{Depot}, \text{Depot})$$
