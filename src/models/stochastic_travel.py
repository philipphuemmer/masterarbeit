"""
Stochastisches Fahrzeitmodell für Monte-Carlo-Routenbewertung.

Modell: T_ij,h ~ Lognormal(mean=m_ij,h, cv=cv_h)
  - m_ij,h : deterministischer Matrixwert für Stunde h (aus traffic_matrices)
  - cv_h   : stundenbezogener Variationskoeffizient (aus config)

Lognormal garantiert positive Fahrzeiten; cv_h modelliert nur die
intra-hour-Variabilität — der stündliche Erwartungswert steckt bereits in m_ij,h.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StochasticRouteMetrics:
    """Stochastische KPIs für eine Teamroute (aggregiert über n_runs MC-Läufe)."""

    team_id: int
    n_runs: int
    det_end_min: float        # deterministisch geplante Rückkehrzeit [min ab 8:00]
    mean_end_min: float       # mittlere stochastische Endzeit
    p95_end_min: float        # 95%-Quantil der Endzeit
    overtime_prob: float      # P(Endzeit > Arbeitszeitende)
    mean_overtime_min: float  # E[max(0, Endzeit − Arbeitszeitende)]

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "n_runs": self.n_runs,
            "det_end_min": round(self.det_end_min, 2),
            "mean_end_min": round(self.mean_end_min, 2),
            "p95_end_min": round(self.p95_end_min, 2),
            "overtime_prob": round(self.overtime_prob, 4),
            "mean_overtime_min": round(self.mean_overtime_min, 2),
        }


def _get_hour(time_min: float, available_hours: list[int], lunch_earliest_min: int = 240) -> int:
    """
    Zeitstempel [min ab 8:00] → passende Matrixstunde.

    Ab lunch_earliest_min wird die Stunde um +1 verschoben — identisch zum VRPSolver.
    Das bildet die Mittagspause im Display-Zeitmodell ab: Rohminute 240 (12:00 raw)
    entspricht 13:00 Anzeige, also verwendet man die 13-Uhr-Matrix.
    """
    hour = 8 + int(max(0.0, time_min)) // 60
    if time_min >= lunch_earliest_min:
        hour += 1
    return max(available_hours[0], min(hour, available_hours[-1]))


def _get_cv(hour: int, cv_by_hour: dict[int, float]) -> float:
    """Variationskoeffizient für eine Stunde; Fallback auf nächste bekannte Stunde."""
    if hour in cv_by_hour:
        return cv_by_hour[hour]
    available = sorted(cv_by_hour.keys())
    if not available:
        return 0.15
    if hour < available[0]:
        return cv_by_hour[available[0]]
    return cv_by_hour[available[-1]]


def evaluate_route(
    node_sequence: list[int],
    service_mins: list[float],
    matrices: dict[int, np.ndarray],
    cv_by_hour: dict[int, float],
    n_runs: int,
    workday_min: int,
    rng: np.random.Generator,
    team_id: int = -1,
    det_end_min: float = float("nan"),
    lunch_duration_min: float = 0.0,
    lunch_earliest_min: int = 240,
) -> StochasticRouteMetrics:
    """
    Bewertet eine Teamroute stochastisch via Monte Carlo.

    Die geplante Stop-Reihenfolge wird als fix angenommen (deterministisch geplant).
    Für jeden Streckenabschnitt wird die Fahrtzeit als Lognormal-Realisierung gezogen.
    Die Stundenzuordnung je Abschnitt basiert auf der deterministisch akkumulierten Zeit
    (Approximation: Matrixauswahl ändert sich nicht pro MC-Run).

    Mittagspausenlogik identisch zum VRPSolver: ab lunch_earliest_min wird die Matrixstunde
    um +1 verschoben. Das stellt sicher, dass nach der Mittagspause die richtige
    Stunden-Matrix verwendet wird — unabhängig davon, ob lunch_duration_min > 0.

    Parameters
    ----------
    node_sequence      : Besuchsreihenfolge [node_idx, ...] ohne Depot am Anfang/Ende
    service_mins       : Wartungszeit je Stop (gleiche Länge wie node_sequence)
    matrices           : Stündliche Fahrzeitmatrizen in Sekunden
    cv_by_hour         : Variationskoeffizient je Stunde
    n_runs             : Anzahl MC-Simulationen
    workday_min        : Länge des Arbeitstags in Minuten (z.B. 480)
    rng                : NumPy-Zufallsgenerator
    team_id            : Team-ID für das Ergebnisobjekt
    det_end_min        : Deterministische geplante Rückkehrzeit (nur für Logging)
    lunch_duration_min : Mittagspausendauer [min] für echte Routing-Pausen (aus config)
    lunch_earliest_min : Ab welcher Rohminute die Matrixstunde um +1 verschoben wird (Default 240)
    """
    if not node_sequence:
        return StochasticRouteMetrics(
            team_id=team_id, n_runs=n_runs,
            det_end_min=0.0, mean_end_min=0.0, p95_end_min=0.0,
            overtime_prob=0.0, mean_overtime_min=0.0,
        )

    available_hours = sorted(matrices.keys())

    # Legs: Depot → s1 → s2 → ... → sK → Depot
    legs_from = [0] + list(node_sequence)
    legs_to   = list(node_sequence) + [0]
    n_legs = len(legs_from)

    # Deterministisch akkumulierte Abfahrtszeit pro Abschnitt (für Matrixstunden-Auswahl)
    # _get_hour verschiebt die Stunde um +1 ab lunch_earliest_min — wie VRPSolver.
    dep_min_per_leg: list[float] = []
    t = 0.0
    for i in range(n_legs):
        dep_min_per_leg.append(t)
        h = _get_hour(t, available_hours, lunch_earliest_min)
        mean_sec = float(matrices[h][legs_from[i], legs_to[i]])
        t += mean_sec / 60.0
        if i < len(service_mins):
            t += service_mins[i]

    # Für jeden Abschnitt N Samples aus Lognormal(mean=m_ij,h, cv=cv_h)
    # Form: (n_legs, n_runs) in Minuten
    samples = np.zeros((n_legs, n_runs))
    for i in range(n_legs):
        h = _get_hour(dep_min_per_leg[i], available_hours, lunch_earliest_min)
        mean_sec = float(matrices[h][legs_from[i], legs_to[i]])
        cv = _get_cv(h, cv_by_hour)
        if mean_sec <= 0.0 or cv <= 0.0:
            samples[i, :] = mean_sec / 60.0
        else:
            sigma_log = float(np.sqrt(np.log(1.0 + cv ** 2)))
            mu_log = float(np.log(mean_sec) - sigma_log ** 2 / 2.0)
            samples[i, :] = rng.lognormal(mu_log, sigma_log, size=n_runs) / 60.0

    total_travel = samples.sum(axis=0)             # (n_runs,) in Minuten
    total_service = float(sum(service_mins)) + lunch_duration_min
    end_times = total_travel + total_service

    overtime = np.maximum(0.0, end_times - workday_min)
    return StochasticRouteMetrics(
        team_id=team_id,
        n_runs=n_runs,
        det_end_min=det_end_min,
        mean_end_min=float(np.mean(end_times)),
        p95_end_min=float(np.percentile(end_times, 95)),
        overtime_prob=float(np.mean(end_times > workday_min)),
        mean_overtime_min=float(np.mean(overtime)),
    )
