"""
Value Function Approximation (VFA) Modell.

Approximiert den zukünftigen Wert eines Zustands durch eine gelernte
Wertfunktion V(s). Die VRP-Zielfunktion wird um V(s') erweitert,
sodass der Solver implizit den langfristigen Nutzen von Entscheidungen
berücksichtigt.

Die Wertfunktion wird durch ADP (Approximate Dynamic Programming)
oder Reinforcement Learning trainiert.
"""
from __future__ import annotations

import numpy as np

from src.planning import VRPSolver, MaintenanceTask, TeamState, DailyPlan


class VFAModel:
    """
    Value Function Approximation Modell.

    Parameters
    ----------
    duration_matrix : Reisezeitmatrix in Sekunden.
    config : Konfigurationsdict.
    value_function : Trainierte Wertfunktion V(state) → float.
                     None = untrainiertes Modell (verhält sich wie Myopic).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        value_function=None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config)
        self.config = config
        self.value_function = value_function

    def _compute_extra_costs(
        self,
        tasks: list[MaintenanceTask],
        team_states: list[TeamState],
    ) -> dict[int, int] | None:
        """
        Berechnet stationsabhängige Zusatzkosten aus der Wertfunktion.

        TODO: Zustandsmerkmale pro Station extrahieren und durch
        value_function bewerten. Differenz V(s') - V(s) als Zusatzkosten
        (negativ = Bonus für wertvolle Stationen).

        Gibt None zurück solange kein Modell trainiert ist.
        """
        if self.value_function is None:
            return None
        # TODO: Implementierung nach Definition der State-Repräsentation
        raise NotImplementedError

    def create_initial_plan(self, tasks: list[MaintenanceTask]) -> DailyPlan:
        extra_costs = self._compute_extra_costs(tasks, [])
        return self.solver.create_initial_plan(tasks, extra_costs=extra_costs)

    def replan(
        self,
        remaining_tasks: list[MaintenanceTask],
        team_states: list[TeamState],
    ) -> DailyPlan:
        extra_costs = self._compute_extra_costs(remaining_tasks, team_states)
        return self.solver.replan(remaining_tasks, team_states, extra_costs=extra_costs)
