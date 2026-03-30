"""
Cost Function Approximation (CFA) Modell.

Erweitert die VRP-Zielfunktion um eine parametrisierte Kostenfunktion,
die zukünftige Auswirkungen von Planentscheidungen approximiert.
Zum Beispiel: Stationen, die schwer erreichbar oder störungsanfällig sind,
erhalten einen Bonus (negativer Zusatzkosten), der ihre Besuchspriorität erhöht.

Die Parameter der Kostenfunktion werden durch Simulation oder
Policy Gradient-Methoden gelernt.
"""
from __future__ import annotations

import numpy as np

from src.planning import VRPSolver, MaintenanceTask, TeamState, DailyPlan


class CFAModel:
    """
    Cost Function Approximation Modell.

    Parameters
    ----------
    duration_matrix : Reisezeitmatrix in Sekunden.
    config : Konfigurationsdict.
    cfa_weights : Gewichtsvektor der Kostenfunktion (wird gelernt).
                  None = untrainiertes Modell (verhält sich wie Myopic).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        cfa_weights: np.ndarray | None = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config)
        self.config = config
        self.cfa_weights = cfa_weights

    def _compute_extra_costs(
        self,
        tasks: list[MaintenanceTask],
        team_states: list[TeamState],
    ) -> dict[int, int] | None:
        """
        Berechnet stationsabhängige Zusatzkosten aus der Kostenfunktion.

        TODO: Features pro Station definieren und mit cfa_weights gewichten.
        Gibt None zurück solange kein Modell trainiert ist.
        """
        if self.cfa_weights is None:
            return None
        # TODO: Implementierung nach Definition der Feature-Basis
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
