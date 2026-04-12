"""
CFA Light Modell – V̂-basierte Drop-Entscheidung + Cheapest-Insertion Routing.

Hierarchie:
    Initialplan  : OR-Tools mit Soft-Deadlines (wie MyopicPlus)
    Drop-Entscheid: nach V̂ (gelernt) – niedrigster Skip-Penalty zuerst droppen
    Routing      : Greedy Cheapest-Insertion (wie Myopic)

Initialplan (identisch zu MyopicPlus):
    deadline(k) = rank(k) / n_tasks × WORKDAY_MINUTES
    deadline_penalty(k) = α × power_kW(k) × p_h × downtime_eur_per_kwh / wage_eur_per_min

Drop-Entscheidung (V̂-basiert):
    V̂(stop) = skip_penalty ∝ power_kW × remaining_hours
    → Stop mit geringstem V̂ wird zuerst aus der Route entfernt
    → Hochwertige Stationen bleiben länger in der Route

Routing nach Drop:
    Cheapest-Insertion: Störungsknoten wird an günstigster Position eingefügt
    → Kein OR-Tools Replan → schneller, deterministischer
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cost_params import CostParams
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    SimRoute,
    SimStop,
    _fmt,
)
from src.planning.clustering import _approx_km
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, VRPSolver

logger = logging.getLogger(__name__)


class CFALightModel:
    """
    CFA Light: V̂-basierte Drop-Entscheidung mit Cheapest-Insertion Routing.

    Parameters
    ----------
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden.
    config : dict
        Konfigurationsdict aus config.yaml.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    stations_df : pd.DataFrame | None
        Stationsdaten mit Spalte „Nennleistung Ladeeinrichtung [kW]".
        None → Fallback-Nennleistung 22 kW für alle Stationen.
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: np.ndarray,
        stations_df: Optional[pd.DataFrame] = None,
        cost_params: Optional[CostParams] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.cost_params = cost_params or CostParams()
        self.n_teams: int = config["maintenance"]["n_teams"]

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60

        # Nennleistung pro node_idx (node_idx = DataFrame-Zeile + 1, Depot = 0)
        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        if stations_df is not None and pwr_col in stations_df.columns:
            self.node_to_power: dict[int, float] = {
                i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
                for i, (_, row) in enumerate(stations_df.iterrows())
            }
        else:
            self.node_to_power = {}

        # Ausfallwahrscheinlichkeit pro Stunde (Typ 1 + Typ 2)
        fail_cfg = config.get("failure_simulation", {})
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )

        # Skalierungsfaktor α
        cfa_cfg = config.get("cfa", {})
        self.alpha: float = float(cfa_cfg.get("alpha", 10.0))

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _deadline_penalty(self, power_kw: float) -> int:
        """Strafkosten in Minuten/Minute Deadline-Überschreitung."""
        cp = self.cost_params
        wage_per_min = cp.wage_eur_per_hour / 60.0
        penalty = self.alpha * power_kw * self.p_failure_per_hour * cp.downtime_eur_per_kwh / wage_per_min
        return max(1, int(round(penalty)))

    def _skip_penalty(self, power_kw: float, remaining_hours: float, service_time: int = 45) -> float:
        """
        V̂(stop): Erwartete Kosten des Überspringens einer Routine-Station.

        Höherer Wert → Station ist wertvoller → nicht droppen.
        """
        cp = self.cost_params
        wage_per_min = cp.wage_eur_per_hour / 60.0
        downtime_eur = self.alpha * power_kw * self.p_failure_per_hour * remaining_hours * cp.downtime_eur_per_kwh
        return service_time + max(0.0, downtime_eur / wage_per_min)

    def _get_matrix(self, time_min: float) -> np.ndarray:
        """Gibt die passende Stundenmatrix für einen Zeitstempel zurück."""
        hour = 8 + int(max(0.0, time_min)) // 60
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Setzt Soft-Deadlines für Routine-Tasks basierend auf Nennleistung.

        Hochleistungs-Stationen erhalten frühe Deadlines mit hohem Penalty
        pro Minute Überschreitung → OR-Tools plant sie bevorzugt früh ein.
        """
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        n = len(routine_tasks)

        if n > 0:
            depot = self.all_coords[0]
            sorted_routine = sorted(
                routine_tasks,
                key=lambda t: _approx_km(self.all_coords[t.node_idx], depot),
            )
            for rank, task in enumerate(sorted_routine):
                task.soft_deadline_min = int((rank + 1) / n * self.WORKDAY_MINUTES)
                task.deadline_penalty = self._deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
                )

        logger.info(
            f"CFA Light Initialplan: {len(tasks)} Tasks, "
            f"{n} Routine mit Soft-Deadlines."
        )
        return self.solver.create_initial_plan(tasks, team_assignment=team_assignment)

    def handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        Fügt Störungen per Cheapest-Insertion ein; Drop-Reihenfolge nach V̂.

        Reihenfolge: aufsteigend nach Insertionskosten (günstigste zuerst).
        """
        matrix = self._get_matrix(time_min)
        carryover: list[DisruptionEvent] = []
        downtime_cost = 0.0
        handled = 0

        # Initiale Kostenbewertung für Sortierung
        queue: list[tuple[float, DisruptionEvent]] = []
        for d in disruptions:
            best = self._find_best_insertion(d, sim_routes, time_min, hour, matrix)
            cost = best[0] if best is not None else np.inf
            queue.append((cost, d))

        queue.sort(key=lambda x: x[0])

        for _, d in queue:
            matrix_fresh = self._get_matrix(time_min)
            best = self._find_best_insertion(d, sim_routes, time_min, hour, matrix_fresh)

            if best is not None:
                cost, team_idx, pos, arrival_at_d = best
                self._insert_disruption(d, sim_routes[team_idx], pos, time_min)
                log.actions.append(
                    f"Eingefuegt (direkt): {d.disruption_type} @ Node {d.node_idx} -> "
                    f"Team {sim_routes[team_idx].team_id}, "
                    f"Ankunft {_fmt(arrival_at_d)}, "
                    f"Zusatzkosten {cost:.2f} EUR"
                )
            else:
                drop_result = self._find_best_drop_and_insert(
                    d, sim_routes, time_min, hour
                )
                if drop_result is not None:
                    cost, team_idx, drop_global_indices, insert_pos, arrival_at_d = drop_result
                    dropped_nodes = [
                        sim_routes[team_idx].stops[i].node_idx for i in drop_global_indices
                    ]
                    for dg in sorted(drop_global_indices, reverse=True):
                        self._remove_stop_and_recompute(sim_routes[team_idx], dg)
                    self._insert_disruption(d, sim_routes[team_idx], insert_pos, time_min)
                    n_d = len(dropped_nodes)
                    log.actions.append(
                        f"Eingefuegt ({n_d} Routine-Stop(s) ausgebaut via V̂: {dropped_nodes}): "
                        f"{d.disruption_type} @ Node {d.node_idx} -> "
                        f"Team {sim_routes[team_idx].team_id}, "
                        f"Ankunft {_fmt(arrival_at_d)}, "
                        f"Zusatzkosten {cost:.2f} EUR"
                    )
                else:
                    carryover.append(d)
                    log.actions.append(
                        f"Carryover (kein Routine-Stop ausreichend): "
                        f"{d.disruption_type} @ Node {d.node_idx}"
                    )
                    continue

            report_min = float((hour - 8) * 60)
            wait_h = max(0.0, (arrival_at_d - report_min) / 60.0)
            d_cost = wait_h * d.power_kw * self.cost_params.downtime_eur_per_kwh
            downtime_cost += d_cost
            if d_cost > 0:
                log.actions.append(f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)")
            handled += 1

        return handled, carryover, downtime_cost

    # ------------------------------------------------------------------
    # Greedy Insertion
    # ------------------------------------------------------------------

    def _find_best_insertion(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        matrix: np.ndarray,
    ) -> Optional[tuple[float, int, int, float]]:
        """Findet die kostengünstigste Einfügeposition über alle Teams."""
        best_cost = np.inf
        best_team_idx: Optional[int] = None
        best_pos: Optional[int] = None
        best_arrival = 0.0

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            current_node = route.current_node_at(time_min)
            current_dep = route.current_departure_at(time_min)
            if route.lunch_end_min is not None and current_dep < route.lunch_end_min:
                current_dep = route.lunch_end_min

            for pos in range(len(remaining) + 1):
                cost, feasible, arrival = self._insertion_cost(
                    d, remaining, current_node, current_dep, pos, hour, matrix
                )
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_team_idx = ti
                    best_pos = pos
                    best_arrival = arrival

        if best_team_idx is None:
            return None
        return best_cost, best_team_idx, best_pos, best_arrival

    def _insertion_cost(
        self,
        d: DisruptionEvent,
        remaining: list[SimStop],
        current_node: int,
        current_dep: float,
        pos: int,
        hour: int,
        matrix: np.ndarray,
    ) -> tuple[float, bool, float]:
        """Berechnet Kosten und Machbarkeit einer Einfügeposition."""
        cp = self.cost_params
        d_node = d.node_idx

        if pos == 0:
            prev_node = current_node
            prev_dep = current_dep
        else:
            prev = remaining[pos - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        next_node = remaining[pos].node_idx if pos < len(remaining) else 0

        t_prev_d = matrix[prev_node, d_node] / 60.0
        t_d_next = matrix[d_node, next_node] / 60.0
        t_prev_next = matrix[prev_node, next_node] / 60.0

        extra_travel = t_prev_d + t_d_next - t_prev_next
        arrival_at_d = prev_dep + t_prev_d
        total_extra = extra_travel + d.service_min

        if remaining:
            if pos < len(remaining):
                new_last_dep = remaining[-1].departure_min + total_extra
                last_node = remaining[-1].node_idx
            else:
                new_last_dep = arrival_at_d + d.service_min
                last_node = d_node
            end_time = new_last_dep + matrix[last_node, 0] / 60.0
        else:
            end_time = arrival_at_d + d.service_min + matrix[d_node, 0] / 60.0

        feasible = end_time <= self.solver.WORKDAY_MINUTES

        extra_km = (
            _approx_km(self.all_coords[prev_node], self.all_coords[d_node])
            + _approx_km(self.all_coords[d_node], self.all_coords[next_node])
            - _approx_km(self.all_coords[prev_node], self.all_coords[next_node])
        )
        extra_time_h = total_extra / 60.0
        report_min = float((hour - 8) * 60)
        downtime_h = max(0.0, (arrival_at_d - report_min) / 60.0)

        cost = (
            extra_time_h * cp.wage_eur_per_hour
            + max(0.0, extra_km) * cp.fuel_eur_per_km
            + downtime_h * d.power_kw * cp.downtime_eur_per_kwh
        )
        return cost, feasible, arrival_at_d

    def _find_best_drop_and_insert(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
    ) -> Optional[tuple[float, int, list[int], int, float]]:
        """
        V̂-basierte Drop-Entscheidung + Cheapest-Insertion.

        Routine-Stops werden nach aufsteigendem V̂-Wert (Skip-Penalty) sortiert.
        Der Stop mit dem niedrigsten V̂ wird zuerst entfernt – d.h. Stationen,
        deren Wartungsausfall am wenigsten kostet, werden geopfert.

        Returns
        -------
        (kosten, team_idx, [drop_global_indices], insert_pos, arrival_at_d)
        oder None falls kein Drop eine Lösung ermöglicht.
        """
        best_cost = np.inf
        best: Optional[tuple[float, int, list[int], int, float]] = None
        matrix = self._get_matrix(time_min)
        remaining_hours = max(0.0, (self.WORKDAY_MINUTES - time_min) / 60.0)

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            routine_in_remaining = [
                (i, s) for i, s in enumerate(remaining) if s.task_type == "routine"
            ]

            if not routine_in_remaining:
                continue

            # V̂-Sortierung: niedrigster Skip-Penalty zuerst droppen
            routine_sorted_by_value = sorted(
                routine_in_remaining,
                key=lambda x: self._skip_penalty(
                    self.node_to_power.get(x[1].node_idx, 22.0),
                    remaining_hours,
                    int(x[1].service_min),
                ),
            )

            current_node = route.current_node_at(time_min)
            current_dep = route.current_departure_at(time_min)
            if route.lunch_end_min is not None and current_dep < route.lunch_end_min:
                current_dep = route.lunch_end_min

            dropped_in_remaining: list[int] = []

            for n_drop in range(1, len(routine_sorted_by_value) + 1):
                # Nächsten billigsten Stop laut V̂ droppen
                dropped_in_remaining.append(routine_sorted_by_value[n_drop - 1][0])
                drop_set = set(dropped_in_remaining)

                # Bereinigte Route mit neu berechneten Zeiten
                trimmed: list[SimStop] = []
                prev_n = current_node
                prev_d = current_dep
                for i, s in enumerate(remaining):
                    if i in drop_set:
                        continue
                    mat_h = self._get_matrix(prev_d)
                    new_arr = prev_d + mat_h[prev_n, s.node_idx] / 60.0
                    trimmed.append(SimStop(
                        node_idx=s.node_idx,
                        task_type=s.task_type,
                        arrival_min=new_arr,
                        service_min=s.service_min,
                    ))
                    prev_n = s.node_idx
                    prev_d = new_arr + s.service_min

                found = False
                for pos in range(len(trimmed) + 1):
                    cost, feasible, arrival = self._insertion_cost(
                        d, trimmed, current_node, current_dep, pos, hour, matrix
                    )
                    if feasible and cost < best_cost:
                        drop_globals = [
                            route.stops.index(remaining[i]) for i in dropped_in_remaining
                        ]
                        best_cost = cost
                        best = (cost, ti, drop_globals, pos, arrival)
                        found = True

                if found:
                    break

        return best

    def _insert_disruption(
        self,
        d: DisruptionEvent,
        route: SimRoute,
        pos: int,
        time_min: float,
    ) -> None:
        """Fügt Störungs-Stop an Position pos (in remaining) ein."""
        remaining = route.remaining_stops_at(time_min)

        if remaining:
            first_global = route.stops.index(remaining[0])
        else:
            first_global = len(route.stops)
        global_insert = first_global + pos

        matrix = self._get_matrix(time_min)
        if pos == 0:
            prev_node = route.current_node_at(time_min)
            prev_dep = route.current_departure_at(time_min)
            if route.lunch_end_min is not None and prev_dep < route.lunch_end_min:
                prev_dep = route.lunch_end_min
        else:
            prev = remaining[pos - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        arrival = prev_dep + matrix[prev_node, d.node_idx] / 60.0
        new_stop = SimStop(
            node_idx=d.node_idx,
            task_type=d.disruption_type,
            arrival_min=arrival,
            service_min=d.service_min,
            days_since_maintenance=0.0,
        )
        route.stops.insert(global_insert, new_stop)

        for i in range(global_insert + 1, len(route.stops)):
            prev_s = route.stops[i - 1]
            curr_s = route.stops[i]
            mat = self._get_matrix(prev_s.departure_min)
            curr_s.arrival_min = prev_s.departure_min + mat[prev_s.node_idx, curr_s.node_idx] / 60.0

    def _remove_stop_and_recompute(self, route: SimRoute, global_idx: int) -> None:
        """Entfernt Stop und berechnet alle Folge-Ankunftszeiten neu."""
        route.stops.pop(global_idx)
        if global_idx >= len(route.stops):
            return

        if global_idx == 0:
            prev_node = 0
            prev_dep = 0.0
        else:
            prev = route.stops[global_idx - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        for i in range(global_idx, len(route.stops)):
            curr = route.stops[i]
            mat = self._get_matrix(prev_dep)
            curr.arrival_min = prev_dep + mat[prev_node, curr.node_idx] / 60.0
            prev_node = curr.node_idx
            prev_dep = curr.departure_min
