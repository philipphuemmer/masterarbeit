"""
Myopic-Policy für die Wartungsoptimierung von E-Ladesäulen.

Strategie: Bei jeder stündlichen Störungsmeldung wird jede Störung in die
Route des Teams mit den geringsten Gesamtzusatzkosten eingefügt (greedy
cheapest insertion). Ist eine Einplanung am selben Tag nicht mehr vor
17:00 Uhr möglich, wird die Störung als Carryover auf den nächsten Tag
verschoben.

Kosten:
  Operational  = (Fahrzeit + Servicezeit) × 40 €/h + km × 0,30 €/km
  Ausfall      = Wartezeit bis Service [h] × Nennleistung [kW] × 0,50 €/kWh

Typ-1-Störung : Servicezeit = 60 min
Typ-2-Störung : Servicezeit = 30 min (Demontage)
                             + Rundfahrt Station→Depot→Station [min]
                             + 5 min (Lagerhandling)
                             + 30 min (Montage)
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
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, VRPSolver

logger = logging.getLogger(__name__)


class MyopicPolicy:
    """
    Myopic-Policy: Greedy cheapest-insertion Replanning bei Störungen.

    Parameters
    ----------
    solver : VRPSolver
        OR-Tools Solver für den Tages-Initialplan.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden.
    config : dict
        Konfigurationsdict aus config.yaml.
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    """

    def __init__(
        self,
        solver: VRPSolver,
        all_coords: np.ndarray,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        cost_params: Optional[CostParams] = None,
        stations_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.solver = solver
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.cost_params = cost_params or CostParams()

        maint = config["maintenance"]
        self._use_or_tools: bool = bool(config.get("solver", {}).get("use_or_tools", True))
        self._workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration_min: int = maint.get("lunch_duration_min", 0)
        self._workday_minutes: int = (maint["workday_end_hour"] - maint["workday_start_hour"]) * 60
        self.n_teams: int = maint["n_teams"]

        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        if stations_df is not None and pwr_col in stations_df.columns:
            self.node_to_power: dict[int, float] = {
                i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
                for i, (_, row) in enumerate(stations_df.iterrows())
            }
        else:
            self.node_to_power: dict[int, float] = {}

        # Drop-Score für PolicyAdapter (RH-Pfad): identisch zur drop_score_fn in handle_disruptions.
        self._drop_score_fn = lambda node, dsm, rem_h, cur, det: 1.0 / max(1.0, det)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _zone_value(self, node_idx: int, days_since_maintenance: float) -> float:
        """Heuristischer Stationswert für V̂-basierte Zonenauswahl.

        Approximiert erwartete Ausfallkosten: power × recovery_curve(dsm).
        recovery_curve = initial_factor + (1 - initial_factor) × dsm / recovery_days
        """
        fail_cfg = self.config.get("failure_simulation", {})
        recovery_days = float(fail_cfg.get("recovery_days", 365))
        initial_factor = float(fail_cfg.get("initial_factor", 0.1))
        dsm = min(days_since_maintenance, recovery_days)
        recovery_curve = initial_factor + (1.0 - initial_factor) * dsm / recovery_days
        return self.node_to_power.get(node_idx, 22.0) * recovery_curve

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        if not self._use_or_tools:
            return greedy_initial_plan(
                tasks=tasks,
                team_assignment=team_assignment,
                all_coords=self.all_coords,
                traffic_matrices=self.traffic_matrices,
                workday_start_hour=self._workday_start_hour,
                workday_minutes=self._workday_minutes,
                lunch_earliest_min=self._lunch_earliest_min,
                lunch_duration_min=self._lunch_duration_min,
                n_teams=self.n_teams,
                route_score_fn=lambda node, dsm, cur, mat: (
                    1.0 / max(1.0, mat[cur, node])
                ),
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
        Fügt Störungen greedy cheapest in die Teamrouten ein.

        Reihenfolge: aufsteigend nach Insertionskosten (günstigste zuerst).
        Nach jeder Einfügung wird die Route neu bewertet, bevor die nächste
        Störung verplant wird.

        Returns
        -------
        (n_handled, carryover_liste, ausfallkosten_eur)
        """
        if not self._use_or_tools:
            return handle_disruptions_greedy(
                disruptions=disruptions,
                sim_routes=sim_routes,
                time_min=time_min,
                hour=hour,
                all_coords=self.all_coords,
                traffic_matrices=self.traffic_matrices,
                workday_start_hour=self._workday_start_hour,
                workday_minutes=self._workday_minutes,
                cost_params=self.cost_params,
                log=log,
                drop_score_fn=lambda node, dsm, rem_h, cur, det: 1.0 / max(1.0, det),
            )
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
                log.notes.append(
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
                    log.notes.append(
                        f"Eingefuegt ({n_d} Routine-Stop(s) ausgebaut: {dropped_nodes}): "
                        f"{d.disruption_type} @ Node {d.node_idx} -> "
                        f"Team {sim_routes[team_idx].team_id}, "
                        f"Ankunft {_fmt(arrival_at_d)}, "
                        f"Zusatzkosten {cost:.2f} EUR"
                    )
                else:
                    carryover.append(d)
                    log.notes.append(
                        f"Carryover (kein Routine-Stop ausreichend): "
                        f"{d.disruption_type} @ Node {d.node_idx}"
                    )
                    continue

            report_min = float((hour - 8) * 60)
            wait_h = max(0.0, (arrival_at_d + d.service_min - report_min) / 60.0)
            d_cost = wait_h * d.power_kw * self.cost_params.downtime_eur_per_kwh
            downtime_cost += d_cost
            if d_cost > 0:
                log.notes.append(f"Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)")
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
        """
        Findet die kostengünstigste Einfügeposition über alle Teams.

        Returns
        -------
        (min_cost, team_idx_in_sim_routes, pos_in_remaining, arrival_at_d)
        oder None, wenn keine Einfügung vor 17:00 möglich ist.
        """
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
        """
        Berechnet Kosten und Machbarkeit einer Einfügeposition.

        pos = 0: direkt nach aktuellem Teamknoten (vor remaining[0])
        pos = k: zwischen remaining[k-1] und remaining[k]
        pos = len(remaining): nach dem letzten verbleibenden Stop

        Returns
        -------
        (kosten_eur, machbar, ankunft_an_stoerung_min)
        """
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
        downtime_h = max(0.0, (arrival_at_d + d.service_min - report_min) / 60.0)

        cost = (
            extra_time_h * cp.wage_eur_per_hour
            + max(0.0, extra_km) * cp.fuel_eur_per_km
            + downtime_h * d.power_kw * cp.downtime_eur_per_kwh
        )
        return cost, feasible, arrival_at_d

    def _insert_disruption(
        self,
        d: DisruptionEvent,
        route: SimRoute,
        pos: int,
        time_min: float,
    ) -> None:
        """
        Fügt Störungs-Stop an Position pos (in remaining) in die Route ein
        und aktualisiert alle Folge-Ankunftszeiten.
        """
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

    def _find_best_drop_and_insert(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
    ) -> Optional[tuple[float, int, list[int], int, float]]:
        """
        Sucht die kostengünstigste Kombination aus 1..N Routine-Drops,
        um Platz für Störung d zu schaffen.

        Strategie: Greedy von hinten – letzte Routine-Stops werden zuerst
        entfernt. Für jede Drop-Anzahl werden alle Einfügepositionen geprüft.

        Returns
        -------
        (kosten, team_idx, [drop_global_indices], insert_pos, arrival_at_d)
        oder None falls kein Drop eine Lösung ermöglicht.
        """
        best_cost = np.inf
        best: Optional[tuple[float, int, list[int], int, float]] = None
        matrix = self._get_matrix(time_min)

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            routine_idx = [i for i, s in enumerate(remaining) if s.task_type == "routine"]

            if not routine_idx:
                continue

            current_node = route.current_node_at(time_min)
            current_dep = route.current_departure_at(time_min)
            if route.lunch_end_min is not None and current_dep < route.lunch_end_min:
                current_dep = route.lunch_end_min

            dropped_in_remaining: list[int] = []

            for n_drop in range(1, len(routine_idx) + 1):
                dropped_in_remaining.append(routine_idx[-n_drop])
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
                        drop_globals = [route.stops.index(remaining[i]) for i in dropped_in_remaining]
                        best_cost = cost
                        best = (cost, ti, drop_globals, pos, arrival)
                        found = True

                if found:
                    break

        return best

    def _remove_stop_and_recompute(self, route: SimRoute, global_idx: int) -> None:
        """
        Entfernt den Stop an global_idx aus der Route und berechnet
        alle nachfolgenden Ankunftszeiten neu.
        """
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

    def _get_matrix(self, time_min: float) -> np.ndarray:
        """Gibt die passende Stundenmatrix für einen Zeitstempel zurück."""
        hour = 8 + int(max(0.0, time_min)) // 60
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]


# Alias für Rückwärtskompatibilität
MyopicModel = MyopicPolicy
