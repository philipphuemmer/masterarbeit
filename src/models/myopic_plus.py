"""
Cost Function Approximation (CFA) Modell – Soft-Deadline-Priorisierung.

Setzt stationsindividuelle Soft-Deadlines in der VRP-Zielfunktion:

Initialplan:
    deadline(k) = rank(k) / n_tasks × WORKDAY_MINUTES
    deadline_penalty(k) = α × power_kW(k) × p_h × downtime_eur_per_kwh / wage_eur_per_min

    rank(k) = Position in nach Depot-Distanz sortierter Routineliste (nah = früh)
    → OR-Tools zahlt deadline_penalty pro Minute Überschreitung der Deadline
    → Hohe kW-Stationen müssen früh besucht sein, sonst hohe Strafkosten

Replan bei Störungen (einheitliche Kostenfunktion):
    Störungsknoten : Soft-Deadline = Meldezeitpunkt, hoher Penalty ∝ power_kW
    Routine-Knoten : optional (AddDisjunction), Skip-Penalty ∝ power_kW
    → OR-Tools wirft niedrig-priorisierte Routine-Stops raus, um Störungen früh zu bedienen
    → Kein manueller Retry-Loop nötig

Formel Deadline-Penalty (Initialplan & Störung):
    penalty_per_min = max(1, round(α × power_kW × p_h × downtime_eur_per_kwh / wage_eur_per_min))

Formel Skip-Penalty (Routine im Replan):
    skip = max(1, round(α × power_kW × p_h × remaining_hours × downtime_eur_per_kwh / wage_eur_per_min))
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
    plan_to_sim_routes,
)
from src.planning.clustering import _approx_km
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)


class MyopicPlusModel:
    """
    Cost Function Approximation Light Modell mit Soft-Deadline-Priorisierung.

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
        self._use_or_tools: bool = bool(config.get("solver", {}).get("use_or_tools", True))
        self._workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration_min: int = maint.get("lunch_duration_min", 0)

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

    def _routine_deadline_penalty(self, power_kw: float) -> int:
        """Soft-Deadline-Penalty für Routine-Tasks in Minuten/Minute.

        Verwendet p_failure, weil die Station noch nicht ausgefallen ist —
        der Erwartungswert der Ausfallkosten pro Minute Verzögerung ist:
        power × p_failure/min × downtime_eur_per_kwh.
        """
        cp = self.cost_params
        wage_per_min = cp.wage_eur_per_hour / 60.0
        penalty = self.alpha * power_kw * self.p_failure_per_hour * cp.downtime_eur_per_kwh / wage_per_min
        return max(1, int(round(penalty)))

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        """Soft-Deadline-Penalty für Störungen in Minuten/Minute.

        Station ist definitiv ausgefallen — direkte Ausfallkosten pro Minute:
        power × downtime_eur_per_kwh / 60 / wage_per_min.
        """
        cp = self.cost_params
        cost_per_min = power_kw * cp.downtime_eur_per_kwh / 60.0
        return max(1, int(round(cost_per_min / (cp.wage_eur_per_hour / 60.0))))

    def _skip_penalty(self, power_kw: float, remaining_hours: float, service_time: int = 45) -> int:
        """
        Kosten in Minuten für das Überspringen einer Routine-Station.

        Setzt sich zusammen aus:
        - service_time: Basiskosten (verhindert unnötige Drops ohne Störungsdruck)
        - Erwartete Ausfallkosten: differenziert nach Nennleistung
        """
        cp = self.cost_params
        wage_per_min = cp.wage_eur_per_hour / 60.0
        downtime_eur = self.alpha * power_kw * self.p_failure_per_hour * remaining_hours * cp.downtime_eur_per_kwh
        return service_time + max(0, int(round(downtime_eur / wage_per_min)))

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
        if not self._use_or_tools:
            n_routine = sum(1 for t in tasks if t.task_type == "routine")
            logger.info(
                f"MyopicPlus Greedy-Initialplan: {len(tasks)} Tasks, "
                f"{n_routine} Routine (Nearest-Neighbor)."
            )
            return greedy_initial_plan(
                tasks=tasks,
                team_assignment=team_assignment,
                all_coords=self.all_coords,
                traffic_matrices=self.solver.traffic_matrices,
                workday_start_hour=self._workday_start_hour,
                workday_minutes=self.WORKDAY_MINUTES,
                lunch_earliest_min=self._lunch_earliest_min,
                lunch_duration_min=self._lunch_duration_min,
                n_teams=self.n_teams,
                route_score_fn=lambda node, dsm, cur: (
                    self.node_to_power.get(node, 22.0)
                    / max(0.1, _approx_km(self.all_coords[cur], self.all_coords[node]))
                ),
            )

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
                task.deadline_penalty = self._routine_deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
                )

        for task in tasks:
            if task.task_type != "routine":
                task.soft_deadline_min = 0
                task.deadline_penalty = self._disruption_deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
                )

        logger.info(
            f"MyopicPlus OR-Tools-Initialplan: {len(tasks)} Tasks, "
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
        Replant den Tag mit einheitlicher Kostenfunktion für Störungen und Routinen.

        Störungsknoten: mandatory, Soft-Deadline = Meldezeitpunkt, Penalty ∝ kW.
        Routine-Knoten: optional (AddDisjunction), Skip-Penalty ∝ kW.
        → OR-Tools wirft niedrig-priorisierte Routinen raus, um Störungen früh zu bedienen.
        """
        if not self._use_or_tools:
            return handle_disruptions_greedy(
                disruptions=disruptions,
                sim_routes=sim_routes,
                time_min=time_min,
                hour=hour,
                all_coords=self.all_coords,
                traffic_matrices=self.solver.traffic_matrices,
                workday_start_hour=self._workday_start_hour,
                workday_minutes=self.WORKDAY_MINUTES,
                cost_params=self.cost_params,
                log=log,
                drop_score_fn=lambda node, dsm, rem_h, cur, det: (
                    self.node_to_power.get(node, 22.0)
                    / max(0.1, _approx_km(self.all_coords[cur], self.all_coords[node]))
                ),
            )

        team_states = [
            TeamState(
                team_id=r.team_id,
                current_node=r.current_node_at(time_min),
                current_time=int(r.lunch_end_min) if (
                    r.lunch_end_min is not None and time_min < r.lunch_end_min
                ) else int(r.current_departure_at(time_min)),
                completed_nodes=r.completed_nodes_at(time_min),
            )
            for r in sim_routes
        ]

        # Routine-Stops: mandatory (kein skip_penalty – kein ungewolltes Droppen)
        remaining_tasks = [
            MaintenanceTask(
                node_idx=s.node_idx,
                task_type=s.task_type,
                priority=1 if s.task_type != "routine" else 2,
                service_time=int(s.service_min),
            )
            for r in sim_routes
            for s in r.remaining_stops_at(time_min)
        ]

        # Störungsknoten: mandatory, Soft-Deadline = Meldezeitpunkt, Penalty ∝ kW
        # → OR-Tools schiebt Störungen nach vorne ohne Routinen zu droppen
        disruption_tasks = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="disruption",
                priority=1,
                service_time=int(round(d.service_min)),
                soft_deadline_min=int(time_min),
                deadline_penalty=self._disruption_deadline_penalty(d.power_kw),
            )
            for d in disruptions
        ]

        all_tasks = remaining_tasks + disruption_tasks
        if not all_tasks:
            return 0, [], 0.0

        new_plan = self.solver.replan(all_tasks, team_states)

        if new_plan.solver_status in ("INFEASIBLE", "NO_SOLUTION"):
            # Retry: letzte Routine-Stops iterativ droppen bis OR-Tools löst
            routine_in_order = [t for t in remaining_tasks if t.task_type == "routine"]
            mandatory = [t for t in remaining_tasks if t.task_type != "routine"] + disruption_tasks
            solved = False
            for n_drop in range(1, len(routine_in_order) + 1):
                retry_tasks = mandatory + routine_in_order[: len(routine_in_order) - n_drop]
                if not retry_tasks:
                    break
                new_plan = self.solver.replan(retry_tasks, team_states)
                if new_plan.solver_status not in ("INFEASIBLE", "NO_SOLUTION"):
                    log.notes.append(
                        f"CFA-Replan Retry: {n_drop} Routine-Stop(s) ausgebaut, "
                        f"Status: {new_plan.solver_status}"
                    )
                    solved = True
                    break
            if not solved:
                log.notes.append(
                    f"CFA-Replan fehlgeschlagen ({new_plan.solver_status}): "
                    f"{len(disruptions)} Störung(en) als Carryover."
                )
                return 0, list(disruptions), 0.0

        new_routes = plan_to_sim_routes(new_plan, all_tasks, self.n_teams)
        for i, route in enumerate(sim_routes):
            completed = [s for s in route.stops if s.arrival_min <= time_min]
            route.stops = completed + (new_routes[i].stops if i < len(new_routes) else [])

        disruption_nodes = {d.node_idx: d for d in disruptions}
        downtime_cost = 0.0
        cp = self.cost_params
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in disruption_nodes:
                    d = disruption_nodes[stop.node_idx]
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (stop.departure_min - report_min) / 60.0)
                    downtime_cost += wait_h * d.power_kw * cp.downtime_eur_per_kwh

        log.notes.append(
            f"CFA-Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost
