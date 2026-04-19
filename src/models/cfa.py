"""
Cost Function Approximation (CFA) – gelernte Wertfunktionsapproximation.

Approximiert die zukünftige Wertfunktion V(s) als lineare Funktion der
stationsindividuellen Dringlichkeit:

    V̂(k) = θ × power_kW[k] × days_since_maintenance[k]

θ wird offline aus Monte-Carlo-Simulationen mit der Myopic-Policy gelernt
(scripts/train_cfa.py) und aus data/cfa/theta.json geladen.

Initialplan:
    Alle Routine-Tasks sind mandatory. Soft-Deadlines nach V̂:
    Höhere Dringlichkeit → frühere Deadline → OR-Tools plant sie früher.

Disruption Handling (CFA-Kern):
    OR-Tools replant die gesamte Restroute (wie CFA Light).
    Falls infeasible: Routine-Stops werden nach aufsteigendem V̂ gedroppt
    (niedrigste Dringlichkeit zuerst) bis OR-Tools eine Lösung findet.
    → Drop-Entscheidung basiert auf gelernten Zukunftskosten, nicht auf Position.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cost_params import CostParams
from src.planning.clustering import _approx_km
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    SimRoute,
    plan_to_sim_routes,
)
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/cfa/theta.json")


class CFAModel:
    """
    Echtes CFA-Modell mit gelernter Wertfunktionsapproximation.

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
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    theta_path : Path | str | None
        Pfad zu data/cfa/theta.json. None → Standardpfad.
    theta_override : float | None
        Direkt übergebener θ-Wert (überschreibt theta_path). Wird für
        iteratives Policy-Training verwendet, um θ ohne Datei-I/O zu setzen.
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: np.ndarray,
        stations_df: Optional[pd.DataFrame] = None,
        cost_params: Optional[CostParams] = None,
        theta_path: Optional[Path | str] = None,
        theta_override: Optional[float] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.cost_params = cost_params or CostParams()

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60

        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        if stations_df is not None and pwr_col in stations_df.columns:
            self.node_to_power: dict[int, float] = {
                i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
                for i, (_, row) in enumerate(stations_df.iterrows())
            }
        else:
            self.node_to_power = {}

        cp = self.cost_params
        self._wage_per_min: float = cp.wage_eur_per_hour / 60.0

        fail_cfg = config.get("failure_simulation", {})
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        cfa_cfg = config.get("cfa", {})
        self.alpha: float = float(cfa_cfg.get("alpha", 10.0))

        if theta_override is not None:
            self.theta: float = float(theta_override)
            logger.info(f"CFA: θ={self.theta:.4e} EUR/(kW·Tag) (direkt übergeben)")
        else:
            path = Path(theta_path) if theta_path else _DEFAULT_THETA_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"CFA-Gewicht nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train_cfa.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self.theta = float(data["theta"])
            logger.info(
                f"CFA: θ={self.theta:.4e} EUR/(kW·Tag) geladen aus {path} "
                f"(R²={data.get('r2', '?'):.4f}, {data.get('n_runs', '?')} Läufe)"
            )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _value(self, node_idx: int, days_since_maintenance: float) -> float:
        """V̂(k) = θ × power_kW[k] × dsm[k] in EUR."""
        return self.theta * self.node_to_power.get(node_idx, 22.0) * days_since_maintenance

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        """Deadline-Penalty für Störungen in Minuten/Minute Überschreitung."""
        penalty_eur = (
            self.alpha * power_kw * self.p_failure_per_hour
            * self.cost_params.downtime_eur_per_kwh
        )
        return max(1, int(round(penalty_eur / self._wage_per_min)))

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Erstellt den Tagesplan mit wertfunktionsbasierter Soft-Deadline.

        Alle Routine-Tasks sind mandatory. V̂ bestimmt die Reihenfolge:
        höhere Dringlichkeit → frühere Soft-Deadline → OR-Tools plant früher.
        """
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        n = len(routine_tasks)

        if n > 0:
            depot = self.all_coords[0]
            urgency = [
                self._value(t.node_idx, t.days_since_maintenance)
                for t in routine_tasks
            ]
            scores = [
                v / max(0.1, _approx_km(self.all_coords[t.node_idx], depot))
                for t, v in zip(routine_tasks, urgency)
            ]
            for rank, idx in enumerate(np.argsort(scores)[::-1]):
                deadline = int((rank + 1) / n * self.WORKDAY_MINUTES)
                penalty = max(1, int(round(urgency[idx] / self._wage_per_min)))
                routine_tasks[idx].soft_deadline_min = deadline
                routine_tasks[idx].deadline_penalty = penalty

        logger.info(
            f"CFA Initialplan: {len(tasks)} Tasks, {n} Routine "
            f"mit Soft-Deadlines (θ={self.theta:.3e})."
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
        OR-Tools Replan mit V̂-basiertem Drop im Retry.

        1. Replan mit allen verbleibenden Stops + Störungen (mandatory).
        2. Falls infeasible: Droppe Routine-Stop mit niedrigstem V̂, repeat.
        """
        team_states = [
            TeamState(
                team_id=r.team_id,
                current_node=r.current_node_at(time_min),
                current_time=int(r.lunch_end_min) if (
                    r.lunch_end_min is not None and time_min < r.lunch_end_min
                ) else int(time_min),
                completed_nodes=r.completed_nodes_at(time_min),
            )
            for r in sim_routes
        ]

        remaining_tasks = [
            MaintenanceTask(
                node_idx=s.node_idx,
                task_type=s.task_type,
                priority=1 if s.task_type != "routine" else 2,
                service_time=int(s.service_min),
                days_since_maintenance=s.days_since_maintenance,
            )
            for r in sim_routes
            for s in r.remaining_stops_at(time_min)
        ]

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
            # CFA-Kern: Routine-Stops nach aufsteigendem V̂ droppen
            routine_tasks = [t for t in remaining_tasks if t.task_type == "routine"]
            mandatory = [t for t in remaining_tasks if t.task_type != "routine"] + disruption_tasks

            # Aufsteigend nach V̂ sortieren: niedrigste Dringlichkeit zuerst droppen
            routine_tasks.sort(key=lambda t: self._value(t.node_idx, t.days_since_maintenance))

            solved = False
            for n_drop in range(1, len(routine_tasks) + 1):
                retry_tasks = mandatory + routine_tasks[n_drop:]
                if not retry_tasks:
                    break
                new_plan = self.solver.replan(retry_tasks, team_states)
                if new_plan.solver_status not in ("INFEASIBLE", "NO_SOLUTION"):
                    dropped = [t.node_idx for t in routine_tasks[:n_drop]]
                    log.notes.append(
                        f"CFA-Replan Retry: {n_drop} Routine-Stop(s) nach V̂ ausgebaut "
                        f"{dropped}, Status: {new_plan.solver_status}"
                    )
                    solved = True
                    break

            if not solved:
                log.notes.append(
                    f"CFA-Replan fehlgeschlagen ({new_plan.solver_status}): "
                    f"{len(disruptions)} Störung(en) als Carryover."
                )
                return 0, list(disruptions), 0.0

        new_routes = plan_to_sim_routes(new_plan, all_tasks, self.solver.n_teams)
        for i, route in enumerate(sim_routes):
            completed = [s for s in route.stops if s.arrival_min <= time_min]
            route.stops = completed + (new_routes[i].stops if i < len(new_routes) else [])

        disruption_nodes = {d.node_idx: d for d in disruptions}
        downtime_cost = 0.0
        cp = self.cost_params
        arrival_at_d = time_min  # Fallback
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in disruption_nodes:
                    d = disruption_nodes[stop.node_idx]
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (stop.arrival_min - report_min) / 60.0)
                    d_cost = wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    downtime_cost += d_cost
                    arrival_at_d = stop.arrival_min
                    if d_cost > 0:
                        log.notes.append(
                            f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)"
                        )

        log.notes.append(
            f"CFA-Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost
