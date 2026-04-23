"""
CFA Real – echte Cost Function Approximation mit AddDisjunction.

Unterschied zu cfa.py (VFA):
    cfa.py setzt V̂(k) als Soft-Deadline-Hint außerhalb des Solvers und
    droppt Stationen manuell in einem Retry-Loop.

    cfa_real.py bettet V̂(k) direkt als skip_penalty (AddDisjunction) in
    die Solver-Zielfunktion ein:

        min  Σ travel_time(route)
           + Σ_k V̂(k)/wage_per_min × 1[Station k wird gedroppt]

    OR-Tools entscheidet simultan, welche Stationen sich lohnen zu besuchen.
    Der manuelle Drop-Loop entfällt.

V̂(k) = θ × power_kW[k] × days_since_maintenance[k]

θ wird aus data/cfa/theta.json geladen (identisches Training wie cfa.py).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
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
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/cfa/theta.json")


class CFARealModel:
    """
    CFA-Modell mit AddDisjunction: V̂ als Solver-Objective-Term.

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
        Direkt übergebener θ-Wert (überschreibt theta_path).
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
        self._mean_service_min: int = maint.get("mean_service_time", 30)

        # Mittlere Fahrtzeit aus der Reisezeitmatrix schätzen (Sekunden → Minuten).
        # Dient als Proxy für future_visit_cost: skip_penalty = (service + travel) + V̂/wage
        any_matrix = next(iter(traffic_matrices.values()))
        n = any_matrix.shape[0]
        off_diag = any_matrix[np.arange(n)[:, None] != np.arange(n)].mean()
        self._mean_travel_min: int = max(5, int(round(float(off_diag) / 60.0)))

        fail_cfg = config.get("failure_simulation", {})
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        cfa_cfg = config.get("cfa", {})
        self.alpha: float = float(cfa_cfg.get("alpha", 10.0))

        if theta_override is not None:
            self.theta: float = float(theta_override)
            logger.info(f"CFA-Real: θ={self.theta:.4e} EUR/(kW·Tag) (direkt übergeben)")
        else:
            path = Path(theta_path) if theta_path else _DEFAULT_THETA_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"CFA-Gewicht nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train/train_cfa.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self.theta = float(data["theta"])
            logger.info(
                f"CFA-Real: θ={self.theta:.4e} EUR/(kW·Tag) geladen aus {path} "
                f"(R²={data.get('r2', '?'):.4f}, {data.get('n_runs', '?')} Läufe)"
            )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _value(self, node_idx: int, days_since_maintenance: float) -> float:
        """V̂(k) = θ × power_kW[k] × dsm[k] in EUR."""
        return self.theta * self.node_to_power.get(node_idx, 22.0) * days_since_maintenance

    def _skip_penalty(self, node_idx: int, days_since_maintenance: float) -> int:
        """Kosten des Weglassens in Minuten.

        skip_penalty = future_visit_cost + V̂(k)/wage_per_min

        future_visit_cost = service_time + mean_travel_time (was ein späterer Besuch kostet)
        V̂(k)/wage_per_min = Ausfallkosten bis zum nächsten Besuch in Minuten

        OR-Tools vergleicht:
            visit heute:  travel + service (~45 min)
            skip heute:   future_visit + V̂/wage (~45 + urgency min)
        → OR-Tools skippt nur wenn V̂ ≈ 0 und Routing eng ist.
        """
        future_visit_min = self._mean_service_min + self._mean_travel_min
        urgency_min = int(round(self._value(node_idx, days_since_maintenance) / self._wage_per_min))
        return future_visit_min + urgency_min

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        """Deadline-Penalty für Störungen in Minuten/Minute.

        Station ist definitiv ausgefallen — direkte Ausfallkosten pro Minute:
        power × downtime_eur_per_kwh / 60 / wage_per_min.
        """
        cost_per_min = power_kw * self.cost_params.downtime_eur_per_kwh / 60.0
        return max(1, int(round(cost_per_min / self._wage_per_min)))

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Erstellt den Tagesplan mit V̂-basierter AddDisjunction.

        Alle Routine-Tasks erhalten skip_penalty = V̂(k) / wage_per_min.
        OR-Tools entscheidet selbst, welche Stationen besucht werden.
        Disruption-Tasks bleiben mandatory (kein skip_penalty).
        """
        n_routine = sum(1 for t in tasks if t.task_type == "routine")
        logger.info(
            f"CFA-Real Initialplan: {len(tasks)} Tasks, {n_routine} Routine "
            f"(θ={self.theta:.3e})."
        )

        # Schritt 1: mandatory — verhindert unnötige Drops wenn feasible
        plan = self.solver.create_initial_plan(
            tasks, team_assignment=team_assignment, internal_retry=False
        )

        if plan.solver_status not in ("OPTIMAL", "FEASIBLE"):
            # Schritt 2: AddDisjunction — OR-Tools droppt simultan nach V̂
            for task in tasks:
                if task.task_type == "routine":
                    task.skip_penalty = self._skip_penalty(
                        task.node_idx, task.days_since_maintenance
                    )
            plan = self.solver.create_initial_plan(
                tasks, team_assignment=team_assignment, internal_retry=False
            )
            logger.info(
                f"CFA-Real Initialplan Retry (AddDisjunction): {plan.solver_status}"
            )

        return plan

    def handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        OR-Tools Replan mit V̂-basierter AddDisjunction.

        1. AddDisjunction-Replan: OR-Tools droppt Routine-Stops simultan nach V̂.
        2. Falls NO_SOLUTION: manueller V̂-Drop-Loop als Fallback (analog CFA).
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
                skip_penalty=(
                    self._skip_penalty(s.node_idx, s.days_since_maintenance)
                    if s.task_type == "routine" else None
                ),
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

        # Schritt 1: AddDisjunction — OR-Tools droppt simultan nach V̂
        new_plan = self.solver.replan(all_tasks, team_states)

        if new_plan.solver_status in ("INFEASIBLE", "NO_SOLUTION"):
            log.notes.append(
                f"CFA-Real Retry (AddDisjunction): {new_plan.solver_status}"
            )
            # Schritt 2: Manueller V̂-Drop-Loop als Fallback (analog CFA)
            # skip_penalty entfernen damit verbleibende Routine-Stops mandatory sind
            routine_tasks = [t for t in remaining_tasks if t.task_type == "routine"]
            for t in routine_tasks:
                t.skip_penalty = None
            mandatory = [t for t in remaining_tasks if t.task_type != "routine"] + disruption_tasks
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
                        f"CFA-Real V̂-Drop Retry: {n_drop} Routine-Stop(s) ausgebaut "
                        f"{dropped}, Status: {new_plan.solver_status}"
                    )
                    all_tasks = retry_tasks
                    solved = True
                    break

            if not solved:
                log.notes.append(
                    f"CFA-Real-Replan fehlgeschlagen: "
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
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in disruption_nodes:
                    d = disruption_nodes[stop.node_idx]
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (stop.arrival_min - report_min) / 60.0)
                    d_cost = wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    downtime_cost += d_cost
                    if d_cost > 0:
                        log.notes.append(
                            f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)"
                        )

        log.notes.append(
            f"CFA-Real-Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost
