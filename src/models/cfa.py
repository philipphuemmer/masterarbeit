"""
Cost Function Approximation (CFA) – mehrdimensionale Wertfunktionsapproximation.

Approximiert die Kosten des Weglassens von Station k als lineares Modell:

    C̃(drop k) = θᵀ φ(k)

    φ(k) = [power_kW, age_years, is_DC, recovery_curve(dsm), mean_dist_to_others]

θ ∈ ℝ⁵ wird per OLS aus Monte-Carlo-Simulationen gelernt.
Features werden pro Station standardisiert (μ, σ aus Trainingsdaten).

Gegenüber skalarem CFA: θ cancelt nicht mehr in Ranking-Entscheidungen,
da verschiedene Features unterschiedlich gewichtet werden.

Initialplan:
    Soft-Deadlines nach C̃-Rang (höhere Kosten → frühere Deadline).
    Penalty = 1 min/min (schwach) damit Disruptions dominieren.

Disruption Handling (CFA-Kern):
    Falls infeasible: Drop nach aufsteigendem C̃(drop k) —
    Station mit niedrigsten Zukunftskosten fliegt zuerst raus.
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
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/training/cfa/theta.json")


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
        Pfad zu data/training/cfa/theta.json. None → Standardpfad.
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

        # Stationsindividuelle Features vorberechnen
        if stations_df is not None:
            from src.data.loader import get_failure_rate_factors
            _factors = get_failure_rate_factors(stations_df)
            self._node_to_failure_factor: dict[int, float] = {
                i + 1: _factors.get(i, 1.0) for i in range(len(stations_df))
            }
            # Alter in Jahren
            date_col = "Inbetriebnahmedatum"
            ref = pd.Timestamp("2026-01-01")
            self._node_to_age: dict[int, float] = {}
            for i, (_, row) in enumerate(stations_df.iterrows()):
                if date_col in stations_df.columns and pd.notna(row.get(date_col)):
                    age = max(0.0, (ref - pd.Timestamp(row[date_col])).days / 365.25)
                else:
                    age = 5.0
                self._node_to_age[i + 1] = age
            # Ladetyp: DC = 1, AC = 0
            type_col = "Art der Ladeeinrichtung"
            self._node_to_is_dc: dict[int, float] = {
                i + 1: float(row.get(type_col, "") == "Schnellladeeinrichtung")
                for i, (_, row) in enumerate(stations_df.iterrows())
            }
        else:
            self._node_to_failure_factor = {}
            self._node_to_age = {}
            self._node_to_is_dc = {}

        # Mittlere Distanz jeder Station zu allen anderen (in km)
        if all_coords is not None and len(all_coords) > 2:
            n = len(all_coords)
            self._node_to_mean_dist: dict[int, float] = {
                i: float(np.mean([
                    _approx_km(all_coords[i], all_coords[j])
                    for j in range(1, n) if j != i
                ]))
                for i in range(1, n)
            }
        else:
            self._node_to_mean_dist = {}

        cp = self.cost_params
        self._wage_per_min: float = cp.wage_eur_per_hour / 60.0

        fail_cfg = config.get("failure_simulation", {})
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        cfa_cfg = config.get("cfa", {})
        self.alpha: float = float(cfa_cfg.get("alpha", 10.0))
        self._use_or_tools: bool = bool(config.get("solver", {}).get("use_or_tools", True))
        self._workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration_min: int = maint.get("lunch_duration_min", 0)

        if theta_override is not None:
            self.theta = np.asarray(theta_override, dtype=float)
            self._feature_means = np.zeros(len(self.theta))
            self._feature_stds = np.ones(len(self.theta))
            logger.info(f"CFA: θ={self.theta} (direkt übergeben)")
        else:
            path = Path(theta_path) if theta_path else _DEFAULT_THETA_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"CFA-Gewicht nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train/train_cfa.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self.theta = np.array(data["theta"], dtype=float)
            self._feature_means = np.array(data.get("feature_means", np.zeros(len(self.theta))))
            self._feature_stds = np.array(data.get("feature_stds", np.ones(len(self.theta))))
            logger.info(
                f"CFA: θ={self.theta} geladen aus {path} "
                f"(R²={data.get('r2', '?'):.4f}, {data.get('n_runs', '?')} Läufe)"
            )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _phi(self, node_idx: int, days_since_maintenance: float) -> np.ndarray:
        """Feature-Vektor φ(k) = [power, age, recovery_curve, mean_dist]."""
        fail_cfg = self.config.get("failure_simulation", {})
        recovery_days = float(fail_cfg.get("recovery_days", 365))
        initial_factor = float(fail_cfg.get("initial_factor", 0.1))
        dsm = min(days_since_maintenance, recovery_days)
        recovery_curve = initial_factor + (1.0 - initial_factor) * dsm / recovery_days
        return np.array([
            self.node_to_power.get(node_idx, 22.0),
            self._node_to_age.get(node_idx, 5.0),
            recovery_curve,
            self._node_to_mean_dist.get(node_idx, 5.0),
        ])

    def _value(self, node_idx: int, days_since_maintenance: float) -> float:
        """C̃(drop k) = θᵀ × φ_scaled(k) — approximierte Kosten des Weglassens."""
        phi = self._phi(node_idx, days_since_maintenance)
        phi_scaled = (phi - self._feature_means) / np.maximum(self._feature_stds, 1e-8)
        return float(self.theta @ phi_scaled)

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
        Erstellt den Tagesplan mit wertfunktionsbasierter Soft-Deadline.

        Alle Routine-Tasks sind mandatory. V̂ bestimmt die Reihenfolge:
        höhere Dringlichkeit → frühere Soft-Deadline → OR-Tools plant früher.
        """
        if not self._use_or_tools:
            n_routine = sum(1 for t in tasks if t.task_type == "routine")
            logger.info(
                f"CFA Greedy-Initialplan: {len(tasks)} Tasks, {n_routine} Routine "
                f"(C̃/dist, θ={np.array2string(self.theta, precision=3)})."
            )
            # Shift so that min(C̃) → 1.0: negative values würden bei kleiner Distanz
            # den Score stark negativ machen und Nearest-Neighbor umkehren.
            min_val = min(
                (self._value(t.node_idx, t.days_since_maintenance) for t in tasks),
                default=0.0,
            )
            shift = max(0.0, -min_val) + 1.0
            return greedy_initial_plan(
                tasks=tasks,
                team_assignment=team_assignment,
                all_coords=self.all_coords,
                traffic_matrices=self.solver.traffic_matrices,
                workday_start_hour=self._workday_start_hour,
                workday_minutes=self.WORKDAY_MINUTES,
                lunch_earliest_min=self._lunch_earliest_min,
                lunch_duration_min=self._lunch_duration_min,
                n_teams=self.solver.n_teams,
                route_score_fn=lambda node, dsm, cur: (
                    (self._value(node, dsm) + shift)
                    / max(0.1, _approx_km(self.all_coords[cur], self.all_coords[node]))
                ),
            )

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
                # Penalty = 1 min/min: V̂ steuert die Reihenfolge der Deadlines,
                # nicht die Durchsetzungsstärke. Disruptions (4+ min/min) dominieren.
                routine_tasks[idx].soft_deadline_min = deadline
                routine_tasks[idx].deadline_penalty = 1

        for task in tasks:
            if task.task_type != "routine":
                task.soft_deadline_min = 0
                task.deadline_penalty = self._disruption_deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
                )

        logger.info(
            f"CFA OR-Tools-Initialplan: {len(tasks)} Tasks, {n} Routine "
            f"mit Soft-Deadlines (θ={np.array2string(self.theta, precision=3)})."
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
                    self._value(node, dsm) - self._wage_per_min * det
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

            # Aufsteigend nach V̂ sortieren; bei gleichem V̂: depotfernere Station zuerst droppen
            # (spart mehr Fahrzeit und ist konsistent mit der Zonenlogik)
            routine_tasks.sort(key=lambda t: (
                self._value(t.node_idx, t.days_since_maintenance),
                -_approx_km(self.all_coords[t.node_idx], self.all_coords[0]),
            ))

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
                    wait_h = max(0.0, (stop.departure_min - report_min) / 60.0)
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
