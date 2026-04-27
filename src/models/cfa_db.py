"""
CFA-DB Policy — OR-Tools Routing mit lernbarem Balance-Parameter α.

Kombiniert:
  - CFA:  OR-Tools Initialplan mit U(k)-basierten Soft-Deadlines
  - DB:   Zustandsabhängiges α ∈ (0,1) via MLP für die Drop-Entscheidung

Score beim Drop (falls OR-Tools infeasible):
    drop_score(k) = (1−α) · U(k) − α · d_depot(k)
    U(k) = power_kW[k] × dsm[k]   (Dringlichkeit)
    d_depot(k)                      (normierte Depotentfernung als Δτ-Proxy)

    niedrigster drop_score → zuerst droppen

α_{S_t} ∈ [0,1] wird durch ein MLP aus dem Zustand berechnet:
    α = σ(W3 · relu(W2 · relu(W1 · φ(S_t) + b1) + b2) + b3)

8 Zustandsmerkmale φ(S_t): identisch mit DBModel.

Training: PPO (scripts/train/train_cfa_db.py)
Ausgabe:  data/training/cfa_db/policy.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.cost_params import CostParams
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    MaintenanceSimulator,
    SimRoute,
    plan_to_sim_routes,
)
from src.planning.clustering import _approx_km
from src.planning.vrp_solver import (
    DailyPlan,
    MaintenanceTask,
    TeamState,
    VRPSolver,
)

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path("data/training/cfa_db/policy.json")

_MAX_DSM = 365.0
_MAX_DEPOT_KM = 30.0
_CARRYOVER_SCALE = 10.0


class CFADBModel:
    """
    CFA-DB Policy: OR-Tools Routing + MLP-gelerntes α für Drop-Entscheidung.

    Parameters
    ----------
    traffic_matrices : Stündliche Reisezeitmatrizen in Sekunden.
    config : Konfigurationsdict aus config.yaml.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
    node_to_power : dict[int, float] — node_idx → Nennleistung [kW].
    n_stations : Gesamtzahl der Stationen (ohne Depot).
    cost_params : Kostenparameter (None → Standardwerte).
    policy_path : Pfad zu data/training/cfa_db/policy.json.
    weights_override : dict | None — direkt übergebene Gewichte (für Training).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: Optional[np.ndarray] = None,
        node_to_power: Optional[dict[int, float]] = None,
        n_stations: int = 397,
        cost_params: Optional[CostParams] = None,
        policy_path: Optional[Path | str] = None,
        weights_override: Optional[dict] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.node_to_power: dict[int, float] = node_to_power or {}
        self.n_stations = n_stations
        self.cost_params = cost_params or CostParams()
        self.n_teams: int = config["maintenance"]["n_teams"]

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60
        self._workday_start_hour: int = maint["workday_start_hour"]

        fail_cfg = config.get("failure_simulation", {})
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        cfa_cfg = config.get("cfa", {})
        self.alpha_cfa: float = float(cfa_cfg.get("alpha", 10.0))
        self._wage_per_min: float = self.cost_params.wage_eur_per_hour / 60.0

        # Zustandskontext: wird von CFADBMaintenanceSimulator vor jedem Tag gesetzt
        self._n_remaining_total: int = n_stations
        self._n_carryover: int = 0
        self._precomputed_alpha: Optional[float] = None

        if weights_override is not None:
            self._load_weights(weights_override)
            logger.info("CFA-DB: Gewichte direkt übergeben.")
        else:
            path = Path(policy_path) if policy_path else _DEFAULT_POLICY_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"CFA-DB Policy nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train/train_cfa_db.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self._load_weights(data)
            logger.info(f"CFA-DB: Policy geladen aus {path}")

    def _load_weights(self, data: dict) -> None:
        self._W1 = np.array(data["W1"], dtype=np.float64)
        self._b1 = np.array(data["b1"], dtype=np.float64)
        self._W2 = np.array(data["W2"], dtype=np.float64)
        self._b2 = np.array(data["b2"], dtype=np.float64)
        self._W3 = np.array(data["W3"], dtype=np.float64)
        self._b3 = np.array(data["b3"], dtype=np.float64)
        self._feat_mean = np.array(
            data.get("feature_mean", np.zeros(8)), dtype=np.float64
        )
        self._feat_std = np.array(
            data.get("feature_std", np.ones(8)), dtype=np.float64
        )
        self.alpha_mean: float = float(data.get("alpha_mean", 0.3))

    # ------------------------------------------------------------------
    # Feature-Extraktion & Vorwärtsdurchlauf (identisch mit DBModel)
    # ------------------------------------------------------------------

    def extract_features(
        self,
        routine_tasks: list[MaintenanceTask],
        n_remaining_total: Optional[int] = None,
        n_carryover: Optional[int] = None,
    ) -> np.ndarray:
        """8-dimensionaler Zustandsvektor φ(S_t)."""
        n_rem = n_remaining_total if n_remaining_total is not None else self._n_remaining_total
        n_carr = n_carryover if n_carryover is not None else self._n_carryover

        f0 = n_rem / max(1, self.n_stations)

        if routine_tasks:
            dsm_vals = np.array(
                [t.days_since_maintenance for t in routine_tasks], dtype=np.float64
            )
            pow_vals = np.array(
                [self.node_to_power.get(t.node_idx, 22.0) for t in routine_tasks],
                dtype=np.float64,
            )
            urgency = pow_vals * dsm_vals

            f1 = float(np.mean(dsm_vals > 90.0))
            f2 = float(np.mean(dsm_vals)) / _MAX_DSM
            max_possible_urgency = 150.0 * _MAX_DSM
            f3 = float(np.sum(urgency)) / (max_possible_urgency * max(1, self.n_stations))
            f4 = float(np.max(urgency)) / max_possible_urgency

            if self.all_coords is not None:
                depot = self.all_coords[0]
                dists = np.array(
                    [_approx_km(self.all_coords[t.node_idx], depot) for t in routine_tasks],
                    dtype=np.float64,
                )
                f5 = float(np.mean(dists)) / _MAX_DEPOT_KM
                f6 = float(np.std(dists)) / _MAX_DEPOT_KM
            else:
                f5 = f6 = 0.0
        else:
            f1 = f2 = f3 = f4 = f5 = f6 = 0.0

        f7 = n_carr / _CARRYOVER_SCALE

        return np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float64)

    def _forward(self, features: np.ndarray) -> float:
        """MLP-Vorwärtsdurchlauf (numpy): φ → α ∈ (0, 1)."""
        std = np.where(self._feat_std > 1e-8, self._feat_std, 1.0)
        x = (features - self._feat_mean) / std
        x = np.maximum(0.0, self._W1 @ x + self._b1)
        x = np.maximum(0.0, self._W2 @ x + self._b2)
        logit = float(self._W3 @ x + self._b3)
        return 1.0 / (1.0 + np.exp(-logit))

    def _station_value(self, node_idx: int, dsm: float) -> float:
        """U(k) = power_kW[k] × dsm — für V̂-basierte Zonenauswahl."""
        return self.node_to_power.get(node_idx, 22.0) * dsm

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _urgency(self, node_idx: int, dsm: float) -> float:
        return self.node_to_power.get(node_idx, 22.0) * dsm

    def _depot_dist_norm(self, node_idx: int) -> float:
        """Normierte Depotentfernung als Δτ-Proxy."""
        if self.all_coords is None:
            return 0.0
        return _approx_km(self.all_coords[node_idx], self.all_coords[0]) / _MAX_DEPOT_KM

    def _drop_score(self, task: MaintenanceTask, alpha: float) -> float:
        """Höherer Score → Station behalten. Niedrigster Score → zuerst droppen."""
        u = self._urgency(task.node_idx, task.days_since_maintenance)
        d = self._depot_dist_norm(task.node_idx) * self.WORKDAY_MINUTES
        return (1.0 - alpha) * u - alpha * d

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        penalty_eur = (
            self.alpha_cfa * power_kw * self.p_failure_per_hour
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
        OR-Tools Initialplan mit U(k)-basierten Soft-Deadlines (wie CFA).

        α beeinflusst die Deadline-Penalties: niedriges α → höhere Penalties
        für dringende Stationen → OR-Tools plant sie früher.
        """
        alpha = self._precomputed_alpha if self._precomputed_alpha is not None else 0.5

        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        n = len(routine_tasks)

        if n > 0 and self.all_coords is not None:
            depot = self.all_coords[0]
            urgency = [
                self._urgency(t.node_idx, t.days_since_maintenance)
                for t in routine_tasks
            ]
            scores = [
                u / max(0.1, _approx_km(self.all_coords[t.node_idx], depot))
                for t, u in zip(routine_tasks, urgency)
            ]
            for rank, idx in enumerate(np.argsort(scores)[::-1]):
                deadline = int((rank + 1) / n * self.WORKDAY_MINUTES)
                # α skaliert Penalties: niedriges α → stärkere Dringlichkeits-Erzwingung
                penalty = max(1, int(round(urgency[idx] * (1.0 - alpha) / self._wage_per_min)))
                routine_tasks[idx].soft_deadline_min = deadline
                routine_tasks[idx].deadline_penalty = max(1, penalty)

        for task in tasks:
            if task.task_type != "routine":
                task.soft_deadline_min = 0
                task.deadline_penalty = self._disruption_deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
                )

        logger.info(
            f"CFA-DB Initialplan: {len(tasks)} Tasks, {n} Routine, α={alpha:.3f}"
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
        OR-Tools Replan mit α-gewichtetem Drop im Retry.

        Drop-Score: (1−α)·U(k) − α·d_depot(k)
        Niedrigster Score → zuerst droppen.
        """
        alpha = self._precomputed_alpha if self._precomputed_alpha is not None else 0.5

        team_states = [
            TeamState(
                team_id=r.team_id,
                current_node=r.current_node_at(time_min),
                current_time=int(r.lunch_end_min)
                if (r.lunch_end_min is not None and time_min < r.lunch_end_min)
                else int(r.current_departure_at(time_min)),
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
            routine_tasks_rem = [t for t in remaining_tasks if t.task_type == "routine"]
            mandatory = (
                [t for t in remaining_tasks if t.task_type != "routine"] + disruption_tasks
            )
            # α-gewichteter Drop: niedrigster drop_score zuerst
            routine_tasks_rem.sort(key=lambda t: self._drop_score(t, alpha))

            solved = False
            for n_drop in range(1, len(routine_tasks_rem) + 1):
                retry_tasks = mandatory + routine_tasks_rem[n_drop:]
                if not retry_tasks:
                    break
                new_plan = self.solver.replan(retry_tasks, team_states)
                if new_plan.solver_status not in ("INFEASIBLE", "NO_SOLUTION"):
                    dropped = [t.node_idx for t in routine_tasks_rem[:n_drop]]
                    log.notes.append(
                        f"CFA-DB Replan Retry: {n_drop} Stop(s) nach α-Score ausgebaut "
                        f"{dropped}, Status: {new_plan.solver_status}"
                    )
                    solved = True
                    all_tasks = retry_tasks
                    break

            if not solved:
                log.notes.append(
                    f"CFA-DB Replan fehlgeschlagen ({new_plan.solver_status}): "
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
                    d_cost = wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    downtime_cost += d_cost
                    if d_cost > 0:
                        log.notes.append(
                            f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)"
                        )

        log.notes.append(
            f"CFA-DB Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost


# ---------------------------------------------------------------------------
# Simulator-Subklasse: setzt globalen Zustandskontext vor jedem Tag
# ---------------------------------------------------------------------------

class CFADBMaintenanceSimulator(MaintenanceSimulator):
    """Setzt α vor jedem Tag aus dem globalen Systemzustand."""

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        self.policy._n_remaining_total = len(remaining)
        self.policy._n_carryover = len(carryover_tasks)

        dsm_map = getattr(self, '_days_since_maintenance', None)
        if dsm_map is not None and len(remaining) > 0:
            global_tasks = [
                MaintenanceTask(
                    node_idx=idx + 1,
                    task_type="routine",
                    service_time=30,
                    days_since_maintenance=float(dsm_map[idx + 1]),
                )
                for idx in remaining
            ]
            phi = self.policy.extract_features(
                global_tasks,
                n_remaining_total=len(remaining),
                n_carryover=len(carryover_tasks),
            )
            self.policy._precomputed_alpha = self.policy._forward(phi)
        else:
            self.policy._precomputed_alpha = None

        return super()._run_day(
            day, remaining, team_states, carryover_tasks, day_disruptions
        )
