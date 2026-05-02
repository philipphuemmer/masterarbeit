"""
Dynamic Balance (DB) Policy — OR-Tools mit state-abhängiger Soft-Deadline-Gewichtung.

Erweiterung gegenüber Stein et al. (2024): statt Greedy-Einfügung verwendet
der Initialplan OR-Tools. α_{S_t} steuert die Penalty-Stärke der Soft-Deadlines:

    deadline_penalty = max(1, round((1 − α_{S_t}) × MAX_PENALTY))

    U(k) = power_kW[k] × dsm[k]   (stationsindividuelle Dringlichkeit)
    Deadline-Position: Rang nach U(k) / Depot-Distanz (wie CFA nach V̂)

    α_{S_t} → 0: hohe Penalties → OR-Tools erzwingt Reihenfolge nach U(k)
    α_{S_t} → 1: niedrige Penalties → OR-Tools optimiert Routing frei

α_{S_t} ∈ [0, 1] wird durch ein gelerntes MLP aus dem Zustand berechnet:
    α = σ(W3 · relu(W2 · relu(W1 · φ(S_t) + b1) + b2) + b3)

8 Zustandsmerkmale φ(S_t):
    f0: n_remaining / n_stations         – Auslastungsgrad
    f1: fraction(dsm > 90)               – Anteil dringlicher Stationen
    f2: mean_dsm / 365                   – Normierte mittlere Überfälligkeit
    f3: sum(power × dsm) / MAX           – Normierte Gesamtdringlichkeit
    f4: max(power × dsm) / MAX           – Normierte Spitzendringlichkeit
    f5: mean_dist_depot / MAX_KM         – Normierte Depotentfernung
    f6: std_dist_depot / MAX_KM          – Räumliche Streuung
    f7: n_carryover / SCALE              – Offene Carryover-Rückstände

Initialplan: OR-Tools mit α-modulierten Soft-Deadline-Penalties.
Disruption Handling: OR-Tools Replan mit U(k)-basiertem Drop (wie db_alt).

Referenz: Stein, D. et al. (2024) — "Learning State-Dependent Policy
Parametrizations for Dynamic Technician Routing with Rework"
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

_DEFAULT_POLICY_PATH = Path("data/training/db/policy.json")

_MAX_DSM = 365.0
_MAX_DEPOT_KM = 30.0
_CARRYOVER_SCALE = 10.0

# Maximale Soft-Deadline-Penalty (bei α=0, volle Dringlichkeitsdurchsetzung).
# Disruptions verwenden 4+ min/min → Routine-Deadlines bleiben nachrangig.
_MAX_ROUTINE_PENALTY = 10


class DBModel:
    """
    Dynamic Balance Policy mit state-abhängigem α-Parameter und OR-Tools Routing.

    Parameters
    ----------
    traffic_matrices : Stündliche Reisezeitmatrizen in Sekunden.
    config : Konfigurationsdict aus config.yaml.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
    node_to_power : dict[int, float] — node_idx → Nennleistung [kW].
    n_stations : Gesamtzahl der Stationen (ohne Depot).
    cost_params : Kostenparameter (None → Standardwerte).
    policy_path : Pfad zu data/training/db/policy.json. None → Standardpfad.
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

        # Zustandskontext: wird von DBMaintenanceSimulator vor jedem Tag gesetzt
        self._n_remaining_total: int = n_stations
        self._n_carryover: int = 0

        if weights_override is not None:
            self._load_weights(weights_override)
            logger.info("DB: Gewichte direkt übergeben.")
        else:
            path = Path(policy_path) if policy_path else _DEFAULT_POLICY_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"DB-Policy nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train/train_db.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self._load_weights(data)
            logger.info(f"DB: Policy geladen aus {path}")

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
    # Feature-Extraktion & Vorwärtsdurchlauf
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
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        OR-Tools Initialplan mit state-abhängiger Soft-Deadline-Gewichtung.

        α moduliert die Penalty-Stärke:
          deadline_penalty = max(1, round((1−α) × MAX_PENALTY))

        Deadline-Position: Rang nach U(k)/Depot-Distanz (höhere Dringlichkeit →
        frühere Deadline). Identisch zu CFA, aber mit α-skalierter Penalty statt
        festem Wert 1.
        """
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        n = len(routine_tasks)

        # α: global vorberechnet (DBMaintenanceSimulator) oder Fallback
        _precomp = getattr(self, '_precomputed_alpha', None)
        if _precomp is not None:
            alpha = _precomp
        else:
            phi = self.extract_features(routine_tasks)
            alpha = self._forward(phi)
        logger.info(f"DB: α={alpha:.4f}, {n} Routine, "
                    f"{len(tasks) - n} Carryover")

        penalty = max(1, int(round((1.0 - alpha) * _MAX_ROUTINE_PENALTY)))

        if n > 0:
            depot = self.all_coords[0]
            urgency = [
                self.node_to_power.get(t.node_idx, 22.0) * t.days_since_maintenance
                for t in routine_tasks
            ]
            scores = [
                u / max(0.1, _approx_km(self.all_coords[t.node_idx], depot))
                for t, u in zip(routine_tasks, urgency)
            ]
            for rank, idx in enumerate(np.argsort(scores)[::-1]):
                routine_tasks[idx].soft_deadline_min = int(
                    (rank + 1) / n * self.WORKDAY_MINUTES
                )
                routine_tasks[idx].deadline_penalty = penalty

        for task in tasks:
            if task.task_type != "routine":
                task.soft_deadline_min = 0
                task.deadline_penalty = self._disruption_deadline_penalty(
                    self.node_to_power.get(task.node_idx, 22.0)
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
        OR-Tools Replan mit U(k)-basiertem Drop im Retry (wie db_alt).
        """
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
            routine_tasks_rem.sort(
                key=lambda t: self.node_to_power.get(t.node_idx, 22.0)
                * t.days_since_maintenance
            )

            solved = False
            for n_drop in range(1, len(routine_tasks_rem) + 1):
                retry_tasks = mandatory + routine_tasks_rem[n_drop:]
                if not retry_tasks:
                    break
                new_plan = self.solver.replan(retry_tasks, team_states)
                if new_plan.solver_status not in ("INFEASIBLE", "NO_SOLUTION"):
                    dropped = [t.node_idx for t in routine_tasks_rem[:n_drop]]
                    log.notes.append(
                        f"DB-Replan Retry: {n_drop} Routine-Stop(s) nach U(k) ausgebaut "
                        f"{dropped}, Status: {new_plan.solver_status}"
                    )
                    solved = True
                    all_tasks = retry_tasks
                    break

            if not solved:
                log.notes.append(
                    f"DB-Replan fehlgeschlagen ({new_plan.solver_status}): "
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
            f"DB-Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        penalty_eur = (
            self.alpha_cfa * power_kw * self.p_failure_per_hour
            * self.cost_params.downtime_eur_per_kwh
        )
        return max(1, int(round(penalty_eur / self._wage_per_min)))


# ---------------------------------------------------------------------------
# Simulator-Subklasse: setzt den globalen Zustandskontext vor jedem Tag
# ---------------------------------------------------------------------------

class DBMaintenanceSimulator(MaintenanceSimulator):
    """
    Setzt vor jedem Tag `policy._n_remaining_total` und `policy._n_carryover`,
    damit die DB-Policy den vollständigen Systemzustand für die Feature-Extraktion
    kennt — analog zum VFATrainingSimulator-Muster.
    """

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
