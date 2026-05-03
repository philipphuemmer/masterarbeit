"""
Dynamic Balance (DB) Policy — Cost Function Approximation (CFA).

Score-basierte Greedy-Einfügung mit state-abhängigem Balance-Parameter α:

    s(k, team) = (1 − α_{S_t}) · U(k) − α_{S_t} · Δτ(team, k)

    U(k) = power_kW[k] × dsm[k]   (stationsindividuelle Dringlichkeit)
    Δτ(team, k)                     (günstigste Einfügekosten in Minuten)

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

U(k) — Stationsindividuelle Dringlichkeit:
    C̃(k) = θᵀ × φ_scaled(k), φ = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others]
    θ aus data/training/cfa/theta.json (identisch mit CFA-Modell).
    Fallback wenn θ fehlt: power × recovery_curve × station_factor.

Initialplan: Kein OR-Tools — reine score-basierte Greedy-Einfügung.
Disruption Handling: OR-Tools Replan (wie CFA/VFA).

Referenz: Stein, D. et al. (2024) — "Learning State-Dependent Policy
Parametrizations for Dynamic Technician Routing with Rework"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cost_params import CostParams
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
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
    PlannedRoute,
    TeamState,
    VRPSolver,
)

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path("data/training/db/policy.json")
_DEFAULT_CFA_THETA_PATH = Path("data/training/cfa/theta.json")

_MAX_DSM = 365.0
_MAX_DEPOT_KM = 30.0
_CARRYOVER_SCALE = 10.0
_DELTA_SCALE_MIN = 60.0  # Referenz für Δτ-Normierung (≈ max sinnvolle Einfügezeit)


class DBModel:
    """
    Dynamic Balance Policy mit state-abhängigem α-Parameter.

    U(k) verwendet dieselbe CFA-Wertfunktion C̃(k) = θᵀφ_scaled(k) mit
    φ = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others].
    θ wird aus data/training/cfa/theta.json geladen (Fallback: power×recovery_curve).

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
        stations_df=None,
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

        # 8:00-Matrix für initiales Routing (einmal gecacht)
        self._mat8: np.ndarray = traffic_matrices.get(
            self._workday_start_hour,
            next(iter(traffic_matrices.values())),
        )

        # Erholungskurven-Parameter
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))
        self._use_or_tools: bool = bool(config.get("solver", {}).get("use_or_tools", True))

        # Stationsindividuelle Ausfallraten-Faktoren (Ladetyp × Alter)
        if stations_df is not None:
            from src.data.loader import get_failure_rate_factors
            factors = get_failure_rate_factors(stations_df)
            self._node_to_failure_factor: dict[int, float] = {
                i + 1: factors.get(i, 1.0) for i in range(len(stations_df))
            }
            date_col = "Inbetriebnahmedatum"
            ref = pd.Timestamp("2026-01-01")
            self._node_to_age: dict[int, float] = {}
            for i, (_, row) in enumerate(stations_df.iterrows()):
                if date_col in stations_df.columns and pd.notna(row.get(date_col)):
                    age = max(0.0, (ref - pd.Timestamp(row[date_col])).days / 365.25)
                else:
                    age = 5.0
                self._node_to_age[i + 1] = age
        else:
            self._node_to_failure_factor = {}
            self._node_to_age = {}

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

        # CFA-θ für verbesserte U(k)-Berechnung (C̃(k) = θᵀφ_scaled)
        self._theta: Optional[np.ndarray] = None
        self._theta_means: np.ndarray = np.zeros(4)
        self._theta_stds: np.ndarray = np.ones(4)
        if _DEFAULT_CFA_THETA_PATH.exists():
            with open(_DEFAULT_CFA_THETA_PATH) as f:
                _cfa = json.load(f)
            self._theta = np.array(_cfa["theta"], dtype=float)
            self._theta_means = np.array(_cfa.get("feature_means", np.zeros(4)), dtype=float)
            self._theta_stds = np.array(_cfa.get("feature_stds", np.ones(4)), dtype=float)
            logger.info(f"DB: CFA-θ={self._theta} geladen für U(k)")
        else:
            logger.warning(
                f"DB: {_DEFAULT_CFA_THETA_PATH} nicht gefunden — "
                "Fallback auf power×recovery_curve für U(k)."
            )

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
            max_possible_urgency = 150.0 * _MAX_DSM  # single-station normalisation
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

    def _phi_cfa(self, node_idx: int, dsm: float) -> np.ndarray:
        """φ(k) = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others] — wie CFA."""
        t = min(dsm, self._recovery_days)
        rc = self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days
        return np.array([
            self.node_to_power.get(node_idx, 22.0),
            self._node_to_age.get(node_idx, 5.0),
            rc,
            self._node_to_mean_dist.get(node_idx, 5.0),
        ])

    def _station_value(self, node_idx: int, dsm: float) -> float:
        """U(k) = C̃(k) = θᵀ × φ_scaled(k) — approximierte Kosten des Weglassens (wie CFA).

        Fallback auf power × recovery_curve × station_factor wenn θ nicht verfügbar.
        """
        if self._theta is not None:
            phi = self._phi_cfa(node_idx, dsm)
            phi_scaled = (phi - self._theta_means) / np.maximum(self._theta_stds, 1e-8)
            return float(self._theta @ phi_scaled)
        power = self.node_to_power.get(node_idx, 22.0)
        t = min(dsm, self._recovery_days)
        rc = self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days
        return power * rc * self._node_to_failure_factor.get(node_idx, 1.0)

    # ------------------------------------------------------------------
    # Routing-Hilfsmethoden
    # ------------------------------------------------------------------

    def _travel_min(self, from_node: int, to_node: int) -> float:
        """Reisezeit in Minuten via 8:00-Matrix (Initialplanung)."""
        return float(self._mat8[from_node, to_node]) / 60.0

    def _route_end_time(
        self,
        route: list[int],
        tasks_map: dict[int, MaintenanceTask],
    ) -> float:
        """Gibt die geschätzte Endzeit (Abfahrt letzter Stop) in Minuten ab 8:00 zurück."""
        t = 0.0
        prev = 0
        for node in route:
            t += self._travel_min(prev, node)
            service = float(tasks_map[node].service_time) if node in tasks_map else 30.0
            t += service
            prev = node
        return t

    def _cheapest_insertion(
        self,
        route: list[int],
        node_k: int,
        tasks_map: dict[int, MaintenanceTask],
        current_end: float,
    ) -> tuple[float, int]:
        """
        Minimale Einfügekosten Δτ(team, k) in Minuten und optimale Position.

        Returns (inf, -1) wenn keine feasible Position existiert.
        """
        service_k = float(tasks_map[node_k].service_time) if node_k in tasks_map else 30.0
        full = [0] + route + [0]
        best_delta = np.inf
        best_pos = -1

        for pos in range(1, len(full)):
            prev_n = full[pos - 1]
            next_n = full[pos]
            delta = (
                self._travel_min(prev_n, node_k)
                + self._travel_min(node_k, next_n)
                - self._travel_min(prev_n, next_n)
            )
            if current_end + delta + service_k <= self.WORKDAY_MINUTES:
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos

        return best_delta, best_pos

    def _compute_arrival_times(
        self,
        route: list[int],
        tasks_map: dict[int, MaintenanceTask],
    ) -> tuple[list[int], list[int]]:
        """Ankunfts- und Abfahrtszeiten (Minuten ab 8:00, gerundet) für eine Route."""
        arrivals: list[int] = []
        departures: list[int] = []
        t = 0.0
        prev = 0
        for node in route:
            t += self._travel_min(prev, node)
            arrival = t
            service = float(tasks_map[node].service_time) if node in tasks_map else 30.0
            t += service
            arrivals.append(int(arrival))
            departures.append(int(t))
            prev = node
        return arrivals, departures

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Score-basierte Greedy-Einfügung ohne OR-Tools.

        Algorithmus (Stein et al., Algorithm 2):
          1. α = MLP(φ(S_t))
          2. Für alle (k, team): score = (1−α)·U(k) − α·Δτ(team, k)
          3. Paar mit höchstem Score feasible einfügen.
          4. Wiederholen bis keine Stationen mehr oder kein Paar feasible.
        """
        tasks_map: dict[int, MaintenanceTask] = {t.node_idx: t for t in tasks}
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        carryover_tasks = [t for t in tasks if t.task_type != "routine"]

        # α: global vorberechnet (DBMaintenanceSimulator) oder Fallback auf heutiges Subset
        _precomp = getattr(self, '_precomputed_alpha', None)
        if _precomp is not None:
            alpha = _precomp
        else:
            phi = self.extract_features(routine_tasks)
            alpha = self._forward(phi)
        logger.info(f"DB: α={alpha:.4f}, {len(routine_tasks)} Routine, {len(carryover_tasks)} Carryover")

        # Routen pro Team initialisieren
        routes: dict[int, list[int]] = {i: [] for i in range(self.n_teams)}
        end_times: dict[int, float] = {i: 0.0 for i in range(self.n_teams)}

        # team_assignment → welche Routine-Stationen gehören welchem Team
        routine_for_team: dict[int, set[int]] = {i: set() for i in range(self.n_teams)}
        use_assignment = (
            team_assignment is not None
            and self.config.get("planning", {}).get("use_team_assignment", True)
        )
        if use_assignment:
            for tid, nodes in team_assignment.items():
                for node in nodes:
                    t = tasks_map.get(node)
                    if t and t.task_type == "routine":
                        routine_for_team[tid].add(node)
        else:
            all_routine_nodes = {t.node_idx for t in routine_tasks}
            for tid in range(self.n_teams):
                routine_for_team[tid] = all_routine_nodes

        # Carryover-Tasks vorab in Teams laden (mandatory, Reihenfolge: FIFO)
        if use_assignment:
            for tid in range(self.n_teams):
                carry_nodes = [
                    node
                    for node in (team_assignment or {}).get(tid, [])
                    if tasks_map.get(node) and tasks_map[node].task_type != "routine"
                ]
                for node in carry_nodes:
                    routes[tid].append(node)
                end_times[tid] = self._route_end_time(routes[tid], tasks_map)
        else:
            # Carryover nach nächster Team-Position (Depot) verteilen
            for task in carryover_tasks:
                best_tid = min(range(self.n_teams), key=lambda i: end_times[i])
                routes[best_tid].append(task.node_idx)
                end_times[best_tid] = self._route_end_time(routes[best_tid], tasks_map)

        # Unzugewiesene Routine-Stationen
        unassigned: set[int] = {t.node_idx for t in routine_tasks}

        # C̃(k) vorberechnen und auf [0,1] normieren: verhindert Skalendominanz gegenüber Δτ.
        # Δτ wird auf [0,1] normiert via _DELTA_SCALE_MIN, sodass α direkt als Balance-Gewicht
        # zwischen Dringlichkeit und Routingeffizienz interpretierbar bleibt.
        c_vals: dict[int, float] = {
            node: self._station_value(node, tasks_map[node].days_since_maintenance)
            for node in unassigned
            if node in tasks_map
        }
        if len(c_vals) > 1:
            c_min = min(c_vals.values())
            c_range = max(max(c_vals.values()) - c_min, 1e-8)
        else:
            c_min, c_range = 0.0, 1.0

        # Score-basierte Greedy-Einfügung
        while unassigned:
            best_score = -np.inf
            best_node: Optional[int] = None
            best_tid: Optional[int] = None
            best_pos: int = -1
            best_delta: float = 0.0

            for node in unassigned:
                task = tasks_map.get(node)
                if task is None:
                    continue
                u_k = (c_vals.get(node, 0.0) - c_min) / c_range  # ∈ [0, 1]

                for tid in range(self.n_teams):
                    # Respektiere Team-Zuordnung
                    if use_assignment and node not in routine_for_team.get(tid, set()):
                        continue

                    delta, pos = self._cheapest_insertion(
                        routes[tid], node, tasks_map, end_times[tid]
                    )
                    if pos == -1:
                        continue  # nicht feasible

                    delta_norm = delta / _DELTA_SCALE_MIN  # ≈ [0, 1]
                    score = (1.0 - alpha) * u_k - alpha * delta_norm
                    if score > best_score:
                        best_score = score
                        best_node = node
                        best_tid = tid
                        best_pos = pos
                        best_delta = delta

            if best_node is None:
                break  # kein feasibles Paar mehr

            unassigned.remove(best_node)
            routes[best_tid].insert(best_pos - 1, best_node)
            end_times[best_tid] += best_delta + tasks_map[best_node].service_time

        # DailyPlan zusammenbauen
        planned_routes: list[PlannedRoute] = []
        total_travel = 0

        for tid in range(self.n_teams):
            route = routes[tid]
            if not route:
                planned_routes.append(
                    PlannedRoute(team_id=tid, stops=[], arrival_times=[], departure_times=[])
                )
                continue

            arr_times, dep_times = self._compute_arrival_times(route, tasks_map)
            planned_routes.append(
                PlannedRoute(
                    team_id=tid,
                    stops=route,
                    arrival_times=arr_times,
                    departure_times=dep_times,
                )
            )
            prev = 0
            for node in route:
                total_travel += int(self._travel_min(prev, node))
                prev = node

        return DailyPlan(
            routes=planned_routes,
            total_travel_time=total_travel,
            solver_status="FEASIBLE",
            objective_value=0,
            n_dropped=0,
            status_before_retry="",
        )

    def handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        OR-Tools Replan mit U(k)-basiertem Drop im Retry (wie CFA, aber nach U statt V̂).
        Greedy-Fallback wenn use_or_tools: false.
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
                drop_score_fn=lambda node, dsm, rem_h, cur: self._station_value(node, dsm),
            )

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
            # Drop nach aufsteigendem U(k) (niedrigste Dringlichkeit zuerst)
            routine_tasks_rem.sort(
                key=lambda t: self._station_value(t.node_idx, t.days_since_maintenance)
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

        # α aus globalem Zustand vorberechnen — alle verbleibenden Stationen,
        # damit create_initial_plan dieselbe Feature-Verteilung sieht wie das Training.
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
