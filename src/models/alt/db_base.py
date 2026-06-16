"""
Dynamic Balance Base (DB-Base) — CFA-Future + zustandsabhängiger Balance-Parameter δ.

C̃(k) = θᵀφ_scaled(k)  aus CFA-Future-Training (data/training/cfa_future/theta.json)
δ(S_t) ∈ [0, 1]        aus DBBalanceModel (data/training/db_base/model.pkl)

Score-Funktion Initialplan (Stein et al., Algorithm 2):
    score(k, team) = (1 − δ) · U(k)_norm − δ · Δτ(team, k)_norm
    Greedy Joint-Insertion über alle (k, team)-Paare; höchster Score zuerst.

Drop-Score Replan (aufsteigend sortiert = zuerst droppen):
    drop_score(k) = (1 − δ) · C̃_norm(k) − δ · G_norm(k)
    G(k) = service_time_k + detour_min_k
    C̃_norm: Min-Max über alle Kandidaten (vorab berechnet per Closure)
    G_norm : (service + detour) / _G_NORM_REF (feste Referenz)

Zustandsfeatures:
    Tagesstart (9d) : Zeit, Teamanzahl, offene Stops, Urgency-Metriken
    Replan    (16d) : obige 9 + Slack, räumliche Verteilung

δ-Berechnung:
    Tagesstart : einmalig via DBBaseMaintenanceSimulator._run_day → _precomputed_delta
    Replan     : bei jeder Störung frisch aus sim_routes + time_min

Initialplan: Score-basierte Greedy-Joint-Insertion (kein OR-Tools).
Replan     : handle_disruptions_greedy + DB-gewichteter drop_score_fn.

Referenz: Stein et al. (2024) — "Learning State-Dependent Policy
Parametrizations for Dynamic Technician Routing with Rework"
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cost_params import CostParams
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    MaintenanceSimulator,
    SimRoute,
    plan_to_sim_routes,
)
from src.planning.clustering import _approx_km
from src.planning.greedy_routing import handle_disruptions_greedy
from src.planning.vrp_solver import (
    DailyPlan,
    MaintenanceTask,
    PlannedRoute,
    TeamState,
    VRPSolver,
)

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/training/cfa_future/theta.json")
_DEFAULT_DB_MODEL_PATH = Path("data/training/db_base/model.pkl")

_DELTA_SCALE_MIN = 60.0    # Δτ-Normierung: ≈ max sinnvolle Einfügezeit (Minuten)
_G_NORM_REF = 90.0         # G-Normierung: service + detour Referenzskala (Minuten)
_CRITICAL_DSM_FRACTION = 0.8  # dsm > 0.8 × recovery_days → kritisch


# ---------------------------------------------------------------------------
# Slack-Hilfsfunktion (öffentlich, für Training nutzbar)
# ---------------------------------------------------------------------------

def compute_route_slack(
    route: SimRoute,
    time_min: float,
    workday_minutes: int,
    mat: np.ndarray,
) -> float:
    """Geschätzte Restpufferzeit (Minuten) vor Depot-Rückkehr-Deadline."""
    remaining = route.remaining_stops_at(time_min)
    if not remaining:
        return float(workday_minutes)
    last = remaining[-1]
    return_min = mat[last.node_idx, 0] / 60.0
    return max(0.0, workday_minutes - last.departure_min - return_min)


# ---------------------------------------------------------------------------
# DB-Balance-Modell
# ---------------------------------------------------------------------------

class DBBalanceModel:
    """
    Sklearn-Wrapper für das gelernte Balance-Modell δ(S_t).

    Erwartet .predict(X) → δ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
    Ohne geladenes Modell: default_delta (statischer Benchmark-Modus).

    Parameters
    ----------
    clf           : sklearn-kompatibles Classifier oder None.
    scaler        : sklearn-kompatibles Scaler oder None.
    default_delta : Fallback-δ wenn kein Modell vorhanden.
    """

    DELTA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

    def __init__(
        self,
        clf=None,
        scaler=None,
        default_delta: float = 0.3,
    ) -> None:
        self.clf = clf
        self.scaler = scaler
        self.default_delta = default_delta

    @classmethod
    def load(cls, path: Path, default_delta: float = 0.5) -> "DBBalanceModel":
        if not path.exists():
            logger.info(f"DB-Base: kein Modell unter {path} — verwende δ={default_delta}")
            return cls(default_delta=default_delta)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            clf=data.get("clf"),
            scaler=data.get("scaler"),
            default_delta=default_delta,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler}, f)

    def predict_delta(self, features: np.ndarray) -> float:
        """Gibt δ ∈ [0, 1] zurück."""
        if self.clf is None:
            return self.default_delta
        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return float(self.clf.predict(x)[0])


# ---------------------------------------------------------------------------
# DB-Base-Policy
# ---------------------------------------------------------------------------

class DBBasePolicy:
    """
    Dynamic Balance Base Policy.

    Kombiniert das CFA-Future-θ als lokale Bewertungsfunktion C̃(k) mit einem
    gelernten globalen Balance-Parameter δ(S_t), der steuert wie stark lokale
    Zukunftskosten gegen operative Routing-Effizienz gewichtet werden.

    Parameters
    ----------
    traffic_matrices : Stündliche Reisezeitmatrizen in Sekunden.
    config           : Konfigurationsdict aus config.yaml.
    all_coords       : np.ndarray shape (n_stations + 1, 2) — inkl. Depot (Index 0).
    node_to_power    : dict[int, float] — node_idx → Nennleistung [kW].
    n_stations       : Gesamtzahl der Stationen (ohne Depot).
    cost_params      : Kostenparameter (None → Standardwerte).
    stations_df      : Stationsdaten (Alter, Ladetyp). None → Fallback-Werte.
    db_model         : DBBalanceModel. None → lädt aus _DEFAULT_DB_MODEL_PATH.
    default_delta    : Fallback-δ wenn kein Modell vorhanden.
    theta_path       : Pfad zu theta.json. None → data/training/cfa_future/theta.json.
    node_to_zone     : dict[int, int] — node_idx → Zone-ID (optional).
    zone_centroids   : np.ndarray shape (n_zones, 2) — Zentroide (optional).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: Optional[np.ndarray] = None,
        node_to_power: Optional[dict[int, float]] = None,
        n_stations: int = 397,
        cost_params: Optional[CostParams] = None,
        stations_df=None,
        db_model: Optional[DBBalanceModel] = None,
        default_delta: float = 0.5,
        theta_path: Optional[Path | str] = None,
        node_to_zone: Optional[dict[int, int]] = None,
        zone_centroids: Optional[np.ndarray] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.node_to_power: dict[int, float] = node_to_power or {}
        self.n_stations = n_stations
        self.cost_params = cost_params or CostParams()
        self.n_teams: int = config["maintenance"]["n_teams"]
        self._node_to_zone = node_to_zone
        self._zone_centroids = zone_centroids

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60
        self._workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration_min: int = maint.get("lunch_duration_min", 0)
        self._use_or_tools: bool = bool(config.get("solver", {}).get("use_or_tools", True))

        fail_cfg = config.get("failure_simulation", {})
        self._p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))

        cfa_cfg = config.get("cfa", {})
        self._alpha_cfa: float = float(cfa_cfg.get("alpha", 10.0))
        self._wage_per_min: float = self.cost_params.wage_eur_per_hour / 60.0

        # 8:00-Matrix gecacht für Greedy-Initialplanung
        self._mat8: np.ndarray = traffic_matrices.get(
            self._workday_start_hour,
            next(iter(traffic_matrices.values())),
        )
        self._traffic_matrices = traffic_matrices

        # Stationsindividuelle Lookups aus stations_df
        if stations_df is not None:
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
            self._node_to_age = {}

        # Mittlere Distanz jeder Station zu allen anderen
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

        # CFA-Future-θ laden
        self._theta: Optional[np.ndarray] = None
        self._theta_means: np.ndarray = np.zeros(4)
        self._theta_stds: np.ndarray = np.ones(4)
        path = Path(theta_path) if theta_path else _DEFAULT_THETA_PATH
        if path.exists():
            with open(path) as f:
                _cfa = json.load(f)
            self._theta = np.array(_cfa["theta"], dtype=float)
            self._theta_means = np.array(_cfa.get("feature_means", np.zeros(4)), dtype=float)
            self._theta_stds = np.array(_cfa.get("feature_stds", np.ones(4)), dtype=float)
            logger.info(f"DB-Base: CFA-Future-θ={self._theta} geladen aus {path}")
        else:
            logger.warning(f"DB-Base: {path} nicht gefunden — Fallback auf power×recovery_curve")

        # DB-Modell
        if db_model is not None:
            self.db_model = db_model
        else:
            self.db_model = DBBalanceModel.load(_DEFAULT_DB_MODEL_PATH, default_delta)

        # Precomputed δ: wird von DBBaseMaintenanceSimulator vor create_initial_plan gesetzt
        self._precomputed_delta: float = self.db_model.default_delta

    def set_precomputed_delta(self, delta: float) -> None:
        self._precomputed_delta = float(delta)

    # ------------------------------------------------------------------
    # Hilfsmethoden: CFA-Future-Scoring
    # ------------------------------------------------------------------

    def _p_failure_curve(self, dsm: float) -> float:
        t = min(dsm, self._recovery_days)
        return self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days

    def _phi_cfa(self, node_idx: int, dsm: float) -> np.ndarray:
        t = min(dsm, self._recovery_days)
        rc = self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days
        return np.array([
            self.node_to_power.get(node_idx, 22.0),
            self._node_to_age.get(node_idx, 5.0),
            rc,
            self._node_to_mean_dist.get(node_idx, 5.0),
        ])

    def _station_value(self, node_idx: int, dsm: float) -> float:
        """C̃(drop k) = θᵀ × φ_scaled(k). Fallback: power × recovery_curve."""
        if self._theta is not None:
            phi = self._phi_cfa(node_idx, dsm)
            phi_scaled = (phi - self._theta_means) / np.maximum(self._theta_stds, 1e-8)
            return float(self._theta @ phi_scaled)
        return self.node_to_power.get(node_idx, 22.0) * self._p_failure_curve(dsm)

    def _value(self, node_idx: int, dsm: float) -> float:
        """Alias für _station_value — kompatibel mit PolicyAdapter.get_drop_score_fn()."""
        return self._station_value(node_idx, dsm)

    def _prepare_day(self, all_tasks: list, n_carryover: int = 0) -> None:
        """Berechnet δ vor dem Tagesstart (analog zu DBBaseMaintenanceSimulator._run_day)."""
        phi = self.extract_replan_features(
            sim_routes=[],
            time_min=0.0,
            n_open_failures=n_carryover,
            tasks=all_tasks,
        )
        delta = self.db_model.predict_delta(phi)
        self.set_precomputed_delta(delta)
        logger.debug(f"DB-Base _prepare_day: δ={delta:.3f}")

    # ------------------------------------------------------------------
    # Hilfsmethoden: Greedy-Routing
    # ------------------------------------------------------------------

    def _travel_min(self, from_node: int, to_node: int) -> float:
        return float(self._mat8[from_node, to_node]) / 60.0

    def _route_end_time(
        self,
        route: list[int],
        tasks_map: dict[int, MaintenanceTask],
    ) -> float:
        t = 0.0
        prev = 0
        for node in route:
            t += self._travel_min(prev, node)
            t += float(tasks_map[node].service_time) if node in tasks_map else 30.0
            prev = node
        return t

    def _cheapest_insertion(
        self,
        route: list[int],
        node_k: int,
        tasks_map: dict[int, MaintenanceTask],
        current_end: float,
        min_pos: int = 1,
    ) -> tuple[float, int]:
        """
        Minimale Einfügekosten Δτ und optimale Position im Tagesplan.
        Gibt (inf, -1) wenn kein feasibles Einfügen möglich.

        min_pos: erste erlaubte Einfügeposition (Standard 1 = direkt nach Depot).
            Auf n_carryover_slots + 1 setzen damit Routine-Tasks nicht vor
            Carryover-Tasks eingefügt werden können.
        """
        service_k = float(tasks_map[node_k].service_time) if node_k in tasks_map else 30.0
        full = [0] + route + [0]
        best_delta = np.inf
        best_pos = -1
        depot_return = self._travel_min(route[-1], 0) if route else 0.0

        for pos in range(min_pos, len(full)):
            prev_n = full[pos - 1]
            next_n = full[pos]
            delta = (
                self._travel_min(prev_n, node_k)
                + self._travel_min(node_k, next_n)
                - self._travel_min(prev_n, next_n)
            )
            if current_end + delta + service_k + depot_return <= self.WORKDAY_MINUTES:
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos

        return best_delta, best_pos

    def _compute_arrival_times(
        self,
        route: list[int],
        tasks_map: dict[int, MaintenanceTask],
    ) -> tuple[list[int], list[int]]:
        arrivals: list[int] = []
        departures: list[int] = []
        t = 0.0
        prev = 0
        for node in route:
            t += self._travel_min(prev, node)
            service = float(tasks_map[node].service_time) if node in tasks_map else 30.0
            arrivals.append(int(t))
            departures.append(int(t + service))
            t += service
            prev = node
        return arrivals, departures

    # ------------------------------------------------------------------
    # Feature-Extraktion
    # ------------------------------------------------------------------

    def extract_day_start_features(
        self,
        tasks: list[MaintenanceTask],
        n_carryover: int = 0,
        time_min: float = 0.0,
    ) -> np.ndarray:
        """
        9-dimensionaler Zustandsvektor — nutzbar bei Tagesstart und Replan.

        Bei Tagesstart: time_min=0 (Standard).
        Bei Replan:     time_min=aktuelle Uhrzeit, tasks=verbleibende Routine-Stops.

        f0: time_now_min
        f1: time_to_end_min
        f2: n_active_teams
        f3: n_remaining_stops
        f4: n_open_failures (Carryover / aktive Störungen)
        f5: n_critical_stations (dsm > 0.8 × recovery_days)
        f6: sum_power_pfail = Σ power_k × p_failure(dsm_k) × p_per_hour
        f7: sum_remaining_hours_inverse = Σ 1/max(1, recovery_days − dsm_k)
        f8: workload_capacity_ratio = Σ service_time / (n_teams × time_to_end)
        """
        time_to_end = max(1.0, float(self.WORKDAY_MINUTES) - time_min)
        routine_tasks = [t for t in tasks if t.task_type == "routine"]

        n_critical = 0
        sum_pf = 0.0
        sum_rhi = 0.0
        total_service = 0.0

        for t in routine_tasks:
            dsm = t.days_since_maintenance
            if dsm > _CRITICAL_DSM_FRACTION * self._recovery_days:
                n_critical += 1
            pf = self._p_failure_curve(dsm)
            power = self.node_to_power.get(t.node_idx, 22.0)
            sum_pf += power * pf * self._p_failure_per_hour
            sum_rhi += 1.0 / max(1.0, self._recovery_days - dsm)
            total_service += float(t.service_time)

        wl_ratio = total_service / max(1.0, float(self.n_teams) * time_to_end)

        return np.array([
            time_min,
            time_to_end,
            float(self.n_teams),
            float(len(tasks)),
            float(n_carryover),
            float(n_critical),
            sum_pf,
            sum_rhi,
            wl_ratio,
        ], dtype=np.float64)

    def extract_replan_features(
        self,
        sim_routes: list[SimRoute],
        time_min: float,
        n_open_failures: int,
        tasks: Optional[list[MaintenanceTask]] = None,
    ) -> np.ndarray:
        """
        16-dimensionaler Zustandsvektor — nutzbar bei Tagesstart und Replan.

        Bei Tagesstart: sim_routes=[], time_min=0, tasks=globale Stationsliste.
            → Route-Features (f9–f15) sind 0; Urgency-Features (f3–f8) aus tasks.
        Bei Replan: sim_routes=aktuelle Routen, tasks=None (aus Routen abgeleitet).

        f0  : time_now_min
        f1  : time_to_end_min
        f2  : n_active_teams
        f3  : n_remaining_stops
        f4  : n_open_failures
        f5  : n_critical_stations (dsm > 0.8 × recovery_days)
        f6  : sum_power_pfail = Σ power_k × p_failure(dsm_k) × p_per_hour
        f7  : sum_remaining_hours_inverse = Σ 1/max(1, recovery_days − dsm_k)
        f8  : workload_capacity_ratio
        f9  : total_slack_min        (0 bei Tagesstart)
        f10 : min_slack_min          (0 bei Tagesstart)
        f11 : mean_slack_min         (0 bei Tagesstart)
        f12 : share_stops_current_zone (0 ohne node_to_zone oder Tagesstart)
        f13 : mean_dist_open_to_team   (0 bei Tagesstart)
        f14 : mean_pairwise_dist_open  (0 bei Tagesstart)
        f15 : mean_dist_open_to_zone_centroid (0 ohne zone_centroids oder Tagesstart)
        """
        time_to_end = max(1.0, float(self.WORKDAY_MINUTES) - time_min)

        slacks = [
            compute_route_slack(r, time_min, self.WORKDAY_MINUTES, self._mat8)
            for r in sim_routes
        ]
        total_slack = sum(slacks)
        min_slack = min(slacks) if slacks else 0.0
        mean_slack = float(np.mean(slacks)) if slacks else 0.0

        # Urgency-Features: aus tasks (Tagesstart) oder aus Routen (Replan)
        if tasks is not None:
            routine_remaining_tasks = [t for t in tasks if t.task_type == "routine"]
            n_remaining_stops = len(tasks)

            n_critical = 0
            sum_pf = 0.0
            sum_rhi = 0.0
            total_service = 0.0
            for t in routine_remaining_tasks:
                dsm = t.days_since_maintenance
                if dsm > _CRITICAL_DSM_FRACTION * self._recovery_days:
                    n_critical += 1
                pf = self._p_failure_curve(dsm)
                power = self.node_to_power.get(t.node_idx, 22.0)
                sum_pf += power * pf * self._p_failure_per_hour
                sum_rhi += 1.0 / max(1.0, self._recovery_days - dsm)
                total_service += float(t.service_time)
            routine_remaining = []  # keine Routen → räumliche Features bleiben 0
        else:
            remaining_stops = [
                s for r in sim_routes for s in r.remaining_stops_at(time_min)
            ]
            routine_remaining = [s for s in remaining_stops if s.task_type == "routine"]
            n_remaining_stops = len(remaining_stops)

            n_critical = 0
            sum_pf = 0.0
            sum_rhi = 0.0
            total_service = 0.0
            for s in routine_remaining:
                dsm = s.days_since_maintenance
                if dsm > _CRITICAL_DSM_FRACTION * self._recovery_days:
                    n_critical += 1
                pf = self._p_failure_curve(dsm)
                power = self.node_to_power.get(s.node_idx, 22.0)
                sum_pf += power * pf * self._p_failure_per_hour
                sum_rhi += 1.0 / max(1.0, self._recovery_days - dsm)
                total_service += float(s.service_min)

        share_current_zone = 0.0
        mean_dist_to_team = 0.0
        mean_pairwise_dist = 0.0
        mean_dist_to_centroid = 0.0

        if self.all_coords is not None and routine_remaining:
            open_coords = np.array([self.all_coords[s.node_idx] for s in routine_remaining])
            team_positions = [r.current_node_at(time_min) for r in sim_routes]
            team_coords = np.array([self.all_coords[n] for n in team_positions])

            dists_to_team = [
                min(_approx_km(oc, tc) for tc in team_coords)
                for oc in open_coords
            ]
            mean_dist_to_team = float(np.mean(dists_to_team))

            if len(open_coords) > 1:
                pairwise = [
                    _approx_km(open_coords[i], open_coords[j])
                    for i in range(len(open_coords))
                    for j in range(i + 1, len(open_coords))
                ]
                mean_pairwise_dist = float(np.mean(pairwise))

            if self._node_to_zone is not None and self._zone_centroids is not None:
                team_zones = {
                    self._node_to_zone.get(r.current_node_at(time_min))
                    for r in sim_routes
                }
                team_zones.discard(None)
                n_in_zone = sum(
                    1 for s in routine_remaining
                    if self._node_to_zone.get(s.node_idx) in team_zones
                )
                share_current_zone = n_in_zone / max(1, len(routine_remaining))

                dists_centroid = []
                for s in routine_remaining:
                    zone_id = self._node_to_zone.get(s.node_idx)
                    if zone_id is not None and zone_id < len(self._zone_centroids):
                        dists_centroid.append(
                            _approx_km(self.all_coords[s.node_idx], self._zone_centroids[zone_id])
                        )
                if dists_centroid:
                    mean_dist_to_centroid = float(np.mean(dists_centroid))

        wl_ratio = total_service / max(1.0, float(self.n_teams) * time_to_end)

        return np.array([
            time_min,
            time_to_end,
            float(self.n_teams),
            float(n_remaining_stops),
            float(n_open_failures),
            float(n_critical),
            sum_pf,
            sum_rhi,
            wl_ratio,
            total_slack,
            min_slack,
            mean_slack,
            share_current_zone,
            mean_dist_to_team,
            mean_pairwise_dist,
            mean_dist_to_centroid,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Greedy-Initialplan — identische Struktur wie CFA-Future:
          1. Carryover: Nearest-Neighbor ab Depot (mandatory, immer zuerst).
          2. Routine: greedy nach δ-gewichtetem route_score_fn.

        route_score_fn = C̃(k) / dist(cur, k)^(1 + δ)
            δ=0 → identisch mit CFA-Future (C̃ / dist)
            δ→1 → Distanz dominiert stärker (Routing-Effizienz)
        """
        delta = self._precomputed_delta
        n_routine = sum(1 for t in tasks if t.task_type == "routine")
        logger.info(
            f"DB-Base Initialplan: δ={delta:.3f}, {n_routine} Routine, "
            f"{len(tasks) - n_routine} Carryover"
        )

        min_val = min(
            (self._station_value(t.node_idx, t.days_since_maintenance) for t in tasks),
            default=0.0,
        )
        shift = max(0.0, -min_val) + 1.0

        from src.planning.greedy_routing import greedy_initial_plan as _greedy_plan
        return _greedy_plan(
            tasks=tasks,
            team_assignment=team_assignment,
            all_coords=self.all_coords,
            traffic_matrices=self._traffic_matrices,
            workday_start_hour=self._workday_start_hour,
            workday_minutes=self.WORKDAY_MINUTES,
            lunch_earliest_min=self._lunch_earliest_min,
            lunch_duration_min=self._lunch_duration_min,
            n_teams=self.n_teams,
            route_score_fn=lambda node, dsm, cur, mat: (
                (self._station_value(node, dsm) + shift)
                / max(0.1, _approx_km(self.all_coords[cur], self.all_coords[node])) ** (1.0 + delta)
            ),
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
        Greedy Cheapest-Insertion mit DB-gewichtetem Drop-Score.

        1. δ aus aktuellem Systemzustand (Replan-Features) berechnen.
        2. Für jede Störung: günstigste Einfügeposition suchen.
        3. Falls infeasible: Routine-Stop nach drop_score droppen.

        drop_score(k) = (1 − δ) · C̃_norm(k) − δ · G_norm(k)
            C̃_norm: Min-Max über alle Routine-Kandidaten (Closure vorab)
            G_norm : (service_time + detour_min) / _G_NORM_REF
        """
        # δ aus aktuellem Systemzustand
        phi_replan = self.extract_replan_features(
            sim_routes=sim_routes,
            time_min=time_min,
            n_open_failures=len(disruptions),
        )
        delta = self.db_model.predict_delta(phi_replan)
        logger.debug(f"DB-Base Replan: δ={delta:.3f} bei t={time_min:.0f}min, h={hour}")

        # C̃(k) vorab für alle Routine-Kandidaten berechnen (für Min-Max-Normierung)
        all_routine = [
            s for r in sim_routes
            for s in r.remaining_stops_at(time_min)
            if s.task_type == "routine"
        ]
        cfa_vals: dict[int, float] = {
            s.node_idx: self._station_value(s.node_idx, s.days_since_maintenance)
            for s in all_routine
        }
        service_vals: dict[int, float] = {s.node_idx: float(s.service_min) for s in all_routine}

        if cfa_vals:
            cfa_arr = np.array(list(cfa_vals.values()))
            cfa_min = float(cfa_arr.min())
            cfa_rng = max(float(cfa_arr.max()) - cfa_min, 1e-8)
        else:
            cfa_min, cfa_rng = 0.0, 1.0

        def drop_score_fn(
            node_idx: int,
            dsm: float,
            remaining_hours: float,
            current_node: int,
            detour_min: float,
        ) -> float:
            c_norm = (cfa_vals.get(node_idx, 0.0) - cfa_min) / cfa_rng
            service = service_vals.get(node_idx, 30.0)
            g_norm = min(1.0, (service + detour_min) / _G_NORM_REF)
            return (1.0 - delta) * c_norm - delta * g_norm

        return handle_disruptions_greedy(
            disruptions=disruptions,
            sim_routes=sim_routes,
            time_min=time_min,
            hour=hour,
            all_coords=self.all_coords,
            traffic_matrices=self._traffic_matrices,
            workday_start_hour=self._workday_start_hour,
            workday_minutes=self.WORKDAY_MINUTES,
            cost_params=self.cost_params,
            log=log,
            drop_score_fn=drop_score_fn,
        )


# ---------------------------------------------------------------------------
# Simulator-Subklasse: berechnet δ_start vor jedem Tag
# ---------------------------------------------------------------------------

class DBBaseMaintenanceSimulator(MaintenanceSimulator):
    """
    Berechnet vor jedem Tagesstart den Balance-Parameter δ aus dem globalen
    Systemzustand und schreibt ihn als _precomputed_delta in die Policy.

    Nutzt extract_replan_features() mit tasks=global_tasks und sim_routes=[]
    (time_min=0, Route-Features auf 0). Konsistent mit dem Replan-Aufruf —
    ein 16d-Modell für beide Kontexte.
    """

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        dsm_map = getattr(self, "_days_since_maintenance", None)

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
            phi = self.policy.extract_replan_features(
                sim_routes=[],
                time_min=0.0,
                n_open_failures=len(carryover_tasks),
                tasks=global_tasks,
            )
            delta = self.policy.db_model.predict_delta(phi)
        else:
            delta = self.policy.db_model.default_delta

        self.policy.set_precomputed_delta(delta)
        logger.debug(f"DB-Base Tag {day}: δ_start={delta:.3f}")

        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)
