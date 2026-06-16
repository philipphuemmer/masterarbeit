"""
Hybrid-VFA – greedy Routing mit lokalem CFA-Future-Term und globalem Zustandswert.

Score(s,a) = α × L(a) + β × ΔV̂_global(s')

    L(a)           = θ_local^T × φ_scaled_local(k)     lokale Drop-Kosten (aus cfa_future)
    V̂_global(s)   = θ_global^T × φ_scaled_state(s)    globaler Zustandswert (neu gelernt)
    ΔV̂_global(k)  = V̂_global(s) − V̂_global(s ohne k)  Marginalwert einer Station

Lokale Features φ_local(k) (4, identisch zu cfa_future):
    [power_kW, age_years, recovery_curve, mean_dist_to_others]

Globale State-Features φ_state(s) (15):
    Demand:      frac_remaining, carryover_ratio, total_urgency, expected_damage,
                 mean_dsm, max_urgency, overdue_frac, critical_frac
    Spatial:     mean_depot_dist_km, std_depot_dist_km
    Team:        mean_slack_ratio, slack_imbalance, time_remaining_ratio
    Anticipation: recovery_weighted_power, risk_weighted_urgency

θ_local  → data/training/cfa_future/theta.json  (kein separates Training)
θ_global → data/training/vfa/theta.json         (train_vfa.py)

α, β konfigurierbar in config.yaml unter vfa.alpha / vfa.beta (Default: 1.0 / 1.0).
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
)
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, VRPSolver

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_THETA_PATH  = Path("data/training/cfa_future/theta.json")
_DEFAULT_GLOBAL_THETA_PATH = Path("data/training/vfa/theta.json")

N_STATE_FEATURES = 10
STATE_FEATURE_NAMES = [
    "frac_remaining",       # f0
    "carryover_ratio",      # f1
    "mean_urgency",         # f2  mean(power×dsm)
    "mean_expected_damage", # f3  mean(failure_risk×power)
    "mean_dsm",             # f4
    "max_urgency",          # f5
    "overdue_frac",         # f6  dsm > 90
    "mean_depot_dist_km",   # f7
    "std_depot_dist_km",    # f8
    "urgency_cv",           # f9  std(power×dsm) / mean(power×dsm)
]


class VFAModel:
    """
    Hybrid-VFA: lokaler CFA-Future-Term + globaler Zustandswert, greedy Routing.

    Parameters
    ----------
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden.
    config : dict
        Konfigurationsdict aus config.yaml.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    stations_df : pd.DataFrame | None
        Stationsdaten mit Nennleistung, Alter und Ladetyp-Spalten.
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    local_theta_path : Path | str | None
        Pfad zu θ_local (Default: data/training/cfa_future/theta.json).
    global_theta_path : Path | str | None
        Pfad zu θ_global (Default: data/training/vfa/theta.json).
    global_theta_override : np.ndarray | None
        Für iteratives Training: θ_global direkt übergeben (überschreibt Datei).
    global_intercept_override : float | None
        Direkt übergebener Intercept (nur mit global_theta_override).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: Optional[np.ndarray] = None,
        stations_df: Optional[pd.DataFrame] = None,
        cost_params: Optional[CostParams] = None,
        local_theta_path: Optional[Path | str] = None,
        global_theta_path: Optional[Path | str] = None,
        global_theta_override: Optional[np.ndarray] = None,
        global_intercept_override: Optional[float] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.cost_params = cost_params or CostParams()

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60
        self._workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration_min: int = maint.get("lunch_duration_min", 0)

        fail_cfg = config.get("failure_simulation", {})
        self.lambda_per_day: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        ) * 24.0
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))

        vfa_cfg = config.get("vfa", {})
        self._alpha: float = float(vfa_cfg.get("alpha", 1.0))
        self._beta:  float = float(vfa_cfg.get("beta",  1.0))

        cp = self.cost_params
        self._wage_per_min: float = cp.wage_eur_per_hour / 60.0

        # Stationslookups
        pwr_col  = "Nennleistung Ladeeinrichtung [kW]"
        date_col = "Inbetriebnahmedatum"
        ref      = pd.Timestamp("2026-01-01")

        if stations_df is not None:
            self.node_to_power: dict[int, float] = {
                i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
                for i, (_, row) in enumerate(stations_df.iterrows())
            }
            self._node_to_age: dict[int, float] = {}
            for i, (_, row) in enumerate(stations_df.iterrows()):
                if date_col in stations_df.columns and pd.notna(row.get(date_col)):
                    age = max(0.0, (ref - pd.Timestamp(row[date_col])).days / 365.25)
                else:
                    age = 5.0
                self._node_to_age[i + 1] = age
            self.n_stations: int = len(stations_df)
        else:
            self.node_to_power = {}
            self._node_to_age  = {}
            self.n_stations    = 397

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

        # --- θ_local aus cfa_future laden ---
        lpath = Path(local_theta_path) if local_theta_path else _DEFAULT_LOCAL_THETA_PATH
        if not lpath.exists():
            raise FileNotFoundError(
                f"θ_local nicht gefunden: {lpath}\n"
                "Bitte zuerst 'python scripts/train/train_cfa_future.py' ausführen."
            )
        with open(lpath) as f:
            local_data = json.load(f)
        self.theta_local             = np.array(local_data["theta"], dtype=float)
        self._local_feature_means    = np.array(
            local_data.get("feature_means", np.zeros(len(self.theta_local)))
        )
        self._local_feature_stds     = np.array(
            local_data.get("feature_stds",  np.ones(len(self.theta_local)))
        )
        logger.info(
            f"VFA: θ_local geladen aus {lpath} "
            f"(R²={local_data.get('r2', '?'):.4f})"
        )

        # --- θ_global laden oder Override ---
        if global_theta_override is not None:
            self.theta_global            = np.array(global_theta_override, dtype=float)
            self.intercept_global        = float(global_intercept_override or 0.0)
            self._global_feature_means   = np.zeros(N_STATE_FEATURES)
            self._global_feature_stds    = np.ones(N_STATE_FEATURES)
            logger.info("VFA: θ_global direkt übergeben (iteratives Training)")
        else:
            gpath = Path(global_theta_path) if global_theta_path else _DEFAULT_GLOBAL_THETA_PATH
            if not gpath.exists():
                self.theta_global          = np.zeros(N_STATE_FEATURES)
                self.intercept_global      = 0.0
                self._global_feature_means = np.zeros(N_STATE_FEATURES)
                self._global_feature_stds  = np.ones(N_STATE_FEATURES)
                logger.info(
                    f"VFA: θ_global nicht gefunden ({gpath}), β-Term inaktiv. "
                    "Bitte 'python scripts/train/train_vfa.py' ausführen."
                )
            else:
                with open(gpath) as f:
                    global_data = json.load(f)
                self.theta_global          = np.array(global_data["theta"], dtype=float)
                self.intercept_global      = float(global_data.get("intercept", 0.0))
                self._global_feature_means = np.array(
                    global_data.get("feature_means", np.zeros(N_STATE_FEATURES))
                )
                self._global_feature_stds  = np.array(
                    global_data.get("feature_stds",  np.ones(N_STATE_FEATURES))
                )
                logger.info(
                    f"VFA: θ_global geladen aus {gpath} "
                    f"(R²={global_data.get('r2', '?'):.4f})"
                )

    # ------------------------------------------------------------------
    # Lokale Wertfunktion (identisch zu cfa_future._value)
    # ------------------------------------------------------------------

    def _phi_local(self, node_idx: int, dsm: float) -> np.ndarray:
        """φ_local(k) = [power, age, recovery_curve, mean_dist]."""
        dsm_c = min(dsm, self._recovery_days)
        rc    = self._initial_factor + (1.0 - self._initial_factor) * dsm_c / self._recovery_days
        return np.array([
            self.node_to_power.get(node_idx, 22.0),
            self._node_to_age.get(node_idx, 5.0),
            rc,
            self._node_to_mean_dist.get(node_idx, 5.0),
        ])

    def _local_value(self, node_idx: int, dsm: float) -> float:
        """C̃(drop k) = θ_local^T × φ_scaled_local(k)."""
        phi = self._phi_local(node_idx, dsm)
        phi_scaled = (phi - self._local_feature_means) / np.maximum(
            self._local_feature_stds, 1e-8
        )
        return float(self.theta_local @ phi_scaled)

    def _value(self, node_idx: int, dsm: float) -> float:
        """Alias für _local_value — kompatibel mit PolicyAdapter.get_drop_score_fn()."""
        return self._local_value(node_idx, dsm)

    def _get_drop_score_fn_at(
        self,
        sim_routes: list[SimRoute],
        time_min: float,
        disruptions: list[DisruptionEvent],
    ):
        """Volle VFA-Drop-Score-Funktion zum Zeitpunkt einer Störung.

        Identische Zustandsextraktion wie handle_disruptions: delta_global
        wird aus den bei time_min noch offenen Stops berechnet.
        """
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
        disruption_stubs = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="disruption",
                priority=1,
                service_time=int(round(d.service_min)),
                days_since_maintenance=0.0,
            )
            for d in disruptions
        ]
        all_tasks = remaining_tasks + disruption_stubs
        n_existing = sum(1 for t in remaining_tasks if t.task_type != "routine")
        delta_global = self._compute_delta_global(
            all_tasks,
            n_carryover=n_existing + len(disruptions),
        )
        alpha, beta, wage = self._alpha, self._beta, self._wage_per_min
        return lambda node, dsm, rem_h, cur, det: (
            alpha * self._local_value(node, dsm)
            + beta * delta_global.get(node, 0.0)
            - wage * det
        )

    # ------------------------------------------------------------------
    # Globale Wertfunktion
    # ------------------------------------------------------------------

    def _phi_state(
        self,
        tasks: list[MaintenanceTask],
        n_carryover: int = 0,
        exclude_node: Optional[int] = None,
    ) -> np.ndarray:
        """
        10 globale Zustandsfeatures (mittelwert-normiert, orthogonal zu frac_remaining).

        f0: frac_remaining, f1: carryover_ratio,
        f2: mean_urgency, f3: mean_expected_damage, f4: mean_dsm,
        f5: max_urgency, f6: overdue_frac,
        f7: mean_depot_dist_km, f8: std_depot_dist_km,
        f9: urgency_cv
        """
        routine = [
            t for t in tasks
            if t.task_type == "routine" and t.node_idx != exclude_node
        ]
        n_remaining = len(routine)

        if routine:
            dsm_vals = np.array([t.days_since_maintenance for t in routine], dtype=np.float64)
            pow_vals = np.array(
                [self.node_to_power.get(t.node_idx, 22.0) for t in routine],
                dtype=np.float64,
            )
            urgency      = pow_vals * dsm_vals
            failure_risk = 1.0 - np.exp(-self.lambda_per_day * dsm_vals)

            f2 = float(np.mean(urgency))
            f3 = float(np.mean(failure_risk * pow_vals))
            f4 = float(np.mean(dsm_vals))
            f5 = float(np.max(urgency))
            f6 = float(np.mean(dsm_vals > 90))
            f9 = float(np.std(urgency) / max(float(np.mean(urgency)), 1e-8))

            if self.all_coords is not None:
                depot = self.all_coords[0]
                dists = np.array(
                    [_approx_km(self.all_coords[t.node_idx], depot) for t in routine],
                    dtype=np.float64,
                )
                f7 = float(np.mean(dists))
                f8 = float(np.std(dists)) if len(dists) > 1 else 0.0
            else:
                f7 = f8 = 0.0
        else:
            f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = 0.0

        f0 = n_remaining / max(1, self.n_stations)
        f1 = n_carryover / 10.0

        return np.array(
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
            dtype=np.float64,
        )

    def _global_value(self, phi_raw: np.ndarray) -> float:
        """V̂_global(s) = θ_global^T × φ_scaled(s) + intercept."""
        phi_scaled = (phi_raw - self._global_feature_means) / np.maximum(
            self._global_feature_stds, 1e-8
        )
        return float(self.theta_global @ phi_scaled) + self.intercept_global

    def _compute_delta_global(
        self,
        tasks: list[MaintenanceTask],
        n_carryover: int = 0,
    ) -> dict[int, float]:
        """ΔV̂_global(k) = V̂(s) − V̂(s ohne k) für alle Routine-Stationen."""
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        if not routine_tasks:
            return {}
        phi_s = self._phi_state(tasks, n_carryover)
        v_s   = self._global_value(phi_s)
        return {
            t.node_idx: v_s - self._global_value(
                self._phi_state(tasks, n_carryover, exclude_node=t.node_idx)
            )
            for t in routine_tasks
        }

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
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
        Greedy Initialplan mit kombiniertem lokalem + globalem Score.

        route_score_fn = (α × (L(k) + shift) + β × max(0, ΔV̂_global(k))) / dist
        """
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        n_carryover   = sum(1 for t in tasks if t.task_type != "routine")

        delta_global = self._compute_delta_global(tasks, n_carryover=n_carryover)

        if routine_tasks:
            min_local = min(
                self._local_value(t.node_idx, t.days_since_maintenance)
                for t in routine_tasks
            )
            shift = max(0.0, -min_local) + 1.0
        else:
            shift = 1.0

        def route_score_fn(node: int, dsm: float, cur: int, mat: np.ndarray) -> float:
            local_v = self._local_value(node, dsm) + shift
            global_delta = max(0.0, delta_global.get(node, 0.0))
            combined = self._alpha * local_v + self._beta * global_delta
            return combined / max(0.1, _approx_km(self.all_coords[cur], self.all_coords[node]))

        logger.info(
            f"VFA Greedy-Initialplan: {len(tasks)} Tasks, {len(routine_tasks)} Routine "
            f"(α={self._alpha}, β={self._beta})."
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
            n_teams=self.solver.n_teams,
            route_score_fn=route_score_fn,
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
        Greedy Replan mit kombiniertem Drop-Score.

        drop_score_fn = α × L(k) + β × ΔV̂_global(k) − wage × detour_min
        Höherer Score → Station wird behalten (wichtiger).
        """
        # Verbleibende Tasks für Zustandsberechnung (inkl. neue Disruptions)
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
        n_existing_disruptions = sum(
            1 for t in remaining_tasks if t.task_type != "routine"
        )
        disruption_stubs = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="disruption",
                priority=1,
                service_time=int(round(d.service_min)),
                days_since_maintenance=0.0,
            )
            for d in disruptions
        ]
        all_tasks    = remaining_tasks + disruption_stubs
        n_carryover  = n_existing_disruptions + len(disruptions)

        delta_global = self._compute_delta_global(
            all_tasks,
            n_carryover=n_carryover,
        )

        def drop_score_fn(node: int, dsm: float, rem_h: float, cur: int, det: float) -> float:
            local_v      = self._local_value(node, dsm)
            global_delta = delta_global.get(node, 0.0)
            return self._alpha * local_v + self._beta * global_delta - self._wage_per_min * det

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
            drop_score_fn=drop_score_fn,
        )
