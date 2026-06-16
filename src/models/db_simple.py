"""
DB-Simple — CFA-Future + zustandsabhängiger Balance-Parameter δ (3-Feature-Modell).

Idee: CFA-Future bleibt funktional 1:1 erhalten. Nur der Distanzexponent im
Greedy-Scoring wird über δ(S_t) ∈ {0.1, 0.3, 0.5, 0.7, 0.9} moduliert.

Score-Funktion Initialplan (Greedy):
    score(k) = (C̃(k) + shift) / dist(cur, k)^(2δ)
    δ=0.5 → dist^1 = identisch mit CFA-Future.
    δ<0.5 → Distanz schwächer bestraft, Zukunftswert dominiert.
    δ>0.5 → Distanz stärker bestraft, Routing-Effizienz dominiert.

Drop-Score Replan (Greedy):
    drop_score(k) = C̃(k) − (2δ) × wage_per_min × detour(k)
    δ=0.5 → β=1 = identisch mit CFA-Future.

Zustandsfeatures (3d):
    f0: day_progress = (day − 1) / 364.0       [0, 1] über 365-Tage-Simulation
    f1: n_remaining_stops                        Anzahl offener Routine-Stationen
    f2: n_critical_stations                      Anzahl mit dsm > 0.8 × recovery_days

OR-Tools-Pfad: unverändert aus CFAFutureModel (super()-Delegation).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cfa_future import CFAFutureModel
from src.models.cost_params import CostParams
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    MaintenanceSimulator,
    SimRoute,
)
from src.planning.greedy_routing import greedy_initial_plan, handle_disruptions_greedy
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/training/cfa_future/theta.json")
_DEFAULT_DB_SIMPLE_MODEL_PATH = Path("data/training/db_simple/model.pkl")

_CRITICAL_DSM_FRACTION = 0.8  # dsm > 0.8 × recovery_days → kritisch


# ---------------------------------------------------------------------------
# Regelbasiertes δ
# ---------------------------------------------------------------------------

def rule_delta(day_progress: float, n_remaining: int) -> float:
    """
    δ-basiertes Balancieren zwischen C̃ (Zukunftswert) und Routingeffizienz.

    Basierend auf empirischer Analyse:
    - Tag 1-10: 3.7-5.2 Störungen/Tag, 340-397 Stops → δ=0.1
    - Tag 11-20: 2.6-3.8 Störungen/Tag, 246-332 Stops → δ=0.3
    - Tag 21-30: 1.5-2.5 Störungen/Tag, 134-235 Stops → δ=0.5
    - Tag 31-36: 1.0-1.3 Störungen/Tag, 38-106 Stops → δ=0.7
    - Tag 37-44: 0-0.9 Störungen/Tag, 2-30 Stops → δ=0.9
    """
    if n_remaining >= 370:    # Tag ~1-12: viele Störungen, C̃ dominiert
        return 0.5
    if n_remaining >= 330:    # Tag ~13-22: noch viel Arbeit1
        return 0.5
    if n_remaining >= 180:    # Tag ~23-31: Balance-Zone
        return 0.5
    if n_remaining >= 130:     # Tag ~32-37: wenig Störungen
        return 0.5
    return 0.9                # Tag 38+: fast fertig, Routing dominiert


# ---------------------------------------------------------------------------
# Balance-Modell
# ---------------------------------------------------------------------------

class DBSimpleBalanceModel:
    """
    Sklearn-Wrapper für das gelernte Balance-Modell δ(S_t) (3-Feature-Version).

    Erwartet .predict(X) → δ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
    Fallback δ=0.5: DBSimplePolicy verhält sich dann exakt wie CFA-Future.
    """

    DELTA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

    def __init__(
        self,
        clf=None,
        scaler=None,
        default_delta: float = 0.5,
    ) -> None:
        self.clf = clf
        self.scaler = scaler
        self.default_delta = default_delta

    @classmethod
    def load(cls, path: Path, default_delta: float = 0.5) -> "DBSimpleBalanceModel":
        if not path.exists():
            logger.info(f"DB-Simple: kein Modell unter {path} — verwende δ={default_delta}")
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
        """Gibt δ ∈ DELTA_GRID zurück. Fallback: default_delta."""
        if self.clf is None:
            return self.default_delta
        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        raw = self.clf.predict(x)[0]
        # clf gibt Strings zurück (sklearn Classifier mit float-Labels als str)
        return float(raw)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DBSimplePolicy(CFAFutureModel):
    """
    DB-Simple-Policy: CFA-Future als Kern, δ moduliert Distanzexponent.

    Erbt alle _phi(), _value(), θ-Lade-Logik und OR-Tools-Pfade von CFAFutureModel.
    Greedy-Pfade werden überschrieben: δ steuert dist^(2δ) und β=2δ im Drop-Score.

    Parameters
    ----------
    db_model      : DBSimpleBalanceModel oder None → lädt aus db_model_path.
    default_delta : Fallback wenn kein Modell geladen (δ=0.5 → cfa_future identisch).
    db_model_path : Pfad zur .pkl-Datei. None → _DEFAULT_DB_SIMPLE_MODEL_PATH.
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
        db_model: Optional[DBSimpleBalanceModel] = None,
        default_delta: float = 0.5,
        db_model_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__(
            traffic_matrices=traffic_matrices,
            config=config,
            all_coords=all_coords,
            stations_df=stations_df,
            cost_params=cost_params,
            theta_path=theta_path,
            theta_override=theta_override,
        )

        if db_model is not None:
            self.db_model = db_model
        else:
            path = Path(db_model_path) if db_model_path else _DEFAULT_DB_SIMPLE_MODEL_PATH
            self.db_model = DBSimpleBalanceModel.load(path, default_delta=default_delta)

        # Wird von DBSimpleMaintenanceSimulator vor create_initial_plan gesetzt.
        self._precomputed_delta: float = self.db_model.default_delta

    def set_precomputed_delta(self, delta: float) -> None:
        self._precomputed_delta = float(delta)

    def _prepare_day(self, _all_tasks: list, n_carryover: int = 0, n_remaining: int = 0) -> None:
        """δ vor dem Tagesstart setzen — wird vom RollingHorizonRunner via prepare_day() aufgerufen.

        Spiegelt die Logik von DBSimpleMaintenanceSimulator._run_day wider:
        delta_mode "rule" nutzt rule_delta(n_remaining); andere Modi behalten default_delta.
        day_progress wird für rule_delta nicht benötigt (0.0 als Platzhalter).
        """
        delta_mode = self.config.get("db_simple", {}).get("delta_mode", "rf")
        if delta_mode == "rule":
            delta = rule_delta(0.0, n_remaining)
        else:
            delta = self.db_model.default_delta
        self.set_precomputed_delta(delta)

    # ------------------------------------------------------------------
    # Feature-Extraktion
    # ------------------------------------------------------------------

    def extract_balance_features(
        self,
        remaining: set,
        dsm_array: np.ndarray,
        day: int,
    ) -> np.ndarray:
        """
        3-dimensionaler Zustandsvektor für δ-Schätzung am Tagesstart.

        f0: day_progress      = (day − 1) / 364.0
        f1: n_remaining_stops = len(remaining)

        Parameters
        ----------
        remaining  : Menge offener Stationsindizes (0-basiert, d.h. node_idx − 1).
        dsm_array  : dsm_array[node_idx] = Tage seit letzter Wartung (node_idx 1-basiert).
        day        : aktueller Simulationstag (1-basiert).
        """
        day_progress = (day - 1) / 364.0
        n_remaining = float(len(remaining))

        return np.array([day_progress, n_remaining], dtype=np.float64)

    # ------------------------------------------------------------------
    # Policy-Schnittstelle (Greedy-Pfad überschrieben, OR-Tools delegiert)
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Greedy-Pfad: score = (C̃ + shift) / dist^(2δ).
        OR-Tools-Pfad: identisch mit CFA-Future (super()-Delegation).
        """
        if not self._use_or_tools:
            delta = self._precomputed_delta
            n_routine = sum(1 for t in tasks if t.task_type == "routine")
            logger.info(
                f"DB-Simple Greedy-Initialplan: δ={delta:.3f}, {len(tasks)} Tasks, "
                f"{n_routine} Routine (C̃ / dist^(2δ={2*delta:.2f}))."
            )
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
                route_score_fn=lambda node, dsm, cur, mat: (
                    (self._value(node, dsm) + shift)
                    / max(1.0, mat[cur, node]) ** (2.0 * delta)
                ),
            )

        # OR-Tools: unverändert aus CFAFutureModel
        return super().create_initial_plan(tasks, team_assignment)

    def _get_drop_score_fn_at(self, sim_routes, time_min: float, disruptions) -> object:
        """Drop-Score-Funktion für PolicyAdapter (RH-Pfad): identisch zu handle_disruptions."""
        delta = self._precomputed_delta
        return lambda node, dsm, rem_h, cur, det: (
            self._value(node, dsm) - (2.0 * delta) * self._wage_per_min * det
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
        Greedy-Pfad: drop_score = C̃(k) − (2δ) × wage_per_min × detour(k).
        OR-Tools-Pfad: identisch mit CFA-Future (super()-Delegation).
        """
        if not self._use_or_tools:
            delta = self._precomputed_delta
            logger.debug(
                f"DB-Simple Greedy-Replan: δ={delta:.3f}, β={2*delta:.3f} "
                f"bei t={time_min:.0f}min, h={hour}"
            )
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
                    self._value(node, dsm) - (2.0 * delta) * self._wage_per_min * det
                ),
            )

        # OR-Tools: unverändert aus CFAFutureModel
        return super().handle_disruptions(disruptions, sim_routes, time_min, hour, log)


# ---------------------------------------------------------------------------
# Simulator-Subklasse: berechnet δ_start vor jedem Tag
# ---------------------------------------------------------------------------

class DBSimpleMaintenanceSimulator(MaintenanceSimulator):
    """
    Berechnet vor jedem Tagesstart δ aus dem 3d-Zustandsvektor und schreibt
    ihn als _precomputed_delta in die Policy.

    Nutzt extract_balance_features(remaining, dsm_array, day).
    Das δ bleibt für den gesamten Tag konstant (Variante A).

    delta_log: dict[int, float] — Tag → δ_start, nach dem Lauf abrufbar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_log: dict[int, float] = {}

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        dsm_map = getattr(self, "_days_since_maintenance", None)
        delta_mode = self.config.get("db_simple", {}).get("delta_mode", "rf")

        if dsm_map is not None and len(remaining) > 0:
            phi = self.policy.extract_balance_features(remaining, dsm_map, day)
            if delta_mode == "rule":
                delta = rule_delta(float(phi[0]), int(phi[1]))
            elif delta_mode == "rf":
                delta = self.policy.db_model.predict_delta(phi)
            else:  # "fixed"
                delta = self.policy.db_model.default_delta
        else:
            phi = None
            delta = self.policy.db_model.default_delta

        self.policy.set_precomputed_delta(delta)
        self.delta_log[day] = delta
        logger.debug(
            f"DB-Simple Tag {day}: δ_start={delta:.3f} (mode={delta_mode}, "
            f"remaining={len(remaining)}, features={phi})"
        )

        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)

    def write_log(self, result, path: str, label: str = "SIMULATION") -> None:
        """Schreibt das Standard-Log und hängt eine δ-Zeittafel an."""
        super().write_log(result, path, label)
        if not self.delta_log:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("DELTA-PROTOKOLL (δ_start pro Tag)\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Tag':>4}  {'δ':>5}\n")
            f.write("-" * 40 + "\n")
            for day in sorted(self.delta_log):
                f.write(f"{day:>4d}  {self.delta_log[day]:.1f}\n")
            f.write("-" * 40 + "\n")
