"""
CFA-Future-Training: Kontrastive C̃(drop k)-Approximation via Suffix-Simulation.

Statt eine globale Zustandswertfunktion V(s) = θᵀ Σφ(k) zu lernen, approximiert
dieses Training direkt die Kosten des Weglassens einer einzelnen Station:

    C̃(drop k) ≈ θᵀ φ(k)

Labels kommen aus gepaarten Suffix-Simulationen (Common-Random-Numbers-Prinzip):

    label_k = cost_drop_k - cost_serve_k

    serve_k: k wird an Tag t gewartet → dsm[k] = 0 nach Tag t
    drop_k:  k wird an Tag t NICHT gewartet → dsm[k] steigt weiter

Exogen/endogen-Trennung:
    - j≠k: identische Störungsereignisse in beiden Pfaden (vorab gesampelt)
    - k:   Störungswahrscheinlichkeit entwickelt sich pfadabhängig (endogen, dsm-basiert)

Hyperparameter:
    H = 30 Tage Suffix-Horizont
    γ = 0.995 Diskontierungsfaktor pro Tag
    Terminalwert: γ^H × (V̂_drop - V̂_serve) mit V̂ aus dem vorherigen θ
    Sampling: jeder SAMPLE_EVERY_N_DAYS-te Tag, K_STATIONS stratifiziert pro Tag
    Stratifikation: je eine Station aus hohem / mittlerem / niedrigem Prioritätsbereich

Iteratives Policy Iteration (wie train_cfa.py):
    Runde 1: Myopic-Policy als Bootstrap
    Runde r: CFA-Future(θ_{r-1})-Policy

Ausführen:
    python scripts/train/train_cfa_future.py
    python scripts/train/train_cfa_future.py --runs 10 --rounds 3 --max-days 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa_future import CFAFutureModel
from src.models.cost_params import CostParams
from src.models.myopic import MyopicPolicy
from src.models.simulator import (
    DisruptionEvent,
    MaintenanceSimulator,
)
from src.planning.clustering import ZoneClusterer, _approx_km
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import MaintenanceTask, TeamState, VRPSolver

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------

SUFFIX_HORIZON = 30       # Tage
DISCOUNT = 0.995          # γ pro Tag
SAMPLE_EVERY_N_DAYS = 5   # Tage zwischen Snapshot-Punkten
K_STATIONS = 3            # Stationen pro gesampeltem Tag (stratifiziert)
N_FEATURES = 4
N_REPS_PER_LABEL = 3      # CRN-Replikationen pro Trainingspunkt (Varianzreduktion)
RIDGE_LAMBDA = 0.1        # Ridge-Regularisierungsparameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# φ(k) als Modulfunktion (geteilt zwischen SnapshotSim und SuffixSim)
# ---------------------------------------------------------------------------

def _compute_phi(
    node_idx: int,
    dsm: float,
    config: dict,
    node_to_power: dict[int, float],
    node_to_age: dict[int, float],
    node_to_mean_dist: dict[int, float],
) -> np.ndarray:
    """Feature vector φ(k) = [power_kW, age_years, recovery_curve(dsm), mean_dist_km]."""
    fail_cfg = config.get("failure_simulation", {})
    recovery_days = float(fail_cfg.get("recovery_days", 365))
    initial_factor = float(fail_cfg.get("initial_factor", 0.1))
    dsm_c = min(dsm, recovery_days)
    recovery_curve = initial_factor + (1.0 - initial_factor) * dsm_c / recovery_days
    return np.array([
        node_to_power.get(node_idx, 22.0),
        node_to_age.get(node_idx, 5.0),
        recovery_curve,
        node_to_mean_dist.get(node_idx, 5.0),
    ])


def _build_station_maps(
    df: pd.DataFrame,
    coords: np.ndarray,
) -> tuple[dict, dict, dict]:
    """Berechnet node_to_power, node_to_age, node_to_mean_dist (unveränderlich)."""
    date_col = "Inbetriebnahmedatum"
    ref = pd.Timestamp("2026-01-01")
    node_to_age: dict[int, float] = {}
    for i, (_, row) in enumerate(df.iterrows()):
        nid = i + 1
        if date_col in df.columns and pd.notna(row.get(date_col)):
            age = max(0.0, (ref - pd.Timestamp(row[date_col])).days / 365.25)
        else:
            age = 5.0
        node_to_age[nid] = age

    n = len(coords)
    node_to_mean_dist: dict[int, float] = {
        i: float(np.mean([
            _approx_km(coords[i], coords[j]) for j in range(1, n) if j != i
        ]))
        for i in range(1, n)
    } if n > 2 else {}

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df.iterrows())
    }

    return node_to_power, node_to_age, node_to_mean_dist


# ---------------------------------------------------------------------------
# Szenario-Sampling (vorab, für Common Random Numbers)
# ---------------------------------------------------------------------------

def sample_disruption_scenario(
    dsm: np.ndarray,
    n_stations: int,
    config: dict,
    rng: np.random.Generator,
    horizon: int,
    traffic_matrices: dict[int, np.ndarray],
    node_to_power: dict[int, float],
    node_to_failure_factor: dict[int, float],
    cost_params: CostParams,
) -> list[list[DisruptionEvent]]:
    """
    Sampelt Störungsereignisse für alle Stationen über `horizon` Tage vorab.
    Gibt scenario[d] = Liste von DisruptionEvents für Tag d zurück.

    Die dsm-Entwicklung im Szenario basiert auf keiner Wartung (worst case für j≠k).
    Diese Events werden für j≠k in beiden Suffix-Pfaden identisch verwendet.
    Für k wird endogen neu gezogen (pfadabhängig).
    """
    fail_cfg = config["failure_simulation"]
    p1_base = float(fail_cfg["p1_per_hour"])
    p2_base = float(fail_cfg["p2_per_hour"])
    recovery_days = float(fail_cfg.get("recovery_days", 365))
    initial_factor = float(fail_cfg.get("initial_factor", 0.1))
    cp = cost_params

    current_dsm = dsm.copy()
    scenario: list[list[DisruptionEvent]] = []

    for d in range(horizon):
        current_dsm += 1.0
        disrupted: set[int] = set()
        day_events: list[DisruptionEvent] = []

        for hour in range(8, 17):
            for node_idx in range(1, n_stations + 1):
                if node_idx in disrupted:
                    continue

                t = min(current_dsm[node_idx], recovery_days)
                factor = initial_factor + (1.0 - initial_factor) * t / recovery_days
                station_factor = node_to_failure_factor.get(node_idx, 1.0)
                power_kw = node_to_power.get(node_idx, 22.0)

                if rng.random() < p1_base * factor * station_factor:
                    day_events.append(DisruptionEvent(
                        day=d, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 1", power_kw=power_kw,
                        service_min=float(cp.typ1_service_min),
                    ))
                    disrupted.add(node_idx)
                    continue

                if rng.random() < p2_base * factor * station_factor:
                    mat = traffic_matrices.get(hour, list(traffic_matrices.values())[0])
                    rt_min = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                    service_min = (
                        cp.typ2_dismount_min + rt_min
                        + cp.typ2_handling_min + cp.typ2_remount_min
                    )
                    day_events.append(DisruptionEvent(
                        day=d, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 2", power_kw=power_kw,
                        service_min=service_min,
                    ))
                    disrupted.add(node_idx)

        scenario.append(day_events)

    return scenario


# ---------------------------------------------------------------------------
# Snapshot-Datenstruktur
# ---------------------------------------------------------------------------

@dataclass
class DaySnapshot:
    """Zustandsschnappschuss nach Ausführung von Tag t."""
    day: int
    remaining_after: set[int]         # station_idx (0-basiert) nach Tag t
    dsm_after: np.ndarray             # dsm nach Tag t (node_idx 1-basiert)
    carryover_after: list[MaintenanceTask]
    visited_today: set[int]           # station_idx der heute abgeschlossenen Routine
    dsm_before: np.ndarray            # dsm VOR Tag t (für drop_k: dsm[k] nicht zurückgesetzt)


# ---------------------------------------------------------------------------
# SnapshotSim: zeichnet Tageszustände an jedem SAMPLE_EVERY_N_DAYS-ten Tag auf
# ---------------------------------------------------------------------------

class SnapshotSim(MaintenanceSimulator):
    """
    Erweitert MaintenanceSimulator um Zustandsschnappschüsse.

    Da die dsm-Aktualisierung in run() (nicht in _run_day()) erfolgt,
    wird dsm_after manuell aus dsm_before + visited_today berechnet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.snapshots: list[DaySnapshot] = []

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        dsm_before = self._days_since_maintenance.copy()
        remaining_before = set(remaining)

        result, sim_routes, new_carryover = super()._run_day(
            day, remaining, team_states, carryover_tasks, day_disruptions
        )

        visited_today: set[int] = {
            stop.node_idx - 1
            for route in sim_routes
            for stop in route.stops
            if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES
        }

        if day % SAMPLE_EVERY_N_DAYS == 0 and remaining_before:
            dsm_after = dsm_before + 1.0
            for station_idx in visited_today:
                dsm_after[station_idx + 1] = 0.0

            remaining_after = remaining_before - visited_today

            carryover_after = [
                MaintenanceTask(
                    node_idx=ev.node_idx,
                    task_type="carryover",
                    priority=1,
                    service_time=int(round(ev.service_min)),
                )
                for ev in new_carryover
            ]

            self.snapshots.append(DaySnapshot(
                day=day,
                remaining_after=remaining_after,
                dsm_after=dsm_after,
                carryover_after=carryover_after,
                visited_today=visited_today,
                dsm_before=dsm_before,
            ))

        return result, sim_routes, new_carryover


# ---------------------------------------------------------------------------
# SuffixSim: Suffix-Simulation aus vorgegebenem Zustand
# ---------------------------------------------------------------------------

class SuffixSim(MaintenanceSimulator):
    """
    Führt eine H-Tage-Simulation aus einem vorgegebenen Zustand durch.

    Störungen für j≠k: aus vorab gesampeltem Szenario (exogen, identisch in beiden Pfaden).
    Störungen für k:   pfadabhängig neu gezogen (endogen).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scenario: list[list[DisruptionEvent]] | None = None
        self._endogen_node: int | None = None
        self._suffix_day_counter: int = 0
        self._suffix_rng: np.random.Generator = np.random.default_rng()
        if not hasattr(self, "_node_to_failure_factor"):
            self._node_to_failure_factor = {}

    def configure(
        self,
        scenario: list[list[DisruptionEvent]],
        endogen_node: int,
        rng_seed: int | None = None,
    ) -> None:
        self._scenario = scenario
        self._endogen_node = endogen_node
        self._suffix_day_counter = 0
        self._suffix_rng = np.random.default_rng(rng_seed)

    def _generate_day_disruptions(self, day: int) -> list[DisruptionEvent]:
        """
        j≠k: Events aus vorab gesampeltem Szenario (identisch in serve und drop).
        k:   endogen neu gezogen basierend auf aktuellem dsm[k].
        """
        if self._scenario is None:
            return super()._generate_day_disruptions(day)

        d = self._suffix_day_counter
        self._suffix_day_counter += 1

        if d >= len(self._scenario):
            return []

        k = self._endogen_node
        events: list[DisruptionEvent] = []
        k_in_scenario = False

        for ev in self._scenario[d]:
            if ev.node_idx == k:
                k_in_scenario = True
                new_ev = self._draw_for_node(k, day)
                if new_ev is not None:
                    events.append(new_ev)
            else:
                events.append(ev)

        # k hatte im Szenario keinen Event — trotzdem endogen prüfen
        if k is not None and not k_in_scenario:
            new_ev = self._draw_for_node(k, day)
            if new_ev is not None:
                events.append(new_ev)

        return events

    def _draw_for_node(self, node_idx: int, day: int) -> DisruptionEvent | None:
        """Endogene Störungsziehung für node_idx basierend auf aktuellem dsm."""
        fail_cfg = self.config["failure_simulation"]
        p1_base = float(fail_cfg["p1_per_hour"])
        p2_base = float(fail_cfg["p2_per_hour"])
        recovery_days = float(fail_cfg.get("recovery_days", 365))
        initial_factor = float(fail_cfg.get("initial_factor", 0.1))

        t = min(self._days_since_maintenance[node_idx], recovery_days)
        factor = initial_factor + (1.0 - initial_factor) * t / recovery_days
        station_factor = self._node_to_failure_factor.get(node_idx, 1.0)
        power_kw = self.node_to_power.get(node_idx, 22.0)
        cp = self.cost_params

        for hour in range(8, 17):
            if self._suffix_rng.random() < p1_base * factor * station_factor:
                return DisruptionEvent(
                    day=day, hour=hour, node_idx=node_idx,
                    disruption_type="Typ 1", power_kw=power_kw,
                    service_min=float(cp.typ1_service_min),
                )
            if self._suffix_rng.random() < p2_base * factor * station_factor:
                mat = self.traffic_matrices.get(hour, list(self.traffic_matrices.values())[0])
                rt_min = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                service_min = (
                    cp.typ2_dismount_min + rt_min
                    + cp.typ2_handling_min + cp.typ2_remount_min
                )
                return DisruptionEvent(
                    day=day, hour=hour, node_idx=node_idx,
                    disruption_type="Typ 2", power_kw=power_kw,
                    service_min=service_min,
                )
        return None

    def run_from_state(
        self,
        initial_remaining: set[int],
        initial_dsm: np.ndarray,
        initial_carryover: list[MaintenanceTask],
        node_to_power: dict[int, float],
        node_to_age: dict[int, float],
        node_to_mean_dist: dict[int, float],
        max_days: int = SUFFIX_HORIZON,
        gamma: float = DISCOUNT,
        terminal_theta: np.ndarray | None = None,
        terminal_mu: np.ndarray | None = None,
        terminal_sigma: np.ndarray | None = None,
    ) -> float:
        """
        Führt Suffix-Simulation durch und gibt diskontierte Gesamtkosten zurück.

        Terminalwert (ab Runde 2): γ^max_days × V̂(s) = γ^max_days × θ_prev^T Σφ_scaled(k).
        """
        self._days_since_maintenance = initial_dsm.copy()
        self._suffix_day_counter = 0

        remaining: set[int] = set(initial_remaining)
        carryover_tasks: list[MaintenanceTask] = list(initial_carryover)

        total_cost = 0.0

        for d in range(max_days):
            if not remaining and not carryover_tasks:
                break

            team_states = [
                TeamState(team_id=i, current_node=0, current_time=0)
                for i in range(self.n_teams)
            ]
            day_disruptions = self._generate_day_disruptions(d + 1)

            result, sim_routes, new_carryover = self._run_day(
                d + 1, remaining, team_states, carryover_tasks, day_disruptions,
            )

            total_cost += gamma**d * result.total_cost_eur

            for route in sim_routes:
                for stop in route.stops:
                    if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES:
                        remaining.discard(stop.node_idx - 1)

            self._days_since_maintenance += 1.0
            for route in sim_routes:
                for stop in route.stops:
                    if stop.departure_min <= self.WORKDAY_MINUTES:
                        self._days_since_maintenance[stop.node_idx] = 0.0

            carryover_tasks = [
                MaintenanceTask(
                    node_idx=ev.node_idx,
                    task_type="carryover",
                    priority=1,
                    service_time=int(round(ev.service_min)),
                )
                for ev in new_carryover
            ]

        # Terminalwert für verbleibende Stationen
        if terminal_theta is not None and remaining:
            mu    = terminal_mu    if terminal_mu    is not None else np.zeros(N_FEATURES)
            sigma = terminal_sigma if terminal_sigma is not None else np.ones(N_FEATURES)
            v_term = 0.0
            for idx in remaining:
                node_idx = idx + 1
                dsm_val = float(self._days_since_maintenance[node_idx])
                phi = _compute_phi(node_idx, dsm_val, self.config, node_to_power, node_to_age, node_to_mean_dist)
                phi_s = (phi - mu) / np.maximum(sigma, 1e-8)
                v_term += float(terminal_theta @ phi_s)
            total_cost += gamma**max_days * v_term

        return total_cost


# ---------------------------------------------------------------------------
# Stratifiziertes Stationen-Sampling
# ---------------------------------------------------------------------------

def _sample_stations_stratified(
    candidates: list[int],
    dsm_before: np.ndarray,
    config: dict,
    node_to_power: dict[int, float],
    node_to_age: dict[int, float],
    node_to_mean_dist: dict[int, float],
    rng: np.random.Generator,
    k: int = K_STATIONS,
) -> list[int]:
    """
    Wählt k Stationen aus candidates stratifiziert nach Priorität.

    Priorität = power × recovery_curve(dsm).
    Je eine Station aus hohem / mittlerem / niedrigem Tertil, damit das
    Trainingsset nicht nur triviale Low-Value-Drops enthält.
    """
    if len(candidates) <= k:
        return list(candidates)

    scores = [
        _compute_phi(
            station_idx + 1, float(dsm_before[station_idx + 1]),
            config, node_to_power, node_to_age, node_to_mean_dist,
        )[0] * _compute_phi(
            station_idx + 1, float(dsm_before[station_idx + 1]),
            config, node_to_power, node_to_age, node_to_mean_dist,
        )[2]
        for station_idx in candidates
    ]

    order = np.argsort(scores)
    n = len(order)
    tertile = max(1, n // 3)
    tertiles = [order[:tertile], order[tertile: 2 * tertile], order[2 * tertile:]]

    selected = []
    for t_range in tertiles[:k]:
        if len(t_range) > 0:
            selected.append(candidates[int(rng.choice(t_range))])

    while len(selected) < k and len(selected) < len(candidates):
        pool = [c for c in candidates if c not in selected]
        selected.append(int(rng.choice(pool)))

    return selected[:k]


# ---------------------------------------------------------------------------
# Kontrastive Trainingsrunde
# ---------------------------------------------------------------------------

def run_contrastive_round(
    round_idx: int,
    cfg: dict,
    coords: np.ndarray,
    df_base: pd.DataFrame,
    mats: dict,
    seeds: list[int],
    max_days: int,
    theta_prev: np.ndarray | None,
    station_mu: np.ndarray,
    station_sigma: np.ndarray,
    node_to_power: dict[int, float],
    node_to_age: dict[int, float],
    node_to_mean_dist: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Führt N Baseline-Simulationen durch und sammelt kontrastive Datenpunkte.

    Pro gesampelten Tag t und K stratifizierten Stationen k:
        1. Vorab-Szenario für H Tage sampeln (Common Random Numbers für j≠k)
        2. N_REPS_PER_LABEL unabhängige serve_k/drop_k Replikationen
        3. label_mean_k = Ø(cost_drop − cost_serve) über Replikationen
        4. label_std_k  = std(cost_drop − cost_serve)  → WLS-Gewicht
    """
    X_all: list[np.ndarray] = []
    y_all: list[float] = []
    w_all: list[float] = []
    use_value_based = cfg["planning"].get("zone_selection_mode", "classic") == "value_based"

    label = "Myopic" if round_idx == 1 else "CFA-Future(θ_prev)"
    print(f"\n  Runde {round_idx}: {label}\n")

    from src.data.loader import get_failure_rate_factors
    _factors = get_failure_rate_factors(df_base)
    node_to_failure_factor = {i + 1: _factors.get(i, 1.0) for i in range(len(df_base))}
    cp = CostParams()

    for run_i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"    Lauf {run_i}/{len(seeds)} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}
        scenario_rng = np.random.default_rng(seed + 10_000)

        clusterer = ZoneClusterer(n_zones=run_cfg["planning"]["n_zones"], random_state=seed)
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))
        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)

        if round_idx == 1:
            policy = MyopicPolicy(VRPSolver(mats, run_cfg, all_coords=coords), coords, mats, run_cfg)
        else:
            policy = CFAFutureModel(
                mats, run_cfg,
                all_coords=coords,
                stations_df=df_base,
                theta_override=theta_prev,
            )
            if use_value_based:
                selector.value_fn = policy._value

        # Baseline-Simulation mit Snapshot-Aufzeichnung
        snap_sim = SnapshotSim(policy, selector, coords, df_base, mats, run_cfg)
        snap_result = snap_sim.run(max_days=max_days)

        n_contrastive = 0

        for snap in snap_sim.snapshots:
            candidates = list(snap.visited_today)
            if not candidates:
                continue

            sampled_k = _sample_stations_stratified(
                candidates=candidates,
                dsm_before=snap.dsm_before,
                config=run_cfg,
                node_to_power=node_to_power,
                node_to_age=node_to_age,
                node_to_mean_dist=node_to_mean_dist,
                rng=scenario_rng,
                k=K_STATIONS,
            )

            for station_idx in sampled_k:
                node_idx = station_idx + 1
                dsm_k = float(snap.dsm_before[node_idx])

                # Szenario vorab sampeln — identisch für j≠k in beiden Pfaden
                scenario = sample_disruption_scenario(
                    dsm=snap.dsm_after,
                    n_stations=snap_sim.n_stations,
                    config=run_cfg,
                    rng=scenario_rng,
                    horizon=SUFFIX_HORIZON,
                    traffic_matrices=mats,
                    node_to_power=node_to_power,
                    node_to_failure_factor=node_to_failure_factor,
                    cost_params=cp,
                )

                suffix_kwargs = dict(
                    initial_carryover=snap.carryover_after,
                    node_to_power=node_to_power,
                    node_to_age=node_to_age,
                    node_to_mean_dist=node_to_mean_dist,
                    max_days=SUFFIX_HORIZON,
                    gamma=DISCOUNT,
                    terminal_theta=theta_prev,
                    terminal_mu=station_mu,
                    terminal_sigma=station_sigma,
                )

                dsm_drop = snap.dsm_after.copy()
                dsm_drop[node_idx] = dsm_k + 1.0
                remaining_drop = set(snap.remaining_after) | {station_idx}

                # N_REPS_PER_LABEL unabhängige CRN-Replikationen für Varianzreduktion
                diffs: list[float] = []
                for rep in range(N_REPS_PER_LABEL):
                    rep_seed = seed + (rep + 1) * 100_000

                    serve_sim = SuffixSim(policy, selector, coords, df_base, mats, run_cfg)
                    serve_sim.configure(scenario=scenario, endogen_node=node_idx, rng_seed=rep_seed)
                    cost_serve = serve_sim.run_from_state(
                        initial_remaining=set(snap.remaining_after),
                        initial_dsm=snap.dsm_after.copy(),
                        **suffix_kwargs,
                    )

                    drop_sim = SuffixSim(policy, selector, coords, df_base, mats, run_cfg)
                    drop_sim.configure(scenario=scenario, endogen_node=node_idx, rng_seed=rep_seed)
                    cost_drop = drop_sim.run_from_state(
                        initial_remaining=remaining_drop,
                        initial_dsm=dsm_drop.copy(),
                        **suffix_kwargs,
                    )

                    diffs.append(cost_drop - cost_serve)

                label_mean = float(np.mean(diffs))
                label_std = float(np.std(diffs)) if len(diffs) > 1 else 0.0

                phi = _compute_phi(node_idx, dsm_k, run_cfg, node_to_power, node_to_age, node_to_mean_dist)
                phi_scaled = (phi - station_mu) / np.maximum(station_sigma, 1e-8)

                X_all.append(phi_scaled)
                y_all.append(label_mean)
                w_all.append(label_std)
                n_contrastive += 1

        elapsed = time.time() - t0
        days = snap_result.days_to_complete or "?"
        print(
            f"fertig ({days} Tage, {snap_result.total_cost_eur:,.0f} €, "
            f"{len(snap_sim.snapshots)} Snapshots, {n_contrastive} Punkte, {elapsed:.0f}s)"
        )

    return np.array(X_all), np.array(y_all), np.array(w_all)


# ---------------------------------------------------------------------------
# WLS-Ridge-Regression
# ---------------------------------------------------------------------------

def fit_theta_wls_ridge(
    X: np.ndarray,
    y: np.ndarray,
    label_stds: np.ndarray,
    lam: float = RIDGE_LAMBDA,
    use_wls: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Ridge-Regression auf label_mean; optional WLS mit Invervarianz-Gewichten.

    use_wls=False (default): gleichmäßige Gewichte → ungewichtete Ridge.
    use_wls=True:            w_k = 1 / (label_std_k² + ε), ε = max(median(std²), 1.0).
    Ridge λ regularisiert nur die Feature-Koeffizienten, nicht den Intercept.
    """
    # NaN/Inf-Punkte filtern
    valid = np.isfinite(y) & np.isfinite(label_stds)
    if not valid.all():
        logger.warning("fit_theta_wls_ridge: %d NaN/Inf-Datenpunkte gefiltert.", int((~valid).sum()))
        X, y, label_stds = X[valid], y[valid], label_stds[valid]
    if len(X) == 0:
        raise ValueError("Keine gültigen Datenpunkte für WLS-Ridge übrig.")

    if use_wls:
        eps = max(float(np.median(label_stds ** 2)), 1.0)
        w = 1.0 / (label_stds ** 2 + eps)
    else:
        w = np.ones(len(y))

    # [X | 1] — Intercept-Spalte wird nicht regularisiert
    A = np.column_stack([X, np.ones(len(X))])

    # Ridge-Matrix: λ auf Feature-Block, 0 auf Intercept
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0

    # Effiziente WLS-Ridge: (AᵀWA + λI)⁻¹ AᵀW y  (kein N×N-Matrixprodukt)
    Aw = A * w[:, None]
    lhs = Aw.T @ A + reg
    rhs = Aw.T @ y
    coeffs = np.linalg.solve(lhs, rhs)

    theta_vec     = coeffs[:-1]
    intercept_val = float(coeffs[-1])

    y_pred = X @ theta_vec + intercept_val
    # Gewichtetes R²
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    y_mean_w = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - y_mean_w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return theta_vec, intercept_val, r2


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFA-Future-Training (kontrastive Suffix-Simulation)"
    )
    parser.add_argument("--runs",     type=int, default=10,
                        help="Anzahl Trainingsläufe pro Runde (Standard: 10)")
    parser.add_argument("--rounds",   type=int, default=3,
                        help="Anzahl Iterationsrunden (Standard: 3)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Baseline-Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/training/cfa_future/theta.json",
                        help="Ausgabepfad für θ")
    parser.add_argument("--fresh",    action="store_true",
                        help="Checkpoints ignorieren")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print("FEHLER: CFA-Future-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    print("  Berechne Stationsmerkmale...")
    node_to_power, node_to_age, node_to_mean_dist = _build_station_maps(df_base, coords)

    fail_cfg = cfg.get("failure_simulation", {})
    rec_days = float(fail_cfg.get("recovery_days", 365))
    all_phi  = np.array([
        _compute_phi(i + 1, rec_days / 2, cfg, node_to_power, node_to_age, node_to_mean_dist)
        for i in range(len(df_base))
    ])
    station_mu    = all_phi.mean(axis=0)
    station_sigma = all_phi.std(axis=0)
    station_sigma = np.where(station_sigma < 1e-8, 1.0, station_sigma)
    print(f"  μ_station = {station_mu}")
    print(f"  σ_station = {station_sigma}")

    seeds = list(range(1, args.runs + 1))
    feature_names = ["power_kW", "age_years", "recovery_curve", "mean_dist_km"]

    theta: np.ndarray | None = None
    intercept_val: float = 0.0
    history: list[dict] = []
    start_round = 1

    existing = sorted(ckpt_dir.glob("round_*.json")) if not args.fresh else []
    if existing:
        latest = existing[-1]
        with open(latest) as f:
            ckpt = json.load(f)
        theta         = np.array(ckpt["theta"], dtype=float)
        intercept_val = float(ckpt["intercept"])
        history       = ckpt.get("history", [])
        start_round   = ckpt["round"] + 1
        print(f"  Checkpoint: {latest.name}  (Runde {ckpt['round']} abgeschlossen)")
        if start_round > args.rounds:
            print(f"  Alle {args.rounds} Runden abgeschlossen.")
            sys.exit(0)

    print(
        f"\nKontrastives CFA-Future-Training:"
        f" Runde {start_round}–{args.rounds}, {args.runs} Läufe/Runde"
    )
    print(
        f"  Suffix H={SUFFIX_HORIZON} Tage, γ={DISCOUNT}, "
        f"Sampling alle {SAMPLE_EVERY_N_DAYS} Tage, K={K_STATIONS} Stationen/Tag"
    )

    t0_total = time.time()
    X = np.zeros((0, N_FEATURES))
    y = np.zeros(0)
    w = np.zeros(0)

    for round_idx in range(start_round, args.rounds + 1):
        X, y, w = run_contrastive_round(
            round_idx=round_idx,
            cfg=cfg,
            coords=coords,
            df_base=df_base,
            mats=mats,
            seeds=seeds,
            max_days=args.max_days,
            theta_prev=theta,
            station_mu=station_mu,
            station_sigma=station_sigma,
            node_to_power=node_to_power,
            node_to_age=node_to_age,
            node_to_mean_dist=node_to_mean_dist,
        )

        if len(X) == 0:
            print("  WARNUNG: Keine Datenpunkte gesammelt — Runde übersprungen.")
            continue

        print(f"\n  Ridge (λ={RIDGE_LAMBDA}) auf {len(X)} Datenpunkten ({X.shape[1]} Features)...")
        theta_new, intercept_new, r2 = fit_theta_wls_ridge(X, y, w)

        print(f"    R² (gewichtet) = {r2:.4f}")
        print(f"    intercept      = {intercept_new:+.4f} EUR")
        print(f"    ȳ (label_mean) = {np.mean(y):.2f} EUR  (σ_labels = {np.std(w):.2f})")
        for name, t in zip(feature_names, theta_new):
            print(f"    θ[{name}] = {t:+.6e}")

        history.append({
            "round": round_idx, "theta": theta_new.tolist(),
            "intercept": intercept_new, "r2": r2,
            "n_samples": len(X),
            "label_mean": float(np.mean(y)),
            "label_std_mean": float(np.mean(w)),
        })
        theta = theta_new
        intercept_val = intercept_new

        ckpt_path = ckpt_dir / f"round_{round_idx}.json"
        with open(ckpt_path, "w") as f:
            json.dump({
                "round": round_idx, "theta": theta.tolist(),
                "intercept": intercept_val, "r2": r2,
                "feature_means": station_mu.tolist(),
                "feature_stds": station_sigma.tolist(),
                "history": history,
            }, f, indent=2)
        print(f"    Checkpoint: {ckpt_path}")

    print(f"\n  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    if theta is None:
        print("FEHLER: Kein θ gelernt.")
        sys.exit(1)

    payload = {
        "theta":           theta.tolist(),
        "feature_names":   feature_names,
        "feature_means":   station_mu.tolist(),
        "feature_stds":    station_sigma.tolist(),
        "intercept":       intercept_val,
        "r2":              history[-1]["r2"] if history else None,
        "n_runs":          args.runs,
        "n_rounds":        args.rounds,
        "n_datapoints":    int(len(X)),
        "history":         history,
        "training_method": "contrastive_suffix",
        "suffix_horizon":  SUFFIX_HORIZON,
        "discount":        DISCOUNT,
        "k_stations":      K_STATIONS,
        "sample_every_n":  SAMPLE_EVERY_N_DAYS,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
