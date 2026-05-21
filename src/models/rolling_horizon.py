"""
Rolling Horizon Runner – aktiver Policy-Improvement-Layer.

Basis-Policies (CFAFutureModel, VFAModel, DBBaseModel, …) laufen unverändert.
Der RH-Runner überschreibt an kritischen Entscheidungspunkten die Basis-Policy,
wenn eine alternative Aktion über den Horizont besser bewertet wird.

Eingriffspunkt: Drop-Entscheidung bei Störungen – wenn eine Routinewartung aus
dem Tagesplan geworfen werden muss, um Platz für eine Störung zu schaffen,
bewertet RH die top_k_candidates alternativen Drop-Entscheidungen und wählt
diejenige mit dem niedrigsten erwarteten Horizont-Gesamtkosten.

Aktivierung via config.yaml:
    rolling_horizon:
      enabled: true
      horizon_days: 7
      n_scenarios: 8
      top_k_candidates: 3
      enable_replan: true       # RH für Drop-Entscheidungen bei Störungen
      enable_initial: false     # RH für Initialplanung (Seed-Rollout)
      top_k_initial: 3
      initial_seed_block_size: 2
      fallback_to_legacy_on_timeout: true
      time_budget_sec: 5.0

enabled: false → exakt dasselbe Verhalten wie der bestehende MaintenanceSimulator.
"""
from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.cost_params import CostParams
from src.models.simulator import (
    DayResult,
    DisruptionEvent,
    HourLog,
    SimRoute,
    SimulationResult,
    _LUNCH_DURATION,
    _LUNCH_START_MIN,
    _fmt,
    _inject_lunch,
    plan_to_sim_routes,
)
from src.planning.clustering import _approx_km
from src.planning.greedy_routing import (
    _extend_route_greedily,
    _find_best_drop_and_insert,
    _find_best_insertion,
    _get_matrix,
    _insert_stop,
    _insertion_cost,
    _remove_stop_and_recompute,
    complete_route_from_partial,
)
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, PlannedRoute, TeamState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zustandsmodell
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    """
    Persistenter Systemzustand über mehrere Tage.

    Wird vom RollingHorizonRunner von Tag zu Tag fortgeschrieben.
    Dient als Eingabe für die HorizonEvaluator-Rollouts.
    """

    day: int
    days_since_maintenance: np.ndarray  # shape (n_stations+1,), Index 0 = Depot
    remaining: set[int]                 # 0-basierte Stationsindizes (node_idx - 1)
    carryover_tasks: list[MaintenanceTask]
    rng: np.random.Generator
    cumulative_cost: float = 0.0
    rh_overrides: int = 0              # Anzahl der von RH überschriebenen Basis-Policy-Entscheidungen


# ---------------------------------------------------------------------------
# Hilfsdatenklasse: Tagesergebnis für die schnelle Horizontbewertung
# ---------------------------------------------------------------------------

@dataclass
class _FastDayResult:
    cost: float
    completed_nodes: set[int]          # node_idx Routine-Stops die heute fertig wurden
    serviced_nodes: set[int]           # node_idx ALLER heute bedienten Stops (Routine + Störungen)
    carried_disruptions: list[DisruptionEvent]


# ---------------------------------------------------------------------------
# PolicyAdapter
# ---------------------------------------------------------------------------

class PolicyAdapter:
    """
    Dünner Wrapper um beliebige Policy-Modelle (CFAFutureModel, VFAModel, …).

    Stellt dem RH-Runner eine einheitliche Schnittstelle zur Verfügung und
    exponiert eine drop_score_fn, die konsistent mit der Basis-Policy ist.
    """

    def __init__(self, model) -> None:
        self.model = model

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        return self.model.create_initial_plan(tasks, team_assignment)

    def handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        return self.model.handle_disruptions(disruptions, sim_routes, time_min, hour, log)

    def get_drop_score_fn(self) -> Callable:
        """
        Gibt die drop_score_fn des zugrunde liegenden Modells zurück.

        Niedrigerer Score = zuerst droppen (konsistent mit handle_disruptions_greedy).
        Priorität: _drop_score_fn (exakte Policy-Logik) > _value/_wage_per_min > Fallback.
        """
        m = self.model
        if hasattr(m, "_drop_score_fn"):
            return m._drop_score_fn
        if hasattr(m, "_value") and hasattr(m, "_wage_per_min"):
            return lambda node, dsm, rem_h, cur, det: (
                m._value(node, dsm) - m._wage_per_min * det
            )
        return lambda node, dsm, rem_h, cur, det: -float(dsm)

    def get_value_fn(self) -> Optional[Callable[[int, float], float]]:
        """C̃(node_idx, dsm) → float; None wenn Modell keine _value-Methode hat."""
        m = self.model
        return m._value if hasattr(m, "_value") else None

    def get_route_score_fn_for_tasks(
        self, tasks: list[MaintenanceTask]
    ) -> Callable[[int, float, int], float]:
        """
        route_score_fn(node_idx, dsm, current_node) → float für greedy_initial_plan.

        Repliziert die Logik aus CFAFutureModel.create_initial_plan (Greedy-Pfad):
        score = (C̃(k) + shift) / dist(cur, k).
        Fallback für Modelle ohne _value: 1.0 (Nearest-Neighbor).
        """
        m = self.model
        if hasattr(m, "_value") and hasattr(m, "all_coords"):
            from src.planning.clustering import _approx_km
            min_val = min(
                (m._value(t.node_idx, t.days_since_maintenance) for t in tasks),
                default=0.0,
            )
            shift = max(0.0, -min_val) + 1.0
            return lambda node, dsm, cur: (
                (m._value(node, dsm) + shift)
                / max(0.1, _approx_km(m.all_coords[cur], m.all_coords[node]))
            )
        return lambda node, dsm, cur: 1.0


# ---------------------------------------------------------------------------
# DailyTaskGenerator
# ---------------------------------------------------------------------------

class DailyTaskGenerator:
    """
    Kapselt den DailyZoneSelector und erzeugt aus einem SystemState
    die tägliche Aufgabenliste (Routine + Carryover).
    """

    def __init__(
        self,
        selector,        # DailyZoneSelector
        failure_mode: str,
        n_stations: int,
    ) -> None:
        self.selector = selector
        self.failure_mode = failure_mode
        self.n_stations = n_stations

    def build_daily_tasks(
        self,
        state: SystemState,
        team_states: list[TeamState],
    ) -> tuple[object, list[MaintenanceTask], dict[int, list[int]]]:
        """
        Gibt (assignment, all_tasks, team_assignment_dict) zurück.

        Schreibt days_since_maintenance aus dem State in die Routine-Tasks.
        """
        dsm_array = (
            state.days_since_maintenance
            if self.failure_mode == "stochastic"
            else None
        )
        assignment = self.selector.select_for_day(
            list(state.remaining), team_states, state.carryover_tasks, dsm_array=dsm_array
        )
        all_tasks = [t for tasks in assignment.team_tasks.values() for t in tasks]

        if self.failure_mode == "stochastic":
            for task in all_tasks:
                if task.task_type == "routine":
                    task.days_since_maintenance = float(
                        state.days_since_maintenance[task.node_idx]
                    )

        team_assignment: dict[int, list[int]] = {
            tid: [t.node_idx for t in tasks]
            for tid, tasks in assignment.team_tasks.items()
        }
        return assignment, all_tasks, team_assignment


# ---------------------------------------------------------------------------
# HorizonEvaluator
# ---------------------------------------------------------------------------

class HorizonEvaluator:
    """
    Bewertet einen SystemState durch schnelle H-Tage-Rollouts ohne Logging.

    Wird vom RollingHorizonRunner genutzt, um alternative Drop-Kandidaten
    zu vergleichen: für jeden Kandidaten wird der erwartete H-Tage-Kostenwert
    über n_scenarios stochastische Szenarien gemittelt.
    """

    def __init__(
        self,
        policy: PolicyAdapter,
        task_generator: DailyTaskGenerator,
        all_coords: np.ndarray,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        node_to_power: dict[int, float],
        node_to_failure_factor: dict[int, float],
        cost_params: CostParams,
        n_stations: int,
        n_teams: int,
    ) -> None:
        self.policy = policy
        self.task_gen = task_generator
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.node_to_power = node_to_power
        self.node_to_failure_factor = node_to_failure_factor
        self.cost_params = cost_params
        self.n_stations = n_stations
        self.n_teams = n_teams

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60
        self.workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration: int = maint.get("lunch_duration_min", 0)

        fail_cfg = config.get("failure_simulation", {})
        self._p1_base: float = fail_cfg.get("p1_per_hour", 0.00084)
        self._p2_base: float = fail_cfg.get("p2_per_hour", 0.00028)
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))

        cp = cost_params
        self._typ1_service_min: float = cp.typ1_service_min
        self._typ2_dismount_min: float = cp.typ2_dismount_min
        self._typ2_handling_min: float = cp.typ2_handling_min
        self._typ2_remount_min: float = cp.typ2_remount_min

        # Fixer RNG nur für die Szenario-Seed-Generierung (läuft unabhängig
        # vom Simulations-RNG, sodass Evaluation reproduzierbar bleibt)
        self._eval_rng = np.random.default_rng(0)

    def draw_scenario_seeds(self, n_scenarios: int) -> list[int]:
        """
        Zieht n_scenarios Zufalls-Seeds für eine Entscheidungssituation.

        Muss *einmal* pro Störungsereignis (nicht pro Kandidat) aufgerufen werden,
        damit alle Kandidaten einer Entscheidung auf denselben Szenarien bewertet
        werden (Common-Random-Numbers-Prinzip → faire Kandidatenvergleiche).
        """
        return [int(self._eval_rng.integers(0, 2**31)) for _ in range(n_scenarios)]

    def evaluate(
        self,
        state: SystemState,
        horizon_days: int,
        scenario_seeds: list[int],
    ) -> float:
        """
        Gibt die mittleren H-Tage-Kosten über die gegebenen Szenario-Seeds zurück.

        scenario_seeds muss für alle Kandidaten einer Entscheidung identisch sein
        (via draw_scenario_seeds() erzeugt), damit der Vergleich fair bleibt.
        """
        costs: list[float] = []
        for seed in scenario_seeds:
            rng = np.random.default_rng(seed)
            s = deepcopy(state)
            s.rng = rng
            total = 0.0
            for _ in range(horizon_days):
                result = self._run_day_fast(s)
                total += result.cost
                s = self._update_state_fast(s, result)
                if not s.remaining and not s.carryover_tasks:
                    break
            costs.append(total)
        return float(np.mean(costs))

    def evaluate_with_forced_day0_plan(
        self,
        state: SystemState,
        forced_plan: DailyPlan,
        forced_tasks: list[MaintenanceTask],
        horizon_days: int,
        scenario_seeds: list[int],
    ) -> float:
        """
        Rollout-Bewertung mit festem Initialplan für Tag 0.

        Tag 0: forced_plan wird direkt simuliert (kein Neuplan durch Policy).
        Tag 1..H-1: Standard-Rollout via evaluate().

        Entspricht dem Rollout-Prinzip: erste Aktion explizit vorgegeben,
        Rest von der Basisheuristik approximiert.
        """
        if not scenario_seeds:
            return 0.0

        costs: list[float] = []
        for seed in scenario_seeds:
            rng = np.random.default_rng(seed)
            s = deepcopy(state)
            s.rng = rng
            total = 0.0
            # Tag 0: forced plan
            result = self._run_day_fast(s, forced_plan=forced_plan, forced_tasks=forced_tasks)
            total += result.cost
            s = self._update_state_fast(s, result)
            # Tage 1..H-1: Standard-Rollout
            for _ in range(horizon_days - 1):
                if not s.remaining and not s.carryover_tasks:
                    break
                result = self._run_day_fast(s)
                total += result.cost
                s = self._update_state_fast(s, result)
            costs.append(total)
        return float(np.mean(costs))

    def _run_day_fast(
        self,
        state: SystemState,
        forced_plan: Optional[DailyPlan] = None,
        forced_tasks: Optional[list[MaintenanceTask]] = None,
    ) -> _FastDayResult:
        """
        Schnelle Tagessimulation ohne Logging für Horizont-Rollouts.

        forced_plan / forced_tasks: wenn gesetzt, wird der Tagesplan nicht
        von der Policy gebaut, sondern direkt verwendet (Initial-RH Day-0-Injection).
        """
        team_states = [
            TeamState(team_id=i, current_node=0, current_time=0)
            for i in range(self.n_teams)
        ]

        if forced_plan is not None:
            all_tasks = forced_tasks or []
            daily_plan = forced_plan
        else:
            _, all_tasks, team_assignment = self.task_gen.build_daily_tasks(state, team_states)
            daily_plan = self.policy.create_initial_plan(all_tasks, team_assignment) if all_tasks else None

        sim_routes = plan_to_sim_routes(daily_plan, all_tasks, self.n_teams)

        if self._lunch_duration > 0:
            for r in sim_routes:
                _inject_lunch(r, self._lunch_earliest, self._lunch_duration)

        disruptions = self._generate_disruptions(state)
        by_hour: dict[int, list[DisruptionEvent]] = {}
        for d in disruptions:
            by_hour.setdefault(d.hour, []).append(d)

        downtime_cost = 0.0
        carried: list[DisruptionEvent] = []

        for hour in range(8, 17):
            time_min = float((hour - 8) * 60)
            if hour not in by_hour:
                continue
            log = HourLog(day=state.day, hour=hour)  # verworfen nach Aufruf
            n_handled, hour_carried, h_cost = self.policy.handle_disruptions(
                by_hour[hour], sim_routes, time_min, hour, log
            )
            downtime_cost += h_cost
            carried.extend(hour_carried)

            report_min = float((hour - 8) * 60)
            remaining_wday_h = (self.WORKDAY_MINUTES - report_min) / 60.0
            for d in hour_carried:
                downtime_cost += remaining_wday_h * d.power_kw * self.cost_params.downtime_eur_per_kwh

        completed_nodes = {
            stop.node_idx
            for route in sim_routes
            for stop in route.stops
            if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES
        }
        serviced_nodes = {
            stop.node_idx
            for route in sim_routes
            for stop in route.stops
            if stop.departure_min <= self.WORKDAY_MINUTES
        }

        op_cost = self._compute_op_cost_fast(sim_routes)
        return _FastDayResult(
            cost=op_cost + downtime_cost,
            completed_nodes=completed_nodes,
            serviced_nodes=serviced_nodes,
            carried_disruptions=carried,
        )

    def _update_state_fast(self, state: SystemState, result: _FastDayResult) -> SystemState:
        """Aktualisiert state nach einem schnellen Simulationstag."""
        s = deepcopy(state)
        s.day += 1
        s.days_since_maintenance += 1.0
        for node in result.serviced_nodes:  # alle bedienten Stops, nicht nur Routine
            s.days_since_maintenance[node] = 0.0
        for node in result.completed_nodes:  # nur Routine: aus remaining entfernen
            s.remaining.discard(node - 1)
        s.carryover_tasks = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="carryover",
                priority=1,
                service_time=int(round(d.service_min)),
            )
            for d in result.carried_disruptions
        ]
        s.cumulative_cost += result.cost
        return s

    def _generate_disruptions(self, state: SystemState) -> list[DisruptionEvent]:
        """Stochastische Störungsgenerierung (identisch zu MaintenanceSimulator)."""
        cp = self.cost_params
        disrupted_today: set[int] = set()
        events: list[DisruptionEvent] = []

        for hour in range(8, 17):
            for node_idx in range(1, self.n_stations + 1):
                if node_idx in disrupted_today:
                    continue
                t = min(state.days_since_maintenance[node_idx], self._recovery_days)
                factor = self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days
                sf = self.node_to_failure_factor.get(node_idx, 1.0)
                power_kw = self.node_to_power.get(node_idx, 22.0)

                if state.rng.random() < self._p1_base * factor * sf:
                    events.append(DisruptionEvent(
                        day=state.day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 1", power_kw=power_kw,
                        service_min=float(self._typ1_service_min),
                    ))
                    disrupted_today.add(node_idx)
                    continue

                if state.rng.random() < self._p2_base * factor * sf:
                    mat = self.traffic_matrices.get(hour, list(self.traffic_matrices.values())[0])
                    rt = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                    svc = self._typ2_dismount_min + rt + self._typ2_handling_min + self._typ2_remount_min
                    events.append(DisruptionEvent(
                        day=state.day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 2", power_kw=power_kw,
                        service_min=svc,
                    ))
                    disrupted_today.add(node_idx)

        return events

    def _compute_op_cost_fast(self, sim_routes: list[SimRoute]) -> float:
        """Betriebskosten (Lohn + Fahrt) für Horizontbewertung."""
        cp = self.cost_params
        total = 0.0
        for route in sim_routes:
            if not route.stops:
                continue
            km = 0.0
            legs = [(0, route.stops[0].node_idx, 0.0)]
            for i in range(len(route.stops) - 1):
                legs.append((
                    route.stops[i].node_idx,
                    route.stops[i + 1].node_idx,
                    route.stops[i].departure_min,
                ))
            legs.append((route.stops[-1].node_idx, 0, route.stops[-1].departure_min))
            for from_n, to_n, _ in legs:
                km += _approx_km(self.all_coords[from_n], self.all_coords[to_n])
            total += km * cp.fuel_eur_per_km + (self.WORKDAY_MINUTES / 60.0) * cp.wage_eur_per_hour
        return total


# ---------------------------------------------------------------------------
# RollingHorizonRunner
# ---------------------------------------------------------------------------

class RollingHorizonRunner:
    """
    Orchestriert die mehrtägige Simulation mit aktivem RH-Policy-Improvement.

    Bei enabled: false im Config-Flag fällt das Laufskript auf den
    bestehenden MaintenanceSimulator zurück. Der Runner selbst ist immer aktiv.

    RH-Eingriffspunkt: wenn eine Störungs-Einfügung einen Routine-Stop
    verdrängen muss, bewertet RH top_k_candidates Alternativen über den
    Horizont und wählt die kostengünstigste.
    """

    def __init__(
        self,
        policy: PolicyAdapter,
        task_generator: DailyTaskGenerator,
        evaluator: HorizonEvaluator,
        all_coords: np.ndarray,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        node_to_power: dict[int, float],
        node_to_failure_factor: dict[int, float],
        cost_params: CostParams,
        n_stations: int,
        n_teams: int,
        stations_df: pd.DataFrame,
    ) -> None:
        self.policy = policy
        self.task_gen = task_generator
        self.evaluator = evaluator
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.node_to_power = node_to_power
        self.node_to_failure_factor = node_to_failure_factor
        self.cost_params = cost_params
        self.n_stations = n_stations
        self.n_teams = n_teams

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60
        self.workday_start_hour: int = maint["workday_start_hour"]
        self._lunch_earliest: int = maint.get("lunch_earliest_min", 240)
        self._lunch_duration: int = maint.get("lunch_duration_min", 0)

        fail_cfg = config.get("failure_simulation", {})
        self._failure_mode: str = fail_cfg.get("mode", "csv")
        self._p1_base: float = fail_cfg.get("p1_per_hour", 0.00084)
        self._p2_base: float = fail_cfg.get("p2_per_hour", 0.00028)
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))

        cp = cost_params
        self._typ1_service_min: float = cp.typ1_service_min
        self._typ2_dismount_min: float = cp.typ2_dismount_min
        self._typ2_handling_min: float = cp.typ2_handling_min
        self._typ2_remount_min: float = cp.typ2_remount_min

        # Station-ID → node_idx (für CSV-Störungen)
        self._id_to_node: dict[int, int] = {
            int(row["ID"]): i + 1
            for i, (_, row) in enumerate(stations_df.iterrows())
        }

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def build_initial_state(
        self,
        seed: Optional[int] = None,
        randomize_initial_dsm: bool = False,
    ) -> SystemState:
        """Erstellt den Anfangszustand für die Simulation."""
        fail_cfg = self.config.get("failure_simulation", {})
        recovery_days = fail_cfg.get("recovery_days", 365)

        if randomize_initial_dsm:
            init_rng = np.random.default_rng(0)
            dsm = init_rng.uniform(0, recovery_days, size=self.n_stations + 1)
            dsm[0] = 0.0
        else:
            dsm = np.full(self.n_stations + 1, float(recovery_days))

        rng_seed = seed if seed is not None else self.config.get("project", {}).get("seed", 42)
        rng = np.random.default_rng(rng_seed)

        return SystemState(
            day=0,
            days_since_maintenance=dsm,
            remaining=set(range(self.n_stations)),
            carryover_tasks=[],
            rng=rng,
        )

    def run(
        self,
        initial_state: SystemState,
        rh_config: dict,
        disruptions_df: Optional[pd.DataFrame] = None,
        max_days: int = 365,
    ) -> tuple[SimulationResult, int]:
        """
        Führt die RH-Simulation durch.

        Returns
        -------
        (SimulationResult, rh_overrides_total)
        """
        # Störungsquellen vorbereiten
        if self._failure_mode == "csv":
            if disruptions_df is None:
                raise ValueError("failure_simulation.mode ist 'csv', aber kein disruptions_df übergeben.")
            by_day = self._load_disruptions_by_day(disruptions_df)
        else:
            by_day = {}

        state = deepcopy(initial_state)
        day_results: list[DayResult] = []
        same_day_handled = 0
        total_carryover = 0
        total_disruptions_generated = 0
        days_to_complete: Optional[int] = None
        last_day = 0

        for day in tqdm(range(1, max_days + 1), desc="RH-Simulation", unit="Tag"):
            last_day = day
            state.day = day

            if days_to_complete is None:
                if self._failure_mode == "stochastic":
                    day_disruptions = self._generate_day_disruptions(state)
                else:
                    day_disruptions = by_day.get(day, [])
                total_disruptions_generated += len(day_disruptions)
            else:
                day_disruptions = []

            day_result, sim_routes, carried_disruptions = self._run_day(
                state, day_disruptions, rh_config
            )
            day_results.append(day_result)
            same_day_handled += day_result.disruptions_handled
            total_carryover += day_result.disruptions_carryover

            # Zustand am Tagesende aktualisieren
            completed_nodes: set[int] = set()
            for route in sim_routes:
                for stop in route.stops:
                    if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES:
                        state.remaining.discard(stop.node_idx - 1)
                        completed_nodes.add(stop.node_idx)

            if not state.remaining and days_to_complete is None:
                days_to_complete = day
                logger.info(f"Alle {self.n_stations} Stationen nach Tag {day} gewartet.")
                # Letzter Tag: tatsächliche Arbeitszeit
                op_cost, wage_cost, fuel_cost = self._compute_op_cost(sim_routes, is_last_day=True)
                day_result.operational_cost_eur = op_cost
                day_result.wage_cost_eur = wage_cost
                day_result.fuel_cost_eur = fuel_cost
                carried_disruptions = []

            if self._failure_mode == "stochastic":
                state.days_since_maintenance += 1.0
                # Alle heute bedienten Stops zurücksetzen — identisch zu MaintenanceSimulator:
                # dort wird dsm für ALLE Stops mit departure_min <= WORKDAY_MINUTES auf 0 gesetzt,
                # nicht nur Routine-Stops. Ohne diesen Reset akkumulieren Stationen mit Störungen
                # fälschlicherweise dsm, was ihre Ausfallwahrscheinlichkeit künstlich erhöht.
                for route in sim_routes:
                    for stop in route.stops:
                        if stop.departure_min <= self.WORKDAY_MINUTES:
                            state.days_since_maintenance[stop.node_idx] = 0.0

            state.carryover_tasks = [
                MaintenanceTask(
                    node_idx=d.node_idx,
                    task_type="carryover",
                    priority=1,
                    service_time=int(round(d.service_min)),
                )
                for d in carried_disruptions
            ]
            state.cumulative_cost += day_result.total_cost_eur

            if days_to_complete is not None and not state.carryover_tasks:
                break

        if self._failure_mode == "csv":
            sim_end = days_to_complete if days_to_complete else last_day
            total_disruptions = sum(len(by_day.get(d, [])) for d in range(1, sim_end + 1))
        else:
            total_disruptions = total_disruptions_generated

        result = SimulationResult(
            day_results=day_results,
            total_disruptions=total_disruptions,
            same_day_handled=same_day_handled,
            total_carryover=total_carryover,
            days_to_complete=days_to_complete,
            remaining_stations_at_end=len(state.remaining),
        )
        return result, state.rh_overrides

    # ------------------------------------------------------------------
    # Initial-RH: Seed-Rollout für Initialplanung
    # ------------------------------------------------------------------

    def _extract_base_seed(
        self,
        route_stops: list[int],
        zone_node_set: set[int],
        block_size: int,
    ) -> list[int]:
        """
        Längster Präfix von route_stops der in zone_node_set liegt, max. block_size Stops.

        Besser als stumpfes stops[:block_size], weil Stops die bereits aus der
        Startzone herausgelaufen sind nicht als Seed interpretiert werden.
        """
        seed = []
        for node in route_stops:
            if node not in zone_node_set or len(seed) >= block_size:
                break
            seed.append(node)
        return seed

    def _seed_prescore(
        self,
        block_nodes: list[int],
        task_by_node: dict[int, MaintenanceTask],
        value_fn: Optional[Callable[[int, float], float]],
        mat8: np.ndarray,
    ) -> float:
        """
        Heuristischer Prescore für einen Seed-Block (1 oder 2 Stops).

        1-Stop:  C̃(k)        - wage_per_min × travel(depot→k)
        2-Stop:  C̃(k1)+C̃(k2) - wage_per_min × (travel(depot→k1) + travel(k1→k2))

        Niedrigere Anfahrtszeit und höherer C̃ → höherer Score → bevorzugt.
        """
        wage = self.cost_params.wage_eur_per_hour / 60.0
        if value_fn is None:
            val_fn = lambda node, dsm: 0.0  # noqa: E731
        else:
            val_fn = value_fn

        if not block_nodes:
            return -np.inf

        score = 0.0
        prev = 0  # Depot
        for node in block_nodes:
            t = task_by_node.get(node)
            dsm = t.days_since_maintenance if t else 0.0
            score += val_fn(node, dsm) - wage * mat8[prev, node] / 60.0
            prev = node
        return score

    def _generate_seed_candidates_for_team(
        self,
        team_id: int,
        base_plan: DailyPlan,
        all_tasks: list[MaintenanceTask],
        team_assignment: dict[int, list[int]],
        block_size: int,
        max_candidates: int,
        value_fn: Optional[Callable[[int, float], float]],
    ) -> list[list[int]]:
        """
        Erzeugt Seed-Kandidaten für ein Team.

        Kandidat 0: Basis-Seed (aus base_plan).
        Kandidaten 1..(max_candidates-1): Top-Alternativen nach Prescore.

        Alternativ-Seeds:
          - 1-Stop-Blöcke: alle Zone-Stationen mit Prescore
          - 2-Stop-Blöcke (nur wenn block_size >= 2): beste Zone-Station + nächste Nachbarin

        Diversitätsfilter: kein alternativer Seed mit identischer erster Station
        wie ein bereits gewählter Kandidat.
        """
        mat8 = self._get_matrix(0.0)  # 8:00-Matrix für Prescore-Berechnung
        zone_nodes: list[int] = team_assignment.get(team_id, [])
        zone_node_set = set(zone_nodes)
        task_by_node: dict[int, MaintenanceTask] = {t.node_idx: t for t in all_tasks}

        # Basis-Seed aus base_plan extrahieren
        base_route_stops: list[int] = []
        for r in base_plan.routes:
            if r.team_id == team_id:
                base_route_stops = list(r.stops)
                break
        base_seed = self._extract_base_seed(base_route_stops, zone_node_set, block_size)

        candidates: list[list[int]] = [base_seed]
        chosen_first_nodes: set[int] = {base_seed[0]} if base_seed else set()

        # Alle Zone-Stationen als potenzielle Seed-Startpunkte bewerten
        scored: list[tuple[float, list[int]]] = []
        for n1 in zone_nodes:
            if n1 not in task_by_node:
                continue
            t1 = task_by_node[n1]
            if t1.task_type == "carryover":
                continue  # Carryover ist immer mandatory, kein Seed-Kandidat

            if block_size >= 2:
                # 2-Stop-Block: n1 + nächste Zone-Nachbarin n2
                best_n2 = None
                best_n2_dist = np.inf
                for n2 in zone_nodes:
                    if n2 == n1 or n2 not in task_by_node:
                        continue
                    t2 = task_by_node[n2]
                    if t2.task_type == "carryover":
                        continue
                    d = mat8[n1, n2]
                    if d < best_n2_dist:
                        best_n2_dist = d
                        best_n2 = n2
                if best_n2 is not None:
                    block = [n1, best_n2]
                    s = self._seed_prescore(block, task_by_node, value_fn, mat8)
                    scored.append((s, block))
            else:
                block = [n1]
                s = self._seed_prescore(block, task_by_node, value_fn, mat8)
                scored.append((s, block))

        # Absteigend nach Prescore, Diversitätsfilter (unterschiedliche erste Station)
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, block in scored:
            if len(candidates) >= max_candidates:
                break
            first = block[0]
            if first in chosen_first_nodes:
                continue
            candidates.append(block)
            chosen_first_nodes.add(first)

        return candidates

    def _build_plan_with_team_seed(
        self,
        target_team_id: int,
        seed_nodes: list[int],
        all_tasks: list[MaintenanceTask],
        team_assignment: dict[int, list[int]],
        reference_plan: DailyPlan,
        route_score_fn: Callable[[int, float, int], float],
    ) -> DailyPlan:
        """
        Baut einen DailyPlan in dem ein Team einen festen Seed-Präfix hat.

        target_team: seed_nodes → complete_route_from_partial → neue Route.
        Andere Teams: Routen aus reference_plan übernehmen.
        """
        node_to_task = {t.node_idx: t for t in all_tasks}
        team_tasks = [
            node_to_task[n]
            for n in team_assignment.get(target_team_id, [])
            if n in node_to_task
        ]

        new_stops, new_arr, new_dep, team_travel = complete_route_from_partial(
            seed_prefix_nodes=seed_nodes,
            all_team_tasks=team_tasks,
            traffic_matrices=self.traffic_matrices,
            workday_start_hour=self.workday_start_hour,
            workday_minutes=self.WORKDAY_MINUTES,
            lunch_earliest_min=self._lunch_earliest,
            lunch_duration_min=self._lunch_duration,
            route_score_fn=route_score_fn,
        )

        routes: list[PlannedRoute] = []
        total_travel = team_travel
        for r in reference_plan.routes:
            if r.team_id == target_team_id:
                routes.append(PlannedRoute(
                    team_id=target_team_id,
                    stops=new_stops,
                    arrival_times=new_arr,
                    departure_times=new_dep,
                ))
            else:
                routes.append(r)
                total_travel += sum(
                    int(round(self._get_matrix(dep)[fr, to] / 60.0))
                    for fr, to, dep in zip(
                        [0] + list(r.stops[:-1]),
                        r.stops,
                        [0.0] + list(r.departure_times[:-1]),
                    )
                ) if r.stops else 0

        # Sicherstellen dass alle Teams vertreten sind
        existing = {r.team_id for r in routes}
        for i in range(self.n_teams):
            if i not in existing:
                routes.append(PlannedRoute(team_id=i, stops=[], arrival_times=[], departure_times=[]))

        routes.sort(key=lambda r: r.team_id)
        return DailyPlan(routes=routes, total_travel_time=total_travel, solver_status="OPTIMAL")

    def create_initial_plan_rh(
        self,
        all_tasks: list[MaintenanceTask],
        team_assignment: dict[int, list[int]],
        state: SystemState,
        rh_config: dict,
    ) -> DailyPlan:
        """
        Initialplanung mit Seed-Rollout (Initial-RH).

        Pro Team:
          1. Basis-Seed aus base_plan + top-(top_k_initial-1) Alternativen nach Prescore.
          2. Jeden Kandidaten per evaluate_with_forced_day0_plan bewerten (CRN).
          3. Bestes Seed für dieses Team wählen.

        Teams werden sequenziell optimiert:
          Team 0: alle Kandidaten gegen Team 1 = Basis-Plan.
          Team 1: alle Kandidaten gegen Team 0 = bester gewählter Seed von Team 0.

        Falls kein brauchbarer Seed gefunden (leere Zone): Basis-Plan für dieses Team.
        """
        top_k = rh_config.get("top_k_initial", 3)
        block_size = rh_config.get("initial_seed_block_size", 2)
        horizon = rh_config.get("horizon_days", 7)
        n_sc = rh_config.get("n_scenarios", 8)

        base_plan = self.policy.create_initial_plan(all_tasks, team_assignment)
        value_fn = self.policy.get_value_fn()
        route_score_fn = self.policy.get_route_score_fn_for_tasks(all_tasks)

        scenario_seeds = self.evaluator.draw_scenario_seeds(n_sc)

        best_plan = base_plan
        team_ids = sorted(team_assignment.keys())

        for team_id in team_ids:
            candidates = self._generate_seed_candidates_for_team(
                team_id=team_id,
                base_plan=best_plan,
                all_tasks=all_tasks,
                team_assignment=team_assignment,
                block_size=block_size,
                max_candidates=top_k,
                value_fn=value_fn,
            )

            best_team_cost = np.inf
            best_team_plan = best_plan
            base_seed = candidates[0] if candidates else []
            chosen_seed = base_seed

            for seed in candidates:
                if not seed:
                    continue
                candidate_plan = self._build_plan_with_team_seed(
                    target_team_id=team_id,
                    seed_nodes=seed,
                    all_tasks=all_tasks,
                    team_assignment=team_assignment,
                    reference_plan=best_plan,
                    route_score_fn=route_score_fn,
                )
                h_cost = self.evaluator.evaluate_with_forced_day0_plan(
                    state=state,
                    forced_plan=candidate_plan,
                    forced_tasks=all_tasks,
                    horizon_days=horizon,
                    scenario_seeds=scenario_seeds,
                )
                if h_cost < best_team_cost:
                    best_team_cost = h_cost
                    best_team_plan = candidate_plan
                    chosen_seed = seed

            if chosen_seed != base_seed:
                logger.info(
                    f"Initial-RH Team {team_id}: Seed {chosen_seed} statt {base_seed} "
                    f"(Horizont {best_team_cost:.2f} EUR)"
                )
            else:
                logger.info(
                    f"Initial-RH Team {team_id}: Basis-Seed bestätigt {base_seed} "
                    f"(Horizont {best_team_cost:.2f} EUR)"
                )

            best_plan = best_team_plan

        return best_plan

    # ------------------------------------------------------------------
    # Tages-Simulation mit RH-Eingriff
    # ------------------------------------------------------------------

    def _run_day(
        self,
        state: SystemState,
        day_disruptions: list[DisruptionEvent],
        rh_config: dict,
    ) -> tuple[DayResult, list[SimRoute], list[DisruptionEvent]]:
        """Simuliert einen vollständigen Arbeitstag mit RH-Intervention."""
        day = state.day
        hourly_logs: list[HourLog] = []
        downtime_cost = 0.0

        team_states = [
            TeamState(team_id=i, current_node=0, current_time=0)
            for i in range(self.n_teams)
        ]
        assignment, all_tasks, team_assignment = self.task_gen.build_daily_tasks(
            state, team_states
        )
        n_routine = sum(1 for t in all_tasks if t.task_type == "routine")

        rh_enabled = rh_config.get("enabled", False)
        enable_replan = rh_config.get("enable_replan", True)
        enable_initial = rh_config.get("enable_initial", False)

        if all_tasks:
            if rh_enabled and enable_initial:
                daily_plan = self.create_initial_plan_rh(
                    all_tasks, team_assignment, state, rh_config
                )
            else:
                daily_plan = self.policy.create_initial_plan(all_tasks, team_assignment)
            logger.info(
                f"RH Tag {day:>3d}: {len(all_tasks)} Aufgaben ({n_routine} Routine, "
                f"{len(all_tasks) - n_routine} Carryover)"
            )
        else:
            daily_plan = None
            logger.info(f"RH Tag {day:>3d}: Keine Aufgaben.")

        sim_routes = plan_to_sim_routes(daily_plan, all_tasks, self.n_teams)

        if self._lunch_duration > 0:
            for r in sim_routes:
                _inject_lunch(r, self._lunch_earliest, self._lunch_duration)

        # Stündliche Simulation
        by_hour: dict[int, list[DisruptionEvent]] = {}
        for d in day_disruptions:
            by_hour.setdefault(d.hour, []).append(d)

        disruptions_handled = 0
        carried_disruptions: list[DisruptionEvent] = []
        drop_score_fn = self.policy.get_drop_score_fn()

        for hour in range(8, 17):
            time_min = float((hour - 8) * 60)
            hour_log = HourLog(day=day, hour=hour)

            # 8:00: Initialplan-Log
            if hour == 8:
                mat8 = self._get_matrix(0.0)
                for r in sim_routes:
                    prev_node = 0
                    route_dicts = []
                    for stop in r.stops:
                        t_min = mat8[prev_node, stop.node_idx] / 60.0
                        km = _approx_km(self.all_coords[prev_node], self.all_coords[stop.node_idx])
                        route_dicts.append({
                            "node_idx": stop.node_idx,
                            "task_type": stop.task_type,
                            "days_since_maintenance": round(float(stop.days_since_maintenance), 2),
                            "from_node": prev_node,
                            "from_label": "Depot" if prev_node == 0 else f"Node {prev_node}",
                            "travel_time_min": round(t_min, 1),
                            "travel_km": round(km, 2),
                            "arrival_time": _fmt(stop.arrival_min),
                            "service_min": int(stop.service_min),
                            "departure_time": _fmt(stop.departure_min),
                        })
                        prev_node = stop.node_idx
                    depot_rt = depot_travel = depot_km = None
                    if r.stops:
                        last = r.stops[-1]
                        t_back = mat8[last.node_idx, 0] / 60.0
                        depot_rt = _fmt(last.departure_min + t_back)
                        depot_travel = round(t_back, 1)
                        depot_km = round(_approx_km(self.all_coords[last.node_idx], self.all_coords[0]), 2)
                    hour_log.initial_plan.append({
                        "team_id": r.team_id,
                        "n_stops": len(r.stops),
                        "route": route_dicts,
                        "depot_return_time": depot_rt,
                        "depot_return_travel_min": depot_travel,
                        "depot_return_km": depot_km,
                    })

            # Störungen
            if hour in by_hour:
                h_disruptions = by_hour[hour]
                hour_log.disruptions = [
                    {
                        "node_idx": d.node_idx,
                        "type": d.disruption_type,
                        "rated_power_kw": d.power_kw,
                        "service_min": d.service_min,
                    }
                    for d in h_disruptions
                ]

                before_replan: dict[int, set[int]] = {
                    r.team_id: {s.node_idx for s in r.remaining_stops_at(time_min)}
                    for r in sim_routes
                }

                if rh_enabled and enable_replan:
                    handled, carried, h_downtime = self._handle_disruptions_rh(
                        h_disruptions, sim_routes, state, all_tasks,
                        time_min, hour, hour_log, rh_config, drop_score_fn
                    )
                else:
                    handled, carried, h_downtime = self.policy.handle_disruptions(
                        h_disruptions, sim_routes, time_min, hour, hour_log
                    )
                disruptions_handled += handled
                carried_disruptions.extend(carried)
                downtime_cost += h_downtime

                report_min = float((hour - 8) * 60)
                remaining_wday_h = (self.WORKDAY_MINUTES - report_min) / 60.0
                for d in carried:
                    downtime_cost += remaining_wday_h * d.power_kw * self.cost_params.downtime_eur_per_kwh

                # Replan-Log
                disruption_nodes = {d.node_idx for d in h_disruptions}
                for r in sim_routes:
                    old_nodes = before_replan.get(r.team_id, set())
                    new_remaining = r.remaining_stops_at(time_min)
                    new_nodes = {s.node_idx for s in new_remaining}
                    dropped_nodes = old_nodes - new_nodes - disruption_nodes
                    route_dicts = []
                    prev_node = r.current_node_at(time_min)
                    for stop in new_remaining:
                        route_dicts.append({
                            "node_idx": stop.node_idx,
                            "task_type": stop.task_type,
                            "from_node": prev_node,
                            "from_label": "Depot" if prev_node == 0 else f"Node {prev_node}",
                            "arrival_time": _fmt(stop.arrival_min),
                            "service_min": int(stop.service_min),
                            "departure_time": _fmt(stop.departure_min),
                        })
                        prev_node = stop.node_idx
                    ref = new_remaining or r.stops
                    depot_rt = None
                    if ref:
                        last = ref[-1]
                        mat = self._get_matrix(last.departure_min)
                        depot_rt = _fmt(last.departure_min + mat[last.node_idx, 0] / 60.0)
                    hour_log.replan.append({
                        "team_id": r.team_id,
                        "dropped_stops": [
                            {"node_idx": n, "task_type": "routine", "reason": "carryover"}
                            for n in dropped_nodes
                        ],
                        "remaining_route": route_dicts,
                        "depot_return_time": depot_rt,
                    })

            # Teamstatus
            for r in sim_routes:
                done_n = len(r.completed_nodes_at(time_min))
                remaining_n = len(r.remaining_stops_at(time_min))
                active = next((s for s in r.stops if s.arrival_min <= time_min < s.departure_min), None)
                nxt = next((s for s in r.stops if s.arrival_min > time_min), None)
                ts: dict = {"team_id": r.team_id, "stops_completed": done_n, "stops_remaining": remaining_n}
                if active:
                    ts.update({
                        "status": "working", "current_node": active.node_idx,
                        "active_task_type": active.task_type,
                        "active_departure_time": _fmt(active.departure_min),
                        "next_node": None, "next_arrival_time": None, "depot_return_time": None,
                    })
                elif nxt:
                    ts.update({
                        "status": "driving", "current_node": r.current_node_at(time_min),
                        "active_task_type": None, "active_departure_time": None,
                        "next_node": nxt.node_idx, "next_arrival_time": _fmt(nxt.arrival_min),
                        "depot_return_time": None,
                    })
                elif r.stops:
                    last = r.stops[-1]
                    mat = self._get_matrix(last.departure_min)
                    ts.update({
                        "status": "returning", "current_node": last.node_idx,
                        "active_task_type": None, "active_departure_time": None,
                        "next_node": None, "next_arrival_time": None,
                        "depot_return_time": _fmt(last.departure_min + mat[last.node_idx, 0] / 60.0),
                    })
                else:
                    ts.update({
                        "status": "idle", "current_node": 0,
                        "active_task_type": None, "active_departure_time": None,
                        "next_node": None, "next_arrival_time": None, "depot_return_time": None,
                    })
                hour_log.team_status.append(ts)

            # 16:00: Ausgeführter Plan
            if hour == 16:
                for r in sim_routes:
                    n_planned = len(r.stops)
                    n_completed = sum(1 for s in r.stops if s.departure_min <= self.WORKDAY_MINUTES)
                    n_disrupt = sum(1 for s in r.stops if s.task_type not in ("routine",))
                    route_dicts = [
                        {
                            "node_idx": s.node_idx, "task_type": s.task_type,
                            "arrival_time": _fmt(s.arrival_min), "service_min": int(s.service_min),
                            "departure_time": _fmt(s.departure_min),
                            "completed": s.departure_min <= self.WORKDAY_MINUTES,
                        }
                        for s in r.stops
                    ]
                    depot_rt = None
                    if r.stops:
                        last = r.stops[-1]
                        mat = self._get_matrix(last.departure_min)
                        depot_rt = _fmt(last.departure_min + mat[last.node_idx, 0] / 60.0)
                    hour_log.executed_plan.append({
                        "team_id": r.team_id,
                        "stops_planned": n_planned,
                        "stops_completed": n_completed,
                        "disruption_stops": n_disrupt,
                        "route": route_dicts,
                        "depot_return_time": depot_rt,
                    })

            hourly_logs.append(hour_log)

        # Ausfallkosten für Carryover-Stops (identisch zum MaintenanceSimulator)
        carryover_nodes = {t.node_idx for t in state.carryover_tasks if t.task_type == "carryover"}
        accounted: set[int] = set()
        cp = self.cost_params
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in carryover_nodes and stop.node_idx not in accounted:
                    accounted.add(stop.node_idx)
                    dep = min(stop.departure_min, float(self.WORKDAY_MINUTES))
                    downtime_cost += (dep / 60.0) * self.node_to_power.get(stop.node_idx, 22.0) * cp.downtime_eur_per_kwh

        op_cost, wage_cost, fuel_cost = self._compute_op_cost(sim_routes)
        n_routine_completed = sum(
            1 for route in sim_routes
            for stop in route.stops
            if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES
        )

        return (
            DayResult(
                day=day,
                n_routine_tasks=n_routine,
                n_routine_completed=n_routine_completed,
                disruptions_handled=disruptions_handled,
                disruptions_carryover=len(carried_disruptions),
                operational_cost_eur=op_cost,
                wage_cost_eur=wage_cost,
                fuel_cost_eur=fuel_cost,
                downtime_cost_eur=downtime_cost,
                hourly_logs=hourly_logs,
            ),
            sim_routes,
            carried_disruptions,
        )

    # ------------------------------------------------------------------
    # RH-Störungshandling
    # ------------------------------------------------------------------

    def _handle_disruptions_rh(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        state: SystemState,
        all_tasks: list[MaintenanceTask],
        time_min: float,
        hour: int,
        log: HourLog,
        rh_config: dict,
        drop_score_fn: Callable,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        Störungshandling mit RH-Kandidatenbewertung.

        Für jede Störung:
        1. Feasible ohne Drop → direkt einfügen (kein RH nötig).
        2. Basis-Policy bestimmt via _find_best_drop_and_insert wie viele Drops nötig sind:
           a. 0 Drops (Pfad 1 already handled above)
           b. 1 Drop → RH darf eingreifen: top_k Einzelkandidaten über Horizont bewerten.
           c. >1 Drops → Basis-Policy direkt übernehmen (RH-Modell nicht anwendbar).
           d. Kein Platz → Carryover.

        RH greift AUSSCHLIESSLICH bei Einzeldrop-Fällen ein. Das verhindert das
        Problem, dass ein RH-gewählter erster Drop + Legacy-Fallback-Drops zusammen
        mehr Stationen opfern als die Basis-Policy allein.
        """
        matrix = self._get_matrix(time_min)
        cp = self.cost_params
        carryover: list[DisruptionEvent] = []
        downtime_cost = 0.0
        handled = 0

        top_k: int = rh_config.get("top_k_candidates", 3)
        horizon: int = rh_config.get("horizon_days", 7)
        n_sc: int = rh_config.get("n_scenarios", 8)
        budget: float = rh_config.get("time_budget_sec", 5.0)
        fallback: bool = rh_config.get("fallback_to_legacy_on_timeout", True)

        # Sortierung: einfach einfügbare Störungen zuerst (identisch zu handle_disruptions_greedy).
        # Wenn 3 Störungen gleichzeitig ankommen, wird die Route-freundlichste zuerst eingebaut —
        # so bleiben mehr Routine-Stops für die schwierigeren Störungen erhalten.
        queue: list[tuple[float, DisruptionEvent]] = []
        for d in disruptions:
            r0 = _find_best_insertion(d, sim_routes, time_min, hour, matrix,
                                      self.all_coords, self.WORKDAY_MINUTES, cp)
            queue.append((r0[0] if r0 is not None else np.inf, d))
        queue.sort(key=lambda x: x[0])

        for _, d in queue:
            # --- Schritt 1: kein Drop nötig ---
            result = _find_best_insertion(
                d, sim_routes, time_min, hour, matrix,
                self.all_coords, self.WORKDAY_MINUTES, cp,
            )
            if result is not None:
                _, team_idx, pos, arrival = result
                _insert_stop(d, sim_routes[team_idx], pos, time_min, self.traffic_matrices, self.workday_start_hour)
                report_min = float((hour - 8) * 60)
                wait_h = max(0.0, (arrival - report_min) / 60.0)
                downtime_cost += wait_h * d.power_kw * cp.downtime_eur_per_kwh
                handled += 1
                log.notes.append(f"RH-Replan (kein Drop): {d.disruption_type} @ {d.node_idx}")
                continue

            # --- Schritt 2: Basis-Policy bestimmt Drop-Entscheidung ---
            # _find_best_drop_and_insert = identische Logik wie im Legacy-Pfad.
            # Das Ergebnis ist der "Kandidat 0" für RH.
            base_drop = _find_best_drop_and_insert(
                d, sim_routes, time_min, hour, self.all_coords,
                self.traffic_matrices, self.workday_start_hour,
                self.WORKDAY_MINUTES, cp, drop_score_fn,
            )
            if base_drop is None:
                carryover.append(d)
                log.notes.append(f"RH-Replan Carryover: {d.disruption_type} @ {d.node_idx} (kein Platz)")
                continue

            _, base_ti, base_drop_globals, base_pos, base_arrival = base_drop

            # --- Schritt 3a: Multi-Drop → Basis-Policy direkt (kein RH-Eingriff) ---
            if len(base_drop_globals) > 1:
                base_dropped = [sim_routes[base_ti].stops[i].node_idx for i in base_drop_globals]
                for gi in sorted(base_drop_globals, reverse=True):
                    _remove_stop_and_recompute(
                        sim_routes[base_ti], gi, self.traffic_matrices, self.workday_start_hour
                    )
                result_md = _find_best_insertion(
                    d, sim_routes, time_min, hour, self._get_matrix(time_min),
                    self.all_coords, self.WORKDAY_MINUTES, cp,
                )
                if result_md is not None:
                    _, ti_md, pos_md, arr_md = result_md
                    _insert_stop(d, sim_routes[ti_md], pos_md, time_min, self.traffic_matrices, self.workday_start_hour)
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (arr_md - report_min) / 60.0)
                    downtime_cost += wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    handled += 1
                    log.notes.append(
                        f"RH-Legacy (Multi-Drop {base_dropped}): {d.disruption_type} @ {d.node_idx}"
                    )
                else:
                    carryover.append(d)
                    log.notes.append(
                        f"RH-Replan Carryover (nach Multi-Drop): {d.disruption_type} @ {d.node_idx}"
                    )
                continue

            # --- Schritt 3b: Einzel-Drop ---
            base_global = base_drop_globals[0]
            base_node = sim_routes[base_ti].stops[base_global].node_idx

            # top_k=1: RH-Evaluation bringt keinen Nutzen (nur 1 Kandidat) →
            # direkt Basis-Policy anwenden → identisches Ergebnis wie Legacy.
            if top_k <= 1:
                _remove_stop_and_recompute(
                    sim_routes[base_ti], base_global,
                    self.traffic_matrices, self.workday_start_hour
                )
                result_s = _find_best_insertion(
                    d, sim_routes, time_min, hour, self._get_matrix(time_min),
                    self.all_coords, self.WORKDAY_MINUTES, cp,
                )
                if result_s is not None:
                    _, ti_s, pos_s, arr_s = result_s
                    _insert_stop(d, sim_routes[ti_s], pos_s, time_min, self.traffic_matrices, self.workday_start_hour)
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (arr_s - report_min) / 60.0)
                    downtime_cost += wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    handled += 1
                    log.notes.append(
                        f"RH-Legacy (top_k=1, drop {base_node}): {d.disruption_type} @ {d.node_idx}"
                    )
                else:
                    carryover.append(d)
                    log.notes.append(
                        f"RH-Replan Carryover (top_k=1): {d.disruption_type} @ {d.node_idx}"
                    )
                continue

            # --- Schritt 3c: RH-Evaluation (top_k > 1) ---
            # Kandidat 0: Basis-Policy-Wahl (von _find_best_drop_and_insert)
            # Kandidaten 1..k-1: Alternativen aus _get_feasible_drop_candidates,
            #                    gefiltert um Basis-Policy-Wahl zu excludieren.
            alternatives = [
                (ti, gi, ni, arr)
                for ti, gi, ni, arr in self._get_feasible_drop_candidates(
                    d, sim_routes, time_min, hour, top_k - 1, drop_score_fn
                )
                if ni != base_node
            ]
            all_candidates = [(base_ti, base_global, base_node, base_arrival)] + alternatives

            scenario_seeds = self.evaluator.draw_scenario_seeds(n_sc)
            t_start = time.perf_counter()
            best_ti, best_gi, best_node_chosen, best_arrival = base_ti, base_global, base_node, base_arrival
            best_horizon_cost = np.inf

            for cand_ti, cand_gi, cand_node, cand_arrival in all_candidates:
                if time.perf_counter() - t_start > budget:
                    if fallback:
                        log.notes.append("RH-Timeout: Fallback auf Basis-Policy")
                    break
                candidate_state = self._build_candidate_state(
                    state, sim_routes, cand_node, time_min
                )
                h_cost = self.evaluator.evaluate(candidate_state, horizon, scenario_seeds)
                if h_cost < best_horizon_cost:
                    best_horizon_cost = h_cost
                    best_ti, best_gi, best_node_chosen = cand_ti, cand_gi, cand_node
                    best_arrival = cand_arrival

            if best_node_chosen != base_node:
                state.rh_overrides += 1
                log.notes.append(
                    f"RH-Override: drop {best_node_chosen} statt {base_node} "
                    f"(Horizont {best_horizon_cost:.2f} EUR)"
                )
            else:
                log.notes.append(
                    f"RH bestätigt Basis-Policy: drop {base_node} "
                    f"(Horizont {best_horizon_cost:.2f} EUR)"
                )

            _remove_stop_and_recompute(
                sim_routes[best_ti], best_gi,
                self.traffic_matrices, self.workday_start_hour
            )
            result2 = _find_best_insertion(
                d, sim_routes, time_min, hour, self._get_matrix(time_min),
                self.all_coords, self.WORKDAY_MINUTES, cp,
            )
            if result2 is not None:
                _, ti2, pos2, arrival_at_d = result2
                _insert_stop(d, sim_routes[ti2], pos2, time_min, self.traffic_matrices, self.workday_start_hour)
                report_min = float((hour - 8) * 60)
                wait_h = max(0.0, (arrival_at_d - report_min) / 60.0)
                downtime_cost += wait_h * d.power_kw * cp.downtime_eur_per_kwh
                handled += 1
                log.notes.append(
                    f"RH-Replan (drop {best_node_chosen}): {d.disruption_type} @ {d.node_idx} → Team {sim_routes[ti2].team_id}"
                )
            else:
                carryover.append(d)
                log.notes.append(
                    f"RH-Replan Carryover (unerwartet): {d.disruption_type} @ {d.node_idx}"
                )

        return handled, carryover, downtime_cost

    def _get_feasible_drop_candidates(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        top_k: int,
        drop_score_fn: Callable,
    ) -> list[tuple[int, int, int, float]]:
        """
        Gibt top-k MACHBARKEITSGEPRÜFTE Einzel-Drop-Kandidaten zurück.

        Jeder Kandidat: (team_idx, global_stop_idx, node_idx, arrival_nach_drop).
        Nur Stops, deren Entfernung allein ausreicht um die Störung d einzufügen.
        Sortiert nach drop_score aufsteigend (niedrigster Score = Basis-Policy-Wahl).

        Durch die Vorprüfung ist garantiert, dass nach dem gewählten Drop
        _find_best_insertion immer erfolgreich ist — keine Multi-Drop-Eskalation.
        """
        from src.models.simulator import SimStop  # lokaler Import verhindert Zirkel
        matrix = self._get_matrix(time_min)
        remaining_hours = max(0.0, (self.WORKDAY_MINUTES - time_min) / 60.0)
        candidates: list[tuple[float, int, int, int, float]] = []

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            cur_node = route.current_node_at(time_min)
            cur_dep = route.current_departure_at(time_min)
            if route.lunch_end_min is not None and cur_dep < route.lunch_end_min:
                cur_dep = route.lunch_end_min

            routine_indices = [i for i, s in enumerate(remaining) if s.task_type == "routine"]

            for local_idx in routine_indices:
                s = remaining[local_idx]

                # Route ohne diesen Stop simulieren (non-destructive)
                trimmed: list[SimStop] = []
                prev_n, prev_d = cur_node, cur_dep
                for j, rs in enumerate(remaining):
                    if j == local_idx:
                        continue
                    mat = _get_matrix(self.traffic_matrices, prev_d, self.workday_start_hour)
                    new_arr = prev_d + mat[prev_n, rs.node_idx] / 60.0
                    trimmed.append(SimStop(
                        node_idx=rs.node_idx,
                        task_type=rs.task_type,
                        arrival_min=new_arr,
                        service_min=rs.service_min,
                        days_since_maintenance=rs.days_since_maintenance,
                    ))
                    prev_n = rs.node_idx
                    prev_d = new_arr + rs.service_min

                # Prüfen ob Störung d in die verkleinerte Route passt
                best_arrival: float | None = None
                best_pos_cost = np.inf
                for pos in range(len(trimmed) + 1):
                    cost, feasible, arrival = _insertion_cost(
                        d, trimmed, cur_node, cur_dep, pos, hour, matrix,
                        self.all_coords, self.WORKDAY_MINUTES, self.cost_params,
                    )
                    if feasible and cost < best_pos_cost:
                        best_pos_cost = cost
                        best_arrival = arrival

                if best_arrival is None:
                    continue  # dieser Drop reicht nicht aus

                # Drop-Score für Reihung (niedrig = zuerst droppen)
                prev_r = remaining[local_idx - 1].node_idx if local_idx > 0 else cur_node
                nxt_r = remaining[local_idx + 1].node_idx if local_idx < len(remaining) - 1 else 0
                detour = max(0.0, (
                    matrix[prev_r, s.node_idx] + matrix[s.node_idx, nxt_r] - matrix[prev_r, nxt_r]
                ) / 60.0)
                score = drop_score_fn(
                    s.node_idx, s.days_since_maintenance, remaining_hours, cur_node, detour
                )
                global_idx = route.stops.index(s)
                candidates.append((score, ti, global_idx, s.node_idx, best_arrival))

        candidates.sort(key=lambda x: x[0])
        return [(ti, gi, ni, arr) for _, ti, gi, ni, arr in candidates[:top_k]]

    def _build_candidate_state(
        self,
        state: SystemState,
        sim_routes: list[SimRoute],
        dropped_node: int,
        time_min: float,
    ) -> SystemState:
        """
        Konstruiert den End-of-Today-State für einen Drop-Kandidaten.

        Annahme (Approximation): alle noch geplanten Routine-Stops außer
        dropped_node werden heute fertig. dropped_node bleibt offen.
        """
        new_state = deepcopy(state)
        new_state.day += 1

        # Bereits heute erledigte Stops
        already_done = {
            stop.node_idx
            for route in sim_routes
            for stop in route.stops
            if stop.task_type == "routine" and stop.departure_min <= time_min
        }
        # Noch geplante Stops (außer dropped_node)
        planned_remaining = {
            stop.node_idx
            for route in sim_routes
            for stop in route.remaining_stops_at(time_min)
            if stop.task_type == "routine" and stop.node_idx != dropped_node
        }

        all_completed_today = already_done | planned_remaining

        new_state.days_since_maintenance += 1.0
        for node in all_completed_today:
            new_state.days_since_maintenance[node] = 0.0
            new_state.remaining.discard(node - 1)

        # dropped_node bleibt in remaining (station_index = node_idx - 1)
        # (ist noch in remaining, da dsm nicht zurückgesetzt)

        new_state.carryover_tasks = []  # Störung wird heute behandelt
        return new_state

    # ------------------------------------------------------------------
    # Hilfsmethoden (analog MaintenanceSimulator)
    # ------------------------------------------------------------------

    def _generate_day_disruptions(self, state: SystemState) -> list[DisruptionEvent]:
        """Stochastische Störungsgenerierung für einen Simulationstag."""
        cp = self.cost_params
        disrupted_today: set[int] = set()
        events: list[DisruptionEvent] = []

        for hour in range(8, 17):
            for node_idx in range(1, self.n_stations + 1):
                if node_idx in disrupted_today:
                    continue
                t = min(state.days_since_maintenance[node_idx], self._recovery_days)
                factor = self._initial_factor + (1.0 - self._initial_factor) * t / self._recovery_days
                sf = self.node_to_failure_factor.get(node_idx, 1.0)
                power_kw = self.node_to_power.get(node_idx, 22.0)

                if state.rng.random() < self._p1_base * factor * sf:
                    events.append(DisruptionEvent(
                        day=state.day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 1", power_kw=power_kw,
                        service_min=float(self._typ1_service_min),
                    ))
                    disrupted_today.add(node_idx)
                    continue

                if state.rng.random() < self._p2_base * factor * sf:
                    mat = self.traffic_matrices.get(hour, list(self.traffic_matrices.values())[0])
                    rt = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                    svc = self._typ2_dismount_min + rt + self._typ2_handling_min + self._typ2_remount_min
                    events.append(DisruptionEvent(
                        day=state.day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 2", power_kw=power_kw,
                        service_min=svc,
                    ))
                    disrupted_today.add(node_idx)

        logger.debug(f"RH Tag {state.day}: {len(events)} Störungen generiert.")
        return events

    def _load_disruptions_by_day(
        self, df: pd.DataFrame
    ) -> dict[int, list[DisruptionEvent]]:
        """Lädt CSV-Störungen gruppiert nach Tag."""
        cp = self.cost_params
        by_day: dict[int, list[DisruptionEvent]] = {}
        for _, row in df.iterrows():
            station_id = int(row["Station_ID"])
            node_idx = self._id_to_node.get(station_id)
            if node_idx is None:
                continue
            power_kw = self.node_to_power.get(node_idx, 22.0)
            d_type = str(row["Typ"])
            hour = int(row["Uhrzeit"])
            if d_type == "Typ 1":
                svc = cp.typ1_service_min
            else:
                mat = self.traffic_matrices.get(hour, list(self.traffic_matrices.values())[0])
                rt = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                svc = cp.typ2_dismount_min + rt + cp.typ2_handling_min + cp.typ2_remount_min
            event = DisruptionEvent(
                day=int(row["Tag"]), hour=hour, node_idx=node_idx,
                disruption_type=d_type, power_kw=power_kw, service_min=svc,
            )
            by_day.setdefault(event.day, []).append(event)
        return by_day

    def _compute_op_cost(
        self, sim_routes: list[SimRoute], is_last_day: bool = False
    ) -> tuple[float, float, float]:
        """Betriebskosten (identisch zu MaintenanceSimulator._compute_operational_cost)."""
        cp = self.cost_params
        wage_total = 0.0
        fuel_total = 0.0
        for route in sim_routes:
            if not route.stops:
                continue
            legs = [(0, route.stops[0].node_idx, 0.0)]
            for i in range(len(route.stops) - 1):
                legs.append((route.stops[i].node_idx, route.stops[i + 1].node_idx, route.stops[i].departure_min))
            legs.append((route.stops[-1].node_idx, 0, route.stops[-1].departure_min))
            travel_min = 0.0
            km = 0.0
            for from_n, to_n, dep_min in legs:
                mat = self._get_matrix(dep_min)
                travel_min += mat[from_n, to_n] / 60.0
                km += _approx_km(self.all_coords[from_n], self.all_coords[to_n])
            fuel_total += km * cp.fuel_eur_per_km
            if is_last_day:
                service_min = sum(s.service_min for s in route.stops)
                work_h = (travel_min + service_min) / 60.0
            else:
                work_h = self.WORKDAY_MINUTES / 60.0
            wage_total += work_h * cp.wage_eur_per_hour
        return wage_total + fuel_total, wage_total, fuel_total

    def _get_matrix(self, time_min: float) -> np.ndarray:
        hour = self.workday_start_hour + int(max(0.0, time_min)) // 60
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]

    # ------------------------------------------------------------------
    # Log-Output (kompatibel mit MaintenanceSimulator.write_log)
    # ------------------------------------------------------------------

    def write_log(
        self,
        result: SimulationResult,
        path: str,
        label: str = "SIMULATION",
    ) -> None:
        from src.models.simulator import MaintenanceSimulator
        MaintenanceSimulator.write_log(self, result, path, label)

    # ------------------------------------------------------------------
    # JSON-Output (kompatibel mit MaintenanceSimulator.write_json)
    # ------------------------------------------------------------------

    def write_json(
        self,
        result: SimulationResult,
        path: str,
        label: str = "RH-SIMULATION",
        run_id: Optional[int] = None,
        model_params: Optional[dict] = None,
        rh_config: Optional[dict] = None,
        rh_overrides: int = 0,
    ) -> None:
        """
        Speichert SimulationResult als JSON – identisches Format zu
        MaintenanceSimulator.write_json plus optionalem rolling_horizon_meta-Block.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        op_total = sum(r.operational_cost_eur for r in result.day_results)
        wage_total = sum(r.wage_cost_eur for r in result.day_results)
        fuel_total = sum(r.fuel_cost_eur for r in result.day_results)
        dt_total = sum(r.downtime_cost_eur for r in result.day_results)

        meta: dict = {
            "label": label,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if model_params:
            meta["model_params"] = model_params

        payload: dict = {
            "meta": meta,
            "summary": {
                "days_simulated": len(result.day_results),
                "days_to_complete": result.days_to_complete,
                "remaining_stations_at_end": result.remaining_stations_at_end,
                "total_disruptions": result.total_disruptions,
                "same_day_handled": result.same_day_handled,
                "total_carryover": result.total_carryover,
                "same_day_rate": round(result.same_day_rate, 6),
                "total_cost_eur": round(result.total_cost_eur, 4),
                "operational_cost_eur": round(op_total, 4),
                "wage_cost_eur": round(wage_total, 4),
                "fuel_cost_eur": round(fuel_total, 4),
                "downtime_cost_eur": round(dt_total, 4),
            },
            "days": [
                {
                    "day": r.day,
                    "n_routine_tasks": r.n_routine_tasks,
                    "n_routine_completed": r.n_routine_completed,
                    "disruptions_handled": r.disruptions_handled,
                    "disruptions_carryover": r.disruptions_carryover,
                    "operational_cost_eur": round(r.operational_cost_eur, 4),
                    "wage_cost_eur": round(r.wage_cost_eur, 4),
                    "fuel_cost_eur": round(r.fuel_cost_eur, 4),
                    "downtime_cost_eur": round(r.downtime_cost_eur, 4),
                    "total_cost_eur": round(r.total_cost_eur, 4),
                }
                for r in result.day_results
            ],
            "hourly": [
                {
                    "day": log.day,
                    "hour": log.hour,
                    **({"initial_plan": log.initial_plan} if log.initial_plan else {}),
                    "disruptions": log.disruptions,
                    **({"replan": log.replan} if log.replan else {}),
                    "team_status": log.team_status,
                    **({"executed_plan": log.executed_plan} if log.executed_plan else {}),
                    **({"notes": log.notes} if log.notes else {}),
                }
                for dr in result.day_results
                for log in dr.hourly_logs
            ],
        }

        if rh_config:
            payload["rolling_horizon_meta"] = {
                "enabled": rh_config.get("enabled", True),
                "enable_replan": rh_config.get("enable_replan", True),
                "enable_initial": rh_config.get("enable_initial", False),
                "horizon_days": rh_config.get("horizon_days"),
                "n_scenarios": rh_config.get("n_scenarios"),
                "top_k_candidates": rh_config.get("top_k_candidates"),
                "top_k_initial": rh_config.get("top_k_initial"),
                "initial_seed_block_size": rh_config.get("initial_seed_block_size"),
                "time_budget_sec": rh_config.get("time_budget_sec"),
                "rh_overrides": rh_overrides,
            }

        def _np_default(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_np_default)

        print(f"RH JSON-Protokoll gespeichert: {out.resolve()}")
