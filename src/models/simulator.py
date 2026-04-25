"""
Gemeinsamer Simulationsrahmen für alle Wartungsplanungs-Policies.

Trennt die Simulationslogik von der Planungsstrategie (Policy):
  - MaintenanceSimulator : Simulation, Evaluierung, Logging
  - MaintenancePolicy    : Protokoll für create_initial_plan + handle_disruptions

Alle Policies (MyopicPolicy, MyopicPlusModel, VFAModel) nutzen denselben Simulator.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.cost_params import CostParams
from src.planning.clustering import _approx_km
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------

def _fmt(minutes_from_8: float) -> str:
    """Minuten ab 8:00 → 'HH:MM'-String."""
    total = int(8 * 60 + minutes_from_8)
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Gemeinsame Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class DisruptionEvent:
    """Eine Störung aus malfunction.csv."""

    day: int
    hour: int               # 8–16
    node_idx: int           # 1-basiert (Depot = 0)
    disruption_type: str    # "Typ 1" | "Typ 2"
    power_kw: float
    service_min: float


@dataclass
class SimStop:
    """Ein Halt in der simulierten Tagesroute."""

    node_idx: int
    task_type: str      # "routine" | "Typ 1" | "Typ 2"
    arrival_min: float
    service_min: float
    days_since_maintenance: float = 0.0
    """Tage seit letzter Wartung zum Zeitpunkt der Tagesplanung (nur Routine-Stops)."""

    @property
    def departure_min(self) -> float:
        return self.arrival_min + self.service_min


@dataclass
class SimRoute:
    """Simulierte Route eines Teams für einen Tag (veränderlich)."""

    team_id: int
    stops: list[SimStop] = field(default_factory=list)
    lunch_start_min: Optional[float] = None
    lunch_end_min: Optional[float] = None

    def current_node_at(self, time_min: float) -> int:
        arrived = [s for s in self.stops if s.arrival_min <= time_min]
        return arrived[-1].node_idx if arrived else 0

    def current_departure_at(self, time_min: float) -> float:
        arrived = [s for s in self.stops if s.arrival_min <= time_min]
        return max(time_min, arrived[-1].departure_min) if arrived else time_min

    def remaining_stops_at(self, time_min: float) -> list[SimStop]:
        return [s for s in self.stops if s.arrival_min > time_min]

    def completed_nodes_at(self, time_min: float) -> list[int]:
        return [s.node_idx for s in self.stops if s.departure_min <= time_min]


@dataclass
class HourLog:
    """Stundenprotokoll für die Simulation."""

    day: int
    hour: int
    initial_plan: list[dict] = field(default_factory=list)   # nur Stunde 8
    disruptions: list[dict] = field(default_factory=list)    # strukturiert
    replan: list[dict] = field(default_factory=list)         # nach Störung, pro Team
    team_status: list[dict] = field(default_factory=list)    # jede Stunde, pro Team
    executed_plan: list[dict] = field(default_factory=list)  # nur Stunde 16
    notes: list[str] = field(default_factory=list)           # Randfall-Meldungen
    solver_debug: dict = field(default_factory=dict)         # nur Stunde 8: Solver-Input/Output

    def __str__(self) -> str:
        lines = [f"[Tag {self.day:>3d}, {self.hour:02d}:00]"]

        if self.solver_debug:
            inp = self.solver_debug.get("input", {})
            out = self.solver_debug.get("output", {})
            lines.append("  SOLVER-DEBUG:")
            use_ta = inp.get("use_team_assignment", True)
            for tid, tasks in inp.get("tasks_per_team", {}).items():
                task_info = ", ".join(
                    f"{t['node_idx']}(dsm={t['dsm']:.0f}d,{t['service_time']}min"
                    + (f",dl={t['soft_deadline_min']}min" if t.get("soft_deadline_min") is not None else "")
                    + ")"
                    for t in tasks
                )
                lines.append(
                    f"    Input  Team {tid} ({len(tasks)} Kandidaten, "
                    f"team_assignment={'an' if use_ta else 'aus'}): {task_info if task_info else '–'}"
                )
            status = out.get("solver_status", "?")
            before = out.get("status_before_retry", "")
            n_drop = out.get("n_dropped", 0)
            status_str = f"{before} → {status} ({n_drop} Drops)" if before else status
            lines.append(f"    Output Status: {status_str}")
            if out.get("dropped_nodes"):
                drops_by_team = out.get("dropped_nodes_by_team", {})
                per_team = ", ".join(
                    f"Team {tid}: {nodes}"
                    for tid, nodes in drops_by_team.items()
                    if nodes
                ) if drops_by_team else ""
                drop_line = f"    Drops  {out['dropped_nodes']}"
                if per_team:
                    drop_line += f"  ({per_team})"
                lines.append(drop_line)
            for tid, route in out.get("routes", {}).items():
                lines.append(f"    Output Team {tid} ({len(route)} Stops): {route}")

        if self.initial_plan:
            lines.append("  INITIALPLAN:")
            for tp in self.initial_plan:
                lines.append(f"    Team {tp['team_id']}: {tp['n_stops']} Stops")
                for s in tp["route"]:
                    lines.append(
                        f"      {s['from_label']:>8} --[{s['travel_time_min']:4.1f} min,"
                        f" {s['travel_km']:4.2f} km]--> Node {s['node_idx']:>3d}"
                        f" ({s['task_type']:10s}) {s['arrival_time']} – {s['departure_time']}"
                    )
                if tp.get("depot_return_time"):
                    lines.append(
                        f"      Node {tp['route'][-1]['node_idx']:>3d} --[{tp['depot_return_travel_min']:.1f} min,"
                        f" {tp['depot_return_km']:.2f} km]--> Depot  ~{tp['depot_return_time']}"
                    )

        for d in self.disruptions:
            lines.append(
                f"  !! STÖRUNG: {d['type']} @ Node {d['node_idx']}"
                f" ({d['rated_power_kw']:.0f} kW, Service {d['service_min']:.0f} min)"
            )

        if self.replan:
            lines.append("  REPLAN:")
            for rp in self.replan:
                dropped = rp.get("dropped_stops", [])
                lines.append(
                    f"    Team {rp['team_id']}: {len(dropped)} ausgebaut"
                    + (f" {[d['node_idx'] for d in dropped]}" if dropped else "")
                )
                for s in rp.get("remaining_route", []):
                    lines.append(
                        f"      Node {s['node_idx']:>3d} ({s['task_type']:10s})"
                        f" {s['arrival_time']} – {s['departure_time']}"
                    )
                if rp.get("depot_return_time"):
                    lines.append(f"      Depot-Rückkehr: {rp['depot_return_time']}")

        for ts in self.team_status:
            status = ts["status"]
            if status == "working":
                detail = (
                    f"arbeitet an Node {ts['current_node']:>3d}"
                    f" ({ts['active_task_type']}, Abfahrt {ts['active_departure_time']})"
                )
            elif status == "driving":
                detail = (
                    f"fährt zu Node {ts['next_node']:>3d}"
                    f" (Ankunft {ts['next_arrival_time']})"
                )
            elif status == "returning":
                detail = f"Rückkehr Depot {ts['depot_return_time']}"
            else:
                detail = "keine Stops geplant"
            lines.append(
                f"  Team {ts['team_id']}: {ts['stops_completed']} erledigt,"
                f" {ts['stops_remaining']} verbleibend – {detail}"
            )

        if self.executed_plan:
            lines.append("  AUSGEFÜHRTER PLAN:")
            for ep in self.executed_plan:
                lines.append(
                    f"    Team {ep['team_id']}: {ep['stops_completed']}/{ep['stops_planned']} Stops"
                    f" ({ep['disruption_stops']} Störungen)"
                )
                for s in ep["route"]:
                    done = "✓" if s["completed"] else "✗"
                    lines.append(
                        f"      {done} Node {s['node_idx']:>3d} ({s['task_type']:10s})"
                        f" {s['arrival_time']} – {s['departure_time']}"
                    )
                if ep.get("depot_return_time"):
                    lines.append(f"      Depot-Rückkehr: {ep['depot_return_time']}")

        for note in self.notes:
            lines.append(f"  >> {note}")

        return "\n".join(lines)


@dataclass
class DayResult:
    """Ergebnis eines Simulationstages."""

    day: int
    n_routine_tasks: int
    n_routine_completed: int
    disruptions_handled: int
    disruptions_carryover: int
    operational_cost_eur: float
    wage_cost_eur: float
    fuel_cost_eur: float
    downtime_cost_eur: float
    hourly_logs: list[HourLog]

    @property
    def total_cost_eur(self) -> float:
        return self.operational_cost_eur + self.downtime_cost_eur


@dataclass
class SimulationResult:
    """Gesamtergebnis der Simulation über alle Tage."""

    day_results: list[DayResult]
    total_disruptions: int
    same_day_handled: int
    total_carryover: int
    days_to_complete: Optional[int]
    remaining_stations_at_end: int

    @property
    def total_cost_eur(self) -> float:
        return sum(r.total_cost_eur for r in self.day_results)

    @property
    def same_day_rate(self) -> float:
        return self.same_day_handled / self.total_disruptions if self.total_disruptions else 1.0


# ---------------------------------------------------------------------------
# Policy-Protokoll
# ---------------------------------------------------------------------------

class MaintenancePolicy(Protocol):
    """
    Schnittstelle für austauschbare Planungsstrategien.

    Implementierungen: MyopicPolicy, MyopicPlusModel, VFAModel.
    """

    def create_initial_plan(self, tasks: list[MaintenanceTask]) -> DailyPlan:
        """Erstellt den Tagesplan zu Tagesbeginn."""
        ...

    def handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        Reagiert auf Störungsmeldungen innerhalb des Arbeitstags.

        Parameters
        ----------
        disruptions : Neue Störungen in dieser Stunde.
        sim_routes  : Aktuelle Teamrouten – werden in-place aktualisiert.
        time_min    : Minuten ab 8:00 zum Zeitpunkt der Meldung.
        hour        : Uhrzeitstunde (8–16).
        log         : Stundenprotokoll für Aktionseinträge.

        Returns
        -------
        (n_handled, carryover_liste, downtime_kosten_eur)
        """
        ...


# ---------------------------------------------------------------------------
# Shared utility: DailyPlan → SimRoutes
# ---------------------------------------------------------------------------

def _inject_lunch(
    route: SimRoute,
    lunch_earliest_min: float,
    lunch_duration_min: float,
) -> None:
    """
    Schiebt Mittagspause nach dem ersten Stop ein, dessen Departure ≥ lunch_earliest_min.
    Alle nachfolgenden Ankunftszeiten werden um lunch_duration_min verschoben.
    """
    if lunch_duration_min <= 0:
        return
    for i, stop in enumerate(route.stops):
        if stop.departure_min >= lunch_earliest_min:
            route.lunch_start_min = stop.departure_min
            route.lunch_end_min = stop.departure_min + lunch_duration_min
            for s in route.stops[i + 1:]:
                s.arrival_min += lunch_duration_min
            return


def plan_to_sim_routes(
    plan: Optional[DailyPlan],
    all_tasks: list[MaintenanceTask],
    n_teams: int,
) -> list[SimRoute]:
    """
    Konvertiert einen DailyPlan in SimRoute-Objekte mit korrekten Servicezeiten.

    Nutzt OR-Tools-Ankunftszeiten und tatsächliche Servicezeiten aus dem
    Task-Lookup (statt des Defaultwerts aus _extract_solution). Kann von
    Simulator und Policies gleichermaßen importiert werden.
    """
    if plan is None or not plan.routes:
        return [SimRoute(team_id=i) for i in range(n_teams)]

    task_map: dict[int, MaintenanceTask] = {t.node_idx: t for t in all_tasks}
    sim_routes: list[SimRoute] = []

    for planned in plan.routes:
        route = SimRoute(team_id=planned.team_id)
        for node, arrival in zip(planned.stops, planned.arrival_times):
            task = task_map.get(node)
            service = float(task.service_time) if task else 30.0
            route.stops.append(SimStop(
                node_idx=node,
                task_type=task.task_type if task else "routine",
                arrival_min=float(arrival),
                service_min=service,
                days_since_maintenance=task.days_since_maintenance if task else 0.0,
            ))
        sim_routes.append(route)

    existing = {r.team_id for r in sim_routes}
    for i in range(n_teams):
        if i not in existing:
            sim_routes.append(SimRoute(team_id=i))

    return sorted(sim_routes, key=lambda r: r.team_id)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MaintenanceSimulator:
    """
    Führt die Tages-Simulation für eine gegebene Policy aus.

    Parameters
    ----------
    policy : MaintenancePolicy
        Planungsstrategie (MyopicPolicy, MyopicPlusModel, VFAModel).
    selector : DailyZoneSelector
        Tägliche Stationsauswahl (gemeinsam für alle Policies).
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    stations_df : pd.DataFrame
        Bereinigter Stationsdatensatz aus load_stations().
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden.
    config : dict
        Konfigurationsdict aus config.yaml.
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    """

    def __init__(
        self,
        policy,
        selector,
        all_coords: np.ndarray,
        stations_df: pd.DataFrame,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        cost_params: Optional[CostParams] = None,
    ) -> None:
        self.policy = policy
        self.selector = selector
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.cost_params = cost_params or CostParams()
        self.n_stations = len(stations_df)
        self.n_teams: int = config["maintenance"]["n_teams"]

        maint = config["maintenance"]
        self.WORKDAY_MINUTES: int = (
            maint["workday_end_hour"] - maint["workday_start_hour"]
        ) * 60

        # Station-ID (CSV-Spalte "ID", 1-basiert) → node_idx
        self.id_to_node: dict[int, int] = {
            int(row["ID"]): i + 1
            for i, (_, row) in enumerate(stations_df.iterrows())
        }

        # node_idx → Nennleistung [kW] für Ausfallkosten
        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        self.node_to_power: dict[int, float] = {
            i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
            for i, (_, row) in enumerate(stations_df.iterrows())
        }

        # Stochastischer Störungsmodus: Tage seit letzter Wartung pro Knoten (1-basiert)
        fail_cfg = config.get("failure_simulation", {})
        self._failure_mode: str = fail_cfg.get("mode", "csv")
        if self._failure_mode == "stochastic":
            recovery_days = fail_cfg.get("recovery_days", 365)
            # Alle Stationen starten bei voller Ausfallwahrscheinlichkeit
            self._days_since_maintenance = np.full(
                self.n_stations + 1, float(recovery_days)
            )
            seed = config.get("project", {}).get("seed", 42)
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def run(
        self,
        disruptions_df: Optional[pd.DataFrame] = None,
        max_days: int = 365,
    ) -> SimulationResult:
        """
        Führt die Simulation durch bis alle Stationen gewartet sind.

        Stoppt automatisch sobald alle Stationen routinemäßig gewartet
        wurden UND keine Carryover-Störungen mehr offen sind.

        Parameters
        ----------
        disruptions_df :
            Stördaten als DataFrame (nur im CSV-Modus benötigt).
            Im stochastischen Modus wird dieser Parameter ignoriert.
        max_days :
            Maximale Anzahl simulierter Tage.
        """
        # --- Störungsquellen vorbereiten ---
        if self._failure_mode == "csv":
            if disruptions_df is None:
                raise ValueError(
                    "failure_simulation.mode ist 'csv', aber kein disruptions_df übergeben."
                )
            disruptions = self._load_disruptions(disruptions_df)
            by_day: dict[int, list[DisruptionEvent]] = {}
            for d in disruptions:
                by_day.setdefault(d.day, []).append(d)
        else:
            by_day = {}  # wird pro Tag stochastisch befüllt

        remaining: set[int] = set(range(self.n_stations))
        carryover_tasks: list[MaintenanceTask] = []

        day_results: list[DayResult] = []
        same_day_handled = 0
        total_carryover = 0
        total_disruptions_generated = 0
        days_to_complete: Optional[int] = None
        last_day = 0

        for day in tqdm(range(1, max_days + 1), desc="Simulation", unit="Tag"):
            last_day = day
            team_states = [
                TeamState(team_id=i, current_node=0, current_time=0)
                for i in range(self.n_teams)
            ]

            if days_to_complete is None:
                if self._failure_mode == "stochastic":
                    day_disruptions = self._generate_day_disruptions(day)
                else:
                    day_disruptions = by_day.get(day, [])
                total_disruptions_generated += len(day_disruptions)
            else:
                day_disruptions = []

            result, sim_routes, new_carryover = self._run_day(
                day, remaining, team_states, carryover_tasks, day_disruptions,
            )
            day_results.append(result)
            same_day_handled += result.disruptions_handled
            total_carryover += result.disruptions_carryover

            for route in sim_routes:
                for stop in route.stops:
                    if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES:
                        remaining.discard(stop.node_idx - 1)

            if not remaining and days_to_complete is None:
                days_to_complete = day
                logger.info(f"Alle {self.n_stations} Stationen nach Tag {day} gewartet.")
                # Lohnkosten für den letzten Tag auf tatsächliche Arbeitszeit umrechnen
                op_cost, wage_cost, fuel_cost = self._compute_operational_cost(
                    sim_routes, is_last_day=True
                )
                result.operational_cost_eur = op_cost
                result.wage_cost_eur = wage_cost
                result.fuel_cost_eur = fuel_cost
                new_carryover = []

            # Stochastik: days_since_maintenance aktualisieren
            if self._failure_mode == "stochastic":
                self._days_since_maintenance += 1.0
                for route in sim_routes:
                    for stop in route.stops:
                        if stop.departure_min <= self.WORKDAY_MINUTES:
                            self._days_since_maintenance[stop.node_idx] = 0.0

            carryover_tasks = [
                MaintenanceTask(
                    node_idx=d.node_idx,
                    task_type="carryover",
                    priority=1,
                    service_time=int(round(d.service_min)),
                )
                for d in new_carryover
            ]

            if days_to_complete is not None and not carryover_tasks:
                break

        if self._failure_mode == "csv":
            sim_end = days_to_complete if days_to_complete else last_day
            total_disruptions = sum(len(by_day.get(d, [])) for d in range(1, sim_end + 1))
        else:
            total_disruptions = total_disruptions_generated

        return SimulationResult(
            day_results=day_results,
            total_disruptions=total_disruptions,
            same_day_handled=same_day_handled,
            total_carryover=total_carryover,
            days_to_complete=days_to_complete,
            remaining_stations_at_end=len(remaining),
        )

    def print_summary(self, result: SimulationResult, label: str = "SIMULATION") -> None:
        """Gibt eine kompakte Zusammenfassung der Simulation aus."""
        sep = "=" * 62
        print(sep)
        print(f"  {label} – ZUSAMMENFASSUNG")
        print(sep)
        print(f"  Tage simuliert          : {len(result.day_results)}")
        if result.days_to_complete:
            print(f"  Alle Stationen gewartet : Tag {result.days_to_complete}")
        else:
            print(f"  Verbleibende Stationen  : {result.remaining_stations_at_end}")
        print()
        total_completed = sum(r.n_routine_completed for r in result.day_results)
        print(f"  Routine-Wartungen       : {total_completed} / {self.n_stations} Stationen")
        print()
        print(f"  Störungen gesamt        : {result.total_disruptions}")
        print(f"    Gleichen Tag erledigt : {result.same_day_handled} "
              f"({result.same_day_rate:.1%})")
        print(f"    Carryover             : {result.total_carryover}")
        print()
        op = sum(r.operational_cost_eur for r in result.day_results)
        wage = sum(r.wage_cost_eur for r in result.day_results)
        fuel = sum(r.fuel_cost_eur for r in result.day_results)
        dt = sum(r.downtime_cost_eur for r in result.day_results)
        print(f"  Gesamtkosten            : {result.total_cost_eur:>10,.2f} €")
        print(f"    Betriebskosten        : {op:>10,.2f} €")
        print(f"      davon Lohn          : {wage:>10,.2f} €")
        print(f"      davon Fahrtkosten   : {fuel:>10,.2f} €")
        print(f"    Ausfallkosten         : {dt:>10,.2f} €")
        print(sep)

    def print_day_log(self, result: SimulationResult, day: int) -> None:
        """Gibt das stündliche Protokoll eines einzelnen Tages aus."""
        day_result = next((r for r in result.day_results if r.day == day), None)
        if not day_result:
            print(f"Tag {day} nicht in den Ergebnissen gefunden.")
            return

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  TAG {day} – DETAIL-LOG")
        print(f"  Routine geplant : {day_result.n_routine_tasks}")
        print(f"  Routine erledigt: {day_result.n_routine_completed}")
        print(f"  Störungen       : {day_result.disruptions_handled} erledigt, "
              f"{day_result.disruptions_carryover} Carryover")
        print(f"  Kosten          : {day_result.total_cost_eur:.2f} €  "
              f"(Betrieb {day_result.operational_cost_eur:.2f} €, "
              f"Ausfall {day_result.downtime_cost_eur:.2f} €)")
        print(sep)
        for log in day_result.hourly_logs:
            print(str(log))

    def write_log(
        self,
        result: SimulationResult,
        path: str,
        label: str = "SIMULATION",
    ) -> None:
        """Speichert das vollständige Simulationsprotokoll als Textdatei."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            sep80 = "=" * 80
            sep40 = "-" * 40

            f.write(f"{sep80}\n")
            f.write(f"  {label} – VOLLSTÄNDIGES PROTOKOLL\n")
            f.write(f"{sep80}\n\n")
            f.write(f"  Tage simuliert          : {len(result.day_results)}\n")
            if result.days_to_complete:
                f.write(f"  Alle Stationen gewartet : Tag {result.days_to_complete}\n")
            else:
                f.write(f"  Verbleibende Stationen  : {result.remaining_stations_at_end}\n")
            f.write(f"\n")
            f.write(f"  Störungen gesamt        : {result.total_disruptions}\n")
            f.write(f"    Gleichen Tag erledigt : {result.same_day_handled} "
                    f"({result.same_day_rate:.1%})\n")
            f.write(f"    Carryover             : {result.total_carryover}\n")
            f.write(f"\n")
            op = sum(r.operational_cost_eur for r in result.day_results)
            wage = sum(r.wage_cost_eur for r in result.day_results)
            fuel = sum(r.fuel_cost_eur for r in result.day_results)
            dt = sum(r.downtime_cost_eur for r in result.day_results)
            f.write(f"  Gesamtkosten            : {result.total_cost_eur:>10,.2f} EUR\n")
            f.write(f"    Betriebskosten        : {op:>10,.2f} EUR\n")
            f.write(f"      Lohnkosten          : {wage:>10,.2f} EUR\n")
            f.write(f"      Fahrtkosten         : {fuel:>10,.2f} EUR\n")
            f.write(f"    Ausfallkosten         : {dt:>10,.2f} EUR\n")
            f.write(f"\n{sep80}\n\n")

            f.write("TAGESÜBERSICHT\n")
            f.write(f"{sep40}\n")
            f.write(f"{'Tag':>4}  {'Geplant':>7}  {'Erledigt':>8}  {'Störg.':>6}  "
                    f"{'Carry':>5}  {'Lohn':>8}  {'Fahrt':>7}  {'Ausfall':>8}  {'Gesamt':>10}\n")
            f.write(f"{sep40}\n")
            for r in result.day_results:
                f.write(
                    f"{r.day:>4d}  {r.n_routine_tasks:>7d}  {r.n_routine_completed:>8d}  "
                    f"{r.disruptions_handled:>6d}  {r.disruptions_carryover:>5d}  "
                    f"{r.wage_cost_eur:>8.2f}  "
                    f"{r.fuel_cost_eur:>7.2f}  "
                    f"{r.downtime_cost_eur:>8.2f}  "
                    f"{r.total_cost_eur:>10.2f}\n"
                )
            f.write(f"{sep40}\n\n")

            f.write(f"{sep80}\n")
            f.write("STUNDEN-PROTOKOLL\n")
            f.write(f"{sep80}\n")
            for dr in result.day_results:
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"TAG {dr.day:>3d}  |  Routine geplant: {dr.n_routine_tasks}, erledigt: {dr.n_routine_completed}  |  "
                    f"Störungen: {dr.disruptions_handled + dr.disruptions_carryover} gesamt, "
                    f"{dr.disruptions_handled} erledigt, "
                    f"{dr.disruptions_carryover} Carryover  |  "
                    f"Kosten: {dr.total_cost_eur:.2f} EUR\n"
                )
                f.write(f"{'=' * 60}\n")
                for log in dr.hourly_logs:
                    f.write(str(log) + "\n")

        print(f"Protokoll gespeichert: {out.resolve()}")

    def write_json(
        self,
        result: SimulationResult,
        path: str,
        label: str = "SIMULATION",
        run_id: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> None:
        """
        Speichert das vollständige Simulationsergebnis als strukturierte JSON-Datei.

        Aufbau:
          meta    – Label, Run-ID, Zeitstempel, Modellparameter
          summary – Skalare Kennzahlen der gesamten Simulation
          days    – Eine Zeile pro Tag (Tagesübersicht)
          hourly  – Eine Zeile pro (Tag, Stunde) mit Störungs- und Aktionslisten
                    (enthält alle exakten Schritte und Entscheidungen als String-Arrays)

        Für Monte-Carlo-Analysen run_id setzen; dann lassen sich N Dateien per
        pd.concat(pd.DataFrame(d["days"]) for d in runs) einfach stapeln.

        Parameters
        ----------
        model_params : dict | None
            Modellspezifische Parameter, die unter meta.model_params gespeichert werden.
            Typischerweise: seed, failure_mode, alpha, theta, cost_params, etc.
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
                    **({"solver_debug": log.solver_debug} if log.solver_debug else {}),
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

        print(f"JSON-Protokoll gespeichert: {out.resolve()}")

    # ------------------------------------------------------------------
    # Tages-Simulation
    # ------------------------------------------------------------------

    def _run_day(
        self,
        day: int,
        remaining: set[int],
        team_states: list[TeamState],
        carryover_tasks: list[MaintenanceTask],
        day_disruptions: list[DisruptionEvent],
    ) -> tuple[DayResult, list[SimRoute], list[DisruptionEvent]]:
        """Simuliert einen vollständigen Arbeitstag."""
        hourly_logs: list[HourLog] = []
        downtime_cost = 0.0

        # Stationsauswahl (dsm_array für V̂-basierte Zonenauswahl bei CFA/VFA)
        dsm_array = (
            self._days_since_maintenance
            if self._failure_mode == "stochastic"
            else None
        )
        assignment = self.selector.select_for_day(
            list(remaining), team_states, carryover_tasks, dsm_array=dsm_array
        )
        all_tasks = [t for tasks in assignment.team_tasks.values() for t in tasks]
        n_routine = sum(1 for t in all_tasks if t.task_type == "routine")

        # Tage seit Wartung in Routine-Tasks einschreiben (für CFA-Policy)
        if self._failure_mode == "stochastic":
            for task in all_tasks:
                if task.task_type == "routine":
                    task.days_since_maintenance = float(
                        self._days_since_maintenance[task.node_idx]
                    )

        # team_assignment: {team_id: [node_idx, ...]} für alle Tasks (Routine + Carryover)
        # Carryover-Störungen werden vom Selector bereits einem Team zugewiesen;
        # diese Zuordnung muss beim Per-Team-Solve respektiert werden.
        team_assignment: dict[int, list[int]] = {
            tid: [t.node_idx for t in tasks]
            for tid, tasks in assignment.team_tasks.items()
        }

        # Initialplan via Policy
        if all_tasks:
            daily_plan = self.policy.create_initial_plan(all_tasks, team_assignment)
            logger.info(
                f"Tag {day:>3d}: {len(all_tasks)} Aufgaben ({n_routine} Routine, "
                f"{len(all_tasks) - n_routine} Carryover) | "
                f"OR-Tools: {daily_plan.solver_status}"
            )
        else:
            daily_plan = None
            logger.info(f"Tag {day:>3d}: Keine Aufgaben.")

        sim_routes = plan_to_sim_routes(daily_plan, all_tasks, self.n_teams)

        # Solver-Debug: Input und Output für den 8:00-HourLog aufzeichnen
        _input_nodes_by_team: dict[int, set[int]] = {
            tid: {t.node_idx for t in tasks if t.task_type == "routine"}
            for tid, tasks in assignment.team_tasks.items()
        }
        _output_nodes: set[int] = {
            stop.node_idx for route in sim_routes for stop in route.stops
        }
        _dropped_nodes: list[int] = sorted(
            node for nodes in _input_nodes_by_team.values()
            for node in nodes
            if node not in _output_nodes
        )
        _dropped_by_team: dict[int, list[int]] = {
            tid: sorted(n for n in nodes if n not in _output_nodes)
            for tid, nodes in _input_nodes_by_team.items()
        }
        _solver_debug = {
            "input": {
                "use_team_assignment": self.config.get("planning", {}).get("use_team_assignment", True),
                "n_routine": n_routine,
                "n_carryover": len(all_tasks) - n_routine,
                "tasks_per_team": {
                    tid: [
                        {
                            "node_idx": t.node_idx,
                            "task_type": t.task_type,
                            "service_time": t.service_time,
                            "soft_deadline_min": t.soft_deadline_min,
                            "deadline_penalty": t.deadline_penalty,
                            "dsm": round(float(t.days_since_maintenance), 1),
                        }
                        for t in tasks
                    ]
                    for tid, tasks in assignment.team_tasks.items()
                },
            },
            "output": {
                "solver_status": daily_plan.solver_status if daily_plan else "NO_PLAN",
                "status_before_retry": daily_plan.status_before_retry if daily_plan else "",
                "n_dropped": daily_plan.n_dropped if daily_plan else 0,
                "dropped_nodes": _dropped_nodes,
                "dropped_nodes_by_team": _dropped_by_team,
                "routes": {
                    route.team_id: [s.node_idx for s in route.stops]
                    for route in sim_routes
                },
            },
        }

        _initial_plan_notes: list[str] = []
        if daily_plan and daily_plan.n_dropped > 0:
            _initial_plan_notes.append(
                f"Initialplan-Retry: {daily_plan.n_dropped} Routine-Stop(s) nach Depot-Distanz "
                f"ausgebaut (erster Status: {daily_plan.status_before_retry}, "
                f"Ergebnis: {daily_plan.solver_status})"
            )
        maint_cfg = self.config["maintenance"]
        _lunch_dur = maint_cfg.get("lunch_duration_min", 0)
        _lunch_early = maint_cfg.get("lunch_earliest_min", 240)
        if _lunch_dur > 0:
            for r in sim_routes:
                _inject_lunch(r, _lunch_early, _lunch_dur)

        # Stündliche Simulation
        by_hour: dict[int, list[DisruptionEvent]] = {}
        for d in day_disruptions:
            by_hour.setdefault(d.hour, []).append(d)

        disruptions_handled = 0
        carried_disruptions: list[DisruptionEvent] = []

        for hour in range(8, 17):
            time_min = float((hour - 8) * 60)
            hour_log = HourLog(day=day, hour=hour)

            # 8:00: Strukturierter Initialplan
            if hour == 8:
                hour_log.solver_debug = _solver_debug
                hour_log.notes.extend(_initial_plan_notes)
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
                    if r.stops:
                        last = r.stops[-1]
                        t_back = mat8[last.node_idx, 0] / 60.0
                        km_back = _approx_km(self.all_coords[last.node_idx], self.all_coords[0])
                        depot_rt = _fmt(last.departure_min + t_back)
                        depot_travel = round(t_back, 1)
                        depot_km = round(km_back, 2)
                    else:
                        depot_rt = depot_travel = depot_km = None
                    hour_log.initial_plan.append({
                        "team_id": r.team_id,
                        "n_stops": len(r.stops),
                        "route": route_dicts,
                        "depot_return_time": depot_rt,
                        "depot_return_travel_min": depot_travel,
                        "depot_return_km": depot_km,
                    })

            # Störungen via Policy behandeln
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

                # Snapshot vor Replan: verbleibende Nodes pro Team
                before_replan: dict[int, set[int]] = {
                    r.team_id: {s.node_idx for s in r.remaining_stops_at(time_min)}
                    for r in sim_routes
                }

                handled, carried, h_downtime = self.policy.handle_disruptions(
                    h_disruptions, sim_routes, time_min, hour, hour_log
                )
                disruptions_handled += handled
                carried_disruptions.extend(carried)
                downtime_cost += h_downtime

                # Ausfallkosten für nicht mehr schaffbare Störungen: Meldung bis 16:00
                report_min = float((hour - 8) * 60)
                remaining_workday_h = (self.WORKDAY_MINUTES - report_min) / 60.0
                for d in carried:
                    downtime_cost += remaining_workday_h * d.power_kw * self.cost_params.downtime_eur_per_kwh

                # Replan-Log: Diff vor/nach, pro Team
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

                    ref_stops = new_remaining if new_remaining else (r.stops or [])
                    if ref_stops:
                        last = ref_stops[-1]
                        mat = self._get_matrix(last.departure_min)
                        t_back = mat[last.node_idx, 0] / 60.0
                        depot_rt = _fmt(last.departure_min + t_back)
                    else:
                        depot_rt = None

                    hour_log.replan.append({
                        "team_id": r.team_id,
                        "dropped_stops": [
                            {"node_idx": n, "task_type": "routine", "reason": "carryover"}
                            for n in dropped_nodes
                        ],
                        "remaining_route": route_dicts,
                        "depot_return_time": depot_rt,
                    })

            # Teamstatus strukturiert
            for r in sim_routes:
                done_n = len(r.completed_nodes_at(time_min))
                remaining_n = len(r.remaining_stops_at(time_min))
                active = next(
                    (s for s in r.stops if s.arrival_min <= time_min < s.departure_min), None
                )
                nxt = next((s for s in r.stops if s.arrival_min > time_min), None)

                ts: dict = {
                    "team_id": r.team_id,
                    "stops_completed": done_n,
                    "stops_remaining": remaining_n,
                }
                if active:
                    ts.update({
                        "status": "working",
                        "current_node": active.node_idx,
                        "active_task_type": active.task_type,
                        "active_departure_time": _fmt(active.departure_min),
                        "next_node": None,
                        "next_arrival_time": None,
                        "depot_return_time": None,
                    })
                elif nxt:
                    ts.update({
                        "status": "driving",
                        "current_node": r.current_node_at(time_min),
                        "active_task_type": None,
                        "active_departure_time": None,
                        "next_node": nxt.node_idx,
                        "next_arrival_time": _fmt(nxt.arrival_min),
                        "depot_return_time": None,
                    })
                elif r.stops:
                    last = r.stops[-1]
                    mat = self._get_matrix(last.departure_min)
                    t_back = mat[last.node_idx, 0] / 60.0
                    ts.update({
                        "status": "returning",
                        "current_node": last.node_idx,
                        "active_task_type": None,
                        "active_departure_time": None,
                        "next_node": None,
                        "next_arrival_time": None,
                        "depot_return_time": _fmt(last.departure_min + t_back),
                    })
                else:
                    ts.update({
                        "status": "idle",
                        "current_node": 0,
                        "active_task_type": None,
                        "active_departure_time": None,
                        "next_node": None,
                        "next_arrival_time": None,
                        "depot_return_time": None,
                    })
                hour_log.team_status.append(ts)

            # 16:00: Ausgeführten Tagesplan speichern
            if hour == 16:
                for r in sim_routes:
                    n_planned = len(r.stops)
                    n_completed = sum(
                        1 for s in r.stops if s.departure_min <= self.WORKDAY_MINUTES
                    )
                    n_disrupt = sum(
                        1 for s in r.stops if s.task_type not in ("routine",)
                    )
                    route_dicts = [
                        {
                            "node_idx": s.node_idx,
                            "task_type": s.task_type,
                            "arrival_time": _fmt(s.arrival_min),
                            "service_min": int(s.service_min),
                            "departure_time": _fmt(s.departure_min),
                            "completed": s.departure_min <= self.WORKDAY_MINUTES,
                        }
                        for s in r.stops
                    ]
                    if r.stops:
                        last = r.stops[-1]
                        mat = self._get_matrix(last.departure_min)
                        t_back = mat[last.node_idx, 0] / 60.0
                        depot_rt = _fmt(last.departure_min + t_back)
                    else:
                        depot_rt = None
                    hour_log.executed_plan.append({
                        "team_id": r.team_id,
                        "stops_planned": n_planned,
                        "stops_completed": n_completed,
                        "disruption_stops": n_disrupt,
                        "route": route_dicts,
                        "depot_return_time": depot_rt,
                    })

            hourly_logs.append(hour_log)

        # Ausfallkosten für Carryover-Störungen: ab 8:00 bis Service abgeschlossen
        carryover_nodes = {t.node_idx for t in carryover_tasks if t.task_type == "carryover"}
        accounted: set[int] = set()
        cp = self.cost_params
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in carryover_nodes and stop.node_idx not in accounted:
                    accounted.add(stop.node_idx)
                    dep = min(stop.departure_min, float(self.WORKDAY_MINUTES))
                    power_kw = self.node_to_power.get(stop.node_idx, 22.0)
                    downtime_cost += (dep / 60.0) * power_kw * cp.downtime_eur_per_kwh

        op_cost, wage_cost, fuel_cost = self._compute_operational_cost(sim_routes)

        n_routine_completed = sum(
            1 for route in sim_routes
            for stop in route.stops
            if stop.task_type == "routine" and stop.departure_min <= self.WORKDAY_MINUTES
        )

        result = DayResult(
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
        )
        return result, sim_routes, carried_disruptions

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _generate_day_disruptions(self, day: int) -> list[DisruptionEvent]:
        """
        Generiert stochastische Störungen für einen ganzen Simulationstag.

        Für jede Stunde (8–16 Uhr) und jede Säule wird unabhängig gewürfelt.
        Eine Säule kann pro Tag maximal eine Störung erhalten.
        Die Ausfallwahrscheinlichkeit skaliert mit der Zeit seit letzter Wartung:

            p(t) = p_base * (initial_factor + (1 - initial_factor) * min(t, recovery_days) / recovery_days)
        """
        fail_cfg = self.config["failure_simulation"]
        p1_base: float = fail_cfg["p1_per_hour"]
        p2_base: float = fail_cfg["p2_per_hour"]
        recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))
        cp = self.cost_params

        disrupted_today: set[int] = set()
        events: list[DisruptionEvent] = []

        for hour in range(8, 17):
            for node_idx in range(1, self.n_stations + 1):
                if node_idx in disrupted_today:
                    continue

                t = min(self._days_since_maintenance[node_idx], recovery_days)
                factor = initial_factor + (1.0 - initial_factor) * t / recovery_days
                power_kw = self.node_to_power.get(node_idx, 22.0)

                # Typ 1 prüfen
                if self._rng.random() < p1_base * factor:
                    events.append(DisruptionEvent(
                        day=day,
                        hour=hour,
                        node_idx=node_idx,
                        disruption_type="Typ 1",
                        power_kw=power_kw,
                        service_min=float(cp.typ1_service_min),
                    ))
                    disrupted_today.add(node_idx)
                    continue

                # Typ 2 prüfen
                if self._rng.random() < p2_base * factor:
                    mat = self.traffic_matrices.get(
                        hour, list(self.traffic_matrices.values())[0]
                    )
                    roundtrip_min = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                    service_min = (
                        cp.typ2_dismount_min
                        + roundtrip_min
                        + cp.typ2_handling_min
                        + cp.typ2_remount_min
                    )
                    events.append(DisruptionEvent(
                        day=day,
                        hour=hour,
                        node_idx=node_idx,
                        disruption_type="Typ 2",
                        power_kw=power_kw,
                        service_min=service_min,
                    ))
                    disrupted_today.add(node_idx)

        logger.debug(
            f"Tag {day}: {len(events)} stochastische Störungen generiert "
            f"({sum(1 for e in events if e.disruption_type == 'Typ 1')} Typ-1, "
            f"{sum(1 for e in events if e.disruption_type == 'Typ 2')} Typ-2)"
        )
        return events

    def _load_disruptions(self, df: pd.DataFrame) -> list[DisruptionEvent]:
        """Lädt und konvertiert Stördaten aus dem DataFrame."""
        cp = self.cost_params
        events: list[DisruptionEvent] = []

        for _, row in df.iterrows():
            station_id = int(row["Station_ID"])
            node_idx = self.id_to_node.get(station_id)
            if node_idx is None:
                logger.warning(f"Station_ID {station_id} nicht in Stationsdaten.")
                continue

            power_kw = self.node_to_power.get(node_idx, 22.0)
            d_type = str(row["Typ"])
            hour = int(row["Uhrzeit"])

            if d_type == "Typ 1":
                service_min = cp.typ1_service_min
            else:  # Typ 2: Demontage + Lagerfahrt + Handling + Montage
                mat = self.traffic_matrices.get(hour, list(self.traffic_matrices.values())[0])
                roundtrip_min = (mat[node_idx, 0] + mat[0, node_idx]) / 60.0
                service_min = (
                    cp.typ2_dismount_min
                    + roundtrip_min
                    + cp.typ2_handling_min
                    + cp.typ2_remount_min
                )

            events.append(DisruptionEvent(
                day=int(row["Tag"]),
                hour=hour,
                node_idx=node_idx,
                disruption_type=d_type,
                power_kw=power_kw,
                service_min=service_min,
            ))

        logger.info(
            f"Störungen geladen: {len(events)} Ereignisse "
            f"({sum(1 for e in events if e.disruption_type == 'Typ 1')} Typ-1, "
            f"{sum(1 for e in events if e.disruption_type == 'Typ 2')} Typ-2)"
        )
        return events

    def _compute_operational_cost(
        self, sim_routes: list[SimRoute], is_last_day: bool = False
    ) -> tuple[float, float, float]:
        """
        Berechnet operative Tageskosten (Lohn + Kraftstoff) aller Teams.

        Normaltage: volle WORKDAY_MINUTES je aktivem Team als Lohnbasis.
        Letzter Tag (is_last_day=True): tatsächliche Arbeitszeit (Fahrt + Service + Depotfahrt).

        Returns
        -------
        (total, wage_cost, fuel_cost)
        """
        cp = self.cost_params
        wage_total = 0.0
        fuel_total = 0.0

        for route in sim_routes:
            if not route.stops:
                continue

            legs: list[tuple[int, int, float]] = []
            legs.append((0, route.stops[0].node_idx, 0.0))
            for i in range(len(route.stops) - 1):
                legs.append((
                    route.stops[i].node_idx,
                    route.stops[i + 1].node_idx,
                    route.stops[i].departure_min,
                ))
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
        """Gibt die passende Stundenmatrix für einen Zeitstempel zurück."""
        hour = 8 + int(max(0.0, time_min)) // 60
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]
