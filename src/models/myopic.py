"""
Myopic-Modell für die Wartungsoptimierung von E-Ladesäulen.

Strategie: Bei jeder stündlichen Störungsmeldung wird jede Störung in die
Route des Teams mit den geringsten Gesamtzusatzkosten eingefügt (greedy
cheapest insertion). Ist eine Einplanung am selben Tag nicht mehr vor
17:00 Uhr möglich, wird die Störung als Carryover auf den nächsten Tag
verschoben.

Kosten:
  Operational  = (Fahrzeit + Servicezeit) × 40 €/h + km × 0,30 €/km
  Ausfall      = Wartezeit bis Service [h] × Nennleistung [kW] × 0,50 €/kWh

Typ-1-Störung : Servicezeit = 60 min
Typ-2-Störung : Servicezeit = 30 min (Demontage)
                             + Rundfahrt Station→Depot→Station [min]
                             + 5 min (Lagerhandling)
                             + 30 min (Montage)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.cost_params import CostParams
from src.planning.clustering import _approx_km
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------

def _fmt(minutes_from_8: float) -> str:
    """Minuten ab 8:00 → 'HH:MM'-String."""
    total = int(8 * 60 + minutes_from_8)
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class DisruptionEvent:
    """Eine Störung aus malfunction.csv."""

    day: int
    hour: int               # 8–16
    node_idx: int           # 1-basiert (Depot = 0)
    disruption_type: str    # "Typ 1" | "Typ 2"
    power_kw: float         # Nennleistung für Ausfallkosten
    service_min: float      # berechnete Servicezeit inkl. Typ-2-Lagerfahrt


@dataclass
class SimStop:
    """Ein Halt in der simulierten Tagesroute."""

    node_idx: int
    task_type: str      # "routine" | "Typ 1" | "Typ 2"
    arrival_min: float  # Minuten ab 8:00
    service_min: float  # Servicezeit an dieser Station

    @property
    def departure_min(self) -> float:
        return self.arrival_min + self.service_min


@dataclass
class SimRoute:
    """Simulierte Route eines Teams für einen Tag (veränderlich)."""

    team_id: int
    stops: list[SimStop] = field(default_factory=list)

    def current_node_at(self, time_min: float) -> int:
        """Letzter angefahrener Knoten zum Zeitpunkt time_min (0 = Depot)."""
        arrived = [s for s in self.stops if s.arrival_min <= time_min]
        return arrived[-1].node_idx if arrived else 0

    def current_departure_at(self, time_min: float) -> float:
        """Frühestmögliche Abfahrtszeit vom aktuellen Knoten."""
        arrived = [s for s in self.stops if s.arrival_min <= time_min]
        return max(time_min, arrived[-1].departure_min) if arrived else time_min

    def remaining_stops_at(self, time_min: float) -> list[SimStop]:
        """Noch nicht angefahrene Stops zum Zeitpunkt time_min."""
        return [s for s in self.stops if s.arrival_min > time_min]

    def completed_nodes_at(self, time_min: float) -> list[int]:
        """Knoten deren Service bis time_min abgeschlossen ist."""
        return [s.node_idx for s in self.stops if s.departure_min <= time_min]


@dataclass
class HourLog:
    """Stundenprotokoll für die Simulation."""

    day: int
    hour: int
    disruptions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"[Tag {self.day:>3d}, {self.hour:02d}:00]"]
        for d in self.disruptions:
            lines.append(f"  !! STÖRUNG: {d}")
        for a in self.actions:
            lines.append(f"  -> {a}")
        return "\n".join(lines)


@dataclass
class DayResult:
    """Ergebnis eines Simulationstages."""

    day: int
    n_routine_tasks: int
    disruptions_handled: int
    disruptions_carryover: int
    operational_cost_eur: float
    downtime_cost_eur: float
    hourly_logs: list[HourLog]

    @property
    def total_cost_eur(self) -> float:
        return self.operational_cost_eur + self.downtime_cost_eur


@dataclass
class SimulationResult:
    """Gesamtergebnis der Myopic-Simulation über alle Tage."""

    day_results: list[DayResult]
    total_disruptions: int
    same_day_handled: int
    total_carryover: int
    days_to_complete: Optional[int]      # Tag, an dem alle Stationen gewartet
    remaining_stations_at_end: int       # ungewartete Stationen nach n_days

    @property
    def total_cost_eur(self) -> float:
        return sum(r.total_cost_eur for r in self.day_results)

    @property
    def same_day_rate(self) -> float:
        return self.same_day_handled / self.total_disruptions if self.total_disruptions else 1.0


# ---------------------------------------------------------------------------
# Myopic-Modell
# ---------------------------------------------------------------------------

class MyopicModel:
    """
    Myopic-Modell: Greedy cheapest-insertion Replanning bei Störungen.

    Parameters
    ----------
    solver : VRPSolver
        OR-Tools Solver für den Tages-Initialplan.
    selector : DailyZoneSelector
        Tägliche Stationsauswahl (gemeinsame Basis für alle Modelle).
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    stations_df : pd.DataFrame
        Bereinigter Stationsdatensatz aus load_stations().
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden, indiziert nach Uhrzeit.
    config : dict
        Konfigurationsdict aus config.yaml.
    cost_params : CostParams | None
        Kostenparameter (None → Standardwerte).
    """

    def __init__(
        self,
        solver: VRPSolver,
        selector: DailyZoneSelector,
        all_coords: np.ndarray,
        stations_df: pd.DataFrame,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        cost_params: Optional[CostParams] = None,
    ) -> None:
        self.solver = solver
        self.selector = selector
        self.all_coords = all_coords
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.cost_params = cost_params or CostParams()
        self.n_stations = len(stations_df)
        self.n_teams: int = config["maintenance"]["n_teams"]

        # Station-ID (ID-Spalte im CSV, 1-basiert) → node_idx
        # node_idx = DataFrame-Zeilenindex + 1  (Depot belegt Index 0)
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

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def run(
        self,
        disruptions_df: pd.DataFrame,
        max_days: int = 365,
    ) -> SimulationResult:
        """
        Führt die Simulation durch bis alle Stationen gewartet sind.

        Die Simulation stoppt automatisch, sobald:
          1. Alle Stationen mindestens einmal routinemäßig gewartet wurden, UND
          2. Keine offenen Carryover-Störungen mehr vorhanden sind.

        Störungen aus malfunction.csv nach dem Abschlusstag werden ignoriert.

        Parameters
        ----------
        disruptions_df : pd.DataFrame
            Stördaten (Pflicht-Spalten: Tag, Uhrzeit, Typ, Station_ID).
        max_days : int
            Maximale Simulationstage als Sicherheitsgrenze (Standard: 365).

        Returns
        -------
        SimulationResult
        """
        disruptions = self._load_disruptions(disruptions_df)
        by_day: dict[int, list[DisruptionEvent]] = {}
        for d in disruptions:
            by_day.setdefault(d.day, []).append(d)

        remaining: set[int] = set(range(self.n_stations))
        carryover_tasks: list[MaintenanceTask] = []

        day_results: list[DayResult] = []
        same_day_handled = 0
        total_carryover = 0
        days_to_complete: Optional[int] = None
        last_day = 0

        for day in tqdm(range(1, max_days + 1), desc="Simulation", unit="Tag"):
            last_day = day
            team_states = [
                TeamState(team_id=i, current_node=0, current_time=0)
                for i in range(self.n_teams)
            ]

            # Nach Abschluss der Routinewartung keine neuen Störungen mehr
            if days_to_complete is None:
                day_disruptions = by_day.get(day, [])
            else:
                day_disruptions = []

            result, sim_routes, new_carryover = self._run_day(
                day, remaining, team_states, carryover_tasks, day_disruptions,
            )
            day_results.append(result)
            same_day_handled += result.disruptions_handled
            total_carryover += result.disruptions_carryover

            # Erledigte Routine-Stationen aus remaining entfernen
            for route in sim_routes:
                for stop in route.stops:
                    if stop.task_type == "routine" and stop.departure_min <= self.solver.WORKDAY_MINUTES:
                        remaining.discard(stop.node_idx - 1)

            if not remaining and days_to_complete is None:
                days_to_complete = day
                logger.info(f"Alle {self.n_stations} Stationen nach Tag {day} gewartet.")

            # Carryover-Störungen als MaintenanceTasks für den nächsten Tag
            carryover_tasks = [
                MaintenanceTask(
                    node_idx=d.node_idx,
                    task_type="disruption",
                    priority=1,
                    service_time=int(round(d.service_min)),
                )
                for d in new_carryover
            ]

            # Stop: alle Stationen fertig UND kein Carryover mehr
            if days_to_complete is not None and not carryover_tasks:
                break

        # Nur Störungen der tatsächlich simulierten Tage zählen
        sim_end = days_to_complete if days_to_complete else last_day
        total_disruptions = sum(len(by_day.get(d, [])) for d in range(1, sim_end + 1))

        return SimulationResult(
            day_results=day_results,
            total_disruptions=total_disruptions,
            same_day_handled=same_day_handled,
            total_carryover=total_carryover,
            days_to_complete=days_to_complete,
            remaining_stations_at_end=len(remaining),
        )

    def print_summary(self, result: SimulationResult) -> None:
        """Gibt eine kompakte Zusammenfassung der Simulation aus."""
        sep = "=" * 62
        print(sep)
        print("  MYOPIC SIMULATION – ZUSAMMENFASSUNG")
        print(sep)
        print(f"  Tage simuliert          : {len(result.day_results)}")
        if result.days_to_complete:
            print(f"  Alle Stationen gewartet : Tag {result.days_to_complete}")
        else:
            print(f"  Verbleibende Stationen  : {result.remaining_stations_at_end}")
        print()
        print(f"  Störungen gesamt        : {result.total_disruptions}")
        print(f"    Gleichen Tag erledigt : {result.same_day_handled} "
              f"({result.same_day_rate:.1%})")
        print(f"    Carryover             : {result.total_carryover}")
        print()
        op = sum(r.operational_cost_eur for r in result.day_results)
        dt = sum(r.downtime_cost_eur for r in result.day_results)
        print(f"  Gesamtkosten            : {result.total_cost_eur:>10,.2f} €")
        print(f"    Betriebskosten        : {op:>10,.2f} €")
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
        print(f"  Routineaufgaben : {day_result.n_routine_tasks}")
        print(f"  Störungen       : {day_result.disruptions_handled} erledigt, "
              f"{day_result.disruptions_carryover} Carryover")
        print(f"  Kosten          : {day_result.total_cost_eur:.2f} €  "
              f"(Betrieb {day_result.operational_cost_eur:.2f} €, "
              f"Ausfall {day_result.downtime_cost_eur:.2f} €)")
        print(sep)
        for log in day_result.hourly_logs:
            print(str(log))

    def write_log(self, result: SimulationResult, path: str) -> None:
        """
        Speichert das vollständige Simulationsprotokoll als Textdatei.

        Parameters
        ----------
        result : SimulationResult
            Ergebnis von run().
        path : str
            Ausgabepfad, z.B. 'logs/myopic_simulation.log'.
        """
        from pathlib import Path
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            sep80 = "=" * 80
            sep40 = "-" * 40

            # --- Zusammenfassung ---
            f.write(f"{sep80}\n")
            f.write("  MYOPIC SIMULATION – VOLLSTÄNDIGES PROTOKOLL\n")
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
            dt = sum(r.downtime_cost_eur for r in result.day_results)
            f.write(f"  Gesamtkosten            : {result.total_cost_eur:>10,.2f} EUR\n")
            f.write(f"    Betriebskosten        : {op:>10,.2f} EUR\n")
            f.write(f"    Ausfallkosten         : {dt:>10,.2f} EUR\n")
            f.write(f"\n{sep80}\n\n")

            # --- Tagesübersicht (kompakt) ---
            f.write("TAGESÜBERSICHT\n")
            f.write(f"{sep40}\n")
            f.write(f"{'Tag':>4}  {'Routine':>7}  {'Störg.':>6}  "
                    f"{'Carry':>5}  {'Op-Kosten':>10}  {'Ausfall':>8}  {'Gesamt':>10}\n")
            f.write(f"{sep40}\n")
            for r in result.day_results:
                f.write(
                    f"{r.day:>4d}  {r.n_routine_tasks:>7d}  "
                    f"{r.disruptions_handled:>6d}  {r.disruptions_carryover:>5d}  "
                    f"{r.operational_cost_eur:>10.2f}  "
                    f"{r.downtime_cost_eur:>8.2f}  "
                    f"{r.total_cost_eur:>10.2f}\n"
                )
            f.write(f"{sep40}\n\n")

            # --- Stunden-Protokoll pro Tag ---
            f.write(f"{sep80}\n")
            f.write("STUNDEN-PROTOKOLL\n")
            f.write(f"{sep80}\n")
            for dr in result.day_results:
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"TAG {dr.day:>3d}  |  Routine: {dr.n_routine_tasks}  |  "
                    f"Störungen: {dr.disruptions_handled} erledigt, "
                    f"{dr.disruptions_carryover} Carryover  |  "
                    f"Kosten: {dr.total_cost_eur:.2f} EUR\n"
                )
                f.write(f"{'=' * 60}\n")
                for log in dr.hourly_logs:
                    f.write(str(log) + "\n")

        print(f"Protokoll gespeichert: {out.resolve()}")

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
        """
        Simuliert einen vollständigen Arbeitstag.

        Returns
        -------
        (DayResult, simulierte Routen, Carryover-Störungen)
        """
        hourly_logs: list[HourLog] = []
        downtime_cost = 0.0

        # --- Schritt 1: Stationsauswahl ---
        assignment = self.selector.select_for_day(
            list(remaining), team_states, carryover_tasks
        )
        all_tasks = [t for tasks in assignment.team_tasks.values() for t in tasks]
        n_routine = sum(1 for t in all_tasks if t.task_type == "routine")

        # --- Schritt 2: Initialplan via OR-Tools ---
        if all_tasks:
            daily_plan = self.solver.create_initial_plan(all_tasks)
            logger.info(
                f"Tag {day:>3d}: {len(all_tasks)} Aufgaben ({n_routine} Routine, "
                f"{len(all_tasks) - n_routine} Carryover) | "
                f"OR-Tools: {daily_plan.solver_status}"
            )
        else:
            daily_plan = None
            logger.info(f"Tag {day:>3d}: Keine Aufgaben.")

        # --- Schritt 3: SimRoutes mit korrekten Servicezeiten aufbauen ---
        sim_routes = self._plan_to_sim_routes(daily_plan, all_tasks)

        # --- Stündliche Simulation ---
        by_hour: dict[int, list[DisruptionEvent]] = {}
        for d in day_disruptions:
            by_hour.setdefault(d.hour, []).append(d)

        disruptions_handled = 0
        carried_disruptions: list[DisruptionEvent] = []

        for hour in range(8, 17):
            time_min = float((hour - 8) * 60)
            hour_log = HourLog(day=day, hour=hour)

            # Initialplan-Detailroute in 8-Uhr-Log
            if hour == 8:
                for r in sim_routes:
                    if r.stops:
                        first = _fmt(r.stops[0].arrival_min)
                        last = _fmt(r.stops[-1].departure_min)
                        hour_log.actions.append(
                            f"Team {r.team_id}: {len(r.stops)} Stops "
                            f"({first}–{last} Uhr)"
                        )
                        mat = self._get_matrix(0.0)
                        prev_node = 0
                        prev_dep = 0.0
                        for stop in r.stops:
                            t_min = mat[prev_node, stop.node_idx] / 60.0
                            km = _approx_km(
                                self.all_coords[prev_node], self.all_coords[stop.node_idx]
                            )
                            prev_label = "Depot" if prev_node == 0 else f"Node {prev_node:>3d}"
                            hour_log.actions.append(
                                f"  {prev_label:>8} --[{t_min:4.1f} min, {km:4.2f} km]--> "
                                f"Node {stop.node_idx:>3d} ({stop.task_type:10s}) "
                                f"Ankunft {_fmt(stop.arrival_min)}, "
                                f"Abfahrt {_fmt(stop.departure_min)}"
                            )
                            prev_node = stop.node_idx
                            prev_dep = stop.departure_min
                        # Rückfahrt zum Depot
                        t_back = mat[prev_node, 0] / 60.0
                        km_back = _approx_km(self.all_coords[prev_node], self.all_coords[0])
                        hour_log.actions.append(
                            f"  Node {prev_node:>3d} --[{t_back:4.1f} min, {km_back:4.2f} km]--> "
                            f"Depot  Ankunft ~{_fmt(prev_dep + t_back)}"
                        )
                    else:
                        hour_log.actions.append(f"Team {r.team_id}: keine Stops geplant")

            # Störungen dieser Stunde behandeln
            if hour in by_hour:
                h_disruptions = by_hour[hour]
                hour_log.disruptions = [
                    f"{d.disruption_type} @ Node {d.node_idx} "
                    f"({d.power_kw:.0f} kW, Service {d.service_min:.0f} min)"
                    for d in h_disruptions
                ]
                handled, carried, h_downtime = self._handle_disruptions(
                    h_disruptions, sim_routes, time_min, hour, hour_log
                )
                disruptions_handled += handled
                carried_disruptions.extend(carried)
                downtime_cost += h_downtime

            # Teamstatus protokollieren
            for r in sim_routes:
                node = r.current_node_at(time_min)
                remaining_n = len(r.remaining_stops_at(time_min))
                done_n = len(r.completed_nodes_at(time_min))
                hour_log.actions.append(
                    f"Team {r.team_id}: Node {node:>3d}, "
                    f"{done_n} erledigt, {remaining_n} verbleibend"
                )

            hourly_logs.append(hour_log)

        # --- Betriebskosten berechnen ---
        op_cost = self._compute_operational_cost(sim_routes)

        result = DayResult(
            day=day,
            n_routine_tasks=n_routine,
            disruptions_handled=disruptions_handled,
            disruptions_carryover=len(carried_disruptions),
            operational_cost_eur=op_cost,
            downtime_cost_eur=downtime_cost,
            hourly_logs=hourly_logs,
        )
        return result, sim_routes, carried_disruptions

    # ------------------------------------------------------------------
    # Störungsbehandlung (Cheapest Insertion)
    # ------------------------------------------------------------------

    def _handle_disruptions(
        self,
        disruptions: list[DisruptionEvent],
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        log: HourLog,
    ) -> tuple[int, list[DisruptionEvent], float]:
        """
        Fügt Störungen greedy cheapest in die Teamrouten ein.

        Reihenfolge: aufsteigend nach Insertionskosten (günstigste zuerst).
        Nach jeder Einfügung wird die Route neu bewertet, bevor die nächste
        Störung verplant wird.

        Returns
        -------
        (n_handled, carryover_liste, ausfallkosten_eur)
        """
        matrix = self._get_matrix(time_min)
        carryover: list[DisruptionEvent] = []
        downtime_cost = 0.0
        handled = 0

        # Initiale Kostenbewertung für Sortierung: direkte Einfügungskosten falls möglich,
        # sonst np.inf (Drop-Versuch folgt im Hauptloop).
        queue: list[tuple[float, DisruptionEvent]] = []
        for d in disruptions:
            best = self._find_best_insertion(d, sim_routes, time_min, hour, matrix)
            cost = best[0] if best is not None else np.inf
            queue.append((cost, d))

        # Günstigste Störung zuerst; nach jeder Einfügung neu bewerten
        queue.sort(key=lambda x: x[0])

        for _, d in queue:
            matrix_fresh = self._get_matrix(time_min)
            best = self._find_best_insertion(d, sim_routes, time_min, hour, matrix_fresh)

            if best is not None:
                # Direkte Einfügung möglich
                cost, team_idx, pos, arrival_at_d = best
                self._insert_disruption(d, sim_routes[team_idx], pos, time_min)
                log.actions.append(
                    f"Eingefuegt (direkt): {d.disruption_type} @ Node {d.node_idx} -> "
                    f"Team {sim_routes[team_idx].team_id}, "
                    f"Ankunft {_fmt(arrival_at_d)}, "
                    f"Zusatzkosten {cost:.2f} EUR"
                )
            else:
                # Plan zu voll → Routine-Stops herausnehmen und Störung einplanen
                drop_result = self._find_best_drop_and_insert(
                    d, sim_routes, time_min, hour
                )
                if drop_result is not None:
                    cost, team_idx, drop_global_indices, insert_pos, arrival_at_d = drop_result
                    dropped_nodes = [
                        sim_routes[team_idx].stops[i].node_idx for i in drop_global_indices
                    ]
                    # Stops von höchstem Index zuerst entfernen (Indexstabilität)
                    for dg in sorted(drop_global_indices, reverse=True):
                        self._remove_stop_and_recompute(sim_routes[team_idx], dg)
                    self._insert_disruption(d, sim_routes[team_idx], insert_pos, time_min)
                    n_d = len(dropped_nodes)
                    log.actions.append(
                        f"Eingefuegt ({n_d} Routine-Stop(s) ausgebaut: {dropped_nodes}): "
                        f"{d.disruption_type} @ Node {d.node_idx} -> "
                        f"Team {sim_routes[team_idx].team_id}, "
                        f"Ankunft {_fmt(arrival_at_d)}, "
                        f"Zusatzkosten {cost:.2f} EUR"
                    )
                else:
                    carryover.append(d)
                    log.actions.append(
                        f"Carryover (kein Routine-Stop ausreichend): "
                        f"{d.disruption_type} @ Node {d.node_idx}"
                    )
                    continue  # kein handled-Increment, kein downtime

            # Ausfallkosten für erfolgreich eingeplante Störung
            report_min = float((hour - 8) * 60)
            wait_h = max(0.0, (arrival_at_d - report_min) / 60.0)
            d_cost = wait_h * d.power_kw * self.cost_params.downtime_eur_per_kwh
            downtime_cost += d_cost
            if d_cost > 0:
                log.actions.append(f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)")
            handled += 1

        return handled, carryover, downtime_cost

    def _find_best_insertion(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
        matrix: np.ndarray,
    ) -> Optional[tuple[float, int, int, float]]:
        """
        Findet die kostengünstigste Einfügeposition über alle Teams.

        Returns
        -------
        (min_cost, team_idx_in_sim_routes, pos_in_remaining, arrival_at_d)
        oder None, wenn keine Team-Route die Einfügung vor 17:00 schafft.
        """
        best_cost = np.inf
        best_team_idx: Optional[int] = None
        best_pos: Optional[int] = None
        best_arrival = 0.0

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            current_node = route.current_node_at(time_min)
            current_dep = route.current_departure_at(time_min)

            for pos in range(len(remaining) + 1):
                cost, feasible, arrival = self._insertion_cost(
                    d, remaining, current_node, current_dep, pos, hour, matrix
                )
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_team_idx = ti
                    best_pos = pos
                    best_arrival = arrival

        if best_team_idx is None:
            return None
        return best_cost, best_team_idx, best_pos, best_arrival

    def _insertion_cost(
        self,
        d: DisruptionEvent,
        remaining: list[SimStop],
        current_node: int,
        current_dep: float,
        pos: int,
        hour: int,
        matrix: np.ndarray,
    ) -> tuple[float, bool, float]:
        """
        Berechnet Kosten und Machbarkeit einer Einfügeposition.

        pos = 0: direkt nach aktuellem Teamknoten (vor remaining[0])
        pos = k: zwischen remaining[k-1] und remaining[k]
        pos = len(remaining): nach dem letzten verbleibenden Stop

        Returns
        -------
        (kosten_eur, machbar, ankunft_an_stoerung_min)
        """
        cp = self.cost_params
        d_node = d.node_idx

        # Vorgänger
        if pos == 0:
            prev_node = current_node
            prev_dep = current_dep
        else:
            prev = remaining[pos - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        # Nachfolger
        next_node = remaining[pos].node_idx if pos < len(remaining) else 0  # 0 = Depot

        # Fahrzeiten [min]
        t_prev_d = matrix[prev_node, d_node] / 60.0
        t_d_next = matrix[d_node, next_node] / 60.0
        t_prev_next = matrix[prev_node, next_node] / 60.0

        extra_travel = t_prev_d + t_d_next - t_prev_next
        arrival_at_d = prev_dep + t_prev_d
        total_extra = extra_travel + d.service_min

        # Machbarkeitsprüfung: endet der letzte Teamstop vor 17:00?
        if remaining:
            if pos < len(remaining):
                # Alle Stops ab pos verschieben sich um total_extra
                new_last_dep = remaining[-1].departure_min + total_extra
                last_node = remaining[-1].node_idx
            else:
                # Störung ist der neue letzte Stop
                new_last_dep = arrival_at_d + d.service_min
                last_node = d_node
            end_time = new_last_dep + matrix[last_node, 0] / 60.0
        else:
            # Route war leer
            end_time = arrival_at_d + d.service_min + matrix[d_node, 0] / 60.0

        feasible = end_time <= self.solver.WORKDAY_MINUTES

        # Kosten: Umweg (Fahrt + Service) + Ausfallzeit
        extra_km = (
            _approx_km(self.all_coords[prev_node], self.all_coords[d_node])
            + _approx_km(self.all_coords[d_node], self.all_coords[next_node])
            - _approx_km(self.all_coords[prev_node], self.all_coords[next_node])
        )
        extra_time_h = total_extra / 60.0
        report_min = float((hour - 8) * 60)
        downtime_h = max(0.0, (arrival_at_d - report_min) / 60.0)

        cost = (
            extra_time_h * cp.wage_eur_per_hour
            + max(0.0, extra_km) * cp.fuel_eur_per_km
            + downtime_h * d.power_kw * cp.downtime_eur_per_kwh
        )
        return cost, feasible, arrival_at_d

    def _insert_disruption(
        self,
        d: DisruptionEvent,
        route: SimRoute,
        pos: int,
        time_min: float,
    ) -> None:
        """
        Fügt Störungs-Stop an Position pos (in remaining) in die Route ein
        und aktualisiert alle Folge-Ankunftszeiten.
        """
        remaining = route.remaining_stops_at(time_min)

        # Globaler Index im route.stops-Array
        if remaining:
            first_global = route.stops.index(remaining[0])
        else:
            first_global = len(route.stops)
        global_insert = first_global + pos

        # Ankunftszeit für den neuen Stop
        matrix = self._get_matrix(time_min)
        if pos == 0:
            prev_node = route.current_node_at(time_min)
            prev_dep = route.current_departure_at(time_min)
        else:
            prev = remaining[pos - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        arrival = prev_dep + matrix[prev_node, d.node_idx] / 60.0
        new_stop = SimStop(
            node_idx=d.node_idx,
            task_type=d.disruption_type,
            arrival_min=arrival,
            service_min=d.service_min,
        )
        route.stops.insert(global_insert, new_stop)

        # Alle nachfolgenden Stops neu berechnen (stündliche Matrix beachten)
        for i in range(global_insert + 1, len(route.stops)):
            prev_s = route.stops[i - 1]
            curr_s = route.stops[i]
            mat = self._get_matrix(prev_s.departure_min)
            curr_s.arrival_min = prev_s.departure_min + mat[prev_s.node_idx, curr_s.node_idx] / 60.0

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _load_disruptions(self, df: pd.DataFrame) -> list[DisruptionEvent]:
        """Lädt und konvertiert Stördaten; berechnet Servicezeiten."""
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

    def _plan_to_sim_routes(
        self,
        plan: Optional[DailyPlan],
        all_tasks: list[MaintenanceTask],
    ) -> list[SimRoute]:
        """
        Konvertiert DailyPlan in SimRoute-Objekte mit korrekten Servicezeiten.

        Verwendet die von OR-Tools berechneten Ankunftszeiten (korrekt) und
        die tatsächlichen Servicezeiten aus dem Task-Lookup (statt dem
        Default-Wert aus _extract_solution).
        """
        if plan is None or not plan.routes:
            return [SimRoute(team_id=i) for i in range(self.n_teams)]

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
                ))
            sim_routes.append(route)

        # Sicherstellen, dass für jedes Team eine Route existiert
        existing = {r.team_id for r in sim_routes}
        for i in range(self.n_teams):
            if i not in existing:
                sim_routes.append(SimRoute(team_id=i))

        return sorted(sim_routes, key=lambda r: r.team_id)

    def _find_best_drop_and_insert(
        self,
        d: DisruptionEvent,
        sim_routes: list[SimRoute],
        time_min: float,
        hour: int,
    ) -> Optional[tuple[float, int, list[int], int, float]]:
        """
        Sucht die kostengünstigste Kombination aus 1..N Routine-Drops,
        um Platz für Störung d zu schaffen.

        Strategie: Greedy von hinten – letzte Routine-Stops werden zuerst
        entfernt (niedrigste Priorität). Für jede Drop-Anzahl werden alle
        Einfügepositionen geprüft.

        Returns
        -------
        (kosten, team_idx, [drop_global_indices], insert_pos, arrival_at_d)
        oder None falls kein Drop eine Lösung ermöglicht.
        """
        best_cost = np.inf
        best: Optional[tuple[float, int, list[int], int, float]] = None
        matrix = self._get_matrix(time_min)

        for ti, route in enumerate(sim_routes):
            remaining = route.remaining_stops_at(time_min)
            routine_idx = [i for i, s in enumerate(remaining) if s.task_type == "routine"]

            if not routine_idx:
                continue

            current_node = route.current_node_at(time_min)
            current_dep = route.current_departure_at(time_min)

            # Iterativ vom Ende droppen, bis Einfügung machbar ist
            dropped_in_remaining: list[int] = []

            for n_drop in range(1, len(routine_idx) + 1):
                dropped_in_remaining.append(routine_idx[-n_drop])
                drop_set = set(dropped_in_remaining)

                # Bereinigte Route mit neu berechneten Zeiten
                trimmed: list[SimStop] = []
                prev_n = current_node
                prev_d = current_dep
                for i, s in enumerate(remaining):
                    if i in drop_set:
                        continue
                    mat_h = self._get_matrix(prev_d)
                    new_arr = prev_d + mat_h[prev_n, s.node_idx] / 60.0
                    trimmed.append(SimStop(
                        node_idx=s.node_idx,
                        task_type=s.task_type,
                        arrival_min=new_arr,
                        service_min=s.service_min,
                    ))
                    prev_n = s.node_idx
                    prev_d = new_arr + s.service_min

                # Alle Einfügepositionen prüfen
                found = False
                for pos in range(len(trimmed) + 1):
                    cost, feasible, arrival = self._insertion_cost(
                        d, trimmed, current_node, current_dep, pos, hour, matrix
                    )
                    if feasible and cost < best_cost:
                        drop_globals = [route.stops.index(remaining[i]) for i in dropped_in_remaining]
                        best_cost = cost
                        best = (cost, ti, drop_globals, pos, arrival)
                        found = True

                if found:
                    break  # minimale Drop-Anzahl für dieses Team gefunden

        return best

    def _remove_stop_and_recompute(self, route: SimRoute, global_idx: int) -> None:
        """
        Entfernt den Stop an global_idx aus der Route und berechnet
        alle nachfolgenden Ankunftszeiten neu.
        """
        route.stops.pop(global_idx)
        if global_idx >= len(route.stops):
            return  # war der letzte Stop, nichts nachfolgendes zu korrigieren

        # Vorgänger des entfernten Stops
        if global_idx == 0:
            prev_node = 0    # Depot
            prev_dep = 0.0
        else:
            prev = route.stops[global_idx - 1]
            prev_node = prev.node_idx
            prev_dep = prev.departure_min

        for i in range(global_idx, len(route.stops)):
            curr = route.stops[i]
            mat = self._get_matrix(prev_dep)
            curr.arrival_min = prev_dep + mat[prev_node, curr.node_idx] / 60.0
            prev_node = curr.node_idx
            prev_dep = curr.departure_min

    def _compute_operational_cost(self, sim_routes: list[SimRoute]) -> float:
        """
        Berechnet die operativen Tageskosten aller Teams.

        Wage: (Fahrzeit + Servicezeit) × 40 €/h
        Fuel: Kilometer × 0,30 €/km
        """
        cp = self.cost_params
        total = 0.0

        for route in sim_routes:
            if not route.stops:
                continue

            # Alle Fahrtabschnitte: Depot → Stops → Depot
            legs: list[tuple[int, int, float]] = []  # (from, to, departure_min)
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

            service_min = sum(s.service_min for s in route.stops)
            work_h = (travel_min + service_min) / 60.0
            total += work_h * cp.wage_eur_per_hour + km * cp.fuel_eur_per_km

        return total

    def _get_matrix(self, time_min: float) -> np.ndarray:
        """Gibt die passende Stundenmatrix für einen Zeitstempel zurück."""
        hour = 8 + int(max(0.0, time_min)) // 60
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]
