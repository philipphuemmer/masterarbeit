"""
OR-Tools basierter VRP-Solver für die tägliche Wartungsplanung.

Gemeinsame Basis für alle Modelle (Myopic, CFA, VFA).
Die Modelle können die Zielfunktion über `extra_costs` erweitern.

Zeitrepräsentation: Minuten ab 8:00 (Tagesbeginn = 0, Tagesende = 540).
Fahrzeiten: stündliche Matrizen in Sekunden. Beim Solve wird die Matrix der
aktuellen Stunde verwendet (z.B. Abfahrt um 8:50 → 8-Uhr-Matrix,
Abfahrt um 9:10 → 9-Uhr-Matrix).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceTask:
    """Offene Wartungsaufgabe."""

    node_idx: int
    """Index in der vollen duration_matrix (0 = Depot, 1..N = Ladesäulen)."""

    task_type: str
    """'routine' | 'disruption'"""

    priority: int = 2
    """1 = hoch (Störung), 2 = normal (Routinewartung)."""

    service_time: int = 30
    """Servicezeit an der Station in Minuten."""

    soft_deadline_min: Optional[int] = None
    """Soft-Deadline in Minuten ab 8:00. OR-Tools zahlt deadline_penalty pro
    Minute Überschreitung (SetCumulVarSoftUpperBound). None → kein Limit."""

    deadline_penalty: int = 0
    """Strafkosten in Minuten pro Minute Überschreitung der soft_deadline_min."""

    skip_penalty: Optional[int] = None
    """Falls gesetzt, ist die Aufgabe optional (AddDisjunction). Kosten in
    Minuten für das Überspringen der Station."""

    days_since_maintenance: float = 0.0
    """Tage seit letzter Wartung dieser Station. Wird vom Simulator im
    stochastischen Modus befüllt und von der CFA-Policy genutzt."""


@dataclass
class TeamState:
    """Aktueller Zustand eines Wartungsteams."""

    team_id: int
    current_node: int
    """Aktueller Knoten in der duration_matrix (0 = Depot)."""

    current_time: int
    """Minuten ab 8:00 (0 = Tagesbeginn)."""

    completed_nodes: list[int] = field(default_factory=list)
    """Bereits besuchte Knoten (node_idx) heute."""


@dataclass
class PlannedRoute:
    """Geplante Route eines Wartungsteams."""

    team_id: int
    stops: list[int]
    """node_idx der Stationen in Besuchsreihenfolge (ohne Depot)."""

    arrival_times: list[int]
    """Ankunftszeiten in Minuten ab 8:00."""

    departure_times: list[int]
    """Abfahrtszeiten in Minuten ab 8:00 (Ankunft + Servicezeit)."""


@dataclass
class DailyPlan:
    """Tages- oder Restplan nach Replanning."""

    routes: list[PlannedRoute]
    total_travel_time: int
    """Gesamtfahrzeit aller Teams in Minuten."""

    solver_status: str
    """'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'NO_SOLUTION'"""

    objective_value: int = 0
    """Zielfunktionswert des Solvers."""

    n_dropped: int = 0
    """Anzahl der Routine-Stops, die im Initialplan-Retry gedroppt wurden (0 = kein Retry)."""

    status_before_retry: str = ""
    """Solver-Status des ersten (gescheiterten) Solve-Versuchs, falls Retry nötig war."""



# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

# OR-Tools Routing-Status → lesbare Strings
_STATUS_MAP = {
    0: "NOT_SOLVED",
    1: "OPTIMAL",
    2: "FEASIBLE",   # local optimum, time limit not reached
    3: "INFEASIBLE",
    4: "FEASIBLE",   # time limit reached, best solution returned
    5: "INVALID",
}


class VRPSolver:
    """
    OR-Tools VRP-Solver für die Wartungsplanung von E-Ladesäulen.

    Erstellt den Initialplan zu Tagesbeginn und führt Replanning nach
    Störungen durch. Alle drei Modelle (Myopic, CFA, VFA) nutzen diese
    Klasse als gemeinsame Optimierungsbasis.

    Parameters
    ----------
    traffic_matrices : dict[int, np.ndarray]
        Stündliche Reisezeitmatrizen in Sekunden, indiziert nach Uhrzeit.
        Beispiel: {8: matrix_8uhr, 9: matrix_9uhr, ..., 16: matrix_16uhr}.
        Index 0 in jeder Matrix = Depot.
    config : dict
        Konfigurationsdict aus config.yaml.
    n_teams : int
        Anzahl Wartungsteams (Standard: 2).
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        n_teams: int = 2,
        all_coords: Optional[np.ndarray] = None,
    ) -> None:
        self.traffic_matrices = traffic_matrices
        self.config = config
        self.n_teams = n_teams
        self.all_coords = all_coords
        maint = config["maintenance"]
        self.default_service_time: int = maint["mean_service_time"]
        self._workday_start: int = maint["workday_start_hour"]
        self._lunch_earliest_min: int = maint.get("lunch_earliest_min", 240)
        self.WORKDAY_MINUTES: int = (
            (maint["workday_end_hour"] - maint["workday_start_hour"]) * 60
            - maint.get("lunch_duration_min", 0)
        )
        self._limit_mode: str = maint.get("solver_limit_mode", "time")
        self._backup_time_limit_initial: int = maint["solver_time_limit_initial"]
        self._backup_time_limit_replan: int  = maint["solver_time_limit_replan"]
        if self._limit_mode == "solution":
            self._time_limit_initial: int = maint.get("solver_solution_limit_initial", 10)
            self._time_limit_replan: int  = maint.get("solver_solution_limit_replan", 5)
        else:
            self._time_limit_initial: int = maint["solver_time_limit_initial"]
            self._time_limit_replan: int  = maint["solver_time_limit_replan"]
        planning = config.get("planning", {})
        self._use_team_assignment: bool = bool(
            planning.get("use_team_assignment", True)
        )

    def _get_matrix(self, current_time_minutes: int) -> np.ndarray:
        """Gibt die passende Stundenmatrix für einen Abfahrtszeitpunkt zurück.

        Parameters
        ----------
        current_time_minutes : Minuten ab 8:00 (z.B. 50 → 8:50 → 8-Uhr-Matrix).
        """
        hour = self._workday_start + current_time_minutes // 60
        # Ab Mittagspause (12:00 = 240 min): nächste Stundenmatrix verwenden
        if current_time_minutes >= self._lunch_earliest_min:
            hour += 1
        # Auf verfügbare Stunden begrenzen
        available = sorted(self.traffic_matrices.keys())
        hour = max(available[0], min(hour, available[-1]))
        return self.traffic_matrices[hour]

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        extra_costs: Optional[dict[int, int]] = None,
        time_limit_seconds: Optional[int] = None,
        team_assignment: Optional[dict[int, list[int]]] = None,
        internal_retry: bool = True,
    ) -> DailyPlan:
        """
        Erstellt den Initialplan für den Tag.

        Carryover-Tasks (Vortags-Störungen) werden vorab greedy geroutet und
        den Teams zugewiesen, bevor OR-Tools die Routine-Tasks plant. Das
        garantiert, dass Carryover-Tasks immer zu Tagesbeginn abgearbeitet
        werden — konsistent mit dem DB-Modell (Stein et al., Algorithm 2).

        Parameters
        ----------
        tasks : list[MaintenanceTask]
            Wartungsaufgaben des Tages (Routinewartungen, ggf. Vortags-Störungen).
        extra_costs : dict[int, int] | None
            Zusatzkosten pro Station {node_idx: cost_in_minutes}.
            Wird von CFA/VFA-Modellen genutzt, um die Zielfunktion zu erweitern.
        time_limit_seconds : int | None
            Solver-Zeitlimit in Sekunden. None → Wert aus config.yaml.
        team_assignment : dict[int, list[int]] | None
            Vorab-Zuweisung von Aufgaben zu Teams: {team_id: [node_idx, ...]}.
            Kommt vom DailyZoneSelector und verhindert, dass OR-Tools geografisch
            unpassende Team-Zuordnungen probiert.

        Returns
        -------
        DailyPlan
        """
        limit = time_limit_seconds if time_limit_seconds is not None else self._time_limit_initial
        effective_assignment = team_assignment if self._use_team_assignment else None

        carryover_tasks = [t for t in tasks if t.task_type == "carryover"]
        routine_tasks = [t for t in tasks if t.task_type != "carryover"]

        carry_routes: Optional[dict[int, PlannedRoute]] = None
        initial_states: Optional[dict[int, TeamState]] = None
        carry_travel = 0

        if carryover_tasks:
            depot_states = [TeamState(team_id=i, current_node=0, current_time=0) for i in range(self.n_teams)]
            carry_routes_list, depot_end_states, carry_travel = self._pre_route_mandatory(
                carryover_tasks, depot_states, assignment=effective_assignment
            )
            carry_routes = carry_routes_list
            initial_states = {s.team_id: s for s in depot_end_states}
            # Reduce assignment to routine nodes only for OR-Tools
            if effective_assignment:
                routine_nodes = {t.node_idx for t in routine_tasks}
                effective_assignment = {
                    tid: [n for n in nodes if n in routine_nodes]
                    for tid, nodes in effective_assignment.items()
                }

        tasks_for_solver = routine_tasks if carryover_tasks else tasks

        if not tasks_for_solver:
            routes = list(carry_routes.values()) if carry_routes else [
                PlannedRoute(i, [], [], []) for i in range(self.n_teams)
            ]
            return DailyPlan(routes=routes, total_travel_time=carry_travel, solver_status="OPTIMAL")

        if effective_assignment:
            plan = self._solve_teams_independently(
                tasks_for_solver, extra_costs, limit, effective_assignment,
                initial_states=initial_states, internal_retry=internal_retry,
            )
        else:
            team_states = [
                initial_states[i] if initial_states and i in initial_states
                else TeamState(team_id=i, current_node=0, current_time=0)
                for i in range(self.n_teams)
            ]
            active_states = [s for s in team_states if s.current_time < self.WORKDAY_MINUTES]
            if not active_states:
                return DailyPlan(
                    routes=list(carry_routes.values()) if carry_routes else [PlannedRoute(i, [], [], []) for i in range(self.n_teams)],
                    total_travel_time=carry_travel,
                    solver_status="OPTIMAL",
                )
            plan = self._solve(tasks_for_solver, active_states, extra_costs, limit, None)
            if internal_retry and plan.solver_status not in ("OPTIMAL", "FEASIBLE") and self.all_coords is not None:
                status_before_retry = plan.solver_status
                depot_coord = self.all_coords[0]
                routine_sorted = sorted(
                    tasks_for_solver,
                    key=lambda t: float(np.linalg.norm(self.all_coords[t.node_idx] - depot_coord)),
                    reverse=True,
                )
                for drop_n in range(1, len(routine_sorted) + 1):
                    retry_tasks = routine_sorted[drop_n:]
                    if not retry_tasks:
                        break
                    plan = self._solve(retry_tasks, team_states, extra_costs, limit, None)
                    if plan.solver_status in ("OPTIMAL", "FEASIBLE"):
                        plan.n_dropped = drop_n
                        plan.status_before_retry = status_before_retry
                        break

        if not carry_routes:
            return plan

        merged_routes = []
        for route in plan.routes:
            tid = route.team_id
            carry = carry_routes.get(tid, PlannedRoute(tid, [], [], []))
            merged_routes.append(PlannedRoute(
                team_id=tid,
                stops=carry.stops + route.stops,
                arrival_times=carry.arrival_times + route.arrival_times,
                departure_times=carry.departure_times + route.departure_times,
            ))
        return DailyPlan(
            routes=merged_routes,
            total_travel_time=carry_travel + plan.total_travel_time,
            solver_status=plan.solver_status,
            n_dropped=plan.n_dropped,
            status_before_retry=plan.status_before_retry,
        )

    def replan(
        self,
        remaining_tasks: list[MaintenanceTask],
        team_states: list[TeamState],
        extra_costs: Optional[dict[int, int]] = None,
        time_limit_seconds: Optional[int] = None,
    ) -> DailyPlan:
        """
        Replanning nach einer Störung oder Planänderung.

        Parameters
        ----------
        remaining_tasks : list[MaintenanceTask]
            Noch offene Aufgaben inklusive neuer Störungen.
            Bereits abgeschlossene Aufgaben werden nicht übergeben.
        team_states : list[TeamState]
            Aktueller Zustand jedes Teams (Position, Zeitstempel).
        extra_costs : dict[int, int] | None
            Modellspezifische Zusatzkosten (CFA/VFA).
        time_limit_seconds : int
            Kürzeres Zeitlimit für schnelles intra-day Replanning.

        Returns
        -------
        DailyPlan
        """
        limit = time_limit_seconds if time_limit_seconds is not None else self._time_limit_replan

        carryover = [t for t in remaining_tasks if t.task_type == "carryover"]
        rest = [t for t in remaining_tasks if t.task_type != "carryover"]

        if not carryover:
            return self._solve(remaining_tasks, team_states, extra_costs, limit)

        carry_routes, updated_states, carry_travel = self._pre_route_mandatory(carryover, team_states)

        if not rest:
            return DailyPlan(
                routes=list(carry_routes.values()),
                total_travel_time=carry_travel,
                solver_status="OPTIMAL",
            )

        # Teams ohne verbleibende Kapazität aus dem OR-Tools-Solve heraushalten
        active_states = [s for s in updated_states if s.current_time < self.WORKDAY_MINUTES]
        exhausted_tids = {s.team_id for s in updated_states if s.current_time >= self.WORKDAY_MINUTES}

        if not active_states:
            return DailyPlan(
                routes=list(carry_routes.values()),
                total_travel_time=carry_travel,
                solver_status="OPTIMAL",
            )

        rest_plan = self._solve(rest, active_states, extra_costs, limit)

        merged_routes = []
        active_route_by_tid = {r.team_id: r for r in rest_plan.routes}
        for s in updated_states:
            tid = s.team_id
            carry = carry_routes.get(tid, PlannedRoute(tid, [], [], []))
            routine_route = active_route_by_tid.get(tid, PlannedRoute(tid, [], [], []))
            merged_routes.append(PlannedRoute(
                team_id=tid,
                stops=carry.stops + routine_route.stops,
                arrival_times=carry.arrival_times + routine_route.arrival_times,
                departure_times=carry.departure_times + routine_route.departure_times,
            ))
        return DailyPlan(
            routes=merged_routes,
            total_travel_time=carry_travel + rest_plan.total_travel_time,
            solver_status=rest_plan.solver_status,
        )

    # ------------------------------------------------------------------
    # Kernimplementierung
    # ------------------------------------------------------------------

    def _pre_route_mandatory(
        self,
        mandatory_tasks: list[MaintenanceTask],
        team_states: list[TeamState],
        assignment: Optional[dict[int, list[int]]] = None,
    ) -> tuple[dict[int, "PlannedRoute"], list["TeamState"], int]:
        """
        Routet nicht-Routine-Tasks (Carryover/Störungen) greedy vor OR-Tools.

        Verwendet die aktuellen Teampositionen als Startpunkte — funktioniert
        sowohl beim Tagesstart (Depot, t=0) als auch beim intra-day Replan
        (beliebige Position und Zeit). Konsistent mit DB-Modell (Stein et al.,
        Algorithm 2).

        Gibt zurück: (Routen pro Team, aktualisierte TeamStates, Gesamtfahrzeit).
        """
        state_by_id = {s.team_id: s for s in team_states}
        matrix = self._get_matrix(min(s.current_time for s in team_states))
        by_team: dict[int, list[MaintenanceTask]] = {s.team_id: [] for s in team_states}

        if assignment:
            node_to_task = {t.node_idx: t for t in mandatory_tasks}
            for tid, nodes in assignment.items():
                for node in nodes:
                    if node in node_to_task:
                        by_team[tid].append(node_to_task[node])
        else:
            end_times = {s.team_id: s.current_time for s in team_states}
            cur_nodes = {s.team_id: s.current_node for s in team_states}
            for task in mandatory_tasks:
                best_tid = min(
                    by_team.keys(),
                    key=lambda tid: end_times[tid] + int(round(
                        matrix[cur_nodes[tid], task.node_idx] / 60
                    )),
                )
                prev = by_team[best_tid][-1].node_idx if by_team[best_tid] else cur_nodes[best_tid]
                end_times[best_tid] += int(round(matrix[prev, task.node_idx] / 60)) + task.service_time
                by_team[best_tid].append(task)

        routes: dict[int, PlannedRoute] = {}
        new_states: list[TeamState] = []
        total_travel = 0

        for s in team_states:
            tid = s.team_id
            stops, arrivals, departures = [], [], []
            cur_node, cur_time = s.current_node, s.current_time
            for task in by_team[tid]:
                travel = int(round(matrix[cur_node, task.node_idx] / 60))
                total_travel += travel
                arr = cur_time + travel
                dep = arr + task.service_time
                stops.append(task.node_idx)
                arrivals.append(arr)
                departures.append(dep)
                cur_node, cur_time = task.node_idx, dep
            routes[tid] = PlannedRoute(team_id=tid, stops=stops, arrival_times=arrivals, departure_times=departures)
            new_states.append(TeamState(team_id=tid, current_node=cur_node, current_time=cur_time))

        return routes, new_states, total_travel

    def _solve_teams_independently(
        self,
        tasks: list[MaintenanceTask],
        extra_costs: Optional[dict[int, int]],
        time_limit_seconds: int,
        team_assignment: dict[int, list[int]],
        internal_retry: bool = True,
        initial_states: Optional[dict[int, "TeamState"]] = None,
    ) -> DailyPlan:
        """
        Löst jedes Team in einem eigenen 1-Fahrzeug-OR-Tools-Modell.

        Bei use_team_assignment=True sind die Teams mathematisch unabhängig.
        Ein 1-Fahrzeug-Modell pro Team ist dann ~4x schneller als ein
        gemeinsames 2-Fahrzeug-Modell und liefert dieselbe Lösung.
        Infeasibility eines Teams führt zum Drop depot-ferner Routine-Tasks
        ausschließlich dieses Teams, ohne das andere Team zu beeinflussen.
        """
        depot_coord = self.all_coords[0] if self.all_coords is not None else None
        routine_by_node = {t.node_idx: t for t in tasks if t.task_type == "routine"}
        mandatory = [t for t in tasks if t.task_type != "routine"]

        # Routine-Tasks nach Team aufteilen (gemäß team_assignment)
        per_team: dict[int, list[MaintenanceTask]] = {
            tid: [routine_by_node[n] for n in nodes if n in routine_by_node]
            for tid, nodes in team_assignment.items()
        }

        # Lookup: node_idx → team_id für explizit zugewiesene Knoten
        # (team_assignment enthält alle Tasks: Routine + Carryover-Störungen)
        assignment_lookup: dict[int, int] = {
            n: tid for tid, nodes in team_assignment.items() for n in nodes
        }

        # Mandatory-Tasks (Carryover-Störungen) zuweisen
        unassigned_mandatory: list[MaintenanceTask] = []
        for task in mandatory:
            if task.node_idx in assignment_lookup:
                # Selector hat diese Störung bereits explizit einem Team zugewiesen
                per_team[assignment_lookup[task.node_idx]].append(task)
            else:
                unassigned_mandatory.append(task)

        # Für unbekannte Mandatory-Tasks: geografisch nächstes Team
        if unassigned_mandatory and self.all_coords is not None:
            routine_nodes_per_team = {
                tid: [n for n in nodes if n in routine_by_node]
                for tid, nodes in team_assignment.items()
            }
            for task in unassigned_mandatory:
                task_coord = self.all_coords[task.node_idx]
                best_tid = min(
                    team_assignment.keys(),
                    key=lambda tid: (
                        float(np.linalg.norm(
                            np.mean([self.all_coords[n] for n in routine_nodes_per_team[tid]], axis=0)
                            - task_coord
                        ))
                        if routine_nodes_per_team[tid] else float("inf")
                    ),
                )
                per_team[best_tid].append(task)
        elif unassigned_mandatory:
            tids = sorted(team_assignment.keys())
            for i, task in enumerate(unassigned_mandatory):
                per_team[tids[i % len(tids)]].append(task)

        all_routes: list[PlannedRoute] = []
        total_travel = 0
        total_dropped = 0
        status_before_retry = ""
        overall_status = "OPTIMAL"

        for tid in sorted(team_assignment.keys()):
            team_tasks = per_team[tid]
            # TeamState mit team_id=tid → _extract_solution setzt PlannedRoute.team_id korrekt
            team_state = [
                initial_states[tid] if initial_states and tid in initial_states
                else TeamState(team_id=tid, current_node=0, current_time=0)
            ]

            # Team hat nach Carryover-Vorroutung keine Kapazität mehr
            if team_state[0].current_time >= self.WORKDAY_MINUTES:
                all_routes.append(PlannedRoute(tid, [], [], []))
                continue

            plan = self._solve(team_tasks, team_state, extra_costs, time_limit_seconds, None)

            if internal_retry and plan.solver_status not in ("OPTIMAL", "FEASIBLE") and self.all_coords is not None:
                if not status_before_retry:
                    status_before_retry = plan.solver_status
                team_mandatory = [t for t in team_tasks if t.task_type != "routine"]
                team_routine_sorted = sorted(
                    [t for t in team_tasks if t.task_type == "routine"],
                    key=lambda t: float(np.linalg.norm(self.all_coords[t.node_idx] - depot_coord)),
                    reverse=True,
                )
                found = False
                for drop_n in range(1, len(team_routine_sorted) + 1):
                    retry_tasks = team_mandatory + team_routine_sorted[drop_n:]
                    # _solve([]) gibt OPTIMAL zurück → Schleife endet spätestens hier
                    plan = self._solve(retry_tasks, team_state, extra_costs, time_limit_seconds, None)
                    if plan.solver_status in ("OPTIMAL", "FEASIBLE"):
                        total_dropped += drop_n
                        found = True
                        break
                if not found:
                    plan = DailyPlan(
                        routes=[PlannedRoute(tid, [], [], [])],
                        total_travel_time=0,
                        solver_status="NO_SOLUTION",
                    )

            # Gesamtstatus: schlechtesten Einzelstatus weiterleiten
            if plan.solver_status == "NO_SOLUTION" or overall_status == "NO_SOLUTION":
                overall_status = "NO_SOLUTION"
            elif plan.solver_status in ("INFEASIBLE", "INVALID"):
                overall_status = plan.solver_status
            elif plan.solver_status == "FEASIBLE" and overall_status == "OPTIMAL":
                overall_status = "FEASIBLE"

            all_routes.extend(plan.routes)
            if plan.total_travel_time > 0:
                total_travel += plan.total_travel_time

        return DailyPlan(
            routes=all_routes,
            total_travel_time=total_travel,
            solver_status=overall_status,
            n_dropped=total_dropped,
            status_before_retry=status_before_retry,
        )

    def _solve_trivial(
        self,
        tasks: list[MaintenanceTask],
        team_states: list[TeamState],
    ) -> Optional[DailyPlan]:
        """Konstruiert für genau 1 Task und 1 Fahrzeug die triviale Route direkt.

        Umgeht OR-Tools für den degenerierten Einzelknoten-Fall, bei dem die
        SAVINGS-Strategie keinen gültigen Status-Code zurückliefert.
        Gibt None zurück wenn die Bedingungen nicht erfüllt sind.
        """
        if len(tasks) != 1 or len(team_states) != 1:
            return None
        task = tasks[0]
        state = team_states[0]
        matrix = self._get_matrix(state.current_time)
        travel_to = int(np.round(matrix[state.current_node, task.node_idx] / 60.0))
        travel_back = int(np.round(matrix[task.node_idx, 0] / 60.0))
        arrival = state.current_time + travel_to
        departure = arrival + task.service_time
        if departure + travel_back > self.WORKDAY_MINUTES:
            return None  # genuiner Infeasibility-Fall → OR-Tools entscheidet
        return DailyPlan(
            routes=[PlannedRoute(
                team_id=state.team_id,
                stops=[task.node_idx],
                arrival_times=[arrival],
                departure_times=[departure],
            )],
            total_travel_time=travel_to + travel_back,
            solver_status="OPTIMAL",
        )

    def _solve(
        self,
        tasks: list[MaintenanceTask],
        team_states: list[TeamState],
        extra_costs: Optional[dict[int, int]],
        time_limit_seconds: int,
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """Baut das OR-Tools Modell auf und löst es."""
        if not tasks:
            return DailyPlan(
                routes=[PlannedRoute(s.team_id, [], [], []) for s in team_states],
                total_travel_time=0,
                solver_status="OPTIMAL",
            )

        trivial = self._solve_trivial(tasks, team_states)
        if trivial is not None:
            return trivial

        # --- Node-Mapping ---
        # Knoten-Reihenfolge: Depot (0) zuerst, dann weitere eindeutige Knoten.
        # Teampositionen und Aufgabenknoten werden in einer gemeinsamen Liste
        # gesammelt, damit die reduzierte Reisezeitmatrix korrekt aufgebaut wird.
        depot_node = 0
        team_start_nodes = [s.current_node for s in team_states]
        task_nodes = [t.node_idx for t in tasks]

        all_nodes: list[int] = [depot_node]
        seen: set[int] = {depot_node}
        for n in team_start_nodes + task_nodes:
            if n not in seen:
                all_nodes.append(n)
                seen.add(n)

        global_to_routing: dict[int, int] = {gn: ri for ri, gn in enumerate(all_nodes)}
        n_routing = len(all_nodes)

        # Stundenmatrix anhand des frühesten Teamzeitpunkts auswählen
        earliest_time = min(s.current_time for s in team_states)
        full_matrix = self._get_matrix(earliest_time)

        # Reisezeitmatrix: Teilmatrix, umgerechnet in Minuten
        time_matrix: np.ndarray = np.round(
            full_matrix[np.ix_(all_nodes, all_nodes)] / 60.0
        ).astype(int)

        # Servicezeit pro Routing-Knoten:
        # - Depot → 0 (Startpunkt, keine Wartung)
        # - Aktuelle Teampositionen → 0 (Teams starten von dort, kein erneuter Service)
        # - Aufgabenknoten → task.service_time
        no_service_r_nodes: set[int] = {global_to_routing[depot_node]}
        for s in team_states:
            no_service_r_nodes.add(global_to_routing[s.current_node])

        task_service_map: dict[int, int] = {
            global_to_routing[t.node_idx]: t.service_time for t in tasks
        }

        def _service_time(r_node: int) -> int:
            if r_node in no_service_r_nodes:
                return 0
            return task_service_map.get(r_node, self.default_service_time)

        # --- OR-Tools Routing Modell ---
        n_vehicles = len(team_states)
        starts = [global_to_routing[s.current_node] for s in team_states]
        ends = [global_to_routing[depot_node]] * n_vehicles

        manager = pywrapcp.RoutingIndexManager(n_routing, n_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Transit-Callback: Fahrzeit (Minuten) + Servicezeit am Quellknoten
        def transit_callback(from_idx: int, to_idx: int) -> int:
            fn = manager.IndexToNode(from_idx)
            tn = manager.IndexToNode(to_idx)
            return int(time_matrix[fn, tn]) + _service_time(fn)

        transit_idx = routing.RegisterTransitCallback(transit_callback)

        # Zielfunktion: Fahrzeit minimieren (ggf. mit Zusatzkosten)
        if extra_costs:
            def combined_callback(from_idx: int, to_idx: int) -> int:
                fn = manager.IndexToNode(from_idx)
                tn = manager.IndexToNode(to_idx)
                travel_and_service = int(time_matrix[fn, tn]) + _service_time(fn)
                arriving_global = all_nodes[tn]
                return travel_and_service + extra_costs.get(arriving_global, 0)

            cost_idx = routing.RegisterTransitCallback(combined_callback)
        else:
            cost_idx = transit_idx

        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        # Zeit-Dimension: verfolgt die kumulative Zeit pro Fahrzeug
        routing.AddDimension(
            transit_idx,
            slack_max=self.WORKDAY_MINUTES,  # max. Wartezeit an einem Knoten
            capacity=self.WORKDAY_MINUTES,   # max. Gesamtzeit pro Fahrzeug
            fix_start_cumul_to_zero=False,
            name="Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Aktuellen Zeitstempel der Teams als festen Startwert setzen
        for v, state in enumerate(team_states):
            start_idx = routing.Start(v)
            t = max(0, min(state.current_time, self.WORKDAY_MINUTES))
            time_dim.CumulVar(start_idx).SetRange(t, t)


        # Soft-Deadlines: SetCumulVarSoftUpperBound für zeitkritische Knoten
        for task in tasks:
            if task.soft_deadline_min is not None and task.deadline_penalty > 0:
                r_node = global_to_routing[task.node_idx]
                routing_idx = manager.NodeToIndex(r_node)
                deadline = max(0, min(task.soft_deadline_min, self.WORKDAY_MINUTES))
                time_dim.SetCumulVarSoftUpperBound(routing_idx, deadline, task.deadline_penalty)

        # Optionale Aufgaben: AddDisjunction erlaubt OR-Tools das Überspringen
        for task in tasks:
            if task.skip_penalty is not None:
                r_node = global_to_routing[task.node_idx]
                routing_idx = manager.NodeToIndex(r_node)
                routing.AddDisjunction([routing_idx], task.skip_penalty)

        # Team-Zuweisung: Knoten dürfen nur vom zugewiesenen Fahrzeug besucht werden
        if team_assignment:
            for vehicle_id, node_list in team_assignment.items():
                for node_idx in node_list:
                    if node_idx not in global_to_routing:
                        continue
                    r_node = global_to_routing[node_idx]
                    routing_idx = manager.NodeToIndex(r_node)
                    # AllowedVehicles beschränkt den Knoten auf genau ein Fahrzeug
                    routing.VehicleVar(routing_idx).SetValues([vehicle_id])

        # Rückkehr zum Depot vor Tagesende
        for v in range(n_vehicles):
            time_dim.CumulVar(routing.End(v)).SetRange(0, self.WORKDAY_MINUTES)

        # Optionaler Makespan-Term für Lastverteilung (konfigurierbar)
        span_coeff = self.config["maintenance"].get("global_span_cost_coefficient", 0)
        if span_coeff > 0:
            time_dim.SetGlobalSpanCostCoefficient(span_coeff)

        # --- Solver-Parameter ---
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.sat_parameters.random_seed = 42
        search_params.sat_parameters.num_search_workers = 4
        if self._limit_mode == "solution":
            search_params.solution_limit = time_limit_seconds  # Wert = solution_limit
            backup = (self._backup_time_limit_replan
                      if time_limit_seconds == self._time_limit_replan
                      else self._backup_time_limit_initial)
            search_params.time_limit.seconds = backup
        else:
            search_params.time_limit.seconds = time_limit_seconds

        _sol_count = [0]
        if getattr(self, "_count_solutions", False):
            routing.AddAtSolutionCallback(lambda: _sol_count.__setitem__(0, _sol_count[0] + 1))

        solution = routing.SolveWithParameters(search_params)
        status = _STATUS_MAP.get(routing.status(), "UNKNOWN")

        if getattr(self, "_count_solutions", False):
            self._solution_counts.append(_sol_count[0])

        return self._extract_solution(solution, routing, manager, all_nodes, team_states, status)

    def _extract_solution(
        self,
        solution,
        routing,
        manager,
        all_nodes: list[int],
        team_states: list[TeamState],
        status: str,
    ) -> DailyPlan:
        """Liest die OR-Tools Lösung aus und erzeugt den DailyPlan."""
        if solution is None:
            return DailyPlan(
                routes=[PlannedRoute(s.team_id, [], [], []) for s in team_states],
                total_travel_time=-1,
                solver_status="NO_SOLUTION",
            )

        time_dim = routing.GetDimensionOrDie("Time")
        routes: list[PlannedRoute] = []
        total_travel = 0

        for v, state in enumerate(team_states):
            stops: list[int] = []
            arrivals: list[int] = []
            departures: list[int] = []

            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                next_idx = solution.Value(routing.NextVar(idx))
                r_node = manager.IndexToNode(idx)
                global_node = all_nodes[r_node]

                # Startknoten (Depot / aktuelle Position) wird nicht als Stop aufgeführt
                if not routing.IsStart(idx):
                    arrival = solution.Value(time_dim.CumulVar(idx))
                    # Service-Zeit für Abfahrtszeitpunkt
                    r_node_next = manager.IndexToNode(next_idx)
                    service = max(
                        0,
                        int(time_dim.CumulVar(idx).Min()) - arrival
                        if routing.IsEnd(next_idx) else 0,
                    )
                    # Einfache Schätzung: departure = arrival + service_time aus der Task
                    # (wird in zukünftigen Iterationen verfeinert)
                    service_at_stop = (
                        self.default_service_time if global_node != 0 else 0
                    )
                    stops.append(global_node)
                    arrivals.append(arrival)
                    departures.append(arrival + service_at_stop)

                # Fahrzeit für Gesamtstatistik (Matrix der aktuellen Ankunftszeit)
                if not routing.IsEnd(next_idx):
                    fn = manager.IndexToNode(idx)
                    tn = manager.IndexToNode(next_idx)
                    departure_time = solution.Value(time_dim.CumulVar(idx))
                    mat = self._get_matrix(departure_time)
                    travel = int(np.round(mat[all_nodes[fn], all_nodes[tn]] / 60.0))
                    total_travel += travel

                idx = next_idx

            routes.append(PlannedRoute(
                team_id=state.team_id,
                stops=stops,
                arrival_times=arrivals,
                departure_times=departures,
            ))

        return DailyPlan(
            routes=routes,
            total_travel_time=total_travel,
            solver_status=status,
            objective_value=solution.ObjectiveValue(),
        )
