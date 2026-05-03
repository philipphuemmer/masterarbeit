"""
Greedy Routing ohne OR-Tools — gemeinsame Basis für CFA und MyopicPlus.

Aktiv wenn solver.use_or_tools: false in config.yaml.

Initialplan
-----------
greedy_initial_plan() baut die Route Schritt für Schritt auf:
  - Carryover-Tasks: Nearest-Neighbor (mandatory, immer zuerst)
  - Routine-Tasks:   route_score_fn(node_idx, dsm, current_node) → höher = als nächstes

    CFA:        route_score_fn = C̃(k) / dist(cur, k)
    MyopicPlus: route_score_fn = 1    / dist(cur, k)   (Nearest-Neighbor)

Störungs-Replan
---------------
handle_disruptions_greedy() fügt Störungen per Cheapest-Insertion ein:
  cost = Δfahrzeit_h × wage + Δkm × fuel + Wartezeit_h × power × downtime

  Falls keine feasible Position:
    Droppe Routine-Stop mit niedrigstem drop_score_fn(node_idx, dsm, remaining_hours)
    (niedrig = weniger wichtig = zuerst opfern)

    Myopic:     drop_score_fn = dist(cur, k)          (depotfernste → niedrigster Score)
    CFA:        drop_score_fn = C̃(k)
    MyopicPlus: drop_score_fn = power × remaining_hours × p_failure
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from src.planning.clustering import _approx_km
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, PlannedRoute

if TYPE_CHECKING:
    from src.models.cost_params import CostParams
    from src.models.simulator import DisruptionEvent, HourLog, SimRoute, SimStop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _get_matrix(
    traffic_matrices: dict[int, np.ndarray],
    time_min: float,
    workday_start_hour: int,
) -> np.ndarray:
    """Passende Stundenmatrix für einen Zeitstempel (Minuten ab 8:00)."""
    hour = workday_start_hour + int(max(0.0, time_min)) // 60
    available = sorted(traffic_matrices.keys())
    return traffic_matrices[max(available[0], min(hour, available[-1]))]


# ---------------------------------------------------------------------------
# Greedy Initialplan
# ---------------------------------------------------------------------------

def _arrive(departure: int, travel_min: int, lunch_start: int, lunch_end: int) -> int:
    """
    Effektive Ankunftszeit unter Berücksichtigung der Mittagspause.

    Liegt die Abfahrt während der Pause → Abfahrt erst nach Pause.
    Überquert die Fahrt die Pausengrenze → komplette Pausendauer addieren.

    Beispiel: Abfahrt 11:55 (235 min), Fahrt 10 min, Pause 12:00–13:00:
      Fahrt läuft 5 min (bis 12:00), Pause 60 min, restliche 5 min → Ankunft 13:05 (305 min).
    """
    lunch_dur = lunch_end - lunch_start
    if lunch_dur <= 0:
        return departure + travel_min
    # Abfahrt mitten in der Pause → erst nach Pause losfahren
    eff_dep = max(departure, lunch_end) if lunch_start <= departure < lunch_end else departure
    arrival = eff_dep + travel_min
    # Fahrt kreuzt den Beginn der Pause (Abfahrt vor Pause, Ankunft nach Pausenbeginn)
    if eff_dep < lunch_start < arrival:
        arrival += lunch_dur
    return arrival


def greedy_initial_plan(
    tasks: list[MaintenanceTask],
    team_assignment: Optional[dict[int, list[int]]],
    all_coords: np.ndarray,
    traffic_matrices: dict[int, np.ndarray],
    workday_start_hour: int,
    workday_minutes: int,
    lunch_earliest_min: int,
    lunch_duration_min: int,
    n_teams: int,
    route_score_fn: Callable[[int, float, int], float],
) -> DailyPlan:
    """
    Baut den Tagesplan greedy auf — kein OR-Tools.

    Berücksichtigt Mittagspause (Fahrt durch Pause verlängert Ankunftszeit)
    und prüft vor jedem Stop ob Depot-Rückkehr bis Arbeitstagesende möglich ist.

    Parameters
    ----------
    route_score_fn
        (node_idx, days_since_maintenance, current_node_idx) → float.
        Höherer Score → Station wird als nächstes gewählt.
        CFA:        C̃(k) / dist(cur, k)
        MyopicPlus: 1    / dist(cur, k)
    """
    lunch_end = lunch_earliest_min + lunch_duration_min
    node_to_task = {t.node_idx: t for t in tasks}

    # Tasks auf Teams verteilen
    if team_assignment:
        per_team: dict[int, list[MaintenanceTask]] = {
            tid: [node_to_task[n] for n in nodes if n in node_to_task]
            for tid, nodes in team_assignment.items()
        }
    else:
        # Ohne Zuweisung: nach Score vom Depot abwechselnd verteilen
        per_team = {i: [] for i in range(n_teams)}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: route_score_fn(t.node_idx, t.days_since_maintenance, 0),
            reverse=True,
        )
        for idx, t in enumerate(sorted_tasks):
            per_team[idx % n_teams].append(t)

    routes: list[PlannedRoute] = []
    total_travel = 0

    for tid in sorted(per_team.keys()):
        team_tasks = per_team[tid]
        carryover = [t for t in team_tasks if t.task_type == "carryover"]
        routine   = [t for t in team_tasks if t.task_type != "carryover"]

        stops: list[int]       = []
        arrivals: list[int]    = []
        departures: list[int]  = []
        current_node = 0   # Depot
        current_time = 0   # Minuten ab 8:00

        # 1. Carryover: Nearest-Neighbor (mandatory, Mittagspause berücksichtigt)
        remaining = list(carryover)
        while remaining:
            matrix = _get_matrix(traffic_matrices, current_time, workday_start_hour)
            best = int(np.argmin([matrix[current_node, t.node_idx] for t in remaining]))
            task = remaining.pop(best)
            travel = int(round(matrix[current_node, task.node_idx] / 60.0))
            total_travel += travel
            arrival = _arrive(current_time, travel, lunch_earliest_min, lunch_end)
            departure = arrival + task.service_time
            stops.append(task.node_idx)
            arrivals.append(arrival)
            departures.append(departure)
            current_node = task.node_idx
            current_time = departure

        # 2. Routine: greedy nach route_score_fn
        #    Mittagspause beachten + Depot-Rückkehr vor Arbeitstagesende prüfen
        remaining = list(routine)
        while remaining:
            matrix = _get_matrix(traffic_matrices, current_time, workday_start_hour)
            cur = current_node

            # Stationen absteigend nach Score sortieren, erste feasible nehmen
            order = sorted(
                range(len(remaining)),
                key=lambda i: route_score_fn(
                    remaining[i].node_idx, remaining[i].days_since_maintenance, cur
                ),
                reverse=True,
            )

            placed = False
            for idx in order:
                task = remaining[idx]
                travel = int(round(matrix[cur, task.node_idx] / 60.0))
                arrival = _arrive(current_time, travel, lunch_earliest_min, lunch_end)
                departure = arrival + task.service_time
                return_travel = int(round(matrix[task.node_idx, 0] / 60.0))
                if departure + return_travel <= workday_minutes:
                    remaining.pop(idx)
                    total_travel += travel
                    stops.append(task.node_idx)
                    arrivals.append(arrival)
                    departures.append(departure)
                    current_node = task.node_idx
                    current_time = departure
                    placed = True
                    break

            if not placed:
                break  # Keine Station passt mehr ins Zeitfenster

        routes.append(PlannedRoute(
            team_id=tid,
            stops=stops,
            arrival_times=arrivals,
            departure_times=departures,
        ))

    return DailyPlan(routes=routes, total_travel_time=total_travel, solver_status="OPTIMAL")


# ---------------------------------------------------------------------------
# Cheapest-Insertion Störungs-Replan
# ---------------------------------------------------------------------------

def handle_disruptions_greedy(
    disruptions: list[DisruptionEvent],
    sim_routes: list[SimRoute],
    time_min: float,
    hour: int,
    all_coords: np.ndarray,
    traffic_matrices: dict[int, np.ndarray],
    workday_start_hour: int,
    workday_minutes: int,
    cost_params: CostParams,
    log: HourLog,
    drop_score_fn: Callable[[int, float, float, int], float],
    travel_time_only: bool = False,
) -> tuple[int, list[DisruptionEvent], float]:
    """
    Cheapest-Insertion Replan ohne OR-Tools.

    Für jede Störung:
      1. Finde günstigste Einfügeposition über alle Teams.
      2. Falls keine feasible Position: droppe Routine-Stops nach aufsteigendem
         drop_score_fn(node_idx, dsm, remaining_hours, current_node) bis Platz entsteht.
      3. Falls immer noch nicht möglich: Carryover.

    Parameters
    ----------
    drop_score_fn
        (node_idx, days_since_maintenance, remaining_hours, current_node) → float.
        Niedrigerer Score → zuerst droppen.
        Myopic:     dist(cur, k)
        CFA:        C̃(k)
        MyopicPlus: power × remaining_hours × p_failure
    """
    matrix = _get_matrix(traffic_matrices, time_min, workday_start_hour)
    carryover: list[DisruptionEvent] = []
    downtime_cost = 0.0
    handled = 0

    # Störungen nach Insertionskosten sortieren (günstigste zuerst)
    queue: list[tuple[float, DisruptionEvent]] = []
    for d in disruptions:
        result = _find_best_insertion(
            d, sim_routes, time_min, hour, matrix, all_coords, workday_minutes, cost_params,
            travel_time_only,
        )
        queue.append((result[0] if result else np.inf, d))
    queue.sort(key=lambda x: x[0])

    for _, d in queue:
        matrix_now = _get_matrix(traffic_matrices, time_min, workday_start_hour)
        result = _find_best_insertion(
            d, sim_routes, time_min, hour, matrix_now, all_coords, workday_minutes, cost_params,
            travel_time_only,
        )

        arrival_at_d: float = time_min

        if result is not None:
            _, team_idx, pos, arrival_at_d = result
            _insert_stop(d, sim_routes[team_idx], pos, time_min, traffic_matrices, workday_start_hour)
            log.notes.append(
                f"Greedy-Replan: {d.disruption_type} @ {d.node_idx} → "
                f"Team {sim_routes[team_idx].team_id}, Ankunft {arrival_at_d:.0f} min"
            )
        else:
            drop_result = _find_best_drop_and_insert(
                d, sim_routes, time_min, hour, all_coords, traffic_matrices,
                workday_start_hour, workday_minutes, cost_params, drop_score_fn,
                travel_time_only,
            )
            if drop_result is not None:
                _, team_idx, drop_globals, pos, arrival_at_d = drop_result
                dropped = [sim_routes[team_idx].stops[i].node_idx for i in drop_globals]
                for gi in sorted(drop_globals, reverse=True):
                    _remove_stop_and_recompute(sim_routes[team_idx], gi, traffic_matrices, workday_start_hour)
                _insert_stop(d, sim_routes[team_idx], pos, time_min, traffic_matrices, workday_start_hour)
                log.notes.append(
                    f"Greedy-Replan (Drop {dropped}): {d.disruption_type} @ {d.node_idx} → "
                    f"Team {sim_routes[team_idx].team_id}, Ankunft {arrival_at_d:.0f} min"
                )
            else:
                carryover.append(d)
                log.notes.append(
                    f"Greedy-Replan Carryover: {d.disruption_type} @ {d.node_idx} "
                    f"(kein feasibler Platz)"
                )
                continue

        report_min = float((hour - 8) * 60)
        wait_h = max(0.0, (arrival_at_d - report_min) / 60.0)
        d_cost = wait_h * d.power_kw * cost_params.downtime_eur_per_kwh
        downtime_cost += d_cost
        if d_cost > 0:
            log.notes.append(f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)")
        handled += 1

    return handled, carryover, downtime_cost


# ---------------------------------------------------------------------------
# Interne Hilfsmethoden für Cheapest-Insertion
# ---------------------------------------------------------------------------

def _insertion_cost(
    d: DisruptionEvent,
    remaining: list[SimStop],
    current_node: int,
    current_dep: float,
    pos: int,
    hour: int,
    matrix: np.ndarray,
    all_coords: np.ndarray,
    workday_minutes: int,
    cost_params: CostParams,
    travel_time_only: bool = False,
) -> tuple[float, bool, float]:
    """Berechnet Kosten und Machbarkeit einer Einfügeposition.

    travel_time_only=True: cost = reine Extra-Fahrzeit (Myopic).
    travel_time_only=False: cost = ökonomische Gesamtkosten (MyopicPlus/CFA).
    Feasibility-Check ist in beiden Modi identisch.
    """
    cp = cost_params
    d_node = d.node_idx

    prev_node = current_node if pos == 0 else remaining[pos - 1].node_idx
    prev_dep  = current_dep  if pos == 0 else remaining[pos - 1].departure_min
    next_node = remaining[pos].node_idx if pos < len(remaining) else 0

    t_prev_d    = matrix[prev_node, d_node]  / 60.0
    t_d_next    = matrix[d_node,    next_node] / 60.0
    t_prev_next = matrix[prev_node, next_node] / 60.0

    extra_travel = t_prev_d + t_d_next - t_prev_next
    arrival_at_d = prev_dep + t_prev_d
    total_extra  = extra_travel + d.service_min

    if remaining:
        last_dep = (remaining[-1].departure_min + total_extra
                    if pos < len(remaining)
                    else arrival_at_d + d.service_min)
        last_node = remaining[-1].node_idx if pos < len(remaining) else d_node
        end_time = last_dep + matrix[last_node, 0] / 60.0
    else:
        end_time = arrival_at_d + d.service_min + matrix[d_node, 0] / 60.0

    feasible = end_time <= workday_minutes

    if travel_time_only:
        cost = extra_travel
    else:
        extra_km = max(0.0,
            _approx_km(all_coords[prev_node], all_coords[d_node])
            + _approx_km(all_coords[d_node],  all_coords[next_node])
            - _approx_km(all_coords[prev_node], all_coords[next_node])
        )
        report_min = float((hour - 8) * 60)
        wait_h = max(0.0, (arrival_at_d - report_min) / 60.0)
        cost = (
            (total_extra / 60.0) * cp.wage_eur_per_hour
            + extra_km            * cp.fuel_eur_per_km
            + wait_h * d.power_kw * cp.downtime_eur_per_kwh
        )
    return cost, feasible, arrival_at_d


def _find_best_insertion(
    d: DisruptionEvent,
    sim_routes: list[SimRoute],
    time_min: float,
    hour: int,
    matrix: np.ndarray,
    all_coords: np.ndarray,
    workday_minutes: int,
    cost_params: CostParams,
    travel_time_only: bool = False,
) -> Optional[tuple[float, int, int, float]]:
    """Günstigste feasible Einfügeposition über alle Teams."""
    best_cost = np.inf
    best: Optional[tuple[float, int, int, float]] = None

    for ti, route in enumerate(sim_routes):
        remaining = route.remaining_stops_at(time_min)
        cur_node = route.current_node_at(time_min)
        cur_dep  = route.current_departure_at(time_min)
        if route.lunch_end_min is not None and cur_dep < route.lunch_end_min:
            cur_dep = route.lunch_end_min

        for pos in range(len(remaining) + 1):
            cost, feasible, arrival = _insertion_cost(
                d, remaining, cur_node, cur_dep, pos, hour, matrix, all_coords,
                workday_minutes, cost_params, travel_time_only,
            )
            if feasible and cost < best_cost:
                best_cost = cost
                best = (cost, ti, pos, arrival)

    return best


def _find_best_drop_and_insert(
    d: DisruptionEvent,
    sim_routes: list[SimRoute],
    time_min: float,
    hour: int,
    all_coords: np.ndarray,
    traffic_matrices: dict[int, np.ndarray],
    workday_start_hour: int,
    workday_minutes: int,
    cost_params: CostParams,
    drop_score_fn: Callable[[int, float, float, int], float],
    travel_time_only: bool = False,
) -> Optional[tuple[float, int, list[int], int, float]]:
    """
    Drop lowest-score Routine-Stops iterativ bis Insertion feasible wird.

    Gibt zurück: (cost, team_idx, [drop_global_indices], insert_pos, arrival_at_d)
    """
    matrix = _get_matrix(traffic_matrices, time_min, workday_start_hour)
    remaining_hours = max(0.0, (workday_minutes - time_min) / 60.0)
    best_cost = np.inf
    best: Optional[tuple[float, int, list[int], int, float]] = None

    for ti, route in enumerate(sim_routes):
        remaining = route.remaining_stops_at(time_min)
        routine_stops = [(i, s) for i, s in enumerate(remaining) if s.task_type == "routine"]
        if not routine_stops:
            continue

        cur_node = route.current_node_at(time_min)
        cur_dep  = route.current_departure_at(time_min)
        if route.lunch_end_min is not None and cur_dep < route.lunch_end_min:
            cur_dep = route.lunch_end_min

        # Aufsteigend nach drop_score_fn sortieren (niedrigster Score = erst droppen)
        routine_sorted = sorted(
            routine_stops,
            key=lambda x: drop_score_fn(x[1].node_idx, x[1].days_since_maintenance, remaining_hours, cur_node),
        )

        dropped_indices: list[int] = []

        for n_drop in range(1, len(routine_sorted) + 1):
            dropped_indices.append(routine_sorted[n_drop - 1][0])
            drop_set = set(dropped_indices)

            # Route ohne gedropte Stops neu berechnen
            from src.models.simulator import SimStop  # lokaler Import vermeidet Zirkel
            trimmed: list[SimStop] = []
            prev_n, prev_d = cur_node, cur_dep
            for i, s in enumerate(remaining):
                if i in drop_set:
                    continue
                mat = _get_matrix(traffic_matrices, prev_d, workday_start_hour)
                new_arr = prev_d + mat[prev_n, s.node_idx] / 60.0
                trimmed.append(SimStop(
                    node_idx=s.node_idx,
                    task_type=s.task_type,
                    arrival_min=new_arr,
                    service_min=s.service_min,
                    days_since_maintenance=s.days_since_maintenance,
                ))
                prev_n = s.node_idx
                prev_d = new_arr + s.service_min

            for pos in range(len(trimmed) + 1):
                cost, feasible, arrival = _insertion_cost(
                    d, trimmed, cur_node, cur_dep, pos, hour, matrix, all_coords,
                    workday_minutes, cost_params, travel_time_only,
                )
                if feasible and cost < best_cost:
                    drop_globals = [
                        route.stops.index(remaining[i]) for i in dropped_indices
                    ]
                    best_cost = cost
                    best = (cost, ti, drop_globals, pos, arrival)

            if best is not None:
                break  # So wenige Drops wie nötig

    return best


def _insert_stop(
    d: DisruptionEvent,
    route: SimRoute,
    pos: int,
    time_min: float,
    traffic_matrices: dict[int, np.ndarray],
    workday_start_hour: int,
) -> None:
    """Fügt Störungs-Stop an Position pos (in remaining) ein und aktualisiert Folgezeiten."""
    from src.models.simulator import SimStop  # lokaler Import vermeidet Zirkel
    remaining = route.remaining_stops_at(time_min)
    # Stops können durch frühere Ankunftszeiten nach einem Drop aus remaining fallen →
    # pos ggf. größer als die neue Liste; dann ans Ende anhängen.
    pos = min(pos, len(remaining))
    first_global = route.stops.index(remaining[0]) if remaining else len(route.stops)
    global_insert = first_global + pos

    if pos == 0:
        prev_node = route.current_node_at(time_min)
        prev_dep  = route.current_departure_at(time_min)
        if route.lunch_end_min is not None and prev_dep < route.lunch_end_min:
            prev_dep = route.lunch_end_min
    else:
        prev = remaining[pos - 1]
        prev_node = prev.node_idx
        prev_dep  = prev.departure_min

    matrix = _get_matrix(traffic_matrices, prev_dep, workday_start_hour)
    arrival = prev_dep + matrix[prev_node, d.node_idx] / 60.0

    new_stop = SimStop(
        node_idx=d.node_idx,
        task_type=d.disruption_type,
        arrival_min=arrival,
        service_min=d.service_min,
        days_since_maintenance=0.0,
    )
    route.stops.insert(global_insert, new_stop)

    # Folge-Stops neu berechnen
    for i in range(global_insert + 1, len(route.stops)):
        prev_s = route.stops[i - 1]
        curr_s = route.stops[i]
        mat = _get_matrix(traffic_matrices, prev_s.departure_min, workday_start_hour)
        curr_s.arrival_min = prev_s.departure_min + mat[prev_s.node_idx, curr_s.node_idx] / 60.0


def _remove_stop_and_recompute(
    route: SimRoute,
    global_idx: int,
    traffic_matrices: dict[int, np.ndarray],
    workday_start_hour: int,
) -> None:
    """Entfernt Stop und berechnet alle Folge-Ankunftszeiten neu."""
    route.stops.pop(global_idx)
    if global_idx >= len(route.stops):
        return

    if global_idx == 0:
        prev_node, prev_dep = 0, 0.0
    else:
        prev = route.stops[global_idx - 1]
        prev_node = prev.node_idx
        prev_dep  = prev.departure_min

    for i in range(global_idx, len(route.stops)):
        curr = route.stops[i]
        mat = _get_matrix(traffic_matrices, prev_dep, workday_start_hour)
        curr.arrival_min = prev_dep + mat[prev_node, curr.node_idx] / 60.0
        prev_node = curr.node_idx
        prev_dep  = curr.departure_min
