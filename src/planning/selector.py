"""
Tägliche Stationsauswahl für den Initialplan.

Algorithmus (gleich für alle Modelle – Myopic, CFA, VFA):
  1. Offene Zonen nach Prioritätsscore ranken (Depot-Entfernung + Fläche).
  2. Jedem Team eine Startzone zuweisen (aus Top-N), mit mindestens
     `min_team_separation_km` Abstand zwischen den Startzonen.
  3. Kandidaten pro Team durch Nearest-Neighbor-Expansion aufbauen:
     Startzone vollständig besuchen, dann iterativ die nächste unbesuchte
     Station (aus beliebiger Zone) hinzufügen bis Kapazitätslimit erreicht.
     → Ergibt eine geografisch zusammenhängende, kompakte Menge ohne
       Quersprünge zwischen weit entfernten Zonen.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.planning.clustering import ZoneClusterer, _approx_km, _pairwise_km
from src.planning.vrp_solver import MaintenanceTask, TeamState

logger = logging.getLogger(__name__)


@dataclass
class ZoneAssignment:
    """Ergebnis der täglichen Zonenauswahl."""

    team_tasks: dict[int, list[MaintenanceTask]]
    """team_id → Liste der Wartungsaufgaben für heute."""

    selected_zones: dict[int, list[int]]
    """team_id → Startzone(n) pro Team (für Logging/Visualisierung)."""


class DailyZoneSelector:
    """
    Wählt täglich Stationen für beide Wartungsteams aus.

    Parameters
    ----------
    clusterer : ZoneClusterer
        Gefitteter Clusterer mit Zoneneigenschaften.
    config : dict
        Konfigurationsdict aus config.yaml.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot an Index 0.
        Entspricht direkt den node_idx-Werten des VRPSolvers.
    """

    def __init__(
        self,
        clusterer: ZoneClusterer,
        config: dict,
        all_coords: np.ndarray,
        charging_points: Optional[np.ndarray] = None,
    ) -> None:
        self.clusterer = clusterer
        self.all_coords = all_coords
        self.station_coords = all_coords[1:]

        planning_cfg = config.get("planning", {})
        self.n_top_candidates: int = planning_cfg.get("n_top_candidates", 20)
        self.min_separation_km: float = planning_cfg.get("min_team_separation_km", 1.0)
        self.max_stations_per_team: int = planning_cfg.get("max_stations_per_team", 16)
        maint_cfg = config.get("maintenance", {})
        self.default_service_time: int = maint_cfg.get("mean_service_time", 30)
        self.service_time_mode: str = maint_cfg.get("service_time_mode", "fixed")
        self.minutes_per_charging_point: int = maint_cfg.get("minutes_per_charging_point", 15)
        # charging_points[station_idx] = Anzahl Ladepunkte (0-basiert, ohne Depot)
        self.charging_points: Optional[np.ndarray] = charging_points
        self._workday_minutes: int = (
            (maint_cfg.get("workday_end_hour", 17) - maint_cfg.get("workday_start_hour", 8)) * 60
            - maint_cfg.get("lunch_duration_min", 0)
        )
        self._travel_reserve_min: int = planning_cfg.get("travel_reserve_min", 60)

        w = planning_cfg.get("priority_weights", {})
        self.w_depot: float = float(w.get("depot_distance", 0.5))
        self.w_area: float = float(w.get("convex_hull_area", 0.5))
        self.w_value: float = float(w.get("zone_value", 0.5))
        self.zone_selection_mode: str = planning_cfg.get("zone_selection_mode", "classic")
        self.zone_expansion_mode: str = planning_cfg.get("zone_expansion_mode", "nearest")

        # Wird nach Konstruktion gesetzt wenn zone_selection_mode == "value_based":
        #   selector.value_fn = lambda node_idx, dsm: model._value(node_idx, dsm)
        self.value_fn: Optional[Callable[[int, float], float]] = None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def select_for_day(
        self,
        remaining_station_indices: list[int],
        team_states: list[TeamState],
        carryover_tasks: list[MaintenanceTask] | None = None,
        dsm_array: Optional[np.ndarray] = None,
    ) -> ZoneAssignment:
        """
        Bestimmt die Wartungsaufgaben für beide Teams am Tagesbeginn.

        Parameters
        ----------
        remaining_station_indices : list[int]
            0-basierte Indices aller noch nicht jährlich gewarteten Stationen.
        team_states : list[TeamState]
            Aktueller Zustand beider Teams.
        carryover_tasks : list[MaintenanceTask] | None
            Unerledigte Störungen vom Vortag – werden priorisiert
            zum jeweiligen Team hinzugefügt.
        dsm_array : np.ndarray | None
            days_since_maintenance pro node_idx (1-basiert). Wird für
            V̂-basierte Zonenauswahl benötigt; None → klassisches Scoring.
        """
        remaining_set = set(remaining_station_indices)
        carryover = carryover_tasks or []

        open_zones = self._open_zones(remaining_set)

        if not open_zones:
            logger.info("Alle Stationen für dieses Jahr bereits gewartet.")
            tasks_per_team = {s.team_id: [] for s in team_states}
            self._distribute_carryover(carryover, team_states, tasks_per_team)
            return ZoneAssignment(team_tasks=tasks_per_team, selected_zones={})

        # Schritt 1: Startzonen zuweisen (1 pro Team, mit Mindestabstand)
        scored_zones = self._score_zones(open_zones, remaining_set, dsm_array)
        top_zones = scored_zones[: self.n_top_candidates]
        starting_zones = self._assign_starting_zones(top_zones, team_states)

        # Schritt 2a: Carryover zuerst verteilen, damit die Kapazität bekannt ist
        tasks_per_team: dict[int, list[MaintenanceTask]] = {
            s.team_id: [] for s in team_states
        }
        self._distribute_carryover(carryover, team_states, tasks_per_team)

        # Schritt 2b: Routine-Kandidaten auswählen.
        # Kapazität: Zeitbudget abzüglich Carryover-Servicezeit, damit OR-Tools
        # nicht mit einem infeasiblen Task-Set konfrontiert wird.
        time_budget = self._workday_minutes - self._travel_reserve_min
        routine_service = self.default_service_time

        # Endspiel: Wenn alle verbleibenden Stationen theoretisch in einem Tag
        # erledigt werden könnten, Obergrenze pro Team gleichmäßig aufteilen.
        # Verhindert dass Team 0 die max. Kapazität ausschöpft und Team 1 leer ausgeht.
        n_teams = len(team_states)
        n_remaining = len(remaining_set)
        if n_remaining <= n_teams * self.max_stations_per_team:
            endgame_cap = math.ceil(n_remaining / n_teams)
        else:
            endgame_cap = self.max_stations_per_team

        claimed: set[int] = set()

        for state in team_states:
            tid = state.team_id
            carryover_service_min = sum(t.service_time for t in tasks_per_team[tid])
            remaining_min = max(0, time_budget - carryover_service_min)
            capacity_by_time = int(remaining_min // routine_service)
            n_routine_assigned = sum(1 for t in tasks_per_team[tid] if t.task_type == "routine")
            capacity_by_count = max(0, endgame_cap - n_routine_assigned)
            capacity = min(capacity_by_time, capacity_by_count)

            start_zone = starting_zones.get(tid)
            stations = self._expand_from_zone(
                start_zone, remaining_set, claimed, state, capacity=capacity,
                scored_zones=scored_zones,
            )
            claimed.update(s - 1 for s in [t.node_idx for t in stations])
            tasks_per_team[tid].extend(stations)

        self._rebalance_if_idle(tasks_per_team, team_states)

        return ZoneAssignment(
            team_tasks=tasks_per_team,
            selected_zones={s.team_id: [starting_zones[s.team_id]] for s in team_states
                            if s.team_id in starting_zones},
        )

    # ------------------------------------------------------------------
    # Interne Schritte
    # ------------------------------------------------------------------

    def _open_zones(self, remaining_set: set[int]) -> list[int]:
        assert self.clusterer.station_indices_per_zone_ is not None
        return [
            z for z, idxs in self.clusterer.station_indices_per_zone_.items()
            if any(i in remaining_set for i in idxs)
        ]

    def _score_zones(
        self,
        open_zones: list[int],
        remaining_set: set[int],
        dsm_array: Optional[np.ndarray],
    ) -> list[int]:
        """Gibt Zonen absteigend nach Prioritätsscore sortiert zurück."""
        assert self.clusterer.mean_depot_distances_ is not None
        assert self.clusterer.convex_hull_areas_ is not None

        dists = self.clusterer.mean_depot_distances_[open_zones]

        def _norm(arr: np.ndarray) -> np.ndarray:
            span = arr.max() - arr.min()
            return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)

        if self.zone_selection_mode == "value_based" and self.value_fn is not None and dsm_array is not None:
            # V̂-basierte Zonenauswahl: Summe der stationsindividuellen Werte pro Zone
            scores = np.array([
                sum(
                    self.value_fn(s + 1, float(dsm_array[s + 1]))
                    for s in self.clusterer.station_indices_per_zone_[z]
                    if s in remaining_set
                )
                for z in open_zones
            ])
        elif self.zone_selection_mode == "centrality":
            assert self.clusterer.mean_dist_to_other_centroids_ is not None
            # Kleinere mittlere Distanz zu anderen Zentroiden = zentralere Zone = höherer Score
            centrality_dists = self.clusterer.mean_dist_to_other_centroids_[open_zones]
            scores = 1 - _norm(centrality_dists)
        elif self.zone_selection_mode == "depot_distance":
            # Nächste Zonen zum Depot zuerst
            scores = 1 - _norm(dists)
        else:
            # classic: depot_distance + convex_hull_area
            areas = self.clusterer.convex_hull_areas_[open_zones]
            scores = self.w_depot * _norm(dists) + self.w_area * _norm(areas)

        return [open_zones[i] for i in np.argsort(scores)[::-1]]

    def _assign_starting_zones(
        self,
        candidates: list[int],
        team_states: list[TeamState],
    ) -> dict[int, int]:
        """
        Weist jedem Team eine Startzone zu.
        Team 0 bekommt die nächste Zone zur seiner Position.
        Team 1 bekommt die nächste Zone die mindestens `min_separation_km`
        von Team 0s Startzone entfernt ist.
        """
        assert self.clusterer.centroids_ is not None

        if not candidates:
            return {}

        result: dict[int, int] = {}
        used_zones: list[int] = []

        for state in team_states:
            team_coord = self.all_coords[state.current_node]

            best_zone, best_dist = None, np.inf
            for z in candidates:
                if z in used_zones:
                    continue
                d_to_team = _approx_km(team_coord, self.clusterer.centroids_[z])
                # Mindestabstand zu bereits zugewiesenen Startzonen prüfen
                too_close = any(
                    _approx_km(
                        self.clusterer.centroids_[z],
                        self.clusterer.centroids_[uz]
                    ) < self.min_separation_km
                    for uz in used_zones
                )
                if too_close:
                    continue
                if d_to_team < best_dist:
                    best_dist = d_to_team
                    best_zone = z

            if best_zone is None:
                # Fallback: Mindestabstand ignorieren, einfach nächste freie Zone
                for z in candidates:
                    if z not in used_zones:
                        best_zone = z
                        break

            if best_zone is not None:
                result[state.team_id] = best_zone
                used_zones.append(best_zone)

        return result

    def _expand_from_zone(
        self,
        start_zone: int | None,
        remaining_set: set[int],
        already_claimed: set[int],
        team_state: TeamState,
        capacity: int | None = None,
        scored_zones: list[int] | None = None,
    ) -> list[MaintenanceTask]:
        """
        Baut die Kandidatenmenge eines Teams auf:
        1. Alle unbesuchten Stationen der Startzone
        2a. "nearest":    Nearest-Neighbor aus allen verfügbaren Stationen
        2b. "score_rank": nächste offene Zone nach Score-Rang komplett laden, usw.

        `already_claimed` verhindert Doppelzuweisungen zwischen Teams.
        """
        assert self.clusterer.station_indices_per_zone_ is not None

        cap = capacity if capacity is not None else self.max_stations_per_team
        available = remaining_set - already_claimed

        if not available or cap <= 0:
            return []

        selected: list[int] = []  # 0-basierte station_indices

        # Startzone vollständig laden (soweit verfügbar)
        if start_zone is not None:
            for s in self.clusterer.station_indices_per_zone_[start_zone]:
                if s in available and len(selected) < cap:
                    selected.append(s)

        if self.zone_expansion_mode == "score_rank" and scored_zones is not None:
            # Zonen in Score-Reihenfolge durchgehen (Startzone überspringen)
            visited_zones = {start_zone} if start_zone is not None else set()
            for z in scored_zones:
                if len(selected) >= cap:
                    break
                if z in visited_zones:
                    continue
                visited_zones.add(z)
                for s in self.clusterer.station_indices_per_zone_[z]:
                    if s in available and s not in selected and len(selected) < cap:
                        selected.append(s)
        else:
            # Nearest-Neighbor-Expansion bis Kapazitätslimit
            while len(selected) < cap:
                remaining_available = available - set(selected)
                if not remaining_available:
                    break

                # Schwerpunkt der bereits gewählten Stationen
                if selected:
                    anchor = self.station_coords[selected].mean(axis=0)
                else:
                    anchor = self.all_coords[team_state.current_node]

                # Nächste Station zum Schwerpunkt
                next_s = min(
                    remaining_available,
                    key=lambda s: _approx_km(anchor, self.station_coords[s]),
                )
                selected.append(next_s)

        # node_idx = station_idx + 1 (Depot belegt Index 0)
        return [
            MaintenanceTask(node_idx=s + 1, task_type="routine", service_time=self._service_time(s))
            for s in selected
        ]

    def _rebalance_if_idle(
        self,
        tasks_per_team: dict[int, list[MaintenanceTask]],
        team_states: list[TeamState],
    ) -> None:
        """Falls ein Team keine Routine-Stops hat, übernimmt es die Hälfte
        der Stationen des beschäftigsten Teams (geografisch nächste zuerst).

        Tritt typischerweise am Jahresende auf, wenn alle verbleibenden
        Stationen in einer einzigen Zone liegen und kein zweites Team
        eine eigene Startzone erhält.
        """
        routine_per_team: dict[int, list[MaintenanceTask]] = {
            tid: [t for t in tasks if t.task_type == "routine"]
            for tid, tasks in tasks_per_team.items()
        }
        idle_tids = [tid for tid, rt in routine_per_team.items() if not rt]
        busy_tids = [tid for tid, rt in routine_per_team.items() if len(rt) >= 2]

        if not idle_tids or not busy_tids:
            return

        for idle_tid in idle_tids:
            idle_state = next(s for s in team_states if s.team_id == idle_tid)
            idle_coord = self.all_coords[idle_state.current_node]

            donor_tid = max(busy_tids, key=lambda tid: len(routine_per_team[tid]))
            donor_routine = routine_per_team[donor_tid]
            n_transfer = len(donor_routine) // 2
            if n_transfer == 0:
                continue

            sorted_by_dist = sorted(
                donor_routine,
                key=lambda t: _approx_km(self.all_coords[t.node_idx], idle_coord),
            )
            to_transfer = sorted_by_dist[:n_transfer]

            for task in to_transfer:
                tasks_per_team[donor_tid].remove(task)
                tasks_per_team[idle_tid].append(task)
                routine_per_team[donor_tid].remove(task)
                routine_per_team[idle_tid].append(task)

    def _service_time(self, station_idx: int) -> int:
        if (
            self.service_time_mode == "per_charging_point"
            and self.charging_points is not None
        ):
            return int(self.charging_points[station_idx]) * self.minutes_per_charging_point
        return self.default_service_time

    def _distribute_carryover(
        self,
        carryover: list[MaintenanceTask],
        team_states: list[TeamState],
        tasks_per_team: dict[int, list[MaintenanceTask]],
    ) -> None:
        """Verteilt Vortags-Störungen balanciert zwischen Teams.

        Primär: Team mit geringster akkumulierter Carryover-Servicezeit.
        Tie-Breaker: geografisch nächstes Team.
        So wird verhindert dass ein Team alle Carryovers bekommt und
        den Arbeitstag weit überzieht.
        """
        carryover_service: dict[int, int] = {s.team_id: 0 for s in team_states}

        for task in carryover:
            task_coord = self.all_coords[task.node_idx]
            best_team = min(
                team_states,
                key=lambda s: (
                    carryover_service[s.team_id],
                    _approx_km(task_coord, self.all_coords[s.current_node]),
                ),
            )
            tasks_per_team[best_team.team_id].insert(0, task)
            carryover_service[best_team.team_id] += task.service_time
