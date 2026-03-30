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
from dataclasses import dataclass

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
        self._workday_minutes: int = (
            maint_cfg.get("workday_end_hour", 17) - maint_cfg.get("workday_start_hour", 8)
        ) * 60
        self._travel_reserve_min: int = planning_cfg.get("travel_reserve_min", 60)

        w = planning_cfg.get("priority_weights", {})
        self.w_depot: float = float(w.get("depot_distance", 0.5))
        self.w_area: float = float(w.get("convex_hull_area", 0.5))

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def select_for_day(
        self,
        remaining_station_indices: list[int],
        team_states: list[TeamState],
        carryover_tasks: list[MaintenanceTask] | None = None,
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
        scored_zones = self._score_zones(open_zones)
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

        claimed: set[int] = set()

        for state in team_states:
            tid = state.team_id
            carryover_service_min = sum(t.service_time for t in tasks_per_team[tid])
            remaining_min = max(0, time_budget - carryover_service_min)
            capacity_by_time = int(remaining_min // routine_service)
            capacity_by_count = max(0, self.max_stations_per_team - len(tasks_per_team[tid]))
            capacity = min(capacity_by_time, capacity_by_count)

            start_zone = starting_zones.get(tid)
            stations = self._expand_from_zone(
                start_zone, remaining_set, claimed, state, capacity=capacity
            )
            claimed.update(s - 1 for s in [t.node_idx for t in stations])
            tasks_per_team[tid].extend(stations)

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

    def _score_zones(self, open_zones: list[int]) -> list[int]:
        """Gibt Zonen absteigend nach Prioritätsscore sortiert zurück."""
        assert self.clusterer.mean_depot_distances_ is not None
        assert self.clusterer.convex_hull_areas_ is not None

        dists = self.clusterer.mean_depot_distances_[open_zones]
        areas = self.clusterer.convex_hull_areas_[open_zones]

        def _norm(arr: np.ndarray) -> np.ndarray:
            span = arr.max() - arr.min()
            return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)

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
    ) -> list[MaintenanceTask]:
        """
        Baut die Kandidatenmenge eines Teams auf:
        1. Alle unbesuchten Stationen der Startzone
        2. Dann iterativ die nächste unbesuchte Station (aus beliebiger Zone)
           zum Cluster-Schwerpunkt hinzufügen bis Kapazitätslimit erreicht.

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
        return [MaintenanceTask(node_idx=s + 1, task_type="routine") for s in selected]

    def _distribute_carryover(
        self,
        carryover: list[MaintenanceTask],
        team_states: list[TeamState],
        tasks_per_team: dict[int, list[MaintenanceTask]],
    ) -> None:
        """Verteilt Vortags-Störungen zum geografisch nächsten Team."""
        for task in carryover:
            task_coord = self.all_coords[task.node_idx]
            best_team = min(
                team_states,
                key=lambda s: _approx_km(task_coord, self.all_coords[s.current_node]),
            )
            tasks_per_team[best_team.team_id].insert(0, task)
