"""
Value Function Approximation (VFA) Modell.

Approximiert den zukünftigen Wert eines Zustands durch eine gelernte
lineare Wertfunktion:

    V̂(s) = θᵀ φ(s) + intercept

Feature-Vektor φ(s) (6 globale Zustandsmerkmale, identisch zu train_vfa.py):
    f0: Σ_k power_kW[k] × dsm[k]           – Gesamtdringlichkeit
    f1: Σ_k failure_risk[k] × power_kW[k]  – Erwarteter Schadenwert
    f2: mean(dsm[k])                         – Mittlere Überfälligkeit
    f3: max(power_kW[k] × dsm[k])           – Größte Einzeldringlichkeit
    f4: n_remaining / n_stations             – Auslastungsgrad
    f5: n_carryover                          – Offene Störungsrückstände

Integration in OR-Tools:
    Für jeden Routine-Task k wird V̂(s') nach Entnahme von k aus der
    verbleibenden Menge berechnet. Die Reduktion ΔV̂ = V̂(s) − V̂(s') ist
    der zukünftige Wert der Wartung, der als negativer Bonus (= Kostensenkung)
    an OR-Tools übergeben wird. Stations mit hohem ΔV̂ werden bevorzugt
    früh eingeplant.

θ wird offline trainiert (scripts/train_vfa.py) und aus
data/vfa/theta.json geladen.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.cost_params import CostParams
from src.models.simulator import (
    DisruptionEvent,
    HourLog,
    SimRoute,
    plan_to_sim_routes,
)
from src.planning.vrp_solver import DailyPlan, MaintenanceTask, TeamState, VRPSolver

logger = logging.getLogger(__name__)

_DEFAULT_THETA_PATH = Path("data/vfa/theta.json")


class VFAModel:
    """
    Value Function Approximation Modell mit gelerntem θ-Vektor.

    Parameters
    ----------
    traffic_matrices : Stündliche Reisezeitmatrizen in Sekunden.
    config : Konfigurationsdict.
    all_coords : np.ndarray, shape (n_stations + 1, 2)
        Koordinaten aller Knoten inkl. Depot (Index 0).
    node_to_power : dict[int, float]
        Mapping node_idx → Nennleistung [kW].
    n_stations : int
        Gesamtzahl der Stationen (ohne Depot).
    cost_params : Kostenparameter (None → Standardwerte).
    theta_path : Pfad zu data/vfa/theta.json. None → Standardpfad.
    theta_override : np.ndarray | list | None
        Direkt übergebener θ-Vektor (überschreibt theta_path). Wird für
        iteratives Policy-Training verwendet, um θ ohne Datei-I/O zu setzen.
    intercept_override : float | None
        Direkt übergebener Intercept-Wert. Wird zusammen mit theta_override
        verwendet; None → 0.0.
    """

    def __init__(
        self,
        traffic_matrices: dict[int, np.ndarray],
        config: dict,
        all_coords: Optional[np.ndarray] = None,
        node_to_power: Optional[dict[int, float]] = None,
        n_stations: int = 397,
        cost_params: Optional[CostParams] = None,
        theta_path: Optional[Path | str] = None,
        theta_override: Optional[np.ndarray] = None,
        intercept_override: Optional[float] = None,
    ) -> None:
        self.solver = VRPSolver(traffic_matrices, config, all_coords=all_coords)
        self.config = config
        self.all_coords = all_coords
        self.node_to_power: dict[int, float] = node_to_power or {}
        self.n_stations = n_stations
        self.cost_params = cost_params or CostParams()
        self.n_teams: int = config["maintenance"]["n_teams"]

        fail_cfg = config.get("failure_simulation", {})
        self.lambda_per_day: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        ) * 24.0

        # Parameter für Disruption-Deadline-Penalty (analog CFA)
        cfa_cfg = config.get("cfa", {})
        self.alpha: float = float(cfa_cfg.get("alpha", 10.0))
        self.p_failure_per_hour: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        )
        self._wage_per_min: float = self.cost_params.wage_eur_per_hour / 60.0

        if theta_override is not None:
            self.theta     = np.array(theta_override, dtype=np.float64)
            self.intercept = float(intercept_override) if intercept_override is not None else 0.0
            logger.info(f"VFA: θ direkt übergeben (iteratives Training)")
        else:
            path = Path(theta_path) if theta_path else _DEFAULT_THETA_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"VFA-Gewichte nicht gefunden: {path}\n"
                    f"Bitte zuerst 'python scripts/train_vfa.py' ausführen."
                )
            with open(path) as f:
                data = json.load(f)
            self.theta     = np.array(data["theta"], dtype=np.float64)
            self.intercept = float(data["intercept"])
            logger.info(
                f"VFA: θ geladen aus {path} "
                f"(R²={data.get('r2', '?'):.4f}, {data.get('n_runs', '?')} Läufe)"
            )

        # Stationsindividuellen Features für ΔV̂: nur die Features die sich
        # beim Entfernen einer einzelnen Station tatsächlich ändern.
        # f4 (frac_remaining) ändert sich für jede Station gleich → konstant,
        # kein Differenzierungssignal. f5 (n_carryover) gehört nicht zu Stationen.
        # Für extra_costs nur f0-f3 verwenden (stations-spezifische Features).
        self._station_feature_mask = np.array([True, True, True, True, False, False])

    # ------------------------------------------------------------------
    # Wertfunktion
    # ------------------------------------------------------------------

    def _phi(
        self,
        tasks: list[MaintenanceTask],
        n_carryover: int = 0,
        exclude_node: Optional[int] = None,
    ) -> np.ndarray:
        """
        Berechnet den Feature-Vektor φ(s) für eine gegebene Task-Menge.

        Parameters
        ----------
        tasks : Verbleibende Routine-Tasks (mit days_since_maintenance).
        n_carryover : Anzahl offener Carryover-Störungen.
        exclude_node : Falls gesetzt, wird dieser node_idx vor der
                       Berechnung aus der Task-Menge entfernt (für ΔV̂).
        """
        routine = [t for t in tasks if t.task_type == "routine"
                   and t.node_idx != exclude_node]

        if routine:
            dsm_vals = np.array([t.days_since_maintenance for t in routine], dtype=np.float64)
            pow_vals = np.array([
                self.node_to_power.get(t.node_idx, 22.0) for t in routine
            ], dtype=np.float64)
            urgency      = pow_vals * dsm_vals
            failure_risk = 1.0 - np.exp(-self.lambda_per_day * dsm_vals)

            f0 = float(np.sum(urgency))
            f1 = float(np.sum(failure_risk * pow_vals))
            f2 = float(np.mean(dsm_vals))
            f3 = float(np.max(urgency))
        else:
            f0 = f1 = f2 = f3 = 0.0

        n_remaining = len(routine)
        f4 = n_remaining / max(1, self.n_stations)
        f5 = float(n_carryover)

        return np.array([f0, f1, f2, f3, f4, f5], dtype=np.float64)

    def _value(self, phi: np.ndarray) -> float:
        """V̂(s) = θᵀ φ + intercept."""
        return float(self.theta @ phi) + self.intercept

    def _station_value(self, node_idx: int, dsm: float) -> float:
        """
        Stationsindividuelle Näherung von ΔV̂ für die V̂-basierte Zonenauswahl.

        Berechnet den marginalen Wertbeitrag einer einzelnen Station, ohne den
        globalen Zustand zu kennen. Verwendet nur die stationsindividuellen
        Features f0 (power × dsm) und f1 (failure_risk × power), da f2 (mean dsm)
        und f3 (max urgency) globalen Kontext erfordern.
        """
        power = self.node_to_power.get(node_idx, 22.0)
        urgency = power * dsm
        failure_risk = 1.0 - np.exp(-self.lambda_per_day * dsm)
        return float(self.theta[0] * urgency + self.theta[1] * failure_risk * power)

    def _disruption_deadline_penalty(self, power_kw: float) -> int:
        """Deadline-Penalty für Störungen in Minuten/Minute Überschreitung (analog CFA)."""
        penalty_eur = (
            self.alpha * power_kw * self.p_failure_per_hour
            * self.cost_params.downtime_eur_per_kwh
        )
        return max(1, int(round(penalty_eur / self._wage_per_min)))

    def _compute_extra_costs(
        self,
        tasks: list[MaintenanceTask],
        n_carryover: int = 0,
    ) -> dict[int, int]:
        """
        Berechnet stationsabhängige Zusatzkosten aus der Wertfunktion.

        OR-Tools erwartet ausschließlich nicht-negative Arc Costs.
        Daher wird die ΔV̂-Skala verschoben:

            extra_cost[k] = max_bonus − bonus[k]  ≥ 0

        Nur stationsindividuelle Features (f0-f3) werden für ΔV̂ verwendet.
        f4 (frac_remaining) und f5 (n_carryover) ändern sich für alle Stationen
        gleich und liefern kein Differenzierungssignal für die Reihenfolge.
        """
        routine_tasks = [t for t in tasks if t.task_type == "routine"]
        if not routine_tasks:
            return {}

        # Theta nur mit stationsindividuellen Features
        theta_station = self.theta * self._station_feature_mask

        phi_s = self._phi(tasks, n_carryover=n_carryover)
        v_s   = float(theta_station @ phi_s)

        raw: dict[int, float] = {}
        for task in routine_tasks:
            phi_prime = self._phi(tasks, n_carryover=n_carryover, exclude_node=task.node_idx)
            delta_v   = v_s - float(theta_station @ phi_prime)
            raw[task.node_idx] = delta_v

        # Verschieben auf ≥ 0, skalieren in Minuten-Einheiten für OR-Tools
        max_v = max(raw.values())
        return {
            node_idx: int(round((max_v - dv) / self._wage_per_min))
            for node_idx, dv in raw.items()
        }

    # ------------------------------------------------------------------
    # Policy-Schnittstelle
    # ------------------------------------------------------------------

    def create_initial_plan(
        self,
        tasks: list[MaintenanceTask],
        team_assignment: Optional[dict[int, list[int]]] = None,
    ) -> DailyPlan:
        """
        Erstellt den Tagesplan mit VFA-basierten Zusatzkosten für OR-Tools.

        Stations mit hohem zukünftigen Ersparnispotenzial (ΔV̂) erhalten
        einen Bonus → werden bevorzugt früh eingeplant.
        """
        extra_costs = self._compute_extra_costs(tasks)
        return self.solver.create_initial_plan(
            tasks, extra_costs=extra_costs, team_assignment=team_assignment
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
        Replant den Tag mit allen offenen + neuen Aufgaben via OR-Tools + VFA.

        1. Vollständiges VRP-Replanning (wie CFA, nicht greedy).
        2. Falls infeasible: Routine-Stops nach aufsteigendem ΔV̂ droppen
           (niedrigstes Ersparnispotenzial zuerst).
        """
        team_states = [
            TeamState(
                team_id=r.team_id,
                current_node=r.current_node_at(time_min),
                current_time=int(r.lunch_end_min) if (
                    r.lunch_end_min is not None and time_min < r.lunch_end_min
                ) else int(time_min),
                completed_nodes=r.completed_nodes_at(time_min),
            )
            for r in sim_routes
        ]

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

        disruption_tasks = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="disruption",
                priority=1,
                service_time=int(round(d.service_min)),
                soft_deadline_min=int(time_min),
                deadline_penalty=self._disruption_deadline_penalty(d.power_kw),
            )
            for d in disruptions
        ]

        all_tasks = remaining_tasks + disruption_tasks
        if not all_tasks:
            return 0, [], 0.0

        extra_costs = self._compute_extra_costs(all_tasks)
        new_plan = self.solver.replan(all_tasks, team_states, extra_costs=extra_costs)

        if new_plan.solver_status in ("INFEASIBLE", "NO_SOLUTION"):
            # VFA-Kern: Routine-Stops nach aufsteigendem ΔV̂ droppen
            routine_tasks = [t for t in remaining_tasks if t.task_type == "routine"]
            mandatory = [t for t in remaining_tasks if t.task_type != "routine"] + disruption_tasks

            # ΔV̂ pro Station berechnen: niedrigstes Ersparnispotenzial zuerst
            phi_s = self._phi(all_tasks)
            v_s   = self._value(phi_s)

            def delta_v(task: MaintenanceTask) -> float:
                phi_prime = self._phi(all_tasks, exclude_node=task.node_idx)
                return v_s - self._value(phi_prime)

            routine_tasks.sort(key=delta_v)

            solved = False
            for n_drop in range(1, len(routine_tasks) + 1):
                retry_tasks = mandatory + routine_tasks[n_drop:]
                if not retry_tasks:
                    break
                retry_extra = self._compute_extra_costs(retry_tasks)
                new_plan = self.solver.replan(retry_tasks, team_states, extra_costs=retry_extra)
                if new_plan.solver_status not in ("INFEASIBLE", "NO_SOLUTION"):
                    dropped = [t.node_idx for t in routine_tasks[:n_drop]]
                    log.notes.append(
                        f"VFA-Replan Retry: {n_drop} Routine-Stop(s) nach ΔV̂ ausgebaut "
                        f"{dropped}, Status: {new_plan.solver_status}"
                    )
                    solved = True
                    all_tasks = retry_tasks
                    break

            if not solved:
                log.notes.append(
                    f"VFA-Replan fehlgeschlagen ({new_plan.solver_status}): "
                    f"{len(disruptions)} Störung(en) als Carryover."
                )
                return 0, list(disruptions), 0.0

        new_routes = plan_to_sim_routes(new_plan, all_tasks, self.n_teams)
        for i, route in enumerate(sim_routes):
            completed = [s for s in route.stops if s.arrival_min <= time_min]
            route.stops = completed + (new_routes[i].stops if i < len(new_routes) else [])

        disruption_nodes = {d.node_idx: d for d in disruptions}
        downtime_cost = 0.0
        cp = self.cost_params
        for route in sim_routes:
            for stop in route.stops:
                if stop.node_idx in disruption_nodes:
                    d = disruption_nodes[stop.node_idx]
                    report_min = float((hour - 8) * 60)
                    wait_h = max(0.0, (stop.arrival_min - report_min) / 60.0)
                    d_cost = wait_h * d.power_kw * cp.downtime_eur_per_kwh
                    downtime_cost += d_cost
                    if d_cost > 0:
                        log.notes.append(
                            f"  Ausfall {d_cost:.2f} EUR ({wait_h:.2f} h Wartezeit)"
                        )

        log.notes.append(
            f"VFA-Replan: {len(disruptions)} Störung(en) eingearbeitet, "
            f"Status: {new_plan.solver_status}"
        )
        return len(disruptions), [], downtime_cost
