"""
Gymnasium-Environment für die Wartungsoptimierung von E-Ladesäulen in Würzburg.

Modelliert zwei Wartungsteams, die täglich von einem Depot aus
defekte/wartungsbedürftige Ladesäulen anfahren.

Zustand (State):
  - Positionen beider Teams (je ein Index in der Koordinatenliste)
  - Zeiten beider Teams (Minuten ab 8:00)
  - Binärer Vektor: welche Stationen müssen gewartet werden
  - Binärer Vektor: welche Stationen wurden bereits besucht

Aktionsraum (Action Space):
  - MultiDiscrete([n_total, n_total]):
    Team 0 fährt zu Station a, Team 1 fährt zu Station b.
    (0 = Depot, 1..N = Ladesäulen)

Reward:
  - +visit_bonus für jede gewartete Station
  - -travel_time_penalty * Fahrzeit in Minuten
  - -revisit_penalty bei Doppelbesuch
  - +completion_bonus wenn alle wartungsbedürftigen Stationen besucht

Das Environment ist bewusst offen gehalten, sodass verschiedene Modellansätze
(Myopic, CFA, VFA) damit arbeiten können.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.data.loader import load_config
from src.planning import TeamState


class MaintenanceEnv(gym.Env):
    """
    Wartungs-Routing-Environment für E-Ladesäulen mit zwei Teams.

    Parameters
    ----------
    duration_matrix : n×n Fahrzeit-Matrix in Sekunden (inkl. Depot an Index 0).
    needs_maintenance : Boolescher Array der Länge n-1 (ohne Depot),
                        True = Station muss gewartet werden.
                        Wenn None, wird zufällig bestimmt.
    config : Konfigurationsdict (aus config.yaml).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        duration_matrix: np.ndarray,
        needs_maintenance: np.ndarray | None = None,
        config: dict | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = load_config()

        self.config = config
        self.render_mode = render_mode
        env_cfg = config["environment"]
        reward_cfg = env_cfg["reward"]
        maint_cfg = config["maintenance"]

        self.duration_matrix = duration_matrix.astype(np.float32)
        self.n_total = duration_matrix.shape[0]     # inkl. Depot
        self.n_stations = self.n_total - 1          # nur Ladesäulen
        self.n_teams = maint_cfg["n_teams"]

        self.max_steps = env_cfg["max_steps"]
        self.visit_bonus = reward_cfg["visit_bonus"]
        self.travel_penalty = reward_cfg["travel_time_penalty"]
        self.revisit_penalty = reward_cfg["revisit_penalty"]
        self.completion_bonus = reward_cfg["completion_bonus"]

        self._needs_maintenance_init = needs_maintenance

        # --- Aktionsraum: jedes Team fährt zu einer der n_total Positionen ---
        self.action_space = spaces.MultiDiscrete([self.n_total] * self.n_teams)

        # --- Zustandsraum ---
        # [team_positions (n_teams, normiert), team_times (n_teams, normiert),
        #  needs_maintenance (binary), visited (binary)]
        obs_size = self.n_teams + self.n_teams + self.n_stations + self.n_stations
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Interne Zustände (werden in reset() initialisiert)
        self.team_positions: list[int] = [0] * self.n_teams   # 0 = Depot
        self.team_times: list[int] = [0] * self.n_teams       # Minuten ab 8:00
        self.needs_maintenance: np.ndarray = np.zeros(self.n_stations, dtype=bool)
        self.visited: np.ndarray = np.zeros(self.n_stations, dtype=bool)
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium-Interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.team_positions = [0] * self.n_teams   # alle Teams starten am Depot
        self.team_times = [0] * self.n_teams       # Tagesbeginn = 8:00
        self.step_count = 0
        self.visited = np.zeros(self.n_stations, dtype=bool)

        if self._needs_maintenance_init is not None:
            self.needs_maintenance = self._needs_maintenance_init.copy()
        else:
            failure_rate = self.config["maintenance"]["failure_rate"]
            self.needs_maintenance = self.np_random.random(self.n_stations) < failure_rate
            if not self.needs_maintenance.any():
                idx = self.np_random.integers(0, self.n_stations)
                self.needs_maintenance[idx] = True

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Ungültige Aktion: {action}"

        reward = 0.0

        for team_id in range(self.n_teams):
            destination = int(action[team_id])
            prev_pos = self.team_positions[team_id]

            # Fahrzeit in Minuten
            travel_seconds = float(self.duration_matrix[prev_pos, destination])
            travel_minutes = travel_seconds / 60.0
            reward -= self.travel_penalty * travel_minutes

            self.team_times[team_id] += int(travel_minutes)
            self.team_positions[team_id] = destination

            # Depot-Besuch zählt nicht als Wartung
            if destination > 0:
                station_idx = destination - 1
                if self.needs_maintenance[station_idx] and not self.visited[station_idx]:
                    reward += self.visit_bonus
                    self.visited[station_idx] = True
                elif self.visited[station_idx]:
                    reward += self.revisit_penalty

        self.step_count += 1

        all_done = bool(np.all(self.visited[self.needs_maintenance]))
        if all_done:
            reward += self.completion_bonus

        terminated = all_done
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            n_done = int(self.visited[self.needs_maintenance].sum())
            n_todo = int(self.needs_maintenance.sum())
            teams_str = ", ".join(
                f"Team{i}@{self.team_positions[i]}(t={self.team_times[i]}min)"
                for i in range(self.n_teams)
            )
            return (
                f"Schritt {self.step_count}/{self.max_steps} | "
                f"{teams_str} | "
                f"Gewartet: {n_done}/{n_todo}"
            )
        if self.render_mode == "human":
            print(self.render())
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def get_team_states(self) -> list[TeamState]:
        """Aktuellen Env-Zustand als TeamState-Liste für den VRPSolver."""
        return [
            TeamState(
                team_id=i,
                current_node=self.team_positions[i],
                current_time=self.team_times[i],
            )
            for i in range(self.n_teams)
        ]

    def _get_obs(self) -> np.ndarray:
        pos_norm = np.array(
            [p / max(self.n_total - 1, 1) for p in self.team_positions],
            dtype=np.float32,
        )
        time_norm = np.array(
            [t / (self.config["maintenance"]["workday_end_hour"] - self.config["maintenance"]["workday_start_hour"]) / 60
             for t in self.team_times],
            dtype=np.float32,
        )
        return np.concatenate([
            pos_norm,
            time_norm,
            self.needs_maintenance.astype(np.float32),
            self.visited.astype(np.float32),
        ])

    def _get_info(self) -> dict[str, Any]:
        return {
            "team_positions": list(self.team_positions),
            "team_times": list(self.team_times),
            "n_to_visit": int(self.needs_maintenance.sum()),
            "n_visited": int(self.visited[self.needs_maintenance].sum()),
            "step": self.step_count,
        }
