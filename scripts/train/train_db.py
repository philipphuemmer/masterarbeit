"""
DB-Policy Training via Proximal Policy Optimization (PPO).

Lernt den state-abhängigen Balance-Parameter α_{S_t} ∈ (0,1) für die
Dynamic Balance Greedy-Insertion-Policy (Stein et al. 2024).

Architektur (Stein Configuration 4):
    Policy: MLP 8 → 16 → 16 → 1, sigmoid output
    Baseline: MLP 8 → 16 → 1, keine Aktivierung (linearer Wert-Prädiktor)
    σ (Explorations-Std): startet bei σ_start, fällt linear auf σ_end
    PPO clipped objective (ε=0.2), kein Value-Clipping
    Observation Normalization: laufende Mean/Std über alle gesehenen φ(S)

Training:
    pro Iteration: N_ROLLOUTS vollständige Jahr-Simulationen
    state : φ(S_t) — 8 Merkmale (vor Tagesplanung)
    action: α_t ~ clip(Normal(μ(φ), σ), 0, 1) — für Exploration
    reward: −tägliche Gesamtkosten (Betriebs- + Ausfallkosten)
    G_t   : cost-to-go (undiskontiert, γ=1)
    A_t   : G_t − V(φ_t) — normiert

Ausgabe: data/training/db/policy.json mit Netzgewichten + Feature-Normierung

Voraussetzung: failure_simulation.mode = stochastic in configs/config.yaml
    (sonst kein dsm-Tracking → triviale Features)

Ausführen:
    python scripts/train/train_db.py
    python scripts/train/train_db.py --iterations 50 --rollouts 10 --max-days 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.db import DBMaintenanceSimulator, DBModel
from src.models.simulator import MaintenanceSimulator
from src.models.myopic import MyopicPolicy
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

N_FEATURES = 8
N_HIDDEN = 16
PPO_EPSILON = 0.2
GAMMA = 1.0  # undiskontiert (kurzfristig vollständige Episode)


# ---------------------------------------------------------------------------
# Netzwerkdefinitionen (PyTorch)
# ---------------------------------------------------------------------------

def _make_policy_net(n_feat: int = N_FEATURES, n_hidden: int = N_HIDDEN):
    """MLP 8 → 16 → 16 → 1 mit Sigmoid-Ausgabe (μ der Gaussverteilung)."""
    return nn.Sequential(
        nn.Linear(n_feat, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1),
    )


def _make_value_net(n_feat: int = N_FEATURES, n_hidden: int = N_HIDDEN):
    """Lineare Baseline: 8 → 16 → 1."""
    return nn.Sequential(
        nn.Linear(n_feat, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1),
    )


# ---------------------------------------------------------------------------
# Trainings-Simulator: zeichnet pro Tag (φ, α, log_prob, cost) auf
# ---------------------------------------------------------------------------

class DBTrainingSimulator(DBMaintenanceSimulator):
    """
    Erweitert DBMaintenanceSimulator: ersetzt die deterministischen Policy-
    Gewichte durch ein PyTorch-Netz und zeichnet Trajektorien für PPO auf.

    Parameters
    ----------
    policy_net : torch.nn.Module
        Policy-MLP (gibt logit für α aus).
    sigma : float
        Aktuelle Explorations-Standardabweichung.
    feat_mean / feat_std : np.ndarray
        Laufende Feature-Normierung.
    """

    def __init__(
        self,
        policy_net,
        sigma: float,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._policy_net = policy_net
        self._sigma = sigma
        self._feat_mean = feat_mean
        self._feat_std = feat_std
        self.trajectory: list[dict] = []  # [(phi, alpha, log_prob, cost)]

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        # Kontext setzen (wie DBMaintenanceSimulator)
        self.policy._n_remaining_total = len(remaining)
        self.policy._n_carryover = len(carryover_tasks)

        # Features extrahieren (vor Tagesplanung, mit aktuellen dsm-Werten)
        if self._failure_mode == "stochastic":
            remaining_nodes = [idx + 1 for idx in remaining]
            dummy_tasks = []
            for node in remaining_nodes:
                from src.planning.vrp_solver import MaintenanceTask
                dummy_tasks.append(MaintenanceTask(
                    node_idx=node,
                    task_type="routine",
                    service_time=30,
                    days_since_maintenance=float(self._days_since_maintenance[node]),
                ))
            phi = self.policy.extract_features(
                dummy_tasks,
                n_remaining_total=len(remaining),
                n_carryover=len(carryover_tasks),
            )
        else:
            phi = np.zeros(N_FEATURES, dtype=np.float64)

        # Normieren
        std = np.where(self._feat_std > 1e-8, self._feat_std, 1.0)
        phi_norm = (phi - self._feat_mean) / std

        # α sampeln via Policy-Netz + Gauss-Exploration
        phi_t = torch.tensor(phi_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = self._policy_net(phi_t).squeeze()
        mu = torch.sigmoid(logit)
        dist = torch.distributions.Normal(mu, self._sigma)
        alpha_sample = dist.sample().clamp(0.01, 0.99)
        log_prob = dist.log_prob(alpha_sample).item()
        alpha = float(alpha_sample.item())

        # Gesampletes α für create_initial_plan setzen (konsistent mit Inference-Pfad)
        self.policy._precomputed_alpha = alpha

        result, sim_routes, new_carryover = super(DBMaintenanceSimulator, self)._run_day(
            day, remaining, team_states, carryover_tasks, day_disruptions
        )

        self.trajectory.append({
            "phi": phi,
            "alpha": alpha,
            "log_prob": log_prob,
            "cost": -result.total_cost_eur,  # negiert: PPO maximiert Reward = -Kosten
        })

        return result, sim_routes, new_carryover


# ---------------------------------------------------------------------------
# Laufende Feature-Normierung
# ---------------------------------------------------------------------------

class RunningNorm:
    """Online-Berechnung von Mean und Std (Welford)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self.M2 = np.zeros(n, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.n, dtype=np.float64)
        return np.sqrt(self.M2 / (self.count - 1))


# ---------------------------------------------------------------------------
# PPO-Update
# ---------------------------------------------------------------------------

def ppo_update(
    policy_net,
    value_net,
    policy_opt: "optim.Optimizer",
    value_opt: "optim.Optimizer",
    phis: np.ndarray,         # (T, 8) normiert
    alphas: np.ndarray,       # (T,)
    old_log_probs: np.ndarray, # (T,)
    advantages: np.ndarray,    # (T,) normiert
    returns: np.ndarray,       # (T,) für Value-Update
    sigma: float,
    n_epochs: int = 4,
) -> tuple[float, float]:
    """PPO clipped surrogate update. Gibt (policy_loss, value_loss) zurück."""
    phi_t = torch.tensor(phis, dtype=torch.float32)
    alpha_t = torch.tensor(alphas, dtype=torch.float32)
    old_lp = torch.tensor(old_log_probs, dtype=torch.float32)
    adv_t = torch.tensor(advantages, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)

    p_loss_last = v_loss_last = 0.0

    for _ in range(n_epochs):
        logits = policy_net(phi_t).squeeze(-1)
        mu = torch.sigmoid(logits)
        dist = torch.distributions.Normal(mu, sigma)
        new_lp = dist.log_prob(alpha_t)
        ratio = torch.exp(new_lp - old_lp)

        # PPO clipped objective (negativ, weil wir minimieren)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * adv_t
        p_loss = -torch.min(surr1, surr2).mean()

        policy_opt.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        policy_opt.step()
        p_loss_last = float(p_loss.item())

        # Value-Update: MSE, kein Clipping (Stein Config 4)
        v_pred = value_net(phi_t).squeeze(-1)
        v_loss = ((v_pred - ret_t) ** 2).mean()
        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()
        v_loss_last = float(v_loss.item())

    return p_loss_last, v_loss_last


# ---------------------------------------------------------------------------
# Numpy-Extraktion der Netzgewichte (für policy.json)
# ---------------------------------------------------------------------------

def _extract_weights(policy_net) -> dict:
    params = list(policy_net.parameters())
    return {
        "W1": params[0].detach().numpy().tolist(),
        "b1": params[1].detach().numpy().tolist(),
        "W2": params[2].detach().numpy().tolist(),
        "b2": params[3].detach().numpy().tolist(),
        "W3": params[4].detach().numpy().tolist(),
        "b3": params[5].detach().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Hilfsfunktion: Myopic-Bootstrap-Rollout
# ---------------------------------------------------------------------------

def _bootstrap_rollout(
    df_base, coords, mats, cfg, mal_df, max_days: int, seed: int
) -> "MaintenanceSimulator":
    """Führt einen Myopic-Rollout als Warm-Start durch."""
    run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}
    clusterer = ZoneClusterer(
        n_zones=run_cfg["planning"]["n_zones"],
        random_state=seed,
    )
    clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))
    charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
    policy = MyopicPolicy(mats, run_cfg, all_coords=coords, stations_df=df_base)
    sim = MaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)
    sim.run(mal_df, max_days=max_days)
    return sim


# ---------------------------------------------------------------------------
# Haupt-Trainingsschleife
# ---------------------------------------------------------------------------

def main() -> None:
    if not _TORCH_AVAILABLE:
        print("FEHLER: PyTorch ist nicht installiert. Bitte 'pip install torch' ausführen.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="DB-Policy Training (PPO)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Anzahl PPO-Iterationen (Standard: 50)")
    parser.add_argument("--rollouts",   type=int, default=8,
                        help="Simulationsläufe pro Iteration (Standard: 8)")
    parser.add_argument("--max-days",   type=int, default=365,
                        help="Max. Simulationstage pro Rollout (Standard: 365)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Lernrate (Standard: 3e-4)")
    parser.add_argument("--sigma-start", type=float, default=0.5,
                        help="Initiale Explorations-Std (Standard: 0.5)")
    parser.add_argument("--sigma-end",   type=float, default=0.05,
                        help="Finale Explorations-Std (Standard: 0.05)")
    parser.add_argument("--ppo-epochs",  type=int, default=4,
                        help="PPO-Update-Epochen pro Iteration (Standard: 4)")
    parser.add_argument("--output",      type=str, default="data/training/db/policy.json",
                        help="Ausgabepfad policy.json (Standard: data/training/db/policy.json)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Ausführliche Ausgabe")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Pfad zu train_state.pt zum Fortsetzen (z.B. data/training/db/checkpoints/train_state.pt)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print(
            "WARNUNG: failure_simulation.mode ist nicht 'stochastic'.\n"
            "         dsm-Tracking ist deaktiviert → Features sind trivial.\n"
            "         Empfehlung: mode: stochastic in config.yaml setzen."
        )

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords = np.array(get_coordinates(df_base, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print("  Störungsmodus: stochastisch")

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df_base.iterrows())
    }

    # Netzwerke initialisieren
    policy_net = _make_policy_net()
    value_net = _make_value_net()
    policy_opt = optim.Adam(policy_net.parameters(), lr=args.lr)
    value_opt = optim.Adam(value_net.parameters(), lr=args.lr)

    norm = RunningNorm(N_FEATURES)
    start_iteration = 0

    # σ-Zeitplan (lineare Decay) — über alle args.iterations, unabhängig vom Resume-Punkt
    sigma_schedule = np.linspace(args.sigma_start, args.sigma_end, args.iterations)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        state_path = Path(args.resume)
        if not state_path.exists():
            print(f"FEHLER: Resume-Datei nicht gefunden: {state_path}")
            sys.exit(1)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        policy_net.load_state_dict(state["policy_net"])
        value_net.load_state_dict(state["value_net"])
        policy_opt.load_state_dict(state["policy_opt"])
        value_opt.load_state_dict(state["value_opt"])
        norm.count = state["norm_count"]
        norm.mean  = state["norm_mean"]
        norm.M2    = state["norm_M2"]
        start_iteration = state["next_iteration"]
        print(f"Resume: fortgesetzt ab Iteration {start_iteration + 1} (aus {state_path})")

    log_path = out_path.parent / "training_log.csv"
    if not args.resume or not log_path.exists():
        with open(log_path, "w") as f:
            f.write("iteration,sigma,mean_daily_cost,alpha_mean,p_loss,v_loss,elapsed_s\n")

    print(f"\nStarte PPO-Training ({args.iterations} Iterationen, {args.rollouts} Rollouts/Iter.)...\n")
    t0 = time.time()

    for iteration in range(start_iteration, args.iterations):
        sigma = float(sigma_schedule[iteration])
        all_phi: list[np.ndarray] = []
        all_alpha: list[float] = []
        all_log_prob: list[float] = []
        all_cost: list[float] = []
        all_day: list[int] = []  # Tag innerhalb der Episode (für cost-to-go)
        all_ep_len: list[int] = []  # Länge der Episode
        ep_costs: list[list[float]] = []  # pro Episode: tägliche Kosten

        # Aktuelle Gewichte → DBModel (numpy)
        weights = _extract_weights(policy_net)
        weights["feature_mean"] = norm.mean.tolist()
        weights["feature_std"] = norm.std.tolist()
        weights["alpha_mean"] = float(args.sigma_start)

        for rollout_idx in range(args.rollouts):
            seed = iteration * args.rollouts + rollout_idx + 1
            run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

            clusterer = ZoneClusterer(
                n_zones=run_cfg["planning"]["n_zones"],
                random_state=seed,
            )
            clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))
            charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
            selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)

            policy = DBModel(
                mats, run_cfg,
                all_coords=coords,
                node_to_power=node_to_power,
                n_stations=len(df_base),
                weights_override=weights,
                stations_df=df_base,
            )

            sim = DBTrainingSimulator(
                policy_net=policy_net,
                sigma=sigma,
                feat_mean=norm.mean.copy(),
                feat_std=norm.std.copy(),
                policy=policy,
                selector=selector,
                all_coords=coords,
                stations_df=df_base,
                traffic_matrices=mats,
                config=run_cfg,
            )

            sim.run(mal_df, max_days=args.max_days)

            ep_len = len(sim.trajectory)
            ep_daily_costs = [rec["cost"] for rec in sim.trajectory]
            ep_costs.append(ep_daily_costs)
            all_ep_len.append(ep_len)

            for t_idx, rec in enumerate(sim.trajectory):
                phi = rec["phi"]
                norm.update(phi)
                all_phi.append(phi)
                all_alpha.append(rec["alpha"])
                all_log_prob.append(rec["log_prob"])
                all_cost.append(rec["cost"])
                all_day.append(t_idx)

        if not all_phi:
            print(f"Iter {iteration + 1:>3d}: keine Trajektorien — übersprungen.")
            continue

        # Cost-to-go G_t = Σ_{t'≥t} cost(t') pro Episode
        returns: list[float] = []
        ep_start = 0
        for ep_idx, ep_len in enumerate(all_ep_len):
            costs = ep_costs[ep_idx]
            G = np.cumsum(costs[::-1])[::-1]  # rückwärtige Kumulation
            returns.extend(G.tolist())
            ep_start += ep_len

        # Normierte Features
        std = np.where(norm.std > 1e-8, norm.std, 1.0)
        phis_norm = (np.array(all_phi) - norm.mean) / std

        # Advantages: A_t = G_t − V(φ_t)
        phi_t = torch.tensor(phis_norm, dtype=torch.float32)
        with torch.no_grad():
            baseline = value_net(phi_t).squeeze(-1).numpy()
        ret_arr = np.array(returns)
        advantages = ret_arr - baseline
        # Normieren
        adv_mean, adv_std = advantages.mean(), advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std

        # Skalierung: Returns auf [-1, 0] normieren (ret_arr ist negativ)
        cost_scale = max(1.0, float(np.abs(ret_arr).max()))
        ret_scaled = ret_arr / cost_scale

        p_loss, v_loss = ppo_update(
            policy_net=policy_net,
            value_net=value_net,
            policy_opt=policy_opt,
            value_opt=value_opt,
            phis=phis_norm,
            alphas=np.array(all_alpha),
            old_log_probs=np.array(all_log_prob),
            advantages=advantages,
            returns=ret_scaled,
            sigma=sigma,
            n_epochs=args.ppo_epochs,
        )

        elapsed = time.time() - t0
        mean_cost = -float(np.mean(all_cost))  # negieren: reward → kosten
        alpha_mean_iter = float(np.mean(all_alpha))
        print(
            f"Iter {iteration + 1:>3d}/{args.iterations}  "
            f"σ={sigma:.3f}  "
            f"mean_daily_cost={mean_cost:>10,.0f} €  "
            f"α={alpha_mean_iter:.3f}  "
            f"p_loss={p_loss:.4f}  v_loss={v_loss:.4f}  "
            f"elapsed={elapsed:.0f}s"
        )
        with open(log_path, "a") as f:
            f.write(f"{iteration + 1},{sigma:.4f},{mean_cost:.2f},{alpha_mean_iter:.4f},"
                    f"{p_loss:.6f},{v_loss:.6f},{elapsed:.0f}\n")

        # Checkpoint nach jeder Iteration
        weights = _extract_weights(policy_net)
        weights["feature_mean"] = norm.mean.tolist()
        weights["feature_std"] = norm.std.tolist()
        weights["alpha_mean"] = float(np.mean(all_alpha))
        weights["n_iterations"] = iteration + 1
        weights["sigma_final"] = sigma

        ckpt_dir = out_path.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / f"iter_{iteration + 1:03d}.json", "w") as f:
            json.dump(weights, f, indent=2)

        with open(out_path, "w") as f:
            json.dump(weights, f, indent=2)

        torch.save(
            {
                "policy_net": policy_net.state_dict(),
                "value_net":  value_net.state_dict(),
                "policy_opt": policy_opt.state_dict(),
                "value_opt":  value_opt.state_dict(),
                "norm_count": norm.count,
                "norm_mean":  norm.mean.copy(),
                "norm_M2":    norm.M2.copy(),
                "next_iteration": iteration + 1,
            },
            ckpt_dir / "train_state.pt",
        )

    # Finale Policy speichern
    final_weights = _extract_weights(policy_net)
    final_weights["feature_mean"] = norm.mean.tolist()
    final_weights["feature_std"] = norm.std.tolist()
    final_weights["alpha_mean"] = float(
        np.mean(all_alpha) if "all_alpha" in dir() and all_alpha else 0.3
    )
    final_weights["n_iterations"] = args.iterations

    with open(out_path, "w") as f:
        json.dump(final_weights, f, indent=2)

    print(f"\nPolicy gespeichert: {out_path.resolve()}")
    print(f"Trainingszeit: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
