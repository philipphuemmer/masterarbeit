"""
CFA-DB Policy Training via Proximal Policy Optimization (PPO).

Lernt den state-abhängigen Balance-Parameter α_{S_t} ∈ (0,1) für die
CFA-DB Policy (OR-Tools Routing + α-gewichtete Drop-Entscheidung).

Architektur: identisch mit DB (MLP 8→16→16→1, Sigmoid)
Training:    identisch mit DB (PPO, γ=1, Advantage-Normierung)
Unterschied: Rollouts verwenden OR-Tools statt Greedy → bessere Basisrouten,
             aber langsamere Simulation pro Iteration.

Ausgabe: data/training/cfa_db/policy.json

Ausführen:
    python scripts/train/train_cfa_db.py
    python scripts/train/train_cfa_db.py --iterations 50 --rollouts 6 --max-days 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa_db import CFADBMaintenanceSimulator, CFADBModel
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import MaintenanceTask

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
GAMMA = 1.0


def _make_policy_net(n_feat: int = N_FEATURES, n_hidden: int = N_HIDDEN):
    return nn.Sequential(
        nn.Linear(n_feat, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1),
    )


def _make_value_net(n_feat: int = N_FEATURES, n_hidden: int = N_HIDDEN):
    return nn.Sequential(
        nn.Linear(n_feat, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1),
    )


class CFADBTrainingSimulator(CFADBMaintenanceSimulator):
    """
    Erweitert CFADBMaintenanceSimulator: sampelt α via PyTorch-Netz
    und zeichnet Trajektorien für PPO auf.
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
        self.trajectory: list[dict] = []

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        self.policy._n_remaining_total = len(remaining)
        self.policy._n_carryover = len(carryover_tasks)

        if self._failure_mode == "stochastic" and len(remaining) > 0:
            dummy_tasks = [
                MaintenanceTask(
                    node_idx=idx + 1,
                    task_type="routine",
                    service_time=30,
                    days_since_maintenance=float(self._days_since_maintenance[idx + 1]),
                )
                for idx in remaining
            ]
            phi = self.policy.extract_features(
                dummy_tasks,
                n_remaining_total=len(remaining),
                n_carryover=len(carryover_tasks),
            )
        else:
            phi = np.zeros(N_FEATURES, dtype=np.float64)

        std = np.where(self._feat_std > 1e-8, self._feat_std, 1.0)
        phi_norm = (phi - self._feat_mean) / std

        phi_t = torch.tensor(phi_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = self._policy_net(phi_t).squeeze()
        mu = torch.sigmoid(logit)
        dist = torch.distributions.Normal(mu, self._sigma)
        alpha_sample = dist.sample().clamp(0.01, 0.99)
        log_prob = dist.log_prob(alpha_sample).item()
        alpha = float(alpha_sample.item())

        self.policy._precomputed_alpha = alpha

        result, sim_routes, new_carryover = super(CFADBMaintenanceSimulator, self)._run_day(
            day, remaining, team_states, carryover_tasks, day_disruptions
        )

        self.trajectory.append({
            "phi": phi,
            "alpha": alpha,
            "log_prob": log_prob,
            "cost": -result.total_cost_eur,
        })

        return result, sim_routes, new_carryover


class RunningNorm:
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


def ppo_update(
    policy_net, value_net, policy_opt, value_opt,
    phis, alphas, old_log_probs, advantages, returns, sigma, n_epochs=4,
) -> tuple[float, float]:
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

        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * adv_t
        p_loss = -torch.min(surr1, surr2).mean()

        policy_opt.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        policy_opt.step()
        p_loss_last = float(p_loss.item())

        v_pred = value_net(phi_t).squeeze(-1)
        v_loss = ((v_pred - ret_t) ** 2).mean()
        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()
        v_loss_last = float(v_loss.item())

    return p_loss_last, v_loss_last


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


def main() -> None:
    if not _TORCH_AVAILABLE:
        print("FEHLER: PyTorch ist nicht installiert.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="CFA-DB Policy Training (PPO)")
    parser.add_argument("--iterations",  type=int,   default=50)
    parser.add_argument("--rollouts",    type=int,   default=6,
                        help="Rollouts pro Iteration (Standard: 6, OR-Tools ist langsamer als Greedy)")
    parser.add_argument("--max-days",    type=int,   default=365)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--sigma-start", type=float, default=0.5)
    parser.add_argument("--sigma-end",   type=float, default=0.05)
    parser.add_argument("--ppo-epochs",  type=int,   default=4)
    parser.add_argument("--output",      type=str,   default="data/training/cfa_db/policy.json")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Pfad zu train_state.pt zum Fortsetzen")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print("WARNUNG: failure_simulation.mode ist nicht 'stochastic' → triviale Features.")

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

    policy_net = _make_policy_net()
    value_net = _make_value_net()
    policy_opt = optim.Adam(policy_net.parameters(), lr=args.lr)
    value_opt = optim.Adam(value_net.parameters(), lr=args.lr)
    norm = RunningNorm(N_FEATURES)
    start_iteration = 0

    sigma_schedule = np.linspace(args.sigma_start, args.sigma_end, args.iterations)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        state_path = Path(args.resume)
        if not state_path.exists():
            print(f"FEHLER: Resume-Datei nicht gefunden: {state_path}")
            sys.exit(1)
        state = torch.load(state_path, map_location="cpu")
        policy_net.load_state_dict(state["policy_net"])
        value_net.load_state_dict(state["value_net"])
        policy_opt.load_state_dict(state["policy_opt"])
        value_opt.load_state_dict(state["value_opt"])
        norm.count = state["norm_count"]
        norm.mean  = state["norm_mean"]
        norm.M2    = state["norm_M2"]
        start_iteration = state["next_iteration"]
        print(f"Resume: fortgesetzt ab Iteration {start_iteration + 1}")

    log_path = out_path.parent / "training_log.csv"
    if not args.resume or not log_path.exists():
        with open(log_path, "w") as f:
            f.write("iteration,sigma,mean_daily_cost,alpha_mean,p_loss,v_loss,elapsed_s\n")

    print(f"\nStarte CFA-DB PPO-Training ({args.iterations} Iterationen, {args.rollouts} Rollouts/Iter.)...\n")
    t0 = time.time()

    for iteration in range(start_iteration, args.iterations):
        sigma = float(sigma_schedule[iteration])
        all_phi, all_alpha, all_log_prob, all_cost = [], [], [], []
        all_ep_len, ep_costs = [], []

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

            policy = CFADBModel(
                mats, run_cfg,
                all_coords=coords,
                node_to_power=node_to_power,
                n_stations=len(df_base),
                weights_override=weights,
            )

            if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
                selector.value_fn = policy._station_value

            sim = CFADBTrainingSimulator(
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

            for rec in sim.trajectory:
                norm.update(rec["phi"])
                all_phi.append(rec["phi"])
                all_alpha.append(rec["alpha"])
                all_log_prob.append(rec["log_prob"])
                all_cost.append(rec["cost"])

        if not all_phi:
            print(f"Iter {iteration + 1:>3d}: keine Trajektorien — übersprungen.")
            continue

        returns: list[float] = []
        for ep_idx, ep_len in enumerate(all_ep_len):
            costs = ep_costs[ep_idx]
            G = np.cumsum(costs[::-1])[::-1]
            returns.extend(G.tolist())

        std = np.where(norm.std > 1e-8, norm.std, 1.0)
        phis_norm = (np.array(all_phi) - norm.mean) / std

        phi_t = torch.tensor(phis_norm, dtype=torch.float32)
        with torch.no_grad():
            baseline = value_net(phi_t).squeeze(-1).numpy()
        ret_arr = np.array(returns)
        advantages = ret_arr - baseline
        adv_mean, adv_std = advantages.mean(), advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std

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
        mean_cost = -float(np.mean(all_cost))
        alpha_mean_iter = float(np.mean(all_alpha))
        print(
            f"Iter {iteration + 1:>3d}/{args.iterations}  "
            f"σ={sigma:.3f}  "
            f"mean_daily_cost={mean_cost:>10,.0f} €  "
            f"α={alpha_mean_iter:.3f}  "
            f"p_loss={p_loss:.4f}  v_loss={v_loss:.4f}  "
            f"elapsed={elapsed:.0f}s"
        )

        weights = _extract_weights(policy_net)
        weights["feature_mean"] = norm.mean.tolist()
        weights["feature_std"] = norm.std.tolist()
        weights["alpha_mean"] = alpha_mean_iter
        weights["n_iterations"] = iteration + 1
        weights["sigma_final"] = sigma

        ckpt_dir = out_path.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(weights, f, indent=2)
        with open(ckpt_dir / f"iter_{iteration + 1:03d}.json", "w") as f:
            json.dump(weights, f, indent=2)
        with open(log_path, "a") as f:
            f.write(f"{iteration + 1},{sigma:.4f},{mean_cost:.2f},{alpha_mean_iter:.4f},"
                    f"{p_loss:.6f},{v_loss:.6f},{elapsed:.0f}\n")

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

    print(f"\nPolicy gespeichert: {out_path.resolve()}")
    print(f"Trainingszeit: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
