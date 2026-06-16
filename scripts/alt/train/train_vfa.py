"""
VFA-Training: Lernt θ_global der globalen Zustandswertfunktion V̂(s).

Architektur:
    V̂_global(s) ≈ θ_global^T × φ_state(s) + intercept

θ_local (lokaler CFA-Future-Term) wird direkt aus data/training/cfa_future/theta.json
geladen und nicht neu trainiert. train_vfa.py lernt ausschließlich θ_global.

Training (iteratives Policy Evaluation):
    Runde 1: N Simulationen mit CFAFutureModel (Bootstrap)
    Runde 2: N Simulationen mit VFAModel(θ_global aus Runde 1)
    Runde r: N Simulationen mit VFAModel(θ_global_{r-1})

    Pro Tag: φ_state(s) vor Tagesplanung + G_t = Σ_{t'≥t} γ^(t'-t) × cost(t')
    Ridge-Regression: G_t ≈ θ_global^T × φ_state(s_t) + intercept

Ausführen:
    python scripts/train/train_vfa.py
    python scripts/train/train_vfa.py --runs 20 --rounds 3 --max-days 200 --gamma 0.995
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
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa_future import CFAFutureModel
from src.models.vfa import VFAModel, N_STATE_FEATURES, STATE_FEATURE_NAMES
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer, _approx_km
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import MaintenanceTask


# ---------------------------------------------------------------------------
# Trainings-Simulator: zeichnet globalen State-Vektor pro Tag auf
# ---------------------------------------------------------------------------

class VFATrainingSimulator(MaintenanceSimulator):
    """
    Erweitert MaintenanceSimulator um Aufzeichnung des täglichen φ_state(s)
    vor der Tagesplanung (Teamfeatures: Tagesstart-Defaults).
    """

    def __init__(self, gamma: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.training_records: list[dict] = []

        fail_cfg = self.config.get("failure_simulation", {})
        self._lambda_per_day: float = (
            fail_cfg.get("p1_per_hour", 0.00084)
            + fail_cfg.get("p2_per_hour", 0.00028)
        ) * 24.0
        self._recovery_days: float = float(fail_cfg.get("recovery_days", 365))
        self._initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))
        maint = self.config.get("maintenance", {})
        self._workday_minutes: int = (
            maint.get("workday_end_hour", 16) - maint.get("workday_start_hour", 8)
        ) * 60

    def _run_day(
        self,
        day: int,
        remaining: set,
        team_states,
        carryover_tasks: list,
        day_disruptions: list,
    ):
        remaining_nodes = sorted([idx + 1 for idx in remaining])
        n_carryover     = len(carryover_tasks)

        if remaining_nodes:
            dsm_vals = np.array(
                [float(self._days_since_maintenance[n]) for n in remaining_nodes],
                dtype=np.float64,
            )
            pow_vals = np.array(
                [self.node_to_power.get(n, 22.0) for n in remaining_nodes],
                dtype=np.float64,
            )
            urgency      = pow_vals * dsm_vals
            failure_risk = 1.0 - np.exp(-self._lambda_per_day * dsm_vals)

            f2 = float(np.mean(urgency))
            f3 = float(np.mean(failure_risk * pow_vals))
            f4 = float(np.mean(dsm_vals))
            f5 = float(np.max(urgency))
            f6 = float(np.mean(dsm_vals > 90))
            f9 = float(np.std(urgency) / max(float(np.mean(urgency)), 1e-8))

            depot = self.all_coords[0]
            dists = np.array(
                [_approx_km(self.all_coords[n], depot) for n in remaining_nodes],
                dtype=np.float64,
            )
            f7 = float(np.mean(dists))
            f8 = float(np.std(dists)) if len(dists) > 1 else 0.0
        else:
            f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = 0.0

        f0 = len(remaining_nodes) / max(1, self.n_stations)
        f1 = n_carryover / 10.0

        phi = np.array(
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
            dtype=np.float64,
        )
        self.training_records.append({"day": day, "phi": phi})
        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)


# ---------------------------------------------------------------------------
# Ridge-Regression
# ---------------------------------------------------------------------------

def fit_theta(
    Phi: np.ndarray,
    y: np.ndarray,
    alpha_ridge: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """
    Ridge-Regression: G_t ≈ θ_global^T × φ_state(s_t) + intercept.

    Returns (theta, intercept, r2).
    """
    reg = Ridge(alpha=alpha_ridge, fit_intercept=True)
    reg.fit(Phi, y)

    y_pred = reg.predict(Phi)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return np.array(reg.coef_, dtype=np.float64), float(reg.intercept_), r2


# ---------------------------------------------------------------------------
# Eine Trainingsrunde
# ---------------------------------------------------------------------------

def run_round(
    round_idx: int,
    cfg: dict,
    coords: np.ndarray,
    df_base: pd.DataFrame,
    mats: dict,
    seeds: list[int],
    max_days: int,
    gamma: float,
    theta_global_prev: np.ndarray | None,
    intercept_prev: float,
    local_theta_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Führt N Simulationen durch und gibt Trainingsdaten (Phi, y) zurück.

    round_idx == 1 → CFAFutureModel als Bootstrap-Policy
    round_idx >= 2 → VFAModel mit theta_global aus Runde r-1
    """
    Phi_all: list[np.ndarray] = []
    y_all:   list[float]      = []

    use_value_based = cfg["planning"].get("zone_selection_mode", "classic") == "value_based"

    label = "CFA-Future (Bootstrap)" if round_idx == 1 else f"VFA (θ_global Runde {round_idx - 1})"
    print(f"\n  Runde {round_idx}: {label}\n")

    for run_i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"    Lauf {run_i}/{len(seeds)} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(n_zones=run_cfg["planning"]["n_zones"], random_state=seed)
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)

        if round_idx == 1:
            policy = CFAFutureModel(
                mats, run_cfg,
                all_coords=coords,
                stations_df=df_base,
                theta_path=str(local_theta_path),
            )
            if use_value_based:
                selector.value_fn = policy._value
        else:
            policy = VFAModel(
                mats, run_cfg,
                all_coords=coords,
                stations_df=df_base,
                local_theta_path=str(local_theta_path),
                global_theta_override=theta_global_prev,
                global_intercept_override=intercept_prev,
            )
            if use_value_based:
                selector.value_fn = policy._local_value

        sim = VFATrainingSimulator(gamma, policy, selector, coords, df_base, mats, run_cfg)
        result = sim.run(max_days=max_days)

        day_costs = [dr.total_cost_eur for dr in result.day_results]
        n_days    = len(sim.training_records)

        for i in range(n_days):
            phi = sim.training_records[i]["phi"]
            # Diskontiertes cost-to-go
            g = 0.0
            for j, c in enumerate(day_costs[i:]):
                g += (gamma ** j) * c
            Phi_all.append(phi)
            y_all.append(g)

        elapsed = time.time() - t0
        days    = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €, {elapsed:.0f}s)")

    return np.stack(Phi_all), np.array(y_all)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VFA-Training: globaler Zustandswert θ_global via Ridge-Regression"
    )
    parser.add_argument("--runs",         type=int,   default=20,
                        help="Anzahl Trainingsläufe pro Runde (Standard: 20)")
    parser.add_argument("--rounds",       type=int,   default=3,
                        help="Anzahl Iterationsrunden (Standard: 3)")
    parser.add_argument("--max-days",     type=int,   default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--gamma",        type=float, default=0.995,
                        help="Diskontfaktor für cost-to-go (Standard: 0.995)")
    parser.add_argument("--ridge-alpha",  type=float, default=1.0,
                        help="Ridge-Regularisierungsstärke (Standard: 1.0)")
    parser.add_argument("--local-theta",  type=str,
                        default="data/training/cfa_future/theta.json",
                        help="Pfad zu θ_local (cfa_future)")
    parser.add_argument("--out",          type=str,
                        default="data/training/vfa/theta.json",
                        help="Ausgabepfad für θ_global")
    parser.add_argument("--fresh",        action="store_true",
                        help="Checkpoints ignorieren und von Runde 1 neu starten")
    parser.add_argument("--verbose",      action="store_true",
                        help="Logging aktivieren")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print("FEHLER: VFA-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

    local_theta_path = Path(args.local_theta)
    if not local_theta_path.exists():
        print(f"FEHLER: θ_local nicht gefunden: {local_theta_path}")
        print("Bitte zuerst 'python scripts/train/train_cfa_future.py' ausführen.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints_vfa"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")
    print(f"  γ = {args.gamma}  |  Ridge α = {args.ridge_alpha}")
    print(f"  θ_local: {local_theta_path}")

    seeds = list(range(1, args.runs + 1))
    t0_total = time.time()

    theta_global: np.ndarray | None = None
    intercept_val: float = 0.0
    history: list[dict] = []
    start_round = 1
    Phi_last: np.ndarray | None = None
    y_last:   np.ndarray | None = None

    existing = sorted(ckpt_dir.glob("round_*.json")) if not args.fresh else []
    if existing:
        latest = existing[-1]
        with open(latest) as f:
            ckpt = json.load(f)
        theta_global  = np.array(ckpt["theta"], dtype=np.float64)
        intercept_val = float(ckpt["intercept"])
        history       = ckpt.get("history", [])
        start_round   = ckpt["round"] + 1
        print(f"  Checkpoint: {latest.name}  (Runde {ckpt['round']} abgeschlossen)")
        if start_round > args.rounds:
            print(f"  Alle {args.rounds} Runden abgeschlossen. Nichts zu tun.")
            sys.exit(0)
    elif args.fresh:
        print("  --fresh: starte von Runde 1 neu.")

    print(f"\nIteratives VFA-Training: Runde {start_round}–{args.rounds}, {args.runs} Läufe/Runde")

    for round_idx in range(start_round, args.rounds + 1):
        Phi, y = run_round(
            round_idx, cfg, coords, df_base, mats, seeds, args.max_days,
            args.gamma, theta_global, intercept_val, local_theta_path,
        )

        print(f"\n  Ridge-Regression auf {len(y)} Datenpunkten...")
        theta_new, intercept_new, r2 = fit_theta(Phi, y, alpha_ridge=args.ridge_alpha)

        print(f"\n  Gelernte Gewichte θ_global (Runde {round_idx}):")
        for name, w_new in zip(STATE_FEATURE_NAMES, theta_new):
            if theta_global is not None:
                delta = w_new - theta_global[STATE_FEATURE_NAMES.index(name)]
                print(f"    {name:<26} = {w_new:+.4e}  (Δ = {delta:+.2e})")
            else:
                print(f"    {name:<26} = {w_new:+.4e}")
        print(f"    {'intercept':<26} = {intercept_new:+.4e}")
        print(f"    R²                         = {r2:.4f}")

        history.append({
            "round":     round_idx,
            "theta":     theta_new.tolist(),
            "intercept": intercept_new,
            "r2":        r2,
        })
        theta_global  = theta_new
        intercept_val = intercept_new
        Phi_last = Phi
        y_last   = y

        ckpt_path = ckpt_dir / f"round_{round_idx}.json"
        with open(ckpt_path, "w") as f:
            json.dump({
                "round": round_idx, "theta": theta_global.tolist(),
                "intercept": intercept_val, "r2": r2, "history": history,
            }, f, indent=2)
        print(f"    Checkpoint: {ckpt_path}")

    print(f"\n  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    if len(history) > 1:
        print("\n  R²-Verlauf:")
        for h in history:
            print(f"    Runde {h['round']}: R² = {h['r2']:.4f}")

    payload = {
        "theta":             theta_global.tolist(),
        "intercept":         intercept_val,
        "feature_names":     STATE_FEATURE_NAMES,
        "r2":                history[-1]["r2"],
        "n_runs":            args.runs,
        "n_rounds":          args.rounds,
        "n_datapoints":      int(len(y_last)),
        "gamma":             args.gamma,
        "ridge_alpha":       args.ridge_alpha,
        "history":           history,
        "feature_means":     Phi_last.mean(axis=0).tolist(),
        "feature_stds":      Phi_last.std(axis=0).tolist(),
        "cost_to_go_mean":   float(np.mean(y_last)),
        "cost_to_go_std":    float(np.std(y_last)),
        "local_theta_path":  str(local_theta_path),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ_global gespeichert: {out_path}")


if __name__ == "__main__":
    main()
