"""
CFA-Training: Lernt den Gewichtsvektor θ der Wertfunktionsapproximation.

Wertfunktion:
    V(s) ≈ θ × Σ_k power_kW[k] × days_since_maintenance[k]

Training (iteratives Policy Iteration):
    Runde 1: N Myopic-Simulationen → θ₁  (Bootstrap)
    Runde 2: N CFA(θ₁)-Simulationen → θ₂
    Runde r: N CFA(θ_{r-1})-Simulationen → θ_r

    Pro Tag / Runde:
        Feature = Σ_k power_kW[k] × dsm[k]  (alle noch offenen Stationen)
        Target  = tatsächliche Restkosten G_t ab diesem Tag (unter aktueller Policy)
    OLS-Regression: G_t ≈ θ × Feature + intercept

Ausführen:
    python scripts/train_cfa.py
    python scripts/train_cfa.py --runs 30 --rounds 3 --max-days 200
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
from src.models.cfa import CFAModel
from src.models.myopic import MyopicPolicy
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


# ---------------------------------------------------------------------------
# Trainings-Simulator: zeichnet Feature pro Tag auf
# ---------------------------------------------------------------------------

class CFATrainingSimulator(MaintenanceSimulator):
    """
    Erweitert MaintenanceSimulator um Aufzeichnung der täglichen Feature-Werte
    für die CFA-Regression.

    Pro Tag wird vor der Planung berechnet:
        feature = Σ_{k ∈ remaining} power_kW[k] × days_since_maintenance[k]
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_records: list[dict] = []

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        # Feature vor dem Tageslauf erfassen
        feature = sum(
            self.node_to_power.get(idx + 1, 22.0)
            * float(self._days_since_maintenance[idx + 1])
            for idx in remaining
        )
        self.training_records.append({"day": day, "feature": feature})
        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def fit_theta(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    OLS-Regression: cost_to_go ≈ θ × feature + intercept.

    Returns
    -------
    (theta, intercept, r2)
    """
    A = np.column_stack([X, np.ones(len(X))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    theta_val     = float(coeffs[0])
    intercept_val = float(coeffs[1])

    y_pred = theta_val * X + intercept_val
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return theta_val, intercept_val, r2


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
    theta_prev: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Führt N Simulationen durch und gibt Trainingsdaten (X, y) zurück.

    round_idx == 1  → Myopic-Policy (Bootstrap)
    round_idx >= 2  → CFA-Policy mit theta_prev
    """
    X_all: list[float] = []
    y_all: list[float] = []
    use_value_based = cfg["planning"].get("value_based_zone_selection", False)

    label = "Myopic" if round_idx == 1 else f"CFA(θ={theta_prev:.3e})"
    print(f"\n  Runde {round_idx}: {label}\n")

    for run_i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"    Lauf {run_i}/{len(seeds)} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        selector = DailyZoneSelector(clusterer, run_cfg, coords)

        if round_idx == 1:
            solver = VRPSolver(mats, run_cfg, all_coords=coords)
            policy = MyopicPolicy(solver, coords, mats, run_cfg)
        else:
            policy = CFAModel(
                mats, run_cfg,
                all_coords=coords,
                stations_df=df_base,
                theta_override=theta_prev,
            )
            if use_value_based:
                selector.value_fn = policy._value

        sim = CFATrainingSimulator(policy, selector, coords, df_base, mats, run_cfg)
        result = sim.run(max_days=max_days)

        day_costs = [dr.total_cost_eur for dr in result.day_results]
        n_days    = len(sim.training_records)

        for i in range(n_days):
            feature    = sim.training_records[i]["feature"]
            cost_to_go = sum(day_costs[i:])
            if feature > 0:
                X_all.append(feature)
                y_all.append(cost_to_go)

        elapsed = time.time() - t0
        days    = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €, {elapsed:.0f}s)")

    return np.array(X_all), np.array(y_all)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CFA-Training (iteratives Policy Iteration)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Anzahl Trainingsläufe pro Runde (Standard: 20)")
    parser.add_argument("--rounds",   type=int, default=3,
                        help="Anzahl Iterationsrunden (Standard: 3; 1 = nur Myopic-Bootstrap)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/cfa/theta.json",
                        help="Ausgabepfad für θ (Standard: data/cfa/theta.json)")
    parser.add_argument("--fresh",    action="store_true",
                        help="Checkpoints ignorieren und von Runde 1 neu starten")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print("FEHLER: CFA-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    seeds = list(range(1, args.runs + 1))
    t0_total = time.time()

    # --- Checkpoint laden (--resume) ---
    theta: float | None = None
    intercept_val: float = 0.0
    history: list[dict] = []
    start_round = 1

    existing = sorted(ckpt_dir.glob("round_*.json")) if not args.fresh else []
    if existing:
        latest = existing[-1]
        with open(latest) as f:
            ckpt = json.load(f)
        theta         = float(ckpt["theta"])
        intercept_val = float(ckpt["intercept"])
        history       = ckpt.get("history", [])
        start_round   = ckpt["round"] + 1
        print(f"  Checkpoint gefunden: {latest.name}  "
              f"(θ={theta:.4e}, Runde {ckpt['round']} abgeschlossen)")
        if start_round > args.rounds:
            print(f"  Alle {args.rounds} Runden bereits abgeschlossen. Nichts zu tun.")
            sys.exit(0)
    elif args.fresh:
        print("  --fresh: starte von Runde 1 (Checkpoints ignoriert).")

    print(f"\nIteratives CFA-Training: Runde {start_round}–{args.rounds}, {args.runs} Läufe/Runde")

    for round_idx in range(start_round, args.rounds + 1):
        X, y = run_round(round_idx, cfg, coords, df_base, mats, seeds, args.max_days, theta)

        print(f"\n  Regression auf {len(X)} Datenpunkten...")
        theta_new, intercept_new, r2 = fit_theta(X, y)

        delta = abs(theta_new - theta) if theta is not None else float("nan")
        print(f"    θ = {theta_new:.6e}  EUR/(kW·Tag)"
              + (f"  (Δ = {delta:+.3e})" if not np.isnan(delta) else "  (Bootstrap)"))
        print(f"    intercept = {intercept_new:,.2f}  EUR")
        print(f"    R²        = {r2:.4f}")

        history.append({"round": round_idx, "theta": theta_new, "intercept": intercept_new, "r2": r2})
        theta = theta_new
        intercept_val = intercept_new

        # Checkpoint nach jeder Runde speichern
        ckpt_path = ckpt_dir / f"round_{round_idx}.json"
        with open(ckpt_path, "w") as f:
            json.dump({"round": round_idx, "theta": theta, "intercept": intercept_val,
                       "r2": r2, "history": history}, f, indent=2)
        print(f"    Checkpoint: {ckpt_path}")

    print(f"\n  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    if len(history) > 1:
        print("\n  θ-Verlauf über Runden:")
        for h in history:
            print(f"    Runde {h['round']}: θ = {h['theta']:.6e}  R² = {h['r2']:.4f}")

    payload = {
        "theta":            theta,
        "intercept":        intercept_val,
        "r2":               history[-1]["r2"],
        "n_runs":           args.runs,
        "n_rounds":         args.rounds,
        "n_datapoints":     int(len(X)),
        "history":          history,
        "feature_mean":     float(np.mean(X)),
        "cost_to_go_mean":  float(np.mean(y)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
