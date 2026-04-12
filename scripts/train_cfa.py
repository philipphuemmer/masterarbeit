"""
CFA-Training: Lernt den Gewichtsvektor θ der Wertfunktionsapproximation.

Wertfunktion:
    V(s) ≈ θ × Σ_k power_kW[k] × days_since_maintenance[k]

Training:
    1. N Myopic-Simulationen mit verschiedenen Seeds
    2. Pro Tag: Feature = Σ_k power_kW[k] × dsm[k] (alle noch offenen Stationen)
               Target  = tatsächliche Restkosten ab diesem Tag
    3. OLS-Regression: cost_to_go ≈ θ × feature + intercept
    4. θ (EUR pro kW·Tag) wird in data/cfa/theta.json gespeichert

Ausführen:
    python scripts/train_cfa.py
    python scripts/train_cfa.py --runs 30 --max-days 200
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.myopic import MyopicPolicy
from src.models.simulator import MaintenanceSimulator, DayResult
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
) -> tuple[float, float]:
    """
    OLS-Regression: cost_to_go ≈ θ × feature + intercept.

    Returns
    -------
    (theta, intercept)
        theta    : EUR pro kW·Tag
        intercept: EUR (Fixanteil, wird nicht in der Policy verwendet)
    """
    A = np.column_stack([X, np.ones(len(X))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CFA-Training (Wertfunktionsapproximation)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Anzahl Trainingsläufe (Standard: 20)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/cfa/theta.json",
                        help="Ausgabepfad für θ (Standard: data/cfa/theta.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Daten laden ---
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

    # --- Trainingsläufe ---
    X_all: list[float] = []
    y_all: list[float] = []

    seeds = list(range(1, args.runs + 1))
    t0_total = time.time()

    print(f"\nStarte {args.runs} Trainingsläufe (Myopic-Baseline)...\n")

    for run_i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"  Lauf {run_i}/{args.runs} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        solver   = VRPSolver(mats, run_cfg, all_coords=coords)
        selector = DailyZoneSelector(clusterer, run_cfg, coords)
        policy   = MyopicPolicy(solver, coords, mats, run_cfg)
        sim      = CFATrainingSimulator(policy, selector, coords, df_base, mats, run_cfg)

        result = sim.run(max_days=args.max_days)

        # Cost-to-go pro Tag berechnen
        day_costs = [dr.total_cost_eur for dr in result.day_results]
        n_days    = len(sim.training_records)

        for i in range(n_days):
            feature     = sim.training_records[i]["feature"]
            cost_to_go  = sum(day_costs[i:])
            if feature > 0:
                X_all.append(feature)
                y_all.append(cost_to_go)

        elapsed = time.time() - t0
        days    = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €, {elapsed:.0f}s)")

    # --- Regression ---
    X = np.array(X_all)
    y = np.array(y_all)

    print(f"\nRegression auf {len(X)} Datenpunkten ({args.runs} Läufe)...")
    theta, intercept = fit_theta(X, y)

    y_pred = theta * X + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"  θ           = {theta:.6e}  EUR / (kW·Tag)")
    print(f"  intercept   = {intercept:,.2f}  EUR")
    print(f"  R²          = {r2:.4f}")
    print(f"  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    # --- Speichern ---
    payload = {
        "theta":      theta,
        "intercept":  intercept,
        "r2":         r2,
        "n_runs":     args.runs,
        "n_datapoints": len(X),
        "feature_mean": float(np.mean(X)),
        "cost_to_go_mean": float(np.mean(y)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
