"""
VFA-Training: Lernt den Gewichtsvektor θ der Wertfunktionsapproximation.

Wertfunktion (linear, mehrere Features):
    V̂(s) ≈ θᵀ φ(s) + intercept

Feature-Vektor φ(s) pro Tag (vor Tagesplanung):
    f0: Σ_k power_kW[k] × dsm[k]           – Gesamtdringlichkeit (wie CFA)
    f1: Σ_k failure_risk[k] × power_kW[k]  – Erwarteter Schadenwert
    f2: mean(dsm[k])                         – Mittlere Wartungsüberfälligkeit
    f3: max(power_kW[k] × dsm[k])           – Größte Einzeldringlichkeit
    f4: n_remaining (normiert)               – Auslastungsgrad
    f5: n_carryover                          – Offene Störungsrückstände

    Dabei: failure_risk[k] = 1 − exp(−λ × dsm[k])
           λ = p1_per_hour × 24 + p2_per_hour × 24  (Tagesrate)

Training:
    1. N Myopic-Simulationen mit verschiedenen Seeds
    2. Pro Tag: φ(s) vor Tagesplanung + cost_to_go G_t = Σ_{t'≥t} cost(t')
    3. OLS-Regression: G_t ≈ θᵀ φ(s_t) + intercept
    4. θ-Vektor wird in data/vfa/theta.json gespeichert

Ausführen:
    python scripts/train_vfa.py
    python scripts/train_vfa.py --runs 30 --max-days 200
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

# Namen der Features (Reihenfolge = Index in θ)
FEATURE_NAMES = [
    "total_urgency",       # Σ kW × dsm
    "expected_damage",     # Σ failure_risk × kW
    "mean_dsm",            # mean(dsm)
    "max_urgency",         # max(kW × dsm)
    "frac_remaining",      # n_remaining / n_stations
    "n_carryover",         # offene Störungen
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Trainings-Simulator: zeichnet Feature-Vektor pro Tag auf
# ---------------------------------------------------------------------------

class VFATrainingSimulator(MaintenanceSimulator):
    """
    Erweitert MaintenanceSimulator um Aufzeichnung des täglichen
    Feature-Vektors φ(s) vor der Tagesplanung.
    """

    def __init__(self, lambda_per_day: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_per_day = lambda_per_day
        self.training_records: list[dict] = []

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        n_stations = self.n_stations

        # --- Feature-Vektor extrahieren ---
        remaining_nodes = [idx + 1 for idx in remaining]  # node_idx (1-basiert)

        if remaining_nodes:
            dsm_vals = np.array([
                float(self._days_since_maintenance[n]) for n in remaining_nodes
            ])
            pow_vals = np.array([
                self.node_to_power.get(n, 22.0) for n in remaining_nodes
            ])

            urgency = pow_vals * dsm_vals
            failure_risk = 1.0 - np.exp(-self.lambda_per_day * dsm_vals)
            expected_damage = failure_risk * pow_vals

            f_total_urgency   = float(np.sum(urgency))
            f_expected_damage = float(np.sum(expected_damage))
            f_mean_dsm        = float(np.mean(dsm_vals))
            f_max_urgency     = float(np.max(urgency))
        else:
            f_total_urgency   = 0.0
            f_expected_damage = 0.0
            f_mean_dsm        = 0.0
            f_max_urgency     = 0.0

        f_frac_remaining = len(remaining_nodes) / max(1, n_stations)
        f_n_carryover    = float(len(carryover_tasks))

        phi = np.array([
            f_total_urgency,
            f_expected_damage,
            f_mean_dsm,
            f_max_urgency,
            f_frac_remaining,
            f_n_carryover,
        ], dtype=np.float64)

        self.training_records.append({"day": day, "phi": phi})

        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def fit_theta(
    Phi: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    OLS-Regression: cost_to_go ≈ θᵀ φ + intercept.

    Parameters
    ----------
    Phi : (n_samples, n_features)
    y   : (n_samples,)

    Returns
    -------
    (theta, intercept, r2)
    """
    A = np.column_stack([Phi, np.ones(len(Phi))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    theta     = coeffs[:-1]
    intercept = float(coeffs[-1])

    y_pred = Phi @ theta + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return theta, intercept, r2


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VFA-Training (Wertfunktionsapproximation)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Anzahl Trainingsläufe (Standard: 20)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/vfa/theta.json",
                        help="Ausgabepfad für θ (Standard: data/vfa/theta.json)")
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
        print("FEHLER: VFA-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

    # Tagesrate für Ausfallwahrscheinlichkeit (stündliche Raten × 24)
    fail_cfg = cfg.get("failure_simulation", {})
    lambda_per_day = (
        fail_cfg.get("p1_per_hour", 0.00084)
        + fail_cfg.get("p2_per_hour", 0.00028)
    ) * 24.0

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")
    print(f"  λ_Tag = {lambda_per_day:.5f}  (Störungsrate pro Station pro Tag)")

    # --- Trainingsläufe ---
    Phi_all: list[np.ndarray] = []
    y_all:   list[float]      = []

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
        sim      = VFATrainingSimulator(
            lambda_per_day, policy, selector, coords, df_base, mats, run_cfg
        )

        result = sim.run(max_days=args.max_days)

        # Cost-to-go pro Tag berechnen
        day_costs = [dr.total_cost_eur for dr in result.day_results]
        n_days    = len(sim.training_records)

        for i in range(n_days):
            phi        = sim.training_records[i]["phi"]
            cost_to_go = float(sum(day_costs[i:]))
            Phi_all.append(phi)
            y_all.append(cost_to_go)

        elapsed = time.time() - t0
        days    = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €, {elapsed:.0f}s)")

    # --- Regression ---
    Phi = np.stack(Phi_all)      # (n_samples, n_features)
    y   = np.array(y_all)        # (n_samples,)

    print(f"\nRegression auf {len(y)} Datenpunkten ({args.runs} Läufe)...")
    theta, intercept, r2 = fit_theta(Phi, y)

    print("\nGelernte Gewichte θ:")
    for name, w in zip(FEATURE_NAMES, theta):
        print(f"  {name:<20} = {w:+.6e}")
    print(f"  {'intercept':<20} = {intercept:+.6e}")
    print(f"  R²                   = {r2:.4f}")
    print(f"  Trainingszeit gesamt : {time.time() - t0_total:.0f}s")

    # --- Speichern ---
    payload = {
        "theta":             theta.tolist(),
        "intercept":         intercept,
        "feature_names":     FEATURE_NAMES,
        "r2":                r2,
        "n_runs":            args.runs,
        "n_datapoints":      int(len(y)),
        "lambda_per_day":    lambda_per_day,
        "feature_means":     Phi.mean(axis=0).tolist(),
        "feature_stds":      Phi.std(axis=0).tolist(),
        "cost_to_go_mean":   float(np.mean(y)),
        "cost_to_go_std":    float(np.std(y)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
