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

Training (iteratives Policy Iteration):
    Runde 1: N Myopic-Simulationen → θ₁  (Bootstrap)
    Runde 2: N VFA(θ₁)-Simulationen → θ₂
    Runde r: N VFA(θ_{r-1})-Simulationen → θ_r

    Pro Tag: φ(s) vor Tagesplanung + cost_to_go G_t = Σ_{t'≥t} cost(t')
    OLS-Regression: G_t ≈ θᵀ φ(s_t) + intercept

Ausführen:
    python scripts/train_vfa.py
    python scripts/train_vfa.py --runs 30 --rounds 3 --max-days 200
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
from src.models.myopic import MyopicPolicy
from src.models.vfa import VFAModel
from src.models.simulator import MaintenanceSimulator
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

        remaining_nodes = [idx + 1 for idx in remaining]  # node_idx (1-basiert)

        if remaining_nodes:
            dsm_vals = np.array([
                float(self._days_since_maintenance[n]) for n in remaining_nodes
            ])
            pow_vals = np.array([
                self.node_to_power.get(n, 22.0) for n in remaining_nodes
            ])

            urgency      = pow_vals * dsm_vals
            failure_risk = 1.0 - np.exp(-self.lambda_per_day * dsm_vals)

            f_total_urgency   = float(np.sum(urgency))
            f_expected_damage = float(np.sum(failure_risk * pow_vals))
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

    Returns
    -------
    (theta, intercept, r2)
    """
    A = np.column_stack([Phi, np.ones(len(Phi))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    theta_vec = coeffs[:-1]
    intercept = float(coeffs[-1])

    y_pred = Phi @ theta_vec + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return theta_vec, intercept, r2


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
    lambda_per_day: float,
    node_to_power: dict[int, float],
    theta_prev: np.ndarray | None,
    intercept_prev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Führt N Simulationen durch und gibt Trainingsdaten (Phi, y) zurück.

    round_idx == 1  → Myopic-Policy (Bootstrap)
    round_idx >= 2  → VFA-Policy mit theta_prev
    """
    Phi_all: list[np.ndarray] = []
    y_all:   list[float]      = []
    use_value_based = cfg["planning"].get("zone_selection_mode", "classic") == "value_based"

    if round_idx == 1:
        label = "Myopic"
    else:
        label = f"VFA(θ=[{', '.join(f'{v:.2e}' for v in theta_prev)}])"
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

        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)

        if round_idx == 1:
            solver = VRPSolver(mats, run_cfg, all_coords=coords)
            policy = MyopicPolicy(solver, coords, mats, run_cfg)
        else:
            policy = VFAModel(
                mats, run_cfg,
                all_coords=coords,
                node_to_power=node_to_power,
                n_stations=len(df_base),
                theta_override=theta_prev,
                intercept_override=intercept_prev,
            )
            if use_value_based:
                selector.value_fn = policy._station_value

        sim = VFATrainingSimulator(
            lambda_per_day, policy, selector, coords, df_base, mats, run_cfg
        )
        result = sim.run(max_days=max_days)

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

    return np.stack(Phi_all), np.array(y_all)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VFA-Training (iteratives Policy Iteration)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Anzahl Trainingsläufe pro Runde (Standard: 20)")
    parser.add_argument("--rounds",   type=int, default=3,
                        help="Anzahl Iterationsrunden (Standard: 3; 1 = nur Myopic-Bootstrap)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/training/vfa/theta.json",
                        help="Ausgabepfad für θ (Standard: data/training/vfa/theta.json)")
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
        print("FEHLER: VFA-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

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

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df_base.iterrows())
    }

    seeds = list(range(1, args.runs + 1))
    t0_total = time.time()

    # --- Checkpoint laden (--resume) ---
    theta: np.ndarray | None = None
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
        theta         = np.array(ckpt["theta"], dtype=np.float64)
        intercept_val = float(ckpt["intercept"])
        history       = ckpt.get("history", [])
        start_round   = ckpt["round"] + 1
        print(f"  Checkpoint gefunden: {latest.name}  (Runde {ckpt['round']} abgeschlossen)")
        if start_round > args.rounds:
            print(f"  Alle {args.rounds} Runden bereits abgeschlossen. Nichts zu tun.")
            sys.exit(0)
    elif args.fresh:
        print("  --fresh: starte von Runde 1 (Checkpoints ignoriert).")

    print(f"\nIteratives VFA-Training: Runde {start_round}–{args.rounds}, {args.runs} Läufe/Runde")

    for round_idx in range(start_round, args.rounds + 1):
        Phi, y = run_round(
            round_idx, cfg, coords, df_base, mats, seeds, args.max_days,
            lambda_per_day, node_to_power, theta, intercept_val,
        )

        print(f"\n  Regression auf {len(y)} Datenpunkten...")
        theta_new, intercept_new, r2 = fit_theta(Phi, y)

        print(f"\n  Gelernte Gewichte θ (Runde {round_idx}):")
        for name, w_new in zip(FEATURE_NAMES, theta_new):
            if theta is not None:
                delta = w_new - theta[FEATURE_NAMES.index(name)]
                print(f"    {name:<20} = {w_new:+.6e}  (Δ = {delta:+.3e})")
            else:
                print(f"    {name:<20} = {w_new:+.6e}")
        print(f"    {'intercept':<20} = {intercept_new:+.6e}")
        print(f"    R²                   = {r2:.4f}")

        history.append({
            "round":     round_idx,
            "theta":     theta_new.tolist(),
            "intercept": intercept_new,
            "r2":        r2,
        })
        theta = theta_new
        intercept_val = intercept_new
        Phi_last = Phi
        y_last   = y

        # Checkpoint nach jeder Runde speichern
        ckpt_path = ckpt_dir / f"round_{round_idx}.json"
        with open(ckpt_path, "w") as f:
            json.dump({"round": round_idx, "theta": theta.tolist(),
                       "intercept": intercept_val, "r2": r2, "history": history}, f, indent=2)
        print(f"    Checkpoint: {ckpt_path}")

    print(f"\n  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    if len(history) > 1:
        print("\n  R²-Verlauf über Runden:")
        for h in history:
            print(f"    Runde {h['round']}: R² = {h['r2']:.4f}")

    payload = {
        "theta":             theta.tolist(),
        "intercept":         intercept_val,
        "feature_names":     FEATURE_NAMES,
        "r2":                history[-1]["r2"],
        "n_runs":            args.runs,
        "n_rounds":          args.rounds,
        "n_datapoints":      int(len(y_last)),
        "history":           history,
        "lambda_per_day":    lambda_per_day,
        "feature_means":     Phi_last.mean(axis=0).tolist(),
        "feature_stds":      Phi_last.std(axis=0).tolist(),
        "cost_to_go_mean":   float(np.mean(y_last)),
        "cost_to_go_std":    float(np.std(y_last)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
