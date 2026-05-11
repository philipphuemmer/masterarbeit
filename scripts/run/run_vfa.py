"""
Startet die Hybrid-VFA-Simulation (lokaler CFA-Future-Term + globaler Zustandswert).

Voraussetzungen:
    python scripts/train/train_cfa_future.py   → data/training/cfa_future/theta.json
    python scripts/train/train_vfa.py          → data/training/vfa/theta.json

Ausführen:
    python scripts/run/run_vfa.py
    python scripts/run/run_vfa.py --max-days 10 --log-day 1 --verbose
    python scripts/run/run_vfa.py --output logs/vfa/run_1.json --seed 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.vfa import VFAModel, STATE_FEATURE_NAMES
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-VFA-Simulation")
    parser.add_argument("--max-days",        type=int,   default=365)
    parser.add_argument("--log-day",         type=int,   default=None)
    parser.add_argument("--output",          type=str,   default="logs/vfa/run_1.json")
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--run-id",          type=int,   default=None)
    parser.add_argument("--local-theta",     type=str,
                        default="data/training/cfa_future/theta.json",
                        help="Pfad zu θ_local (cfa_future)")
    parser.add_argument("--global-theta",    type=str,
                        default="data/training/vfa/theta.json",
                        help="Pfad zu θ_global (vfa)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["project"]["seed"] = None if args.seed == -1 else args.seed

    print("Lade Stationsdaten...")
    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats   = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print("  Störungsmodus: stochastisch")

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)

    policy = VFAModel(
        mats, cfg,
        all_coords=coords,
        stations_df=df,
        local_theta_path=args.local_theta,
        global_theta_path=args.global_theta,
    )
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._local_value
        print("  V̂-basierte Zonenauswahl aktiv (VFA, lokaler Term).")

    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    vfa_cfg = cfg.get("vfa", {})
    print(f"  θ_local  = {policy.theta_local.tolist()}")
    print(f"  θ_global = {policy.theta_global.tolist()}")
    print(f"  α={policy._alpha}, β={policy._beta}")
    print(f"\nStarte VFA-Simulation (max. {args.max_days} Tage)...\n")

    result = sim.run(mal_df, max_days=args.max_days)
    sim.print_summary(result, label="VFA")

    if args.log_day is not None:
        sim.print_day_log(result, args.log_day)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cp       = policy.cost_params
    fail_cfg = cfg.get("failure_simulation", {})
    model_params = {
        "seed":               cfg["project"].get("seed"),
        "failure_mode":       fail_cfg.get("mode", "csv"),
        "n_zones":            cfg["planning"]["n_zones"],
        "n_teams":            cfg["maintenance"]["n_teams"],
        "max_stations_per_team": cfg["planning"].get("max_stations_per_team"),
        "zone_selection_mode":   cfg["planning"].get("zone_selection_mode", "classic"),
        "use_team_assignment":   cfg["planning"].get("use_team_assignment", True),
        "theta_local":           policy.theta_local.tolist(),
        "theta_global":          policy.theta_global.tolist(),
        "intercept_global":      policy.intercept_global,
        "feature_names_global":  STATE_FEATURE_NAMES,
        "alpha":                 policy._alpha,
        "beta":                  policy._beta,
        "lambda_per_day":        policy.lambda_per_day,
        "local_theta_path":      args.local_theta,
        "global_theta_path":     args.global_theta,
        "cost_params": {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        },
    }
    if fail_cfg.get("mode") == "stochastic":
        model_params["failure_simulation"] = {
            "p1_per_hour":   fail_cfg.get("p1_per_hour"),
            "p2_per_hour":   fail_cfg.get("p2_per_hour"),
            "recovery_days": fail_cfg.get("recovery_days"),
            "initial_factor": fail_cfg.get("initial_factor"),
        }

    sim.write_json(result, str(out_path), label="VFA SIMULATION", run_id=args.run_id,
                   model_params=model_params)


if __name__ == "__main__":
    main()
