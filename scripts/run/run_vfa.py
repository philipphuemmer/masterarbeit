"""
Startet die VFA-Simulation mit gelernter Wertfunktionsapproximation.

Voraussetzung: θ muss zuerst trainiert werden:
    python scripts/train_vfa.py

Ausführen:
    python scripts/run_vfa.py
    python scripts/run_vfa.py --max-days 10 --log-day 1 --verbose

Monte Carlo:
    python scripts/run_vfa.py --output logs/vfa/run_1.json --run-id 1 --seed 1
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
from src.models.vfa import VFAModel
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="VFA-Simulation (gelernte Wertfunktion)")
    parser.add_argument("--max-days",   type=int,   default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day",    type=int,   default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output",     type=str,   default="logs/vfa/run_1.json",
                        help="Ausgabedatei (.json)")
    parser.add_argument("--verbose",    action="store_true",
                        help="OR-Tools Logging aktivieren")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id",     type=int,   default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    parser.add_argument("--theta-path", type=str,   default="data/training/vfa/theta.json",
                        help="Pfad zur theta.json (Standard: data/training/vfa/theta.json)")
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
        print(f"  Störungsmodus: stochastisch")

    # node_to_power aufbauen (für Wertfunktion)
    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df.iterrows())
    }

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords)
    policy   = VFAModel(
        mats, cfg,
        all_coords=coords,
        node_to_power=node_to_power,
        n_stations=len(df),
        theta_path=args.theta_path,
    )
    if cfg["planning"].get("value_based_zone_selection", False):
        selector.value_fn = policy._station_value
        print("  V̂-basierte Zonenauswahl aktiv (VFA).")
    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    print(f"  θ = {policy.theta.tolist()}")
    print(f"  R² (Training): siehe {args.theta_path}")
    print(f"\nStarte VFA-Simulation (max. {args.max_days} Tage)...\n")

    result = sim.run(mal_df, max_days=args.max_days)
    sim.print_summary(result, label="VFA")

    if args.log_day is not None:
        sim.print_day_log(result, args.log_day)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cp = policy.cost_params
    fail_cfg = cfg.get("failure_simulation", {})
    model_params = {
        "seed": cfg["project"].get("seed"),
        "failure_mode": fail_cfg.get("mode", "csv"),
        "n_zones": cfg["planning"]["n_zones"],
        "n_teams": cfg["maintenance"]["n_teams"],
        "max_stations_per_team": cfg["planning"].get("max_stations_per_team"),
        "value_based_zone_selection": cfg["planning"].get("value_based_zone_selection", False),
        "use_team_assignment": cfg["planning"].get("use_team_assignment", True),
        "theta": policy.theta.tolist(),
        "intercept": policy.intercept,
        "feature_names": ["total_urgency", "expected_damage", "mean_dsm",
                          "max_urgency", "frac_remaining", "n_carryover"],
        "alpha": policy.alpha,
        "lambda_per_day": policy.lambda_per_day,
        "p_failure_per_hour": policy.p_failure_per_hour,
        "theta_path": str(args.theta_path),
        "cost_params": {
            "wage_eur_per_hour": cp.wage_eur_per_hour,
            "fuel_eur_per_km": cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        },
    }
    if fail_cfg.get("mode") == "stochastic":
        model_params["failure_simulation"] = {
            "p1_per_hour": fail_cfg.get("p1_per_hour"),
            "p2_per_hour": fail_cfg.get("p2_per_hour"),
            "recovery_days": fail_cfg.get("recovery_days"),
            "initial_factor": fail_cfg.get("initial_factor"),
        }
    sim.write_json(result, str(out_path), label="VFA SIMULATION", run_id=args.run_id,
                   model_params=model_params)


if __name__ == "__main__":
    main()
