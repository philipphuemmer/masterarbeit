"""
Startet die DB-Simulation mit gelerntem Balance-Parameter α.

Voraussetzung: Policy muss zuerst trainiert werden:
    python scripts/train/train_db.py

Ausführen:
    python scripts/run/run_db.py
    python scripts/run/run_db.py --max-days 10 --log-day 1 --verbose

Monte Carlo:
    python scripts/run/run_db.py --output logs/db/run_1.json --run-id 1 --seed 1
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
from src.models.db import DBMaintenanceSimulator, DBModel
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="DB-Simulation (Dynamic Balance Policy)")
    parser.add_argument("--max-days",    type=int,  default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day",     type=int,  default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output",      type=str,  default="logs/db/run_1.json",
                        help="Ausgabedatei (.json)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Ausführliche Logging-Ausgabe")
    parser.add_argument("--seed",        type=int,  default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id",      type=int,  default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    parser.add_argument("--policy-path", type=str,  default="data/training/db/policy.json",
                        help="Pfad zur policy.json (Standard: data/training/db/policy.json)")
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
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

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
        for i, (_, row) in enumerate(df.iterrows())
    }

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords)
    policy = DBModel(
        mats, cfg,
        all_coords=coords,
        node_to_power=node_to_power,
        n_stations=len(df),
        policy_path=args.policy_path,
    )

    if cfg["planning"].get("value_based_zone_selection", False):
        selector.value_fn = policy._station_value
        print("  V̂-basierte Zonenauswahl aktiv (DB: U(k) = power × dsm).")

    sim = DBMaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    print(f"\nStarte DB-Simulation (max. {args.max_days} Tage)...\n")
    result = sim.run(mal_df, max_days=args.max_days)
    sim.print_summary(result, label="DB")

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
        "policy_path": str(args.policy_path),
        "alpha_mean": policy.alpha_mean,
        "feature_names": [
            "frac_remaining", "frac_urgent", "mean_dsm_norm", "total_urgency_norm",
            "max_urgency_norm", "mean_dist_depot_norm", "std_dist_depot_norm",
            "carryover_pressure",
        ],
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

    sim.write_json(result, str(out_path), label="DB SIMULATION", run_id=args.run_id,
                   model_params=model_params)


if __name__ == "__main__":
    main()
