"""
Startet die Myopic-Plus-Simulation für alle Stationen.

Ausführen:
    .venv/bin/python3 scripts/run_myopic_plus.py
    .venv/bin/python3 scripts/run_myopic_plus.py --max-days 10 --log-day 1 --verbose

Monte Carlo:
    .venv/bin/python3 scripts/run_myopic_plus.py --output logs/myopic_plus/json/run_1.json --run-id 1 --seed 1
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
from src.models.myopic_plus import MyopicPlusModel
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Myopic-Plus-Simulation (Soft-Deadline-Priorisierung)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day", type=int, default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output", type=str, default="logs/myopic_plus/json/simulation.json",
                        help="Ausgabedatei (.json oder .log)")
    parser.add_argument("--verbose", action="store_true", help="OR-Tools Logging aktivieren")
    parser.add_argument("--seed", type=int, default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["project"]["seed"] = None if args.seed == -1 else args.seed
        print(f"  Seed: {cfg['project']['seed']} (überschrieben via --seed)")

    cfa_cfg = cfg.get("cfa", {})
    print(f"  α = {cfa_cfg.get('alpha', 10.0)} (Skalierungsfaktor Ausfallkosten)")

    print("Lade Stationsdaten...")
    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords)
    policy = MyopicPlusModel(mats, cfg, all_coords=coords, stations_df=df)
    if cfg["planning"].get("value_based_zone_selection", False):
        selector.value_fn = policy._zone_value
    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print(f"  Störungsmodus: stochastisch")

    print(f"\nStarte Myopic-Plus-Simulation (max. {args.max_days} Tage)...\n")
    result = sim.run(mal_df, max_days=args.max_days)

    sim.print_summary(result, label="MYOPIC PLUS")

    if args.log_day is not None:
        sim.print_day_log(result, args.log_day)

    out_path = Path(args.output)
    if out_path.suffix == ".log":
        sim.write_log(result, args.output, label="MYOPIC PLUS SIMULATION")
    else:
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
            "alpha": policy.alpha,
            "p_failure_per_hour": policy.p_failure_per_hour,
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
        sim.write_json(result, str(out_path.with_suffix(".json")),
                       label="MYOPIC PLUS SIMULATION", run_id=args.run_id,
                       model_params=model_params)


if __name__ == "__main__":
    main()
