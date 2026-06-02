"""
Startet die DB-Simple-Simulation (CFA-Future + 3-Feature-δ(S_t)).

Voraussetzung (optional): DB-Modell trainieren:
    python scripts/train/train_db_simple.py --collect --train

Ohne Modell läuft die Policy mit festem δ = --delta (Fallback δ=0.5 → identisch zu CFA-Future).

Ausführen:
    python scripts/run/run_db_simple.py
    python scripts/run/run_db_simple.py --max-days 10 --log-day 1 --verbose
    python scripts/run/run_db_simple.py --delta 0.3          # fester δ-Wert (Benchmark)
    python scripts/run/run_db_simple.py --no-model           # explizit kein Modell laden
    python scripts/run/run_db_simple.py --delta 0.5 --no-model  # == CFA-Future (Sanity)

Monte Carlo:
    python scripts/run/run_db_simple.py --output logs/db_simple/run_1.json --run-id 1 --seed 1
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
from src.models.db_simple import (
    DBSimpleBalanceModel,
    DBSimpleMaintenanceSimulator,
    DBSimplePolicy,
)
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="DB-Simple-Simulation (CFA-Future + 3-Feature-δ)")
    parser.add_argument("--max-days",   type=int,   default=365)
    parser.add_argument("--log-day",    type=int,   default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output",     type=str,   default="logs/db_simple/run_1.json")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--run-id",     type=int,   default=None)
    parser.add_argument("--theta-path", type=str,   default="data/training/cfa_future/theta.json")
    parser.add_argument("--model-path", type=str,   default="data/training/db_simple/model.pkl")
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Fester δ-Wert als Fallback wenn kein Modell geladen (δ=0.5 → CFA-Future)")
    parser.add_argument("--no-model",   action="store_true",
                        help="Explizit kein Modell laden — festes δ verwenden")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
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

    # DB-Modell laden
    if args.no_model:
        db_model = DBSimpleBalanceModel(default_delta=args.delta)
        print(f"  DB-Modell deaktiviert — festes δ = {args.delta}")
    else:
        model_path = Path(args.model_path)
        db_model = DBSimpleBalanceModel.load(model_path, default_delta=args.delta)
        if db_model.clf is not None:
            print(f"  DB-Modell geladen aus {model_path}")
        else:
            print(f"  Kein Modell unter {model_path} — festes δ = {args.delta}")

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values

    policy = DBSimplePolicy(
        traffic_matrices=mats,
        config=cfg,
        all_coords=coords,
        stations_df=df,
        db_model=db_model,
        default_delta=args.delta,
        theta_path=args.theta_path,
    )

    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._value
        print("  V̂-basierte Zonenauswahl aktiv (DB-Simple).")

    print(f"  CFA-Future θ = {policy.theta}")
    print(f"  δ Fallback   = {db_model.default_delta}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cp = policy.cost_params
    fail_cfg = cfg.get("failure_simulation", {})
    model_params = {
        "seed": cfg["project"].get("seed"),
        "failure_mode": fail_cfg.get("mode", "csv"),
        "n_zones": cfg["planning"]["n_zones"],
        "n_teams": cfg["maintenance"]["n_teams"],
        "zone_selection_mode": cfg["planning"].get("zone_selection_mode", "classic"),
        "theta": policy.theta.tolist(),
        "theta_path": str(args.theta_path),
        "db_model_trained": db_model.clf is not None,
        "delta_default": db_model.default_delta,
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

    print(f"\nStarte DB-Simple-Simulation (max. {args.max_days} Tage)...\n")
    sim = DBSimpleMaintenanceSimulator(policy, selector, coords, df, mats, cfg)
    result = sim.run(mal_df, max_days=args.max_days)
    model_params["delta_by_day"] = {str(d): v for d, v in sim.delta_log.items()}
    sim.print_summary(result, label="DB-Simple")

    if args.log_day is not None:
        sim.print_day_log(result, args.log_day)

    sim.write_json(result, str(out_path), label="DB-SIMPLE SIMULATION",
                   run_id=args.run_id, model_params=model_params)


if __name__ == "__main__":
    main()
