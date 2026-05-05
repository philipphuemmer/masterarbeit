"""
Startet die DB-Base-Simulation (CFA-Future + zustandsabhängiger Balance-Parameter δ).

Voraussetzung (optional): DB-Modell trainieren:
    python scripts/train/train_db_base.py

Ohne Modell läuft die Policy mit festem δ = default_delta (statischer Benchmark).

Ausführen:
    python scripts/run/run_db_base.py
    python scripts/run/run_db_base.py --max-days 10 --log-day 1 --verbose
    python scripts/run/run_db_base.py --delta 0.3          # fester δ-Wert (Benchmark)
    python scripts/run/run_db_base.py --no-model           # explizit kein Modell laden

Monte Carlo:
    python scripts/run/run_db_base.py --output logs/db_base/run_1.json --run-id 1 --seed 1
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
from src.models.db_base import DBBalanceModel, DBBaseMaintenanceSimulator, DBBasePolicy
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="DB-Base-Simulation (CFA-Future + δ-Modell)")
    parser.add_argument("--max-days",   type=int,   default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day",    type=int,   default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output",     type=str,   default="logs/db_base/run_1.json",
                        help="Ausgabedatei (.json)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Debug-Logging aktivieren")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id",     type=int,   default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    parser.add_argument("--theta-path", type=str,   default="data/training/cfa_future/theta.json",
                        help="Pfad zur CFA-Future theta.json")
    parser.add_argument("--model-path", type=str,   default="data/training/db_base/model.pkl",
                        help="Pfad zum trainierten DB-Modell (.pkl)")
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Fester δ-Wert als Fallback wenn kein Modell geladen (Standard: 0.5)")
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
        print(f"  Störungsmodus: stochastisch")

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df.iterrows())
    }

    # DB-Modell laden (oder statischer Fallback)
    if args.no_model:
        db_model = DBBalanceModel(default_delta=args.delta)
        print(f"  DB-Modell deaktiviert — festes δ = {args.delta}")
    else:
        model_path = Path(args.model_path)
        db_model = DBBalanceModel.load(model_path, default_delta=args.delta)
        if db_model.clf is not None:
            print(f"  DB-Modell geladen aus {model_path}")
        else:
            print(f"  Kein Modell unter {model_path} — festes δ = {args.delta}")

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)

    policy = DBBasePolicy(
        traffic_matrices=mats,
        config=cfg,
        all_coords=coords,
        node_to_power=node_to_power,
        n_stations=len(df),
        stations_df=df,
        db_model=db_model,
        default_delta=args.delta,
        theta_path=args.theta_path,
    )

    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._station_value
        print("  V̂-basierte Zonenauswahl aktiv (DB-Base).")

    sim = DBBaseMaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    print(f"  CFA-Future θ = {policy._theta}")
    print(f"  δ Fallback   = {db_model.default_delta}")
    print(f"\nStarte DB-Base-Simulation (max. {args.max_days} Tage)...\n")

    result = sim.run(mal_df, max_days=args.max_days)
    sim.print_summary(result, label="DB-Base")

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
        "zone_selection_mode": cfg["planning"].get("zone_selection_mode", "classic"),
        "use_team_assignment": cfg["planning"].get("use_team_assignment", True),
        "theta": policy._theta.tolist() if policy._theta is not None else None,
        "theta_path": str(args.theta_path),
        "db_model_trained": db_model.clf is not None,
        "delta_default": db_model.default_delta,
        "p_failure_per_hour": policy._p_failure_per_hour,
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

    sim.write_json(result, str(out_path), label="DB-BASE SIMULATION", run_id=args.run_id,
                   model_params=model_params)


if __name__ == "__main__":
    main()
