"""
Vergleicht OR-Tools Zeitlimits für MyopicPlus oder CFA.
Läuft N Tage mit jedem Limit und gibt Gesamtkosten aus.

Ausführen:
    .venv/bin/python3 scripts/benchmark_timelimit.py [--days 20] [--model cfa|myopic_plus]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa import CFAModel
from src.models.myopic_plus import MyopicPlusModel
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def run_with_limit(cfg: dict, mal_df, limit_seconds: int, max_days: int, model: str) -> dict:
    cfg = {**cfg}
    cfg["maintenance"] = {**cfg["maintenance"],
                          "solver_time_limit_initial": limit_seconds,
                          "solver_time_limit_replan": max(1, limit_seconds // 2)}

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)

    if model == "cfa":
        policy = CFAModel(mats, cfg, all_coords=coords, stations_df=df)
        if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._value
    else:
        policy = MyopicPlusModel(mats, cfg, all_coords=coords, stations_df=df)

    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    t0 = time.perf_counter()
    result = sim.run(mal_df, max_days=max_days)
    elapsed = time.perf_counter() - t0

    travel = sum(r.fuel_cost_eur for r in result.day_results)
    downtime = sum(r.downtime_cost_eur for r in result.day_results)
    serviced = sum(r.n_routine_completed for r in result.day_results)
    return {
        "total_cost_eur": result.total_cost_eur,
        "travel_cost_eur": travel,
        "disruption_cost_eur": downtime,
        "stations_serviced": serviced,
        "wall_time_s": round(elapsed, 1),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=20)
    parser.add_argument("--limits", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--model",  type=str, default="myopic_plus",
                        choices=["myopic_plus", "cfa"],
                        help="Modell für den Benchmark (Standard: myopic_plus)")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
    else:
        mal_df = None

    label = "CFA" if args.model == "cfa" else "MyopicPlus"
    print(f"Benchmark: {label}, {args.days} Tage, seed={cfg['project']['seed']}")
    print(f"{'Limit':>6}  {'Gesamtkosten':>14}  {'Fahrtkosten':>12}  {'Störungskosten':>15}  {'Stationen':>10}  {'Laufzeit':>10}")
    print("-" * 80)

    for limit in args.limits:
        print(f"\n--- Limit {limit}s ---")
        r = run_with_limit(cfg, mal_df, limit, args.days, args.model)
        print(
            f"{limit:>5}s  "
            f"{r['total_cost_eur']:>13.0f}€  "
            f"{r['travel_cost_eur']:>11.0f}€  "
            f"{r['disruption_cost_eur']:>14.0f}€  "
            f"{r['stations_serviced']:>10}  "
            f"{r['wall_time_s']:>9.1f}s"
        )


if __name__ == "__main__":
    main()
