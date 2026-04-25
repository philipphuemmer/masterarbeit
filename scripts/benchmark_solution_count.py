"""
Misst wie viele OR-Tools-Lösungen pro Solve-Aufruf im echten CFA-Modell
gefunden werden (mit Soft-Deadlines, Extra-Costs etc.).

Ausführen:
    .venv/bin/python3 scripts/benchmark_solution_count.py --days 5 --time-limits 2 5
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa import CFAModel
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


def run_and_count(cfg: dict, time_limit: int, max_days: int) -> dict:
    cfg = {**cfg}
    cfg["maintenance"] = {
        **cfg["maintenance"],
        "solver_limit_mode": "time",
        "solver_time_limit_initial": time_limit,
        "solver_time_limit_replan": max(1, time_limit // 2),
    }

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"],
                              random_state=cfg["project"]["seed"])
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords)
    policy = CFAModel(mats, cfg, all_coords=coords, stations_df=df)
    if cfg["planning"].get("value_based_zone_selection", False):
        selector.value_fn = policy._value
    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    # Solution-Counting aktivieren
    policy.solver._count_solutions = True
    policy.solver._solution_counts = []

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    mal_df = None
    if failure_mode == "csv":
        import pandas as pd
        mal_df = pd.read_csv("data/malfunction.csv")

    t0 = time.perf_counter()
    sim.run(mal_df, max_days=max_days)
    elapsed = time.perf_counter() - t0

    counts = policy.solver._solution_counts
    return {
        "time_limit_s": time_limit,
        "n_solve_calls": len(counts),
        "total_solutions": sum(counts),
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": round(statistics.mean(counts), 1) if counts else 0,
        "median": statistics.median(counts) if counts else 0,
        "wall_time_s": round(elapsed, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--time-limits", nargs="+", type=int, default=[2, 5])
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"CFA Solution-Count Benchmark | {args.days} Tage | seed={cfg['project']['seed']}")
    print(f"(misst Lösungen pro _solve-Aufruf inkl. Soft-Deadlines)\n")

    for tl in args.time_limits:
        print(f"▶ Zeitlimit {tl}s ...")
        r = run_and_count(cfg, tl, args.days)
        print(f"  Solve-Aufrufe gesamt : {r['n_solve_calls']}")
        print(f"  Lösungen gesamt      : {r['total_solutions']}")
        print(f"  Lösungen pro Aufruf  : min={r['min']}  max={r['max']}  "
              f"Ø={r['mean']}  Median={r['median']}")
        print(f"  Gesamtlaufzeit       : {r['wall_time_s']}s\n")


if __name__ == "__main__":
    main()
