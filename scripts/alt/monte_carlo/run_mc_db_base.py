"""
Monte-Carlo-Simulation für die DB-Base-Policy (CFA-Future + gelerntes δ(S_t)).

Führt N Läufe durch (Seed 1 … N), speichert jeden Lauf als
logs/db_base/json/run_<N>.json und logs/db_base/log/run_<N>.log.
Am Ende wird eine aggregierte Analyse als logs/db_base/log/db_base_overview.log gespeichert.

Voraussetzung (optional): DB-Modell trainieren:
    python scripts/train/train_db_base.py --collect --train

Ohne Modell läuft die Policy mit festem δ = --delta (Standard: 0.3).

Ausführen:
    python scripts/monte_carlo/run_mc_db_base.py --runs 30
    python scripts/monte_carlo/run_mc_db_base.py --runs 30 --delta 0.3   # statischer Benchmark
    python scripts/monte_carlo/run_mc_db_base.py --runs 10 --max-days 50 --verbose
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.alt.db_base import DBBalanceModel, DBBaseMaintenanceSimulator, DBBasePolicy
from src.models.simulator import SimulationResult, DayResult
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def _load_result_from_json(path: Path) -> SimulationResult:
    """Rekonstruiert SimulationResult aus gespeicherter JSON-Datei (für --resume)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    day_results = [
        DayResult(
            day=d["day"],
            n_routine_tasks=d["n_routine_tasks"],
            n_routine_completed=d["n_routine_completed"],
            disruptions_handled=d["disruptions_handled"],
            disruptions_carryover=d["disruptions_carryover"],
            operational_cost_eur=d["operational_cost_eur"],
            wage_cost_eur=d["wage_cost_eur"],
            fuel_cost_eur=d["fuel_cost_eur"],
            downtime_cost_eur=d["downtime_cost_eur"],
            hourly_logs=[],
        )
        for d in data["days"]
    ]
    return SimulationResult(
        day_results=day_results,
        total_disruptions=s["total_disruptions"],
        same_day_handled=s["same_day_handled"],
        total_carryover=s["total_carryover"],
        days_to_complete=s["days_to_complete"],
        remaining_stations_at_end=s["remaining_stations_at_end"],
    )


from src.utils.mc_analyse import analyse


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo DB-Base")
    parser.add_argument("--runs",       type=int,   default=30)
    parser.add_argument("--max-days",   type=int,   default=365)
    parser.add_argument("--delta",      type=float, default=0.3,
                        help="Fallback-δ wenn kein Modell geladen (Standard: 0.3)")
    parser.add_argument("--no-model",   action="store_true",
                        help="Kein Modell laden — festes δ verwenden")
    parser.add_argument("--model-path", type=str,   default="data/training/db_base/model.pkl")
    parser.add_argument("--theta-path", type=str,   default="data/training/cfa_future/theta.json")
    parser.add_argument("--log-dir",    type=str,   default="logs/db_base")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Vorhandene run_N.json laden und fehlende Läufe fortsetzen")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df     = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats   = load_traffic_matrices(cfg)

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
    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    mal_df = pd.read_csv("data/malfunction.csv") if failure_mode == "csv" else None

    if args.no_model:
        db_model = DBBalanceModel(default_delta=args.delta)
        model_trained = False
    else:
        db_model = DBBalanceModel.load(Path(args.model_path), default_delta=args.delta)
        model_trained = db_model.clf is not None

    label = "DB-BASE (dynamisch)" if model_trained else f"DB-BASE (δ={args.delta:.1f} statisch)"
    log_dir  = Path(args.log_dir) / "log"
    json_dir = Path(args.log_dir) / "json"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    overview_path = log_dir / f"{Path(args.log_dir).name}_overview.log"

    print(f"{label} — {args.runs} Läufe, max. {args.max_days} Tage")

    seeds = list(range(1, args.runs + 1))
    completed: dict[int, SimulationResult] = {}

    if args.resume:
        for s in seeds:
            p = json_dir / f"run_{s}.json"
            if p.exists():
                try:
                    completed[s] = _load_result_from_json(p)
                    print(f"  Lauf {s} geladen ({p.name}).")
                except Exception as e:
                    print(f"  Warnung: Lauf {s} übersprungen ({e}).")
        if completed:
            print(f"  {len(completed)} Läufe aus JSON geladen.")

    seeds_to_run = [s for s in seeds if s not in completed]
    if not seeds_to_run:
        print("Alle Läufe bereits vorhanden. Overview wird neu geschrieben.")
        sorted_seeds = sorted(completed.keys())
        overview_path.write_text(
            analyse([completed[s] for s in sorted_seeds], sorted_seeds, cfg=cfg),
            encoding="utf-8",
        )
        return

    for seed in seeds_to_run:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["project"]["seed"] = seed

        policy = DBBasePolicy(
            traffic_matrices=mats,
            config=run_cfg,
            all_coords=coords,
            node_to_power=node_to_power,
            n_stations=len(df),
            stations_df=df,
            db_model=db_model,
            default_delta=args.delta,
            theta_path=args.theta_path,
        )
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
        if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._station_value

        sim = DBBaseMaintenanceSimulator(policy, selector, coords, df, mats, run_cfg)
        result = sim.run(mal_df, max_days=args.max_days)

        cp = policy.cost_params
        fail_cfg = run_cfg.get("failure_simulation", {})
        model_params = {
            "seed": seed,
            "failure_mode": fail_cfg.get("mode", "csv"),
            "n_zones": run_cfg["planning"]["n_zones"],
            "n_teams": run_cfg["maintenance"]["n_teams"],
            "zone_selection_mode": run_cfg["planning"].get("zone_selection_mode", "classic"),
            "theta": policy._theta.tolist() if policy._theta is not None else None,
            "theta_path": str(args.theta_path),
            "db_model_trained": model_trained,
            "delta_default": args.delta,
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

        sim.write_json(result, str(json_dir / f"run_{seed}.json"),
                       label=label.upper(), run_id=seed, model_params=model_params)
        sim.write_log(result, str(log_dir / f"run_{seed}.log"), label=label.upper())

        completed[seed] = result
        days = result.days_to_complete or "?"
        cp_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        sorted_seeds = sorted(completed.keys())
        overview_path.write_text(
            analyse([completed[s] for s in sorted_seeds], sorted_seeds, cfg=cfg, cost_params=cp_dict),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
