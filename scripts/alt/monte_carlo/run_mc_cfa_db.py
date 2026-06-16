"""
Monte-Carlo-Simulation für die CFA-DB-Policy.

Führt N Läufe durch (Seed 1…N), speichert jeden Lauf als
logs/cfa_db/json/run_<N>.json und logs/cfa_db/log/run_<N>.log.
Am Ende wird eine aggregierte Analyse als logs/cfa_db/log/cfa_db_overview.log gespeichert.

Voraussetzung: Policy muss trainiert sein:
    python scripts/train/train_cfa_db.py

Ausführen:
    .venv/bin/python3 scripts/monte_carlo/run_mc_cfa_db.py --runs 30
    .venv/bin/python3 scripts/monte_carlo/run_mc_cfa_db.py --runs 10 --max-days 50 --verbose
"""
from __future__ import annotations

import argparse
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
from src.models.alt.cfa_db import CFADBMaintenanceSimulator, CFADBModel
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
    parser = argparse.ArgumentParser(description="Monte-Carlo-Simulation (CFA-DB)")
    parser.add_argument("--runs",        type=int, default=30,
                        help="Letzter Seed (inklusiv, Standard: 30)")
    parser.add_argument("--start-run",   type=int, default=1,
                        help="Erster Seed zum Fortfahren (Standard: 1)")
    parser.add_argument("--max-days",    type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Ausführliche Logging-Ausgabe")
    parser.add_argument("--policy-path", type=str, default="data/training/cfa_db/policy.json",
                        help="Pfad zur policy.json (Standard: data/training/cfa_db/policy.json)")
    parser.add_argument("--log-dir",     type=str, default="logs/cfa_db",
                        help="Basisordner für JSON- und Log-Ausgaben (Standard: logs/cfa_db)")
    parser.add_argument("--resume", action="store_true",
                        help="Vorhandene run_N.json laden und fehlende Läufe fortsetzen")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords = np.array(get_coordinates(df_base, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print("  Störungsmodus: stochastisch")

    with open(args.policy_path) as f:
        policy_data = json.load(f)
    print(f"  α_mean={policy_data.get('alpha_mean', '?'):.3f}  "
          f"({policy_data.get('n_iterations', '?')} Trainingsiterationen)")

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df_base.iterrows())
    }

    json_dir = Path(args.log_dir) / "json"
    log_dir  = Path(args.log_dir) / "log"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.start_run > args.runs:
        print(f"Fehler: --start-run ({args.start_run}) > --runs ({args.runs})")
        sys.exit(1)
    seeds = list(range(args.start_run, args.runs + 1))
    completed: dict[int, SimulationResult] = {}
    overview_path = log_dir / f"{Path(args.log_dir).name}_overview.log"

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

    print(f"\nStarte {len(seeds_to_run)} Monte-Carlo-Läufe (Seeds {seeds_to_run[0]}–{seeds_to_run[-1]})...\n")
    for seed in seeds_to_run:
        print(f"  Lauf {seed}/{args.runs} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
        policy = CFADBModel(
            mats, run_cfg,
            all_coords=coords,
            node_to_power=node_to_power,
            n_stations=len(df_base),
            policy_path=args.policy_path,
        )

        if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._station_value

        sim = CFADBMaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)
        result = sim.run(mal_df, max_days=args.max_days)

        cp = policy.cost_params
        fail_cfg = run_cfg.get("failure_simulation", {})
        model_params = {
            "seed": seed,
            "failure_mode": fail_cfg.get("mode", "csv"),
            "n_zones": run_cfg["planning"]["n_zones"],
            "n_teams": run_cfg["maintenance"]["n_teams"],
            "max_stations_per_team": run_cfg["planning"].get("max_stations_per_team"),
            "zone_selection_mode": run_cfg["planning"].get("zone_selection_mode", "classic"),
            "alpha_mean": policy.alpha_mean,
            "policy_path": str(args.policy_path),
            "cost_params": {
                "wage_eur_per_hour": cp.wage_eur_per_hour,
                "fuel_eur_per_km": cp.fuel_eur_per_km,
                "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
            },
        }
        pl_cfg = run_cfg.get("planning", {})
        mt_cfg = run_cfg.get("maintenance", {})
        _slm   = mt_cfg.get("solver_limit_mode", "time")
        model_params["zone_selection"] = {
            "n_zones":                    pl_cfg.get("n_zones"),
            "n_top_candidates":           pl_cfg.get("n_top_candidates"),
            "min_team_separation_km":     pl_cfg.get("min_team_separation_km"),
            "max_stations_per_team":      pl_cfg.get("max_stations_per_team"),
            "travel_reserve_min":         pl_cfg.get("travel_reserve_min"),
            "zone_selection_mode": pl_cfg.get("zone_selection_mode", "classic"),
            "use_team_assignment":        pl_cfg.get("use_team_assignment", True),
            "priority_weights":           pl_cfg.get("priority_weights", {}),
        }
        model_params["solver"] = {
            "n_teams":                       mt_cfg.get("n_teams"),
            "workday_start_hour":            mt_cfg.get("workday_start_hour", 8),
            "workday_end_hour":              mt_cfg.get("workday_end_hour", 16),
            "mean_service_time":             mt_cfg.get("mean_service_time"),
            "global_span_cost_coefficient":  mt_cfg.get("global_span_cost_coefficient", 0),
            "lunch_duration_min":            mt_cfg.get("lunch_duration_min", 0),
            "solver_limit_mode":             _slm,
            "solver_solution_limit_initial": mt_cfg.get("solver_solution_limit_initial") if _slm == "solution" else None,
            "solver_solution_limit_replan":  mt_cfg.get("solver_solution_limit_replan")  if _slm == "solution" else None,
            "solver_time_limit_initial":     mt_cfg.get("solver_time_limit_initial")     if _slm == "time"     else None,
            "solver_time_limit_replan":      mt_cfg.get("solver_time_limit_replan")      if _slm == "time"     else None,
        }
        if fail_cfg.get("mode") == "stochastic":
            model_params["failure_simulation"] = {
                "p1_per_hour": fail_cfg.get("p1_per_hour"),
                "p2_per_hour": fail_cfg.get("p2_per_hour"),
                "recovery_days": fail_cfg.get("recovery_days"),
                "initial_factor": fail_cfg.get("initial_factor"),
            }

        out_path = json_dir / f"run_{seed}.json"
        sim.write_json(result, str(out_path), label="CFA-DB SIMULATION", run_id=seed,
                       model_params=model_params)

        log_path = log_dir / f"run_{seed}.log"
        sim.write_log(result, str(log_path), label="CFA-DB SIMULATION")

        completed[seed] = result
        days = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €)")

        cost_params_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        sorted_seeds = sorted(completed.keys())
        overview_text = analyse(
            [completed[s] for s in sorted_seeds], sorted_seeds,
            cfg=cfg, cost_params=cost_params_dict,
        )
        overview_path.write_text(overview_text, encoding="utf-8")

    print(f"\nOverview gespeichert: {overview_path.resolve()}")


if __name__ == "__main__":
    main()
