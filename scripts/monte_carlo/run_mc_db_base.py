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
from src.models.db_base import DBBalanceModel, DBBaseMaintenanceSimulator, DBBasePolicy
from src.models.simulator import SimulationResult
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def analyse(results: list[SimulationResult], seeds: list[int], cfg: dict | None = None, cost_params: dict | None = None) -> str:
    total_costs   = np.array([r.total_cost_eur for r in results])
    op_costs      = np.array([sum(d.operational_cost_eur for d in r.day_results) for r in results])
    wage_costs    = np.array([sum(d.wage_cost_eur for d in r.day_results) for r in results])
    fuel_costs    = np.array([sum(d.fuel_cost_eur for d in r.day_results) for r in results])
    dt_costs      = np.array([sum(d.downtime_cost_eur for d in r.day_results) for r in results])
    days_done     = np.array([r.days_to_complete if r.days_to_complete is not None else np.nan for r in results])
    same_day_rate = np.array([r.same_day_rate for r in results])
    total_disrupt = np.array([r.total_disruptions for r in results])
    carryovers    = np.array([r.total_carryover for r in results])

    buf = io.StringIO()

    def out(line: str = "") -> None:
        buf.write(line + "\n")

    def row(label: str, arr: np.ndarray, unit: str = "") -> None:
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            out(f"  {label:<32}  (keine Daten)")
            return
        out(
            f"  {label:<32}  "
            f"MW {np.mean(finite):>10,.2f}  "
            f"SD {np.std(finite):>9,.2f}  "
            f"Min {np.min(finite):>10,.2f}  "
            f"Max {np.max(finite):>10,.2f}"
            + (f"  {unit}" if unit else "")
        )

    sep = "=" * 70
    out(f"\n{sep}")
    out(f"  MONTE-CARLO-ANALYSE  –  {len(results)} Läufe (Seeds {seeds[0]}–{seeds[-1]})")
    out(sep)
    row("Gesamtkosten (€)",       total_costs,   "€")
    row("  Betriebskosten (€)",   op_costs,      "€")
    row("    Lohnkosten (€)",     wage_costs,    "€")
    row("    Fahrtkosten (€)",    fuel_costs,    "€")
    row("  Ausfallkosten (€)",    dt_costs,      "€")
    row("Simulationstage",        days_done)
    row("Same-Day-Rate",          same_day_rate * 100, "%")
    row("Gesamtstörungen",        total_disrupt)
    row("Gesamtcarryover",        carryovers)
    out(sep)

    out(f"\n  {'Seed':>5}  {'Tage':>5}  {'Gesamt (€)':>12}  "
        f"{'Lohn (€)':>10}  {'Fahrt (€)':>10}  {'Ausfall (€)':>11}  "
        f"{'Same-Day %':>10}  {'Störungen':>9}  {'Carryover':>9}")
    out(f"  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*9}")
    for i, r in enumerate(results):
        wage = sum(d.wage_cost_eur for d in r.day_results)
        fuel = sum(d.fuel_cost_eur for d in r.day_results)
        dt   = sum(d.downtime_cost_eur for d in r.day_results)
        d_   = r.days_to_complete if r.days_to_complete is not None else "-"
        out(
            f"  {seeds[i]:>5}  {str(d_):>5}  {r.total_cost_eur:>12,.2f}  "
            f"{wage:>10,.2f}  {fuel:>10,.2f}  {dt:>11,.2f}  "
            f"{r.same_day_rate * 100:>9.1f}%  "
            f"{r.total_disruptions:>9}  {r.total_carryover:>9}"
        )
    out()

    if cfg:
        pl = cfg.get("planning", {})
        mt = cfg.get("maintenance", {})
        fs = cfg.get("failure_simulation", {})

        out(f"\n{sep}")
        out("  SIMULATIONSPARAMETER")
        out(sep)
        out(f"    Störungsmodus            : {fs.get('mode', '–')}")
        out(f"    Anzahl Zonen             : {pl.get('n_zones', '–')}")
        out(f"    Max. Stationen/Team      : {pl.get('max_stations_per_team', '–')}")
        out(f"    Zonenauswahl             : {pl.get('zone_selection_mode', 'classic')}")
        if fs.get("mode") == "stochastic":
            out(f"    p1_per_hour              : {fs.get('p1_per_hour', '–')}")
            out(f"    p2_per_hour              : {fs.get('p2_per_hour', '–')}")
            out(f"    recovery_days            : {fs.get('recovery_days', '–')}")
            out(f"    initial_factor           : {fs.get('initial_factor', '–')}")

    return buf.getvalue()


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
    parser.add_argument("--verbose",    action="store_true")
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
    log_dir  = Path("logs/db_base/log")
    json_dir = Path("logs/db_base/json")
    log_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    overview_path = log_dir / "db_base_overview.log"

    print(f"{label} — {args.runs} Läufe, max. {args.max_days} Tage")

    seeds   = list(range(1, args.runs + 1))
    results = []

    for seed in seeds:
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

        results.append(result)
        days = result.days_to_complete or "?"
        cp_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        overview_path.write_text(
            analyse(results, seeds[:len(results)], cfg=cfg, cost_params=cp_dict),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
