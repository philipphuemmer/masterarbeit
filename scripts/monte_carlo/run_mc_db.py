"""
Monte-Carlo-Simulation für die DB-Policy (Dynamic Balance).

Führt N Läufe durch (Seed 1…N), speichert jeden Lauf als
logs/db/json/run_<N>.json und logs/db/log/run_<N>.log.
Am Ende wird eine aggregierte Analyse als logs/db/log/db_overview.log gespeichert.

Voraussetzung: Policy muss trainiert sein:
    python scripts/train/train_db.py

Ausführen:
    .venv/bin/python3 scripts/monte_carlo/run_mc_db.py --runs 30
    .venv/bin/python3 scripts/monte_carlo/run_mc_db.py --runs 10 --max-days 50 --verbose
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
from src.models.db import DBMaintenanceSimulator, DBModel
from src.models.simulator import SimulationResult
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def analyse(results: list[SimulationResult], seeds: list[int], cfg: dict | None = None, cost_params: dict | None = None) -> str:
    """Aggregierte Statistiken über alle Läufe."""
    total_costs   = np.array([r.total_cost_eur for r in results])
    op_costs      = np.array([sum(d.operational_cost_eur for d in r.day_results) for r in results])
    wage_costs    = np.array([sum(d.wage_cost_eur for d in r.day_results) for r in results])
    fuel_costs    = np.array([sum(d.fuel_cost_eur for d in r.day_results) for r in results])
    dt_costs      = np.array([sum(d.downtime_cost_eur for d in r.day_results) for r in results])
    days_done     = np.array([r.days_to_complete if r.days_to_complete is not None else np.nan
                              for r in results])
    same_day_rate = np.array([r.same_day_rate for r in results])
    total_disrupt = np.array([r.total_disruptions for r in results])
    carryovers    = np.array([r.total_carryover for r in results])

    buf = io.StringIO()

    def out(line: str = "") -> None:
        #print(line)
        buf.write(line + "\n")

    sep = "=" * 70
    out(f"\n{sep}")
    out(f"  MONTE-CARLO-ANALYSE  –  {len(results)} Läufe (Seeds {seeds[0]}–{seeds[-1]})")
    out(sep)

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

    row("Gesamtkosten (€)",           total_costs,   "€")
    row("  Betriebskosten (€)",       op_costs,      "€")
    row("    Lohnkosten (€)",         wage_costs,    "€")
    row("    Fahrtkosten (€)",        fuel_costs,    "€")
    row("  Ausfallkosten (€)",        dt_costs,      "€")
    row("Simulationstage",            days_done)
    row("Same-Day-Rate",              same_day_rate * 100, "%")
    row("Gesamtstörungen",            total_disrupt)
    row("Gesamtcarryover",            carryovers)

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
        pw = pl.get("priority_weights", {})

        out(f"\n{sep}")
        out("  SIMULATIONSPARAMETER")
        out(sep)

        out("\n  Zonenauswahl")
        out(f"    Anzahl Zonen             : {pl.get('n_zones', '–')}")
        out(f"    Top-Kandidaten           : {pl.get('n_top_candidates', '–')}")
        out(f"    Min. Teamabstand         : {pl.get('min_team_separation_km', '–')} km")
        out(f"    Max. Stationen/Team      : {pl.get('max_stations_per_team', '–')}")
        out(f"    Zeitpuffer Depot         : {pl.get('travel_reserve_min', '–')} min")
        out(f"    V̂-basierte Zonenauswahl  : {pl.get('zone_selection_mode', 'classic')}")
        out(f"    Team-Zuweisung           : {'Ja' if pl.get('use_team_assignment', True) else 'Nein'}")
        out(f"    Gewicht Depot-Entfernung : {pw.get('depot_distance', '–')}")
        out(f"    Gewicht Fläche           : {pw.get('convex_hull_area', '–')}")
        out(f"    Gewicht Zonenwert (V̂)    : {pw.get('zone_value', '–')}")

        out("\n  Solver (OR-Tools)")
        slm = mt.get("solver_limit_mode", "time")
        out(f"    Abbruchkriterium         : {slm}")
        if slm == "solution":
            out(f"    Lösungslimit initial     : {mt.get('solver_solution_limit_initial', '–')}")
            out(f"    Lösungslimit Replan      : {mt.get('solver_solution_limit_replan', '–')}")
        else:
            out(f"    Zeitlimit initial        : {mt.get('solver_time_limit_initial', '–')} s")
            out(f"    Zeitlimit Replan         : {mt.get('solver_time_limit_replan', '–')} s")
        out(f"    Makespan-Koeffizient     : {mt.get('global_span_cost_coefficient', 0)}")
        out(f"    Mittagspause             : {mt.get('lunch_duration_min', 0)} min")

        out("\n  Wartung")
        out(f"    Teams                    : {mt.get('n_teams', '–')}")
        sh = mt.get("workday_start_hour", 8)
        eh = mt.get("workday_end_hour", 16)
        out(f"    Arbeitstag               : {sh:02d}:00–{eh:02d}:00")
        out(f"    Mittlere Servicezeit     : {mt.get('mean_service_time', '–')} min")

        if cost_params:
            out("\n  Kosten")
            out(f"    Lohn                     : {cost_params.get('wage_eur_per_hour', '–'):.2f} €/h")
            out(f"    Fahrtkosten              : {cost_params.get('fuel_eur_per_km', '–'):.2f} €/km")
            out(f"    Ausfallkosten            : {cost_params.get('downtime_eur_per_kwh', '–'):.2f} €/kWh")

        out("\n  Störungssimulation")
        out(f"    Modus                    : {fs.get('mode', '–')}")
        if fs.get("mode") == "stochastic":
            out(f"    p(Typ-1)/h               : {fs.get('p1_per_hour', 0):.5f}")
            out(f"    p(Typ-2)/h               : {fs.get('p2_per_hour', 0):.5f}")
            out(f"    Erholungsdauer           : {fs.get('recovery_days', '–')} Tage")
            out(f"    Initialfaktor            : {fs.get('initial_factor', 0):.2f}")
        out("")

    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo-Simulation (DB)")
    parser.add_argument("--runs",        type=int, default=30,
                        help="Letzter Seed (inklusiv, Standard: 30)")
    parser.add_argument("--start-run",   type=int, default=1,
                        help="Erster Seed zum Fortfahren (Standard: 1)")
    parser.add_argument("--max-days",    type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Ausführliche Logging-Ausgabe")
    parser.add_argument("--policy-path", type=str, default="data/training/db/policy.json",
                        help="Pfad zur policy.json (Standard: data/training/db/policy.json)")
    parser.add_argument("--log-dir",     type=str, default="logs/db",
                        help="Basisordner für JSON- und Log-Ausgaben (Standard: logs/db)")
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
    results: list[SimulationResult] = []
    overview_path = log_dir / f"{Path(args.log_dir).name}_overview.log"

    print(f"\nStarte {len(seeds)} Monte-Carlo-Läufe (Seeds {seeds[0]}–{seeds[-1]})...\n")
    for seed in seeds:
        print(f"  Lauf {seed}/{args.runs} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
        policy = DBModel(
            mats, run_cfg,
            all_coords=coords,
            node_to_power=node_to_power,
            n_stations=len(df_base),
            policy_path=args.policy_path,
            stations_df=df_base,
        )

        if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._station_value

        sim = DBMaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)
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
        sim.write_json(result, str(out_path), label="DB SIMULATION", run_id=seed,
                       model_params=model_params)

        log_path = log_dir / f"run_{seed}.log"
        sim.write_log(result, str(log_path), label="DB SIMULATION")

        results.append(result)
        days = result.days_to_complete or "?"
        #print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €)")

        cost_params_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        overview_text = analyse(results, seeds[:len(results)], cfg=cfg, cost_params=cost_params_dict)
        overview_path.write_text(overview_text, encoding="utf-8")

    #print(f"\nOverview gespeichert: {overview_path.resolve()}")


if __name__ == "__main__":
    main()
