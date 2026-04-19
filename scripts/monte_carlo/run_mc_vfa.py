"""
Monte-Carlo-Simulation für die VFA-Policy (gelernte Wertfunktionsapproximation).

Führt N Läufe durch (Seed 1 … N), speichert jeden Lauf als
logs/vfa/json/run_<N>.json und logs/vfa/log/run_<N>.log.
Am Ende wird eine aggregierte Analyse als logs/vfa/log/vfa_overview.log gespeichert.

Voraussetzung: theta muss trainiert sein:
    python scripts/train_vfa.py

Ausführen:
    .venv/bin/python3 scripts/monte_carlo/run_mc_vfa.py --runs 30
    .venv/bin/python3 scripts/monte_carlo/run_mc_vfa.py --runs 10 --max-days 50 --verbose
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
from src.models.vfa import VFAModel
from src.models.simulator import MaintenanceSimulator, SimulationResult
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def analyse(results: list[SimulationResult], seeds: list[int]) -> str:
    """Gibt aggregierte Statistiken über alle Läufe aus und gibt den Text zurück."""
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
        print(line)
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

    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo-Simulation (VFA)")
    parser.add_argument("--runs",       type=int, default=30,
                        help="Anzahl der Simulationsläufe N (Seeds 1…N, Standard: 30)")
    parser.add_argument("--max-days",   type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",    action="store_true",
                        help="OR-Tools Logging aktivieren")
    parser.add_argument("--theta-path", type=str, default="data/vfa/theta.json",
                        help="Pfad zur theta.json (Standard: data/vfa/theta.json)")
    parser.add_argument("--log-dir",    type=str, default="logs/vfa",
                        help="Basisordner für JSON- und Log-Ausgaben (Standard: logs/vfa)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print(f"  Störungsmodus: stochastisch")

    with open(args.theta_path) as f:
        theta_data = json.load(f)
    print(f"  θ = {theta_data['theta']}  "
          f"(R²={theta_data.get('r2', '?'):.4f}, {theta_data.get('n_runs', '?')} Trainingsläufe)")

    # node_to_power einmalig aufbauen (wird pro Lauf weitergegeben)
    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df_base.iterrows())
    }

    json_dir = Path(args.log_dir) / "json"
    log_dir  = Path(args.log_dir) / "log"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(1, args.runs + 1))
    results: list[SimulationResult] = []
    overview_path = log_dir / f"{Path(args.log_dir).name}_overview.log"

    print(f"\nStarte {args.runs} Monte-Carlo-Läufe...\n")
    for seed in seeds:
        print(f"  Lauf {seed}/{args.runs} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        selector = DailyZoneSelector(clusterer, run_cfg, coords)
        policy   = VFAModel(
            mats, run_cfg,
            all_coords=coords,
            node_to_power=node_to_power,
            n_stations=len(df_base),
            theta_path=args.theta_path,
        )
        if run_cfg["planning"].get("value_based_zone_selection", False):
            selector.value_fn = policy._station_value
        sim = MaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)

        result = sim.run(mal_df, max_days=args.max_days)

        cp = policy.cost_params
        fail_cfg = run_cfg.get("failure_simulation", {})
        model_params = {
            "seed": seed,
            "failure_mode": fail_cfg.get("mode", "csv"),
            "n_zones": run_cfg["planning"]["n_zones"],
            "n_teams": run_cfg["maintenance"]["n_teams"],
            "max_stations_per_team": run_cfg["planning"].get("max_stations_per_team"),
            "value_based_zone_selection": run_cfg["planning"].get("value_based_zone_selection", False),
            "theta": policy.theta.tolist(),
            "intercept": policy.intercept,
            "feature_names": ["total_urgency", "expected_damage", "mean_dsm",
                              "max_urgency", "frac_remaining", "n_carryover"],
            "alpha": policy.alpha,
            "lambda_per_day": policy.lambda_per_day,
            "p_failure_per_hour": policy.p_failure_per_hour,
            "theta_path": str(args.theta_path),
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

        out_path = json_dir / f"run_{seed}.json"
        sim.write_json(result, str(out_path), label="VFA SIMULATION", run_id=seed,
                       model_params=model_params)

        log_path = log_dir / f"run_{seed}.log"
        sim.write_log(result, str(log_path), label="VFA SIMULATION")

        results.append(result)
        days = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €)")

    overview_text = analyse(results, seeds)
    overview_path.write_text(overview_text, encoding="utf-8")
    print(f"\nOverview gespeichert: {overview_path.resolve()}")


if __name__ == "__main__":
    main()
