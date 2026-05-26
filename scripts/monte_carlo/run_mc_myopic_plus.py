"""
Monte-Carlo-Simulation für die Myopic-Plus-Policy (Soft-Deadline-Priorisierung).

Führt N Läufe durch (Seed 1 … N), speichert jeden Lauf als
logs/myopic_plus/json/run_<N>.json und gibt am Ende eine aggregierte
Analyse aus, die auch als logs/myopic_plus/log/mc_overview.log gespeichert wird.

rolling_horizon.enabled: false → Legacy-Pfad (MaintenanceSimulator), Log-Dir: logs/myopic_plus
rolling_horizon.enabled: true  → RH-Pfad (RollingHorizonRunner),     Log-Dir: logs/myopic_plus_rh

Ausführen:
    python scripts/monte_carlo/run_mc_myopic_plus.py --runs 30
    python scripts/monte_carlo/run_mc_myopic_plus.py --runs 10 --max-days 50 --verbose
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

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices, get_failure_rate_factors
from src.models.myopic_plus import MyopicPlusModel
from src.models.rolling_horizon import (
    DailyTaskGenerator,
    HorizonEvaluator,
    PolicyAdapter,
    RollingHorizonRunner,
)
from src.models.simulator import MaintenanceSimulator, SimulationResult, DayResult
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


def _load_rh_overrides_from_json(path: Path) -> tuple[int, int]:
    """Gibt (initial_overrides, replan_overrides) aus rolling_horizon_meta zurück."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("rolling_horizon_meta", {})
    legacy = meta.get("rh_overrides", 0)
    return (
        meta.get("initial_overrides", 0),
        meta.get("replan_overrides", legacy),
    )


def analyse(
    results: list[SimulationResult],
    seeds: list[int],
    cfg: dict | None = None,
    cost_params: dict | None = None,
    initial_overrides: list[int] | None = None,
    replan_overrides: list[int] | None = None,
) -> str:
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
    if replan_overrides is not None:
        row("  Replan-Rollout-Overrides", np.array(replan_overrides, dtype=float))
    if initial_overrides is not None:
        row("  Initialplan-Rollout-Overrides", np.array(initial_overrides, dtype=float))

    out(sep)

    rh_col = replan_overrides is not None
    out(f"\n  {'Seed':>5}  {'Tage':>5}  {'Gesamt (€)':>12}  "
        f"{'Lohn (€)':>10}  {'Fahrt (€)':>10}  {'Ausfall (€)':>11}  "
        f"{'Same-Day %':>10}  {'Störungen':>9}  {'Carryover':>9}"
        + (f"  {'Ov-R':>5}  {'Ov-I':>5}" if rh_col else ""))
    out(f"  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*9}"
        + (f"  {'-'*5}  {'-'*5}" if rh_col else ""))
    for i, r in enumerate(results):
        wage = sum(d.wage_cost_eur for d in r.day_results)
        fuel = sum(d.fuel_cost_eur for d in r.day_results)
        dt   = sum(d.downtime_cost_eur for d in r.day_results)
        d_   = r.days_to_complete if r.days_to_complete is not None else "-"
        rh_str = (
            f"  {replan_overrides[i]:>5}  {initial_overrides[i]:>5}"
            if rh_col else ""
        )
        out(
            f"  {seeds[i]:>5}  {str(d_):>5}  {r.total_cost_eur:>12,.2f}  "
            f"{wage:>10,.2f}  {fuel:>10,.2f}  {dt:>11,.2f}  "
            f"{r.same_day_rate * 100:>9.1f}%  "
            f"{r.total_disruptions:>9}  {r.total_carryover:>9}"
            + rh_str
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
    parser = argparse.ArgumentParser(description="Monte-Carlo-Simulation (Myopic Plus)")
    parser.add_argument("--runs", type=int, default=30,
                        help="Letzter Seed (inklusiv, Standard: 30)")
    parser.add_argument("--start-run", type=int, default=1,
                        help="Erster Seed zum Fortfahren (Standard: 1)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose", action="store_true",
                        help="Logging aktivieren")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Basisordner für JSON- und Log-Ausgaben "
                             "(Standard: logs/myopic_plus bzw. logs/myopic_plus_rh)")
    parser.add_argument("--resume", action="store_true",
                        help="Vorhandene run_N.json laden und fehlende Läufe fortsetzen")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    rh_cfg     = cfg.get("rolling_horizon", {})
    rh_enabled = bool(rh_cfg.get("enabled", False))

    if args.log_dir is None:
        args.log_dir = "logs/myopic_plus_rh" if rh_enabled else "logs/myopic_plus"

    cfa_cfg = cfg.get("cfa", {})
    print(f"  α = {cfa_cfg.get('alpha', 10.0)} (Skalierungsfaktor Ausfallkosten)")

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

    if rh_enabled:
        failure_factors = get_failure_rate_factors(df_base)
        node_to_failure_factor = {i + 1: failure_factors.get(i, 1.0) for i in range(len(df_base))}
        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        node_to_power = {
            i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
            for i, (_, row) in enumerate(df_base.iterrows())
        }
        print(f"  Rolling Horizon aktiv: H={rh_cfg.get('horizon_days', 7)}, "
              f"k={rh_cfg.get('top_k_candidates', 3)}, S={rh_cfg.get('n_scenarios', 8)}")

    json_dir = Path(args.log_dir) / "json"
    log_dir  = Path(args.log_dir) / "log"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.start_run > args.runs:
        print(f"Fehler: --start-run ({args.start_run}) > --runs ({args.runs})")
        sys.exit(1)
    seeds = list(range(args.start_run, args.runs + 1))
    completed: dict[int, SimulationResult] = {}
    completed_rh: dict[int, tuple[int, int]] = {}  # (initial_ov, replan_ov)
    overview_path = log_dir / f"{Path(args.log_dir).name}_overview.log"

    if args.resume:
        for s in seeds:
            p = json_dir / f"run_{s}.json"
            if p.exists():
                try:
                    completed[s] = _load_result_from_json(p)
                    completed_rh[s] = _load_rh_overrides_from_json(p)
                    print(f"  Lauf {s} geladen ({p.name}).")
                except Exception as e:
                    print(f"  Warnung: Lauf {s} übersprungen ({e}).")
        if completed:
            print(f"  {len(completed)} Läufe aus JSON geladen.")

    seeds_to_run = [s for s in seeds if s not in completed]
    if not seeds_to_run:
        print("Alle Läufe bereits vorhanden. Overview wird neu geschrieben.")
        sorted_seeds = sorted(completed.keys())
        sorted_init = [completed_rh[s][0] for s in sorted_seeds]
        sorted_replan = [completed_rh[s][1] for s in sorted_seeds]
        overview_path.write_text(
            analyse(
                [completed[s] for s in sorted_seeds], sorted_seeds, cfg=cfg,
                initial_overrides=sorted_init if rh_enabled else None,
                replan_overrides=sorted_replan if rh_enabled else None,
            ),
            encoding="utf-8",
        )
        return

    mode_label = "[Rolling Horizon]" if rh_enabled else "[Legacy]"
    print(f"\nStarte {len(seeds_to_run)} Monte-Carlo-Läufe {mode_label} (Seeds {seeds_to_run[0]}–{seeds_to_run[-1]})...\n")

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
        policy   = MyopicPlusModel(mats, run_cfg, all_coords=coords, stations_df=df_base)
        if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._zone_value

        cp = policy.cost_params
        fail_cfg = run_cfg.get("failure_simulation", {})
        pl_cfg = run_cfg.get("planning", {})
        mt_cfg = run_cfg.get("maintenance", {})
        _slm   = mt_cfg.get("solver_limit_mode", "time")

        model_params = {
            "seed": seed,
            "failure_mode": fail_cfg.get("mode", "csv"),
            "n_zones": run_cfg["planning"]["n_zones"],
            "n_teams": run_cfg["maintenance"]["n_teams"],
            "max_stations_per_team": pl_cfg.get("max_stations_per_team"),
            "zone_selection_mode": pl_cfg.get("zone_selection_mode", "classic"),
            "use_team_assignment": pl_cfg.get("use_team_assignment", True),
            "alpha": policy.alpha,
            "p_failure_per_hour": policy.p_failure_per_hour,
            "cost_params": {
                "wage_eur_per_hour": cp.wage_eur_per_hour,
                "fuel_eur_per_km": cp.fuel_eur_per_km,
                "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
            },
            "zone_selection": {
                "n_zones":                    pl_cfg.get("n_zones"),
                "n_top_candidates":           pl_cfg.get("n_top_candidates"),
                "min_team_separation_km":     pl_cfg.get("min_team_separation_km"),
                "max_stations_per_team":      pl_cfg.get("max_stations_per_team"),
                "travel_reserve_min":         pl_cfg.get("travel_reserve_min"),
                "zone_selection_mode":        pl_cfg.get("zone_selection_mode", "classic"),
                "use_team_assignment":        pl_cfg.get("use_team_assignment", True),
                "priority_weights":           pl_cfg.get("priority_weights", {}),
            },
            "solver": {
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
        log_path = log_dir / f"run_{seed}.log"

        if not rh_enabled:
            # ------------------------------------------------------------------
            # Legacy-Pfad
            # ------------------------------------------------------------------
            sim    = MaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)
            result = sim.run(mal_df, max_days=args.max_days)

            sim.write_json(result, str(out_path), label="MYOPIC PLUS SIMULATION",
                           run_id=seed, model_params=model_params)
            sim.write_log(result, str(log_path), label="MYOPIC PLUS SIMULATION")

            completed[seed] = result
            completed_rh[seed] = (0, 0)

        else:
            # ------------------------------------------------------------------
            # RH-Pfad
            # ------------------------------------------------------------------
            policy_adapter = PolicyAdapter(policy)
            task_gen = DailyTaskGenerator(
                selector=selector,
                failure_mode=failure_mode,
                n_stations=len(df_base),
            )
            evaluator = HorizonEvaluator(
                policy=policy_adapter,
                task_generator=task_gen,
                all_coords=coords,
                traffic_matrices=mats,
                config=run_cfg,
                node_to_power=node_to_power,
                node_to_failure_factor=node_to_failure_factor,
                cost_params=cp,
                n_stations=len(df_base),
                n_teams=run_cfg["maintenance"]["n_teams"],
            )
            runner = RollingHorizonRunner(
                policy=policy_adapter,
                task_generator=task_gen,
                evaluator=evaluator,
                all_coords=coords,
                traffic_matrices=mats,
                config=run_cfg,
                node_to_power=node_to_power,
                node_to_failure_factor=node_to_failure_factor,
                cost_params=cp,
                n_stations=len(df_base),
                n_teams=run_cfg["maintenance"]["n_teams"],
                stations_df=df_base,
            )
            initial_state = runner.build_initial_state(
                seed=seed,
                randomize_initial_dsm=fail_cfg.get("randomize_initial_dsm", False),
            )
            result, initial_ov, replan_ov = runner.run(
                initial_state=initial_state,
                rh_config=rh_cfg,
                disruptions_df=mal_df,
                max_days=args.max_days,
            )
            model_params["rolling_horizon"] = rh_cfg

            runner.write_json(result, str(out_path),
                              label="MYOPIC PLUS SIMULATION [Rolling Horizon]",
                              run_id=seed, model_params=model_params,
                              rh_config=rh_cfg,
                              replan_overrides=replan_ov,
                              initial_overrides=initial_ov)
            runner.write_log(result, str(log_path),
                             label="MYOPIC PLUS SIMULATION [Rolling Horizon]")

            completed[seed] = result
            completed_rh[seed] = (initial_ov, replan_ov)

        cost_params_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        sorted_seeds = sorted(completed.keys())
        sorted_init = [completed_rh[s][0] for s in sorted_seeds]
        sorted_replan = [completed_rh[s][1] for s in sorted_seeds]
        overview_text = analyse(
            [completed[s] for s in sorted_seeds], sorted_seeds, cfg=run_cfg,
            cost_params=cost_params_dict,
            initial_overrides=sorted_init if rh_enabled else None,
            replan_overrides=sorted_replan if rh_enabled else None,
        )
        overview_path.write_text(overview_text, encoding="utf-8")


if __name__ == "__main__":
    main()
