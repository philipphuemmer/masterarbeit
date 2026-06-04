"""
Monte-Carlo-Simulation für die DB-Simple-Policy (CFA-Future + 3-Feature-δ(S_t)).

Führt N Läufe durch (Seed 1 … N), speichert jeden Lauf als
logs/db_simple/json/run_<N>.json und logs/db_simple/log/run_<N>.log.
Am Ende wird eine aggregierte Analyse als logs/db_simple/log/db_simple_overview.log gespeichert.

rolling_horizon.enabled: false → Legacy-Pfad (DBSimpleMaintenanceSimulator), Log-Dir: logs/db_simple
rolling_horizon.enabled: true  → RH-Pfad (RollingHorizonRunner),             Log-Dir: logs/db_simple_rh

Voraussetzung (optional): DB-Modell trainieren:
    python scripts/train/train_db_simple.py --collect --train

Ohne Modell läuft die Policy mit festem δ = --delta (Standard: 0.5 → identisch zu CFA-Future).

Ausführen:
    python scripts/monte_carlo/run_mc_db_simple.py --runs 30
    python scripts/monte_carlo/run_mc_db_simple.py --runs 30 --delta 0.5   # CFA-Future Referenz
    python scripts/monte_carlo/run_mc_db_simple.py --runs 10 --max-days 50 --verbose
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
from src.models.db_simple import (
    DBSimpleBalanceModel,
    DBSimpleMaintenanceSimulator,
    DBSimplePolicy,
)
from src.models.rolling_horizon import (
    DailyTaskGenerator,
    HorizonEvaluator,
    PolicyAdapter,
    RollingHorizonRunner,
)
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


from src.utils.mc_analyse import analyse


def _db_simple_cfg_lines(cfg: dict | None) -> list[str]:
    if not cfg:
        return []
    ds = cfg.get("db_simple", {})
    return [f"    δ-Modus                  : {ds.get('delta_mode', 'rf')}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo-Simulation (DB-Simple)")
    parser.add_argument("--runs",       type=int, default=30,
                        help="Letzter Seed (inklusiv, Standard: 30)")
    parser.add_argument("--start-run",  type=int, default=1,
                        help="Erster Seed zum Fortfahren (Standard: 1)")
    parser.add_argument("--max-days",   type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Fallback-δ wenn kein Modell geladen (0.5 → CFA-Future identisch)")
    parser.add_argument("--no-model",   action="store_true",
                        help="Kein Modell laden — festes δ verwenden")
    parser.add_argument("--model-path", type=str, default="data/training/db_simple/model.pkl",
                        help="Pfad zur DB-Simple-Modell-Datei (.pkl)")
    parser.add_argument("--theta-path", type=str, default="data/training/cfa_future/theta.json",
                        help="Pfad zur theta.json")
    parser.add_argument("--log-dir",    type=str, default=None,
                        help="Basisordner für JSON- und Log-Ausgaben "
                             "(Standard: logs/db_simple bzw. logs/db_simple_rh)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Logging aktivieren")
    parser.add_argument("--resume",     action="store_true",
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

    # Log-Dir: auto je nach Modus, überschreibbar per --log-dir
    if args.log_dir is None:
        args.log_dir = "logs/db_simple_rh" if rh_enabled else "logs/db_simple"

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
        print("  Störungsmodus: stochastisch")

    if args.no_model:
        db_model = DBSimpleBalanceModel(default_delta=args.delta)
        model_trained = False
        print(f"  DB-Modell deaktiviert — festes δ = {args.delta}")
    else:
        db_model = DBSimpleBalanceModel.load(Path(args.model_path), default_delta=args.delta)
        model_trained = db_model.clf is not None
        if model_trained:
            print(f"  DB-Modell geladen aus {args.model_path}")
        else:
            print(f"  Kein Modell unter {args.model_path} — festes δ = {args.delta}")

    delta_mode = cfg.get("db_simple", {}).get("delta_mode", "rf")

    print("Clustering...")
    clusterer_base = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer_base.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    if rh_enabled:
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
        sorted_init   = [completed_rh[s][0] for s in sorted_seeds]
        sorted_replan = [completed_rh[s][1] for s in sorted_seeds]
        overview_path.write_text(
            analyse(
                [completed[s] for s in sorted_seeds], sorted_seeds, cfg=cfg,
                initial_overrides=sorted_init   if rh_enabled else None,
                replan_overrides=sorted_replan  if rh_enabled else None,
                model_cfg_lines=_db_simple_cfg_lines(cfg),
            ),
            encoding="utf-8",
        )
        return

    # Störungswahrscheinlichkeitsfaktoren für RH-Evaluator (einmal laden)
    if rh_enabled:
        failure_factors = get_failure_rate_factors(df_base)
        node_to_failure_factor = {i + 1: failure_factors.get(i, 1.0) for i in range(len(df_base))}
        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        node_to_power = {
            i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
            for i, (_, row) in enumerate(df_base.iterrows())
        }

    mode_label = "[Rolling Horizon]" if rh_enabled else "[Legacy]"
    print(f"\nStarte {len(seeds_to_run)} Monte-Carlo-Läufe {mode_label} (Seeds {seeds_to_run[0]}–{seeds_to_run[-1]})...\n")

    for seed in seeds_to_run:
        print(f"  Lauf {seed}/{args.runs} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = clusterer_base

        charging_points = df_base["Anzahl Ladepunkte"].fillna(1).astype(int).values
        selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
        policy = DBSimplePolicy(
            traffic_matrices=mats,
            config=run_cfg,
            all_coords=coords,
            stations_df=df_base,
            db_model=db_model,
            default_delta=args.delta,
            theta_path=args.theta_path,
        )
        if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
            selector.value_fn = policy._value

        cp = policy.cost_params
        fail_cfg = run_cfg.get("failure_simulation", {})
        pl_cfg   = run_cfg.get("planning", {})
        mt_cfg   = run_cfg.get("maintenance", {})
        _slm     = mt_cfg.get("solver_limit_mode", "time")

        model_params = {
            "seed": seed,
            "failure_mode": fail_cfg.get("mode", "csv"),
            "n_zones": run_cfg["planning"]["n_zones"],
            "n_teams": run_cfg["maintenance"]["n_teams"],
            "max_stations_per_team": pl_cfg.get("max_stations_per_team"),
            "zone_selection_mode": pl_cfg.get("zone_selection_mode", "classic"),
            "use_team_assignment": pl_cfg.get("use_team_assignment", True),
            "theta": policy.theta.tolist() if hasattr(policy.theta, "tolist") else policy.theta,
            "alpha": policy.alpha,
            "p_failure_per_hour": policy.p_failure_per_hour,
            "theta_path": str(args.theta_path),
            "db_model_trained": model_trained,
            "delta_default": args.delta,
            "delta_mode": delta_mode,
            "cost_params": {
                "wage_eur_per_hour": cp.wage_eur_per_hour,
                "fuel_eur_per_km": cp.fuel_eur_per_km,
                "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
            },
            "zone_selection": {
                "n_zones": pl_cfg.get("n_zones"),
                "n_top_candidates": pl_cfg.get("n_top_candidates"),
                "min_team_separation_km": pl_cfg.get("min_team_separation_km"),
                "max_stations_per_team": pl_cfg.get("max_stations_per_team"),
                "travel_reserve_min": pl_cfg.get("travel_reserve_min"),
                "zone_selection_mode": pl_cfg.get("zone_selection_mode", "classic"),
                "use_team_assignment": pl_cfg.get("use_team_assignment", True),
                "priority_weights": pl_cfg.get("priority_weights", {}),
            },
            "solver": {
                "n_teams": mt_cfg.get("n_teams"),
                "workday_start_hour": mt_cfg.get("workday_start_hour", 8),
                "workday_end_hour": mt_cfg.get("workday_end_hour", 16),
                "mean_service_time": mt_cfg.get("mean_service_time"),
                "global_span_cost_coefficient": mt_cfg.get("global_span_cost_coefficient", 0),
                "lunch_duration_min": mt_cfg.get("lunch_duration_min", 0),
                "solver_limit_mode": _slm,
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

        if not rh_enabled:
            # ------------------------------------------------------------------
            # Legacy-Pfad: DBSimpleMaintenanceSimulator mit δ-Berechnung pro Tag
            # ------------------------------------------------------------------
            sim    = DBSimpleMaintenanceSimulator(policy, selector, coords, df_base, mats, run_cfg)
            result = sim.run(mal_df, max_days=args.max_days)
            model_params["delta_by_day"] = {str(d): v for d, v in sim.delta_log.items()}

            sim.write_json(result, str(out_path),
                           label="DB-SIMPLE SIMULATION", run_id=seed, model_params=model_params)
            log_path = log_dir / f"run_{seed}.log"
            sim.write_log(result, str(log_path), label="DB-SIMPLE SIMULATION")

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

            def _db_simple_pre_day_hook(state, _policy=policy, _cfg=run_cfg) -> None:
                phi = _policy.extract_balance_features(state.remaining, state.days_since_maintenance, state.day)
                _delta_mode = _cfg.get("db_simple", {}).get("delta_mode", "rf")
                if _delta_mode == "rule":
                    from src.models.db_simple import rule_delta as _rule_delta
                    delta = _rule_delta(float(phi[0]), int(phi[1]))
                elif _delta_mode == "rf":
                    delta = _policy.db_model.predict_delta(phi)
                else:
                    delta = _policy.db_model.default_delta
                _policy.set_precomputed_delta(delta)

            runner.pre_day_hook = _db_simple_pre_day_hook

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
                              label="DB-SIMPLE SIMULATION [Rolling Horizon]",
                              run_id=seed, model_params=model_params,
                              rh_config=rh_cfg,
                              replan_overrides=replan_ov,
                              initial_overrides=initial_ov)
            log_path = log_dir / f"run_{seed}.log"
            runner.write_log(result, str(log_path),
                             label="DB-SIMPLE SIMULATION [Rolling Horizon]")

            completed[seed] = result
            completed_rh[seed] = (initial_ov, replan_ov)

        cost_params_dict = {
            "wage_eur_per_hour":    cp.wage_eur_per_hour,
            "fuel_eur_per_km":      cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        }
        sorted_seeds  = sorted(completed.keys())
        sorted_init   = [completed_rh[s][0] for s in sorted_seeds]
        sorted_replan = [completed_rh[s][1] for s in sorted_seeds]
        overview_text = analyse(
            [completed[s] for s in sorted_seeds], sorted_seeds, cfg=run_cfg,
            cost_params=cost_params_dict,
            initial_overrides=sorted_init   if rh_enabled else None,
            replan_overrides=sorted_replan  if rh_enabled else None,
            model_cfg_lines=_db_simple_cfg_lines(run_cfg),
        )
        overview_path.write_text(overview_text, encoding="utf-8")

        days = result.days_to_complete or "?"
        print(f"{result.total_cost_eur:>12,.2f} € | Tage: {str(days):>4}")


if __name__ == "__main__":
    main()
