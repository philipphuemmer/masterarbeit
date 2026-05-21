"""
Startet die Myopic-Plus-Simulation für alle Stationen.

Ausführen:
    python scripts/run/run_myopic_plus.py
    python scripts/run/run_myopic_plus.py --max-days 10 --log-day 1 --verbose

Rolling Horizon:
    rolling_horizon.enabled: true in configs/config.yaml setzen.
    Parameter: horizon_days, n_scenarios, top_k_candidates, time_budget_sec.

Monte Carlo:
    python scripts/run/run_myopic_plus.py --output logs/myopic_plus/run_1.json --run-id 1 --seed 1
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

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices, get_failure_rate_factors
from src.models.myopic_plus import MyopicPlusModel
from src.models.rolling_horizon import (
    DailyTaskGenerator,
    HorizonEvaluator,
    PolicyAdapter,
    RollingHorizonRunner,
)
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


def _build_model_params(cfg: dict, policy: MyopicPlusModel) -> dict:
    cp = policy.cost_params
    fail_cfg = cfg.get("failure_simulation", {})
    params = {
        "seed": cfg["project"].get("seed"),
        "failure_mode": fail_cfg.get("mode", "csv"),
        "n_zones": cfg["planning"]["n_zones"],
        "n_teams": cfg["maintenance"]["n_teams"],
        "max_stations_per_team": cfg["planning"].get("max_stations_per_team"),
        "zone_selection_mode": cfg["planning"].get("zone_selection_mode", "classic"),
        "use_team_assignment": cfg["planning"].get("use_team_assignment", True),
        "alpha": policy.alpha,
        "p_failure_per_hour": policy.p_failure_per_hour,
        "cost_params": {
            "wage_eur_per_hour": cp.wage_eur_per_hour,
            "fuel_eur_per_km": cp.fuel_eur_per_km,
            "downtime_eur_per_kwh": cp.downtime_eur_per_kwh,
        },
    }
    if fail_cfg.get("mode") == "stochastic":
        params["failure_simulation"] = {
            "p1_per_hour": fail_cfg.get("p1_per_hour"),
            "p2_per_hour": fail_cfg.get("p2_per_hour"),
            "recovery_days": fail_cfg.get("recovery_days"),
            "initial_factor": fail_cfg.get("initial_factor"),
        }
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Myopic-Plus-Simulation (Soft-Deadline-Priorisierung)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day", type=int, default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output", type=str, default="logs/myopic_plus/json/simulation.json",
                        help="Ausgabedatei (.json oder .log)")
    parser.add_argument("--verbose", action="store_true", help="Logging aktivieren")
    parser.add_argument("--seed", type=int, default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["project"]["seed"] = None if args.seed == -1 else args.seed
        print(f"  Seed: {cfg['project']['seed']} (überschrieben via --seed)")

    cfa_cfg = cfg.get("cfa", {})
    print(f"  α = {cfa_cfg.get('alpha', 10.0)} (Skalierungsfaktor Ausfallkosten)")

    print("Lade Stationsdaten...")
    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse aus malfunction.csv geladen.")
    else:
        mal_df = None
        print("  Störungsmodus: stochastisch")

    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    policy = MyopicPlusModel(mats, cfg, all_coords=coords, stations_df=df)
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._zone_value
        print("  V̂-basierte Zonenauswahl aktiv (Myopic Plus).")

    rh_cfg = cfg.get("rolling_horizon", {})
    rh_enabled = bool(rh_cfg.get("enabled", False))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_params = _build_model_params(cfg, policy)

    if not rh_enabled:
        # ----------------------------------------------------------------
        # Legacy-Pfad: bestehender MaintenanceSimulator
        # ----------------------------------------------------------------
        print(f"\nStarte Myopic-Plus-Simulation [Legacy] (max. {args.max_days} Tage)...\n")
        sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)
        result = sim.run(mal_df, max_days=args.max_days)
        sim.print_summary(result, label="MYOPIC PLUS")

        if args.log_day is not None:
            sim.print_day_log(result, args.log_day)

        if out_path.suffix == ".log":
            sim.write_log(result, str(out_path), label="MYOPIC PLUS SIMULATION")
        else:
            sim.write_json(result, str(out_path.with_suffix(".json")),
                           label="MYOPIC PLUS SIMULATION", run_id=args.run_id,
                           model_params=model_params)

    else:
        # ----------------------------------------------------------------
        # RH-Pfad: RollingHorizonRunner mit Policy-Improvement
        # ----------------------------------------------------------------
        print(
            f"\nStarte Myopic-Plus-Simulation [Rolling Horizon] "
            f"(H={rh_cfg.get('horizon_days', 7)}, "
            f"k={rh_cfg.get('top_k_candidates', 3)}, "
            f"S={rh_cfg.get('n_scenarios', 8)}, "
            f"max. {args.max_days} Tage)...\n"
        )

        failure_factors = get_failure_rate_factors(df)
        node_to_failure_factor = {i + 1: failure_factors.get(i, 1.0) for i in range(len(df))}
        pwr_col = "Nennleistung Ladeeinrichtung [kW]"
        node_to_power = {
            i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
            for i, (_, row) in enumerate(df.iterrows())
        }

        policy_adapter = PolicyAdapter(policy)
        task_gen = DailyTaskGenerator(
            selector=selector,
            failure_mode=failure_mode,
            n_stations=len(df),
        )
        evaluator = HorizonEvaluator(
            policy=policy_adapter,
            task_generator=task_gen,
            all_coords=coords,
            traffic_matrices=mats,
            config=cfg,
            node_to_power=node_to_power,
            node_to_failure_factor=node_to_failure_factor,
            cost_params=policy.cost_params,
            n_stations=len(df),
            n_teams=cfg["maintenance"]["n_teams"],
        )
        runner = RollingHorizonRunner(
            policy=policy_adapter,
            task_generator=task_gen,
            evaluator=evaluator,
            all_coords=coords,
            traffic_matrices=mats,
            config=cfg,
            node_to_power=node_to_power,
            node_to_failure_factor=node_to_failure_factor,
            cost_params=policy.cost_params,
            n_stations=len(df),
            n_teams=cfg["maintenance"]["n_teams"],
            stations_df=df,
        )

        fail_cfg_dict = cfg.get("failure_simulation", {})
        initial_state = runner.build_initial_state(
            seed=cfg["project"].get("seed"),
            randomize_initial_dsm=fail_cfg_dict.get("randomize_initial_dsm", False),
        )

        result, rh_overrides = runner.run(
            initial_state=initial_state,
            rh_config=rh_cfg,
            disruptions_df=mal_df,
            max_days=args.max_days,
        )

        sep = "=" * 62
        print(sep)
        print("  MYOPIC PLUS [Rolling Horizon] – ZUSAMMENFASSUNG")
        print(sep)
        print(f"  Tage simuliert          : {len(result.day_results)}")
        if result.days_to_complete:
            print(f"  Alle Stationen gewartet : Tag {result.days_to_complete}")
        else:
            print(f"  Verbleibende Stationen  : {result.remaining_stations_at_end}")
        total_completed = sum(r.n_routine_completed for r in result.day_results)
        print(f"  Routine-Wartungen       : {total_completed} / {len(df)}")
        print(f"  Störungen gesamt        : {result.total_disruptions}")
        print(f"    Gleichen Tag erledigt : {result.same_day_handled} ({result.same_day_rate:.1%})")
        print(f"    Carryover             : {result.total_carryover}")
        print(f"  RH-Overrides            : {rh_overrides}")
        print(f"  Gesamtkosten            : {result.total_cost_eur:>10,.2f} €")
        print(sep)

        if args.log_day is not None:
            day_result = next((r for r in result.day_results if r.day == args.log_day), None)
            if day_result:
                print(f"\nTag {args.log_day} – Detail-Log:")
                for log in day_result.hourly_logs:
                    print(str(log))

        model_params["rolling_horizon"] = rh_cfg
        runner.write_json(
            result, str(out_path.with_suffix(".json")),
            label="MYOPIC PLUS SIMULATION [Rolling Horizon]",
            run_id=args.run_id,
            model_params=model_params,
            rh_config=rh_cfg,
            rh_overrides=rh_overrides,
        )


if __name__ == "__main__":
    main()
