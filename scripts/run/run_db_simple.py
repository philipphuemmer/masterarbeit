"""
Startet die DB-Simple-Simulation (CFA-Future + 3-Feature-δ(S_t)).

Voraussetzung (optional): DB-Modell trainieren:
    python scripts/train/train_db_simple.py --collect --train

Ohne Modell läuft die Policy mit festem δ = --delta (Fallback δ=0.5 → identisch zu CFA-Future).

Ausführen:
    python scripts/run/run_db_simple.py
    python scripts/run/run_db_simple.py --max-days 10 --log-day 1 --verbose
    python scripts/run/run_db_simple.py --delta 0.3          # fester δ-Wert (Benchmark)
    python scripts/run/run_db_simple.py --no-model           # explizit kein Modell laden
    python scripts/run/run_db_simple.py --delta 0.5 --no-model  # == CFA-Future (Sanity)

Rolling Horizon:
    rolling_horizon.enabled: true in configs/config.yaml setzen.
    Parameter: horizon_days, n_scenarios, top_k_candidates, time_budget_sec.

Monte Carlo:
    python scripts/run/run_db_simple.py --output logs/db_simple/run_1.json --run-id 1 --seed 1
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
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector


def _build_model_params(cfg: dict, policy: DBSimplePolicy, theta_path: str, db_model: DBSimpleBalanceModel) -> dict:
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
        "theta": policy.theta.tolist() if hasattr(policy.theta, "tolist") else policy.theta,
        "alpha": policy.alpha,
        "p_failure_per_hour": policy.p_failure_per_hour,
        "theta_path": str(theta_path),
        "db_model_trained": db_model.clf is not None,
        "delta_default": db_model.default_delta,
        "delta_mode": cfg.get("db_simple", {}).get("delta_mode", "rf"),
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
    parser = argparse.ArgumentParser(description="DB-Simple-Simulation (CFA-Future + 3-Feature-δ)")
    parser.add_argument("--max-days",   type=int,   default=365,
                        help="Maximale Simulationstage (Standard: 365)")
    parser.add_argument("--log-day",    type=int,   default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output",     type=str,   default="logs/db_simple/run_1.json",
                        help="Ausgabedatei (.json)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Logging aktivieren")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Zufallsseed (-1 = zufällig)")
    parser.add_argument("--run-id",     type=int,   default=None,
                        help="Run-ID für Monte-Carlo-Läufe")
    parser.add_argument("--theta-path", type=str,   default="data/training/cfa_future/theta.json",
                        help="Pfad zur theta.json")
    parser.add_argument("--model-path", type=str,   default="data/training/db_simple/model.pkl",
                        help="Pfad zur DB-Simple-Modell-Datei (.pkl)")
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Fester δ-Wert als Fallback wenn kein Modell geladen (δ=0.5 → CFA-Future)")
    parser.add_argument("--no-model",   action="store_true",
                        help="Explizit kein Modell laden — festes δ verwenden")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["project"]["seed"] = None if args.seed == -1 else args.seed

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

    # DB-Modell laden
    if args.no_model:
        db_model = DBSimpleBalanceModel(default_delta=args.delta)
        print(f"  DB-Modell deaktiviert — festes δ = {args.delta}")
    else:
        model_path = Path(args.model_path)
        db_model = DBSimpleBalanceModel.load(model_path, default_delta=args.delta)
        if db_model.clf is not None:
            print(f"  DB-Modell geladen aus {model_path}")
        else:
            print(f"  Kein Modell unter {model_path} — festes δ = {args.delta}")

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    policy = DBSimplePolicy(
        traffic_matrices=mats,
        config=cfg,
        all_coords=coords,
        stations_df=df,
        db_model=db_model,
        default_delta=args.delta,
        theta_path=args.theta_path,
    )
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._value
        print("  V̂-basierte Zonenauswahl aktiv (DB-Simple).")

    print(f"  CFA-Future θ = {policy.theta}")
    print(f"  δ Fallback   = {db_model.default_delta}")

    rh_cfg = cfg.get("rolling_horizon", {})
    rh_enabled = bool(rh_cfg.get("enabled", False))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_params = _build_model_params(cfg, policy, args.theta_path, db_model)

    if not rh_enabled:
        # ----------------------------------------------------------------
        # Legacy-Pfad: DBSimpleMaintenanceSimulator mit δ-Berechnung pro Tag
        # ----------------------------------------------------------------
        print(f"\nStarte DB-Simple-Simulation [Legacy] (max. {args.max_days} Tage)...\n")
        sim = DBSimpleMaintenanceSimulator(policy, selector, coords, df, mats, cfg)
        result = sim.run(mal_df, max_days=args.max_days)
        model_params["delta_by_day"] = {str(d): v for d, v in sim.delta_log.items()}
        sim.print_summary(result, label="DB-Simple")

        if args.log_day is not None:
            sim.print_day_log(result, args.log_day)

        sim.write_json(
            result, str(out_path),
            label="DB-SIMPLE SIMULATION",
            run_id=args.run_id,
            model_params=model_params,
        )

    else:
        # ----------------------------------------------------------------
        # RH-Pfad: RollingHorizonRunner mit Policy-Improvement
        # ----------------------------------------------------------------
        print(
            f"\nStarte DB-Simple-Simulation [Rolling Horizon] "
            f"(H={rh_cfg.get('horizon_days', 7)}, "
            f"k={rh_cfg.get('top_k_candidates', 3)}, "
            f"S={rh_cfg.get('n_scenarios', 8)}, "
            f"max. {args.max_days} Tage)...\n"
        )

        # Störungswahrscheinlichkeitsfaktoren (für HorizonEvaluator)
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

        def _db_simple_pre_day_hook(state) -> None:
            phi = policy.extract_balance_features(state.remaining, state.days_since_maintenance, state.day)
            delta_mode_rh = cfg.get("db_simple", {}).get("delta_mode", "rf")
            if delta_mode_rh == "rule":
                from src.models.db_simple import rule_delta as _rule_delta
                delta = _rule_delta(float(phi[0]), int(phi[1]))
            elif delta_mode_rh == "rf":
                delta = policy.db_model.predict_delta(phi)
            else:
                delta = policy.db_model.default_delta
            policy.set_precomputed_delta(delta)

        runner.pre_day_hook = _db_simple_pre_day_hook

        fail_cfg_dict = cfg.get("failure_simulation", {})
        initial_state = runner.build_initial_state(
            seed=cfg["project"].get("seed"),
            randomize_initial_dsm=fail_cfg_dict.get("randomize_initial_dsm", False),
        )

        result, initial_overrides, replan_overrides = runner.run(
            initial_state=initial_state,
            rh_config=rh_cfg,
            disruptions_df=mal_df,
            max_days=args.max_days,
        )

        # Zusammenfassung (identisches Format)
        sep = "=" * 62
        print(sep)
        print("  DB-SIMPLE [Rolling Horizon] – ZUSAMMENFASSUNG")
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
        print(f"  Rollout-Overrides       : {initial_overrides + replan_overrides} (Initial: {initial_overrides}, Replan: {replan_overrides})")
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
            result, str(out_path),
            label="DB-SIMPLE SIMULATION [Rolling Horizon]",
            run_id=args.run_id,
            model_params=model_params,
            rh_config=rh_cfg,
            replan_overrides=replan_overrides,
            initial_overrides=initial_overrides,
        )


if __name__ == "__main__":
    main()
