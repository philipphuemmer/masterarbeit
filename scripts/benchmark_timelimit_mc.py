"""
MC-Benchmark: CFA mit verschiedenen OR-Tools Zeitlimits, je 10 Runs (Seed 1–10).
Reihenfolge: erst alle Seeds für Limit 2s, dann 5s, dann 10s, dann 20s.
Unterbrechbar — abgeschlossene Runs werden beim Neustart übersprungen.

Ausführen:
    .venv/bin/python3 scripts/benchmark_timelimit_mc.py
    .venv/bin/python3 scripts/benchmark_timelimit_mc.py --limits 2 5 10 20 --seeds 10
"""
from __future__ import annotations

import json
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

CHECKPOINT = Path("logs/benchmark/timelimit_mc_cfa.json")
LOGFILE    = Path("logs/benchmark/timelimit_mc_cfa.log")


def run_single(cfg: dict, limit_seconds: int, seed: int) -> dict:
    cfg = {**cfg}
    cfg["project"] = {**cfg["project"], "seed": seed}
    cfg["maintenance"] = {
        **cfg["maintenance"],
        "solver_time_limit_initial": limit_seconds,
        "solver_time_limit_replan": max(1, limit_seconds // 2),
    }

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        import pandas as pd
        mal_df = pd.read_csv("data/malfunction.csv")
    else:
        mal_df = None

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"], random_state=seed)
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords)
    policy = CFAModel(mats, cfg, all_coords=coords, stations_df=df)
    if cfg["planning"].get("value_based_zone_selection", False):
        selector.value_fn = policy._value

    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    t0 = time.perf_counter()
    result = sim.run(mal_df, max_days=365)
    elapsed = time.perf_counter() - t0

    return {
        "limit_s": limit_seconds,
        "seed": seed,
        "total_cost_eur": round(result.total_cost_eur, 2),
        "travel_cost_eur": round(sum(r.fuel_cost_eur for r in result.day_results), 2),
        "disruption_cost_eur": round(sum(r.downtime_cost_eur for r in result.day_results), 2),
        "stations_serviced": sum(r.n_routine_completed for r in result.day_results),
        "days_to_complete": result.days_to_complete,
        "same_day_rate": round(result.same_day_rate, 4),
        "wall_time_s": round(elapsed, 1),
    }


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def save_checkpoint(data: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2, cls=_Encoder)


def log(msg: str) -> None:
    """Schreibt in Konsole und Logdatei."""
    print(msg)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")


def run_key(limit: int, seed: int) -> str:
    return f"{limit}s_seed{seed}"


def print_summary(data: dict, limits: list[int], n_seeds: int) -> None:
    import statistics
    print("\n" + "=" * 90)
    print(f"{'Limit':>6}  {'Runs':>4}  {'Ø Gesamt':>12}  {'Ø Fahrt':>10}  "
          f"{'Ø Störung':>12}  {'Ø Tage':>8}  {'Ø Laufzeit':>11}")
    print("-" * 90)
    for limit in limits:
        runs = [v for k, v in data.items() if k.startswith(f"{limit}s_seed")]
        if not runs:
            continue
        costs = [r["total_cost_eur"] for r in runs]
        travel = [r["travel_cost_eur"] for r in runs]
        disrupt = [r["disruption_cost_eur"] for r in runs]
        days = [r["days_to_complete"] for r in runs if r["days_to_complete"]]
        wallt = [r["wall_time_s"] for r in runs]
        print(
            f"  {limit:>3}s  {len(runs):>4}  "
            f"{statistics.mean(costs):>11,.0f}€  "
            f"{statistics.mean(travel):>9,.0f}€  "
            f"{statistics.mean(disrupt):>11,.0f}€  "
            f"{statistics.mean(days) if days else 0:>7.1f}  "
            f"{statistics.mean(wallt):>10.1f}s"
        )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limits", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--seeds",  type=int, default=10,
                        help="Anzahl Seeds (1..N, Standard: 10)")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data = load_checkpoint()
    total = len(args.limits) * len(seeds)
    done = len([k for k in data if any(k.startswith(f"{l}s_") for l in args.limits)])

    import statistics

    header = (f"CFA Zeitlimit-Benchmark | Limits: {args.limits}s | Seeds: 1–{args.seeds} "
              f"| {time.strftime('%Y-%m-%d %H:%M')}")
    log("=" * 90)
    log(header)
    log(f"Gesamt: {total} Runs | Abgeschlossen: {done} | Ausstehend: {total - done}")
    if data:
        print_summary(data, args.limits, args.seeds)
    log("")

    try:
        for limit in args.limits:
            pending_seeds = [s for s in seeds if run_key(limit, s) not in data]
            if not pending_seeds:
                log(f"✓ Limit {limit}s: alle {len(seeds)} Runs bereits abgeschlossen.")
                continue

            log(f"▶ Limit {limit}s — {len(pending_seeds)} Seeds ausstehend: {pending_seeds}")
            for seed in pending_seeds:
                print(f"  Seed {seed:>2} ...", end=" ", flush=True)
                r = run_single(cfg, limit, seed)
                data[run_key(limit, seed)] = r
                save_checkpoint(data)
                msg = (f"Gesamt {r['total_cost_eur']:>9,.0f}€  "
                       f"Störung {r['disruption_cost_eur']:>9,.0f}€  "
                       f"Tage {r['days_to_complete']}  "
                       f"({r['wall_time_s']:.0f}s)")
                log(f"  Seed {seed:>2}  {msg}")

            # Übersicht nach allen Seeds dieses Limits
            runs = [v for k, v in data.items() if k.startswith(f"{limit}s_seed")]
            log(f"\n  ── Übersicht Limit {limit}s ({len(runs)} Runs) ──")
            log(f"     Ø Gesamtkosten  : {statistics.mean(r['total_cost_eur'] for r in runs):>10,.0f} €  "
                f"(±{statistics.stdev(r['total_cost_eur'] for r in runs):,.0f})")
            log(f"     Ø Fahrtkosten   : {statistics.mean(r['travel_cost_eur'] for r in runs):>10,.0f} €  "
                f"(±{statistics.stdev(r['travel_cost_eur'] for r in runs):,.0f})")
            log(f"     Ø Störungskosten: {statistics.mean(r['disruption_cost_eur'] for r in runs):>10,.0f} €  "
                f"(±{statistics.stdev(r['disruption_cost_eur'] for r in runs):,.0f})")
            days = [r["days_to_complete"] for r in runs if r["days_to_complete"]]
            if days:
                log(f"     Ø Tage          : {statistics.mean(days):>10.1f}")
            log(f"     Ø Laufzeit      : {statistics.mean(r['wall_time_s'] for r in runs):>10.1f} s\n")

    except KeyboardInterrupt:
        log("\n⚠ Unterbrochen. Bisherige Ergebnisse gespeichert.")

    print_summary(data, args.limits, args.seeds)
    log(f"\nCheckpoint: {CHECKPOINT}")
    log(f"Log:        {LOGFILE}")


if __name__ == "__main__":
    main()
