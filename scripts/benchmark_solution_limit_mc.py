"""
Konvergenzanalyse: CFA-Qualität in Abhängigkeit des Solution-Limits.
Läuft je 10 Seeds pro Limit und zeigt ab wo die Qualität plateaut.

Methodik (Masterarbeit): Das Solution-Limit wird als kleinster Wert gewählt,
bei dem sich Ø-Gesamtkosten nicht mehr signifikant verbessern (Konvergenzpunkt).

Ausführen:
    .venv/bin/python3 scripts/benchmark_solution_limit_mc.py
    .venv/bin/python3 scripts/benchmark_solution_limit_mc.py  # Fortsetzung nach Unterbrechung
"""
from __future__ import annotations

import json
import statistics
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

CHECKPOINT = Path("logs/benchmark/solution_limit_mc_cfa.json")
LOGFILE    = Path("logs/benchmark/solution_limit_mc_cfa.log")


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def log(msg: str) -> None:
    print(msg)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")


def run_single(cfg: dict, solution_limit: int, seed: int) -> dict:
    cfg = {**cfg}
    cfg["project"] = {**cfg["project"], "seed": seed}
    cfg["maintenance"] = {
        **cfg["maintenance"],
        "solver_limit_mode": "solution",
        "solver_solution_limit_initial": solution_limit,
        "solver_solution_limit_replan": max(1, solution_limit // 2),
        # Backup-Zeitlimit: identisch mit time-Modus — kein Call dauert länger als bisher
        "solver_time_limit_initial": 2,
        "solver_time_limit_replan": 2,
    }

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    mal_df = None
    if failure_mode == "csv":
        import pandas as pd
        mal_df = pd.read_csv("data/malfunction.csv")

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"], random_state=seed)
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    selector = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    policy = CFAModel(mats, cfg, all_coords=coords, stations_df=df)
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._value

    sim = MaintenanceSimulator(policy, selector, coords, df, mats, cfg)

    t0 = time.perf_counter()
    result = sim.run(mal_df, max_days=365)
    elapsed = time.perf_counter() - t0

    return {
        "solution_limit": solution_limit,
        "seed": seed,
        "total_cost_eur": round(result.total_cost_eur, 2),
        "travel_cost_eur": round(sum(r.fuel_cost_eur for r in result.day_results), 2),
        "disruption_cost_eur": round(sum(r.downtime_cost_eur for r in result.day_results), 2),
        "days_to_complete": result.days_to_complete,
        "same_day_rate": round(result.same_day_rate, 4),
        "wall_time_s": round(elapsed, 1),
    }


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2, cls=_Encoder)


def run_key(limit: int, seed: int) -> str:
    return f"sol{limit}_seed{seed}"


def print_convergence_table(data: dict, limits: list[int]) -> None:
    log("\n" + "=" * 85)
    log("KONVERGENZTABELLE — Ø über alle Seeds")
    log(f"{'Limit':>7}  {'Runs':>4}  {'Ø Gesamt':>12}  {'StdAbw':>8}  "
        f"{'Ø Störung':>11}  {'Ø Tage':>7}  {'Ø Zeit':>8}")
    log("-" * 85)
    prev_mean = None
    for limit in limits:
        runs = [v for k, v in data.items() if k.startswith(f"sol{limit}_seed")]
        if not runs:
            continue
        costs = [r["total_cost_eur"] for r in runs]
        disrupt = [r["disruption_cost_eur"] for r in runs]
        days = [r["days_to_complete"] for r in runs if r["days_to_complete"]]
        wallt = [r["wall_time_s"] for r in runs]
        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
        delta = f"  Δ{mean_cost - prev_mean:+,.0f}€" if prev_mean is not None else ""
        log(f"  {limit:>5}   {len(runs):>4}  "
            f"{mean_cost:>11,.0f}€  "
            f"{std_cost:>7,.0f}€  "
            f"{statistics.mean(disrupt):>10,.0f}€  "
            f"{statistics.mean(days) if days else 0:>6.1f}  "
            f"{statistics.mean(wallt):>7.1f}s"
            f"{delta}")
        prev_mean = mean_cost


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limits", nargs="+", type=int,
                        default=[1, 2, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data = load_checkpoint()
    total = len(args.limits) * len(seeds)
    done = sum(1 for k in data if any(k.startswith(f"sol{l}_") for l in args.limits))

    log("=" * 85)
    log(f"CFA Konvergenzanalyse Solution-Limit | Limits: {args.limits} | Seeds: 1–{args.seeds} "
        f"| {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"Gesamt: {total} Runs | Abgeschlossen: {done} | Ausstehend: {total - done}")
    if data:
        print_convergence_table(data, args.limits)
    log("")

    try:
        for limit in args.limits:
            pending = [s for s in seeds if run_key(limit, s) not in data]
            if not pending:
                log(f"✓ Solution-Limit {limit}: alle {len(seeds)} Runs abgeschlossen.")
                continue

            log(f"▶ Solution-Limit {limit} (replan {max(1, limit//2)}) — {len(pending)} Seeds ausstehend")
            for seed in pending:
                print(f"  Seed {seed:>2} ...", end=" ", flush=True)
                r = run_single(cfg, limit, seed)
                data[run_key(limit, seed)] = r
                save_checkpoint(data)
                log(f"  Seed {seed:>2}  "
                    f"Gesamt {r['total_cost_eur']:>9,.0f}€  "
                    f"Störung {r['disruption_cost_eur']:>9,.0f}€  "
                    f"Tage {r['days_to_complete']}  "
                    f"({r['wall_time_s']:.0f}s)")

            runs = [v for k, v in data.items() if k.startswith(f"sol{limit}_seed")]
            costs = [r["total_cost_eur"] for r in runs]
            log(f"\n  ── Solution-Limit {limit} abgeschlossen ──")
            log(f"     Ø Gesamtkosten: {statistics.mean(costs):>10,.0f}€  "
                f"(±{statistics.stdev(costs) if len(costs)>1 else 0:,.0f}€)\n")

    except KeyboardInterrupt:
        log("\n⚠ Unterbrochen. Bisherige Ergebnisse gespeichert.")

    print_convergence_table(data, args.limits)
    log(f"\nCheckpoint : {CHECKPOINT}")
    log(f"Log        : {LOGFILE}")


if __name__ == "__main__":
    main()
