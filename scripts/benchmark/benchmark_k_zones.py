"""
Zweiphasen-Benchmark zur wissenschaftlichen Bestimmung der optimalen Zonenanzahl k.

Kombiniert interne Cluster-Validierung (Silhouette-Score) mit
Downstream-Performance (CFA- und DB-Wartungskosten) über drei adaptive
Granularitätsstufen:

  Phase 1 — Breites Screening  : k ∈ [20, 200] in 20er-Schritten (10 Werte)
  Phase 2 — Feinanalyse        : k ∈ [center ± 20] in 10er-Schritten (5 Werte)
  Phase 3 — Feinstanalyse      : k ∈ [center ± 2]  in 1er-Schritten  (5 Werte)

Ausgabe:
  data/benchmark_k/benchmark_results.csv    — Alle Messungen
  data/benchmark_k/benchmark_summary.json  — Phasen-Übersicht und Finale Entscheidung
  data/benchmark_k/silhouette_plot.png
  data/benchmark_k/cfa_costs_plot.png
  data/benchmark_k/db_costs_plot.png
  data/benchmark_k/combined_plot.png

Ausführen:
  python scripts/benchmark/benchmark_k_zones.py
  python scripts/benchmark/benchmark_k_zones.py --runs-per-k 5 --max-days 365
  python scripts/benchmark/benchmark_k_zones.py --phase 1          # nur Phase 1
  python scripts/benchmark/benchmark_k_zones.py --resume           # fehlende k ergänzen
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import silhouette_score

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa_future import CFAFutureModel
from src.models.db_simple import DBSimpleBalanceModel, DBSimpleMaintenanceSimulator, DBSimplePolicy
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

PHASE1_K_VALUES = list(range(20, 201, 20))   # [20, 40, ..., 200]
DEFAULT_RUNS_PER_K = 100
DEFAULT_MAX_DAYS = 365
OUTPUT_DIR = Path("data/benchmark_k")
CSV_FILE = OUTPUT_DIR / "benchmark_results.csv"
JSON_FILE = OUTPUT_DIR / "benchmark_summary.json"

CSV_FIELDS = [
    "k", "phase", "seed",
    "silhouette_score",
    "cfa_total_costs", "cfa_runtime",
    "db_simple_total_costs", "db_simple_runtime",
    "timestamp",
]

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _load_summary() -> dict:
    if JSON_FILE.exists():
        with open(JSON_FILE) as f:
            return json.load(f)
    return {}


def _save_summary(summary: dict) -> None:
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_FILE, "w") as f:
        json.dump(summary, f, indent=2)


def _read_csv_results() -> list[dict]:
    if not CSV_FILE.exists():
        return []
    with open(CSV_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _already_measured(k: int, seed: int, existing: list[dict]) -> bool:
    return any(
        int(r["k"]) == k and int(r["seed"]) == seed
        for r in existing
    )


def _silhouette_for_k(coords_stations: np.ndarray, k: int, seed: int) -> float:
    """Berechnet den Silhouette-Score für gegebenes k (nur Clustering, keine Simulation)."""
    clusterer = ZoneClusterer(n_zones=k, random_state=seed)
    clusterer.fit(coords_stations, (0.0, 0.0))  # depot_coords irrelevant für Silhouette
    labels = clusterer.zone_labels_
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(coords_stations, labels, metric="euclidean", sample_size=min(1000, len(coords_stations))))


def _run_cfa_once(
    coords: np.ndarray,
    df: pd.DataFrame,
    mats: list,
    cfg: dict,
    mal_df,
    seed: int,
    max_days: int,
    theta_path: str,
) -> float:
    """Einzelner CFA-Lauf; gibt Gesamtkosten zurück."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["project"]["seed"] = seed

    clusterer = ZoneClusterer(n_zones=run_cfg["planning"]["n_zones"], random_state=seed)
    clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)
    policy = CFAFutureModel(mats, run_cfg, all_coords=coords, stations_df=df, theta_path=theta_path)

    if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._value

    sim = MaintenanceSimulator(policy, selector, coords, df, mats, run_cfg)
    result = sim.run(mal_df, max_days=max_days)
    return float(result.total_cost_eur)


def _run_db_simple_once(
    coords: np.ndarray,
    df: pd.DataFrame,
    mats: list,
    cfg: dict,
    mal_df,
    seed: int,
    max_days: int,
    model_path: str,
    theta_path: str,
) -> float:
    """Einzelner DB-Simple-Lauf; gibt Gesamtkosten zurück."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["project"]["seed"] = seed

    clusterer = ZoneClusterer(n_zones=run_cfg["planning"]["n_zones"], random_state=seed)
    clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    selector = DailyZoneSelector(clusterer, run_cfg, coords, charging_points)

    db_model = DBSimpleBalanceModel.load(Path(model_path), default_delta=0.5)
    policy = DBSimplePolicy(
        traffic_matrices=mats,
        config=run_cfg,
        all_coords=coords,
        stations_df=df,
        db_model=db_model,
        default_delta=0.5,
        theta_path=theta_path,
    )

    if run_cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        selector.value_fn = policy._value

    sim = DBSimpleMaintenanceSimulator(policy, selector, coords, df, mats, run_cfg)
    result = sim.run(mal_df, max_days=max_days)
    return float(result.total_cost_eur)


def _measure_k(
    k: int,
    phase: int,
    seeds: list[int],
    coords: np.ndarray,
    df: pd.DataFrame,
    mats: list,
    cfg: dict,
    mal_df,
    max_days: int,
    theta_path: str,
    db_simple_model_path: str,
    db_simple_theta_path: str,
    resume: bool,
    existing_rows: list[dict],
) -> dict:
    """
    Führt alle Messungen für ein k durch (über mehrere Seeds).
    Gibt aggregierte Ergebnisse zurück: mean silhouette, mean cfa_costs, mean db_costs.
    """
    # Silhouette: seed-unabhängig → einmal mit seed=42
    sil = _silhouette_for_k(coords[1:], k, seed=42)

    cfa_costs_list = []
    db_simple_costs_list = []
    cfa_rt_list = []
    db_simple_rt_list = []

    k_cfg = copy.deepcopy(cfg)
    k_cfg["planning"]["n_zones"] = k

    for seed in seeds:
        if resume and _already_measured(k, seed, existing_rows):
            # Werte aus CSV laden
            row = next(
                r for r in existing_rows
                if int(r["k"]) == k and int(r["seed"]) == seed
            )
            cfa_costs_list.append(float(row["cfa_total_costs"]))
            db_simple_costs_list.append(float(row["db_simple_total_costs"]))
            cfa_rt_list.append(float(row["cfa_runtime"]))
            db_simple_rt_list.append(float(row["db_simple_runtime"]))
            print(f"    Seed {seed}: aus CSV geladen.")
            continue

        print(f"    Seed {seed}: ", end="", flush=True)

        # CFA
        t0 = time.perf_counter()
        cfa_cost = _run_cfa_once(coords, df, mats, k_cfg, mal_df, seed, max_days, theta_path)
        cfa_rt = time.perf_counter() - t0
        cfa_costs_list.append(cfa_cost)
        cfa_rt_list.append(cfa_rt)
        print(f"CFA={cfa_cost:,.0f}€ ({cfa_rt:.1f}s)  ", end="", flush=True)

        # DB-Simple
        t0 = time.perf_counter()
        db_simple_cost = _run_db_simple_once(
            coords, df, mats, k_cfg, mal_df, seed, max_days,
            db_simple_model_path, db_simple_theta_path,
        )
        db_simple_rt = time.perf_counter() - t0
        db_simple_costs_list.append(db_simple_cost)
        db_simple_rt_list.append(db_simple_rt)
        print(f"DB-Simple={db_simple_cost:,.0f}€ ({db_simple_rt:.1f}s)")

        _append_csv({
            "k": k,
            "phase": phase,
            "seed": seed,
            "silhouette_score": round(sil, 6),
            "cfa_total_costs": round(cfa_cost, 2),
            "cfa_runtime": round(cfa_rt, 2),
            "db_simple_total_costs": round(db_simple_cost, 2),
            "db_simple_runtime": round(db_simple_rt, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    return {
        "k": k,
        "phase": phase,
        "silhouette_score": sil,
        "cfa_mean_costs": float(np.mean(cfa_costs_list)),
        "cfa_std_costs": float(np.std(cfa_costs_list)),
        "cfa_mean_runtime": float(np.mean(cfa_rt_list)),
        "db_simple_mean_costs": float(np.mean(db_simple_costs_list)),
        "db_simple_std_costs": float(np.std(db_simple_costs_list)),
        "db_simple_mean_runtime": float(np.mean(db_simple_rt_list)),
    }


def _best_k_by(results: list[dict], metric: str, mode: str = "min") -> int:
    """Gibt das k mit dem besten (min oder max) Metrikwert zurück."""
    fn = min if mode == "min" else max
    best = fn(results, key=lambda r: r[metric])
    return best["k"]


def _phase2_range(center: int) -> list[int]:
    candidates = sorted({center - 20, center - 10, center, center + 10, center + 20})
    return [k for k in candidates if k >= 2]


def _phase3_range(center: int) -> list[int]:
    candidates = list(range(center - 5, center + 6))
    return [k for k in candidates if k >= 2]


# ---------------------------------------------------------------------------
# Plot-Funktionen
# ---------------------------------------------------------------------------


def _generate_plots(all_results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib nicht verfügbar — Plots übersprungen.")
        return

    ks = [r["k"] for r in all_results]
    sils = [r["silhouette_score"] for r in all_results]
    cfa_costs = [r["cfa_mean_costs"] for r in all_results]
    db_costs = [r["db_simple_mean_costs"] for r in all_results]

    phase_colors = {1: "steelblue", 2: "darkorange", 3: "forestgreen"}
    phases = [r["phase"] for r in all_results]

    def _scatter(ax, x, y, p_list, **kwargs):
        for phase, color in phase_colors.items():
            mask = [i for i, p in enumerate(p_list) if p == phase]
            if mask:
                ax.scatter([x[i] for i in mask], [y[i] for i in mask],
                           color=color, label=f"Phase {phase}", zorder=3, **kwargs)
        ax.plot(x, y, color="gray", linewidth=0.8, zorder=1)

    # --- Silhouette ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _scatter(ax, ks, sils, phases, s=60)
    ax.set_xlabel("Zonenanzahl k")
    ax.set_ylabel("Silhouette-Score")
    ax.set_title("K-Means Silhouette-Score vs. Zonenanzahl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "silhouette_plot.png", dpi=150)
    plt.close(fig)

    # --- CFA ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _scatter(ax, ks, cfa_costs, phases, s=60)
    ax.set_xlabel("Zonenanzahl k")
    ax.set_ylabel("Gesamtkosten (€)")
    ax.set_title("CFA-Gesamtkosten vs. Zonenanzahl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cfa_costs_plot.png", dpi=150)
    plt.close(fig)

    # --- DB ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _scatter(ax, ks, db_costs, phases, s=60)
    ax.set_xlabel("Zonenanzahl k")
    ax.set_ylabel("Gesamtkosten (€)")
    ax.set_title("DB-Simple-Gesamtkosten vs. Zonenanzahl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "db_costs_plot.png", dpi=150)
    plt.close(fig)

    # --- Combined ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, y_data, ylabel, title in [
        (axes[0], sils,      "Silhouette-Score",  "Silhouette-Score"),
        (axes[1], cfa_costs, "Gesamtkosten (€)",  "CFA-Future-Kosten"),
        (axes[2], db_costs,  "Gesamtkosten (€)",  "DB-Simple-Kosten"),
    ]:
        _scatter(ax, ks, y_data, phases, s=50)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if "€" in ylabel:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].legend(loc="upper right")
    axes[2].set_xlabel("Zonenanzahl k")
    fig.suptitle("Benchmark Zonenanzahl k — Alle Metriken", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "combined_plot.png", dpi=150)
    plt.close(fig)

    print(f"  Plots gespeichert in {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Haupt-Benchmark-Logik
# ---------------------------------------------------------------------------


def run_phase(
    phase: int,
    k_values: list[int],
    seeds: list[int],
    coords: np.ndarray,
    df: pd.DataFrame,
    mats: list,
    cfg: dict,
    mal_df,
    max_days: int,
    theta_path: str,
    db_simple_model_path: str,
    db_simple_theta_path: str,
    resume: bool,
    existing_rows: list[dict],
) -> list[dict]:
    """Führt alle k-Messungen einer Phase durch; gibt Liste von Ergebnis-Dicts zurück."""
    print(f"\n{'='*60}")
    print(f"  PHASE {phase}  —  k ∈ {k_values}")
    print(f"{'='*60}")
    results = []
    for k in k_values:
        print(f"\n  k={k}:")
        res = _measure_k(
            k=k, phase=phase,
            seeds=seeds, coords=coords, df=df, mats=mats,
            cfg=cfg, mal_df=mal_df, max_days=max_days,
            theta_path=theta_path,
            db_simple_model_path=db_simple_model_path,
            db_simple_theta_path=db_simple_theta_path,
            resume=resume, existing_rows=existing_rows,
        )
        results.append(res)
        print(
            f"    → Silhouette={res['silhouette_score']:.4f}  "
            f"CFA={res['cfa_mean_costs']:,.0f}€  "
            f"DB-Simple={res['db_simple_mean_costs']:,.0f}€"
        )
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark zur Bestimmung der optimalen Zonenanzahl k"
    )
    parser.add_argument(
        "--runs-per-k", type=int, default=DEFAULT_RUNS_PER_K,
        help=f"Anzahl Seeds pro k-Wert (Standard: {DEFAULT_RUNS_PER_K})"
    )
    parser.add_argument(
        "--max-days", type=int, default=DEFAULT_MAX_DAYS,
        help=f"Maximale Simulationstage pro Lauf (Standard: {DEFAULT_MAX_DAYS})"
    )
    parser.add_argument(
        "--theta-path", type=str, default="data/training/cfa_future/theta.json",
        help="Pfad zur CFA-Future theta.json"
    )
    parser.add_argument(
        "--db-simple-model-path", type=str, default="data/training/db_simple/model.pkl",
        help="Pfad zur DB-Simple model.pkl"
    )
    parser.add_argument(
        "--db-simple-theta-path", type=str, default=None,
        help="Pfad zur DB-Simple theta.json (Standard: identisch mit --theta-path)"
    )
    parser.add_argument(
        "--phase", type=int, default=None, choices=[1, 2, 3],
        help="Nur diese Phase ausführen (ohne automatische Phasenadaption)"
    )
    parser.add_argument(
        "--phase1-k", type=str, default=None,
        help="Komma-getrennte k-Werte für Phase 1 (überschreibt Standard)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Vorhandene Messungen aus CSV laden; nur fehlende k/Seeds nachholen"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Keine Simulationen — nur Plots aus vorhandener CSV generieren"
    )
    parser.add_argument(
        "--seed-start", type=int, default=1,
        help="Erster Seed (Standard: 1)"
    )
    args = parser.parse_args()
    if args.db_simple_theta_path is None:
        args.db_simple_theta_path = args.theta_path

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_csv_results() if args.resume else []

    # --- Plots-only Modus ---
    if args.plots_only:
        rows = _read_csv_results()
        if not rows:
            print("Keine CSV-Daten vorhanden.")
            sys.exit(1)
        # Aggregieren: pro k Mittelwert berechnen
        from collections import defaultdict
        buckets: dict[int, list] = defaultdict(list)
        for r in rows:
            buckets[int(r["k"])].append(r)
        all_results = []
        for k, krows in sorted(buckets.items()):
            all_results.append({
                "k": k,
                "phase": int(krows[0]["phase"]),
                "silhouette_score": float(krows[0]["silhouette_score"]),
                "cfa_mean_costs": float(np.mean([float(r["cfa_total_costs"]) for r in krows])),
                "db_simple_mean_costs": float(np.mean([float(r["db_simple_total_costs"]) for r in krows])),
            })
        _generate_plots(all_results)
        return

    # --- Daten laden (einmalig) ---
    print("Lade Stationsdaten...")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    if failure_mode == "csv":
        mal_df = pd.read_csv("data/malfunction.csv")
        print(f"  {len(mal_df)} Störereignisse geladen.")
    else:
        mal_df = None
        print("  Störungsmodus: stochastisch")

    seeds = list(range(args.seed_start, args.seed_start + args.runs_per_k))

    # ---------------------------------------------------------------------------
    # Phase 1
    # ---------------------------------------------------------------------------
    if args.phase1_k:
        phase1_k_values = [int(x) for x in args.phase1_k.split(",")]
    else:
        phase1_k_values = PHASE1_K_VALUES

    if args.phase in (None, 1):
        phase1_results = run_phase(
            phase=1, k_values=phase1_k_values,
            seeds=seeds, coords=coords, df=df, mats=mats,
            cfg=cfg, mal_df=mal_df, max_days=args.max_days,
            theta_path=args.theta_path,
            db_simple_model_path=args.db_simple_model_path,
            db_simple_theta_path=args.db_simple_theta_path,
            resume=args.resume, existing_rows=existing_rows,
        )

        k_best_sil_p1 = _best_k_by(phase1_results, "silhouette_score", "max")
        k_best_cfa_p1 = _best_k_by(phase1_results, "cfa_mean_costs", "min")
        k_best_db_p1  = _best_k_by(phase1_results, "db_simple_mean_costs",  "min")
        k_center_p1 = k_best_cfa_p1  # CFA als primärer Proxy

        summary = _load_summary()
        summary["phase1_results"] = {
            "k_range": phase1_k_values,
            "best_silhouette": {
                "k": k_best_sil_p1,
                "score": next(r["silhouette_score"] for r in phase1_results if r["k"] == k_best_sil_p1),
            },
            "best_cfa": {
                "k": k_best_cfa_p1,
                "costs": next(r["cfa_mean_costs"] for r in phase1_results if r["k"] == k_best_cfa_p1),
            },
            "best_db_simple": {
                "k": k_best_db_p1,
                "costs": next(r["db_simple_mean_costs"] for r in phase1_results if r["k"] == k_best_db_p1),
            },
            "phase2_k_center": k_center_p1,
        }
        _save_summary(summary)

        print(f"\nPhase 1 abgeschlossen:")
        print(f"  Bestes Silhouette-k = {k_best_sil_p1}")
        print(f"  Bestes CFA-k        = {k_best_cfa_p1}")
        print(f"  Bestes DB-k         = {k_best_db_p1}")
        print(f"  Phase-2-Zentrum     = {k_center_p1}")

        if args.phase == 1:
            _generate_plots(phase1_results)
            return
    else:
        # Phase 1 aus Summary laden
        summary = _load_summary()
        p1 = summary.get("phase1_results", {})
        k_center_p1 = p1.get("phase2_k_center")
        if k_center_p1 is None:
            print("Fehler: Phase-1-Ergebnisse nicht gefunden. Zuerst --phase 1 ausführen.")
            sys.exit(1)
        phase1_results = []

    all_results: list[dict] = list(phase1_results)

    # ---------------------------------------------------------------------------
    # Phase 2
    # ---------------------------------------------------------------------------
    phase2_k_values = _phase2_range(k_center_p1)
    # Werte aus Phase 1 nicht erneut messen
    phase2_k_new = [k for k in phase2_k_values if k not in [r["k"] for r in phase1_results]]

    if args.phase in (None, 2):
        if phase2_k_new:
            phase2_results_new = run_phase(
                phase=2, k_values=phase2_k_new,
                seeds=seeds, coords=coords, df=df, mats=mats,
                cfg=cfg, mal_df=mal_df, max_days=args.max_days,
                theta_path=args.theta_path,
            db_simple_model_path=args.db_simple_model_path,
            db_simple_theta_path=args.db_simple_theta_path,
                resume=args.resume, existing_rows=existing_rows,
            )
        else:
            phase2_results_new = []

        # Phase-2-Ergebnisse: neue Messungen + wiederverwendete Phase-1-Werte
        phase2_results = [
            r for r in phase1_results if r["k"] in phase2_k_values
        ] + phase2_results_new

        k_best_sil_p2 = _best_k_by(phase2_results, "silhouette_score", "max")
        k_best_cfa_p2 = _best_k_by(phase2_results, "cfa_mean_costs", "min")
        k_best_db_p2  = _best_k_by(phase2_results, "db_simple_mean_costs",  "min")
        k_center_p2 = k_best_cfa_p2

        summary = _load_summary()
        summary["phase2_results"] = {
            "k_range": phase2_k_values,
            "best_silhouette": {
                "k": k_best_sil_p2,
                "score": next(r["silhouette_score"] for r in phase2_results if r["k"] == k_best_sil_p2),
            },
            "best_cfa": {
                "k": k_best_cfa_p2,
                "costs": next(r["cfa_mean_costs"] for r in phase2_results if r["k"] == k_best_cfa_p2),
            },
            "best_db_simple": {
                "k": k_best_db_p2,
                "costs": next(r["db_simple_mean_costs"] for r in phase2_results if r["k"] == k_best_db_p2),
            },
            "phase3_k_center": k_center_p2,
        }
        _save_summary(summary)

        print(f"\nPhase 2 abgeschlossen:")
        print(f"  Bestes Silhouette-k = {k_best_sil_p2}")
        print(f"  Bestes CFA-k        = {k_best_cfa_p2}")
        print(f"  Bestes DB-k         = {k_best_db_p2}")
        print(f"  Phase-3-Zentrum     = {k_center_p2}")

        all_results.extend(phase2_results_new)

        if args.phase == 2:
            _generate_plots(all_results)
            return
    else:
        summary = _load_summary()
        p2 = summary.get("phase2_results", {})
        k_center_p2 = p2.get("phase3_k_center")
        if k_center_p2 is None:
            print("Fehler: Phase-2-Ergebnisse nicht gefunden. Zuerst --phase 2 ausführen.")
            sys.exit(1)

    # ---------------------------------------------------------------------------
    # Phase 3
    # ---------------------------------------------------------------------------
    phase3_k_values = _phase3_range(k_center_p2)
    phase3_k_new = [
        k for k in phase3_k_values
        if k not in [r["k"] for r in all_results]
    ]

    if args.phase in (None, 3):
        if phase3_k_new:
            phase3_results_new = run_phase(
                phase=3, k_values=phase3_k_new,
                seeds=seeds, coords=coords, df=df, mats=mats,
                cfg=cfg, mal_df=mal_df, max_days=args.max_days,
                theta_path=args.theta_path,
            db_simple_model_path=args.db_simple_model_path,
            db_simple_theta_path=args.db_simple_theta_path,
                resume=args.resume, existing_rows=existing_rows,
            )
        else:
            phase3_results_new = []

        phase3_results = [
            r for r in all_results if r["k"] in phase3_k_values
        ] + phase3_results_new

        k_best_sil_p3 = _best_k_by(phase3_results, "silhouette_score", "max")
        k_best_cfa_p3 = _best_k_by(phase3_results, "cfa_mean_costs", "min")
        k_best_db_p3  = _best_k_by(phase3_results, "db_simple_mean_costs",  "min")

        if k_best_sil_p3 == k_best_cfa_p3:
            k_final = k_best_sil_p3
            decision_logic = "consensus_silhouette_cfa"
        else:
            k_final = k_best_cfa_p3
            decision_logic = "priority_cfa"

        final_result = next(r for r in phase3_results if r["k"] == k_final)

        summary = _load_summary()
        summary["phase3_results"] = {
            "k_range": phase3_k_values,
            "best_silhouette": {
                "k": k_best_sil_p3,
                "score": next(r["silhouette_score"] for r in phase3_results if r["k"] == k_best_sil_p3),
            },
            "best_cfa": {
                "k": k_best_cfa_p3,
                "costs": next(r["cfa_mean_costs"] for r in phase3_results if r["k"] == k_best_cfa_p3),
            },
            "best_db_simple": {
                "k": k_best_db_p3,
                "costs": next(r["db_simple_mean_costs"] for r in phase3_results if r["k"] == k_best_db_p3),
            },
        }
        summary["final_decision"] = {
            "k_final": k_final,
            "silhouette_score": final_result["silhouette_score"],
            "cfa_costs": final_result["cfa_mean_costs"],
            "db_simple_costs": final_result["db_simple_mean_costs"],
            "decision_logic": decision_logic,
            "runs_per_k": args.runs_per_k,
            "max_days": args.max_days,
        }
        _save_summary(summary)

        all_results.extend(phase3_results_new)

        print(f"\n{'='*60}")
        print(f"  FINALE ENTSCHEIDUNG")
        print(f"{'='*60}")
        print(f"  Optimales k          = {k_final}")
        print(f"  Silhouette-Score     = {final_result['silhouette_score']:.4f}")
        print(f"  CFA-Kosten (MW)      = {final_result['cfa_mean_costs']:,.0f} €")
        print(f"  DB-Simple-Kosten (MW)= {final_result['db_simple_mean_costs']:,.0f} €")
        print(f"  Entscheidungslogik   = {decision_logic}")
        print(f"  Summary → {JSON_FILE}")

    _generate_plots(all_results)
    print(f"\nCSV   → {CSV_FILE}")
    print(f"JSON  → {JSON_FILE}")


if __name__ == "__main__":
    main()
