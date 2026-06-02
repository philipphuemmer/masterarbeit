"""
DB-Simple Training: Label-Erzeugung per CRN-Rollout + Modelltraining.

Phase 1 — collect (--collect):
    Für jeden Tagesstart-Snapshot aus Pilot-Simulationen:
        Für jedes δ ∈ DELTA_GRID:
            Rollout von diesem Zustand über H Tage mit CRN (gleiche Störungen).
        Label = δ mit niedrigsten H-Tage-Gesamtkosten.
    → (φ_3d, δ*)-Paare in data/training/db_simple/labels.pkl

Phase 2 — train (--train):
    RandomForestClassifier auf den Labels trainieren.
    → data/training/db_simple/model.pkl

Statischer Benchmark (--benchmark):
    Alle δ ∈ DELTA_GRID über mehrere Simulationsläufe evaluieren.

Sanity-Check (--sanity):
    Vergleich DBSimplePolicy(δ=0.5) vs. CFAFutureModel auf identischen Daten.
    Soll identische Ergebnisse liefern (Regressionstest).

Ausführen:
    python scripts/train/train_db_simple.py --benchmark --max-days 50 --n-runs 5
    python scripts/train/train_db_simple.py --collect --n-outer 10 --horizon 20
    python scripts/train/train_db_simple.py --train
    python scripts/train/train_db_simple.py --sanity --max-days 5
"""
from __future__ import annotations

import argparse
import copy
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cost_params import CostParams
from src.models.db_simple import (
    DBSimpleBalanceModel,
    DBSimpleMaintenanceSimulator,
    DBSimplePolicy,
)
from src.models.cfa_future import CFAFutureModel
from src.models.simulator import DisruptionEvent, MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import MaintenanceTask, TeamState

logger = logging.getLogger(__name__)

_DEFAULT_LABELS_PATH = Path("data/training/db_simple/labels.pkl")
_DEFAULT_MODEL_PATH  = Path("data/training/db_simple/model.pkl")

DELTA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _build_policy(cfg, mats, coords, df, default_delta):
    """Erzeugt eine frische DBSimplePolicy mit festem δ = default_delta."""
    db_model = DBSimpleBalanceModel(default_delta=default_delta)
    return DBSimplePolicy(
        traffic_matrices=mats,
        config=cfg,
        all_coords=coords,
        stations_df=df,
        db_model=db_model,
        default_delta=default_delta,
    )


def _build_selector(clusterer, cfg, coords, charging_points, policy):
    sel = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        sel.value_fn = policy._value
    return sel


def _pregenerate_disruptions(
    dsm_array: np.ndarray,
    n_days: int,
    start_day: int,
    cfg: dict,
    node_to_power: dict[int, float],
    node_to_failure_factor: dict[int, float],
    n_stations: int,
    crn_seed: int,
) -> dict[int, list[DisruptionEvent]]:
    """Generiert Störungen für n_days mit festem CRN-Seed (identisch zu train_db_base)."""
    fail_cfg = cfg["failure_simulation"]
    p1_base: float = fail_cfg["p1_per_hour"]
    p2_base: float = fail_cfg["p2_per_hour"]
    recovery_days: float = float(fail_cfg.get("recovery_days", 365))
    initial_factor: float = float(fail_cfg.get("initial_factor", 0.1))
    cp = CostParams()

    rng = np.random.default_rng(crn_seed)
    disps_by_day: dict[int, list[DisruptionEvent]] = {}

    for day_offset in range(n_days):
        day = start_day + day_offset
        events: list[DisruptionEvent] = []
        disrupted_today: set[int] = set()

        for hour in range(8, 17):
            for node_idx in range(1, n_stations + 1):
                if node_idx in disrupted_today:
                    continue
                t = min(dsm_array[node_idx], recovery_days)
                curve = initial_factor + (1.0 - initial_factor) * t / recovery_days
                station_factor = node_to_failure_factor.get(node_idx, 1.0)
                power_kw = node_to_power.get(node_idx, 22.0)

                if rng.random() < p1_base * curve * station_factor:
                    events.append(DisruptionEvent(
                        day=day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 1", power_kw=power_kw,
                        service_min=float(cp.typ1_service_min),
                    ))
                    disrupted_today.add(node_idx)
                    continue

                if rng.random() < p2_base * curve * station_factor:
                    service_min = (
                        cp.typ2_dismount_min + 60.0
                        + cp.typ2_handling_min + cp.typ2_remount_min
                    )
                    events.append(DisruptionEvent(
                        day=day, hour=hour, node_idx=node_idx,
                        disruption_type="Typ 2", power_kw=power_kw,
                        service_min=service_min,
                    ))
                    disrupted_today.add(node_idx)

        disps_by_day[day] = events

    return disps_by_day


def _rollout_from_snapshot(
    snapshot: dict,
    delta: float,
    disps_by_day: dict[int, list[DisruptionEvent]],
    horizon_days: int,
    cfg: dict,
    coords: np.ndarray,
    df: pd.DataFrame,
    mats: dict,
    clusterer,
    charging_points: np.ndarray,
    node_to_power: dict[int, float],
) -> float:
    """Rollout von einem Snapshot aus über horizon_days mit festem δ (CRN)."""
    policy = _build_policy(cfg, mats, coords, df, delta)
    selector = _build_selector(clusterer, cfg, coords, charging_points, policy)

    sim = DBSimpleMaintenanceSimulator(policy, selector, coords, df, mats, cfg)
    sim._days_since_maintenance = snapshot["dsm_array"].copy()

    remaining: set[int] = set(snapshot["remaining"])
    carryover: list[MaintenanceTask] = list(snapshot["carryover"])
    n_teams = cfg["maintenance"]["n_teams"]
    workday_min = sim.WORKDAY_MINUTES
    total_cost = 0.0

    for day_offset in range(horizon_days):
        if not remaining:
            break
        day = snapshot["day"] + day_offset
        team_states = [
            TeamState(team_id=i, current_node=0, current_time=0)
            for i in range(n_teams)
        ]
        day_disruptions = disps_by_day.get(day, [])

        result, sim_routes, new_carryover = sim._run_day(
            day, remaining, team_states, carryover, day_disruptions
        )
        total_cost += result.total_cost_eur

        for route in sim_routes:
            for stop in route.stops:
                if stop.task_type == "routine" and stop.departure_min <= workday_min:
                    remaining.discard(stop.node_idx - 1)

        sim._days_since_maintenance += 1.0
        for route in sim_routes:
            for stop in route.stops:
                if stop.departure_min <= workday_min:
                    sim._days_since_maintenance[stop.node_idx] = 0.0

        carryover = [
            MaintenanceTask(
                node_idx=d.node_idx,
                task_type="carryover",
                priority=1,
                service_time=int(round(d.service_min)),
            )
            for d in new_carryover
        ]

    return total_cost


# ---------------------------------------------------------------------------
# Phase 1: Label-Sammlung
# ---------------------------------------------------------------------------

def collect_labels(
    n_outer: int = 10,
    horizon: int = 20,
    sample_every: int = 5,
    output_path: Path = _DEFAULT_LABELS_PATH,
    max_days: int = 365,
    crn_seed_base: int = 9000,
    pilot_delta: float = 0.5,
) -> None:
    """
    Erzeugt (φ_3d, δ*)-Labels per CRN-Rollout.

    Pilot-Simulation mit pilot_delta → Snapshots alle sample_every Tage →
    für jeden Snapshot H-Tage-Rollout mit jedem δ ∈ DELTA_GRID (CRN) →
    Label = bestes δ.
    """
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("failure_simulation", {}).get("mode") == "stochastic", (
        "collect_labels() benötigt failure_simulation.mode: stochastic"
    )

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)
    n_stations = len(df)

    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power: dict[int, float] = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df.iterrows())
    }
    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values

    from src.data.loader import get_failure_rate_factors
    _factors = get_failure_rate_factors(df)
    node_to_failure_factor: dict[int, float] = {
        i + 1: _factors.get(i, 1.0) for i in range(n_stations)
    }

    X_all: list[np.ndarray] = []
    y_all: list[float] = []

    for outer in range(n_outer):
        seed = cfg["project"]["seed"] + outer
        cfg_run = copy.deepcopy(cfg)
        cfg_run["project"]["seed"] = seed

        snapshots: list[dict] = []

        class _SnapshotSim(DBSimpleMaintenanceSimulator):
            def _run_day(self, day, remaining, team_states, carryover, day_disps):
                if day % sample_every == 0 and len(remaining) > 0:
                    dsm_map = getattr(self, "_days_since_maintenance", None)
                    if dsm_map is not None:
                        snapshots.append({
                            "day": day,
                            "remaining": frozenset(remaining),
                            "carryover": list(carryover),
                            "dsm_array": dsm_map.copy(),
                        })
                return super()._run_day(day, remaining, team_states, carryover, day_disps)

        pilot_policy = _build_policy(cfg_run, mats, coords, df, pilot_delta)
        selector = _build_selector(clusterer, cfg_run, coords, charging_points, pilot_policy)
        pilot_sim = _SnapshotSim(pilot_policy, selector, coords, df, mats, cfg_run)
        pilot_sim.run(None, max_days=max_days)

        print(f"  Outer {outer + 1}/{n_outer}: {len(snapshots)} Snapshots", flush=True)

        for snap in snapshots:
            crn_seed = crn_seed_base + outer * 100000 + snap["day"]
            disps = _pregenerate_disruptions(
                dsm_array=snap["dsm_array"],
                n_days=horizon,
                start_day=snap["day"],
                cfg=cfg_run,
                node_to_power=node_to_power,
                node_to_failure_factor=node_to_failure_factor,
                n_stations=n_stations,
                crn_seed=crn_seed,
            )

            delta_costs: dict[float, float] = {}
            for delta in DELTA_GRID:
                cost = _rollout_from_snapshot(
                    snapshot=snap,
                    delta=delta,
                    disps_by_day=disps,
                    horizon_days=horizon,
                    cfg=cfg_run,
                    coords=coords,
                    df=df,
                    mats=mats,
                    clusterer=clusterer,
                    charging_points=charging_points,
                    node_to_power=node_to_power,
                )
                delta_costs[delta] = cost

            best_delta = min(delta_costs, key=delta_costs.get)

            # 3d-Feature aus Snapshot-Zustand
            feat_policy = _build_policy(cfg_run, mats, coords, df, 0.5)
            phi = feat_policy.extract_balance_features(
                remaining=snap["remaining"],
                dsm_array=snap["dsm_array"],
                day=snap["day"],
            )
            X_all.append(phi)
            y_all.append(best_delta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"X": np.array(X_all), "y": np.array(y_all)}, f)

    dist = {d: int(np.sum(np.array(y_all) == d)) for d in DELTA_GRID}
    print(f"\nLabels gespeichert: {len(X_all)} Samples → {output_path}")
    print(f"δ-Verteilung: {dist}")


# ---------------------------------------------------------------------------
# Phase 2: Modelltraining
# ---------------------------------------------------------------------------

def train_model(
    labels_path: Path = _DEFAULT_LABELS_PATH,
    model_path: Path = _DEFAULT_MODEL_PATH,
    classifier: str = "rf",
) -> None:
    """Trainiert DBSimpleBalanceModel aus (φ_3d, δ*)-Paaren."""
    if not labels_path.exists():
        print(f"Labels nicht gefunden: {labels_path}")
        print("Zuerst: python scripts/train/train_db_simple.py --collect")
        return

    with open(labels_path, "rb") as f:
        data = pickle.load(f)

    X: np.ndarray = np.array(data["X"], dtype=float)
    y_raw: np.ndarray = np.array(data["y"], dtype=float)
    y = y_raw.astype(str)
    print(f"Labels: {len(X)} Samples, {X.shape[1]} Features (3d)")
    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"δ-Verteilung: {dict(zip(unique.tolist(), counts.tolist()))}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42
        )
    elif classifier == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unbekannter Classifier: {classifier}")

    clf.fit(X_scaled, y)

    n_cv = min(5, len(X) // 5)
    if n_cv >= 2:
        from sklearn.model_selection import cross_val_score
        cv = cross_val_score(clf, X_scaled, y, cv=n_cv, scoring="accuracy")
        print(f"CV-Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
    else:
        train_acc = float(np.mean(clf.predict(X_scaled) == y))
        print(f"Train-Accuracy: {train_acc:.3f} (zu wenige Samples für CV)")

    model = DBSimpleBalanceModel(clf=clf, scaler=scaler, default_delta=0.5)
    model.save(model_path)
    print(f"Modell gespeichert: {model_path}")


# ---------------------------------------------------------------------------
# Statischer Benchmark
# ---------------------------------------------------------------------------

def run_delta_benchmark(
    delta_values: list[float] = DELTA_GRID,
    max_days: int = 365,
    n_runs: int = 5,
    output_dir: str = "logs/db_simple/benchmark",
) -> None:
    """Evaluiert alle festen δ-Werte und bestimmt die beste statische Baseline."""
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"], random_state=42)
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    pwr_col = "Nennleistung Ladeeinrichtung [kW]"
    node_to_power = {
        i + 1: (float(row[pwr_col]) if pd.notna(row.get(pwr_col)) else 22.0)
        for i, (_, row) in enumerate(df.iterrows())
    }
    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    mal_df = pd.read_csv("data/malfunction.csv") if failure_mode == "csv" else None

    results = []
    for delta in delta_values:
        run_costs = []
        for run in range(n_runs):
            cfg_run = copy.deepcopy(cfg)
            cfg_run["project"]["seed"] = 42 + run

            policy = _build_policy(cfg_run, mats, coords, df, delta)
            selector = _build_selector(clusterer, cfg_run, coords, charging_points, policy)
            sim = DBSimpleMaintenanceSimulator(policy, selector, coords, df, mats, cfg_run)
            result = sim.run(mal_df, max_days=max_days)
            run_costs.append(result.total_cost_eur)

            out = Path(output_dir) / f"delta_{delta:.1f}_run_{run + 1}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            sim.write_json(result, str(out), label=f"DB-SIMPLE δ={delta:.1f}", run_id=run + 1)

        mean_c, std_c = np.mean(run_costs), np.std(run_costs)
        print(f"δ={delta:.1f}: {mean_c:>10,.0f} ± {std_c:>6,.0f} EUR ({n_runs} Runs)")
        results.append({"delta": delta, "mean": mean_c, "std": std_c})

    best = min(results, key=lambda r: r["mean"])
    print(f"\nBestes δ: {best['delta']:.1f} ({best['mean']:,.0f} EUR)")


# ---------------------------------------------------------------------------
# Sanity-Check: δ=0.5 muss identisch zu CFA-Future sein
# ---------------------------------------------------------------------------

def run_sanity_check(max_days: int = 5) -> None:
    """
    Vergleich DBSimplePolicy(δ=0.5) vs. CFAFutureModel auf identischem Seed.
    Identische Gesamtkosten bestätigen den Referenzfall.
    """
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)

    clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"], random_state=42)
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    charging_points = df["Anzahl Ladepunkte"].fillna(1).astype(int).values
    failure_mode = cfg.get("failure_simulation", {}).get("mode", "csv")
    mal_df = pd.read_csv("data/malfunction.csv") if failure_mode == "csv" else None

    # Run 1: DBSimplePolicy mit δ=0.5
    policy_simple = _build_policy(cfg, mats, coords, df, 0.5)
    sel_simple = _build_selector(clusterer, cfg, coords, charging_points, policy_simple)
    sim_simple = DBSimpleMaintenanceSimulator(policy_simple, sel_simple, coords, df, mats, cfg)
    res_simple = sim_simple.run(mal_df, max_days=max_days)

    # Run 2: CFAFutureModel
    policy_cfa = CFAFutureModel(
        traffic_matrices=mats,
        config=cfg,
        all_coords=coords,
        stations_df=df,
    )
    sel_cfa = DailyZoneSelector(clusterer, cfg, coords, charging_points)
    if cfg["planning"].get("zone_selection_mode", "classic") == "value_based":
        sel_cfa.value_fn = policy_cfa._value
    sim_cfa = MaintenanceSimulator(policy_cfa, sel_cfa, coords, df, mats, cfg)
    res_cfa = sim_cfa.run(mal_df, max_days=max_days)

    print("\nSanity-Check: DBSimple(δ=0.5) vs. CFA-Future")
    print(f"  DB-Simple  Gesamtkosten: {res_simple.total_cost_eur:>12,.2f} €")
    print(f"  CFA-Future Gesamtkosten: {res_cfa.total_cost_eur:>12,.2f} €")
    diff = abs(res_simple.total_cost_eur - res_cfa.total_cost_eur)
    if diff < 0.01:
        print("  ✓ IDENTISCH — Referenzfall bestätigt.")
    else:
        print(f"  ✗ ABWEICHUNG: {diff:.4f} € — Implementierung prüfen!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DB-Simple Training")
    parser.add_argument("--collect",      action="store_true", help="Labels sammeln (Phase 1)")
    parser.add_argument("--train",        action="store_true", help="Modell trainieren (Phase 2)")
    parser.add_argument("--benchmark",    action="store_true", help="Statischen δ-Benchmark laufen")
    parser.add_argument("--sanity",       action="store_true", help="Sanity-Check: δ=0.5 == CFA-Future")
    parser.add_argument("--classifier",   type=str, default="rf", choices=["rf", "mlp"])
    parser.add_argument("--n-outer",      type=int, default=10, help="Pilot-Simulationen")
    parser.add_argument("--horizon",      type=int, default=20, help="Rollout-Horizont (Tage)")
    parser.add_argument("--sample-every", type=int, default=5,  help="Snapshot alle N Tage")
    parser.add_argument("--max-days",     type=int, default=365)
    parser.add_argument("--n-runs",       type=int, default=5,  help="Runs pro δ (Benchmark)")
    parser.add_argument("--labels-path",  type=str, default=str(_DEFAULT_LABELS_PATH))
    parser.add_argument("--model-path",   type=str, default=str(_DEFAULT_MODEL_PATH))
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.sanity:
        run_sanity_check(max_days=args.max_days)
        return

    if args.benchmark:
        print("Statischer δ-Benchmark...")
        run_delta_benchmark(max_days=args.max_days, n_runs=args.n_runs)
        return

    if not args.collect and not args.train:
        parser.print_help()
        return

    if args.collect:
        print(f"Phase 1: Labels sammeln ({args.n_outer} Läufe, H={args.horizon} Tage)...")
        collect_labels(
            n_outer=args.n_outer,
            horizon=args.horizon,
            sample_every=args.sample_every,
            output_path=Path(args.labels_path),
            max_days=args.max_days,
        )

    if args.train:
        print("Phase 2: Modell trainieren...")
        train_model(
            labels_path=Path(args.labels_path),
            model_path=Path(args.model_path),
            classifier=args.classifier,
        )


if __name__ == "__main__":
    main()
