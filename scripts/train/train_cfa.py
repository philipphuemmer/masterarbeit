"""
CFA-Training: Lernt den Gewichtsvektor θ der Wertfunktionsapproximation.

Wertfunktion:
    V(s) ≈ θ × Σ_k station_factor[k] × power_kW[k] × recovery_curve(dsm[k]) × recovery_days

Training (iteratives Policy Iteration):
    Runde 1: N Myopic-Simulationen → θ₁  (Bootstrap)
    Runde 2: N CFA(θ₁)-Simulationen → θ₂
    Runde r: N CFA(θ_{r-1})-Simulationen → θ_r

    Pro Tag / Runde:
        Feature = Σ_k station_factor[k] × power_kW[k] × recovery_curve(dsm[k]) × recovery_days
        Target  = tatsächliche Restkosten G_t ab diesem Tag (unter aktueller Policy)
    OLS-Regression: G_t ≈ θ × Feature + intercept

Ausführen:
    python scripts/train_cfa.py
    python scripts/train_cfa.py --runs 30 --rounds 3 --max-days 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.models.cfa import CFAModel
from src.models.myopic import MyopicPolicy
from src.models.simulator import MaintenanceSimulator
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver


# ---------------------------------------------------------------------------
# Trainings-Simulator: zeichnet Feature pro Tag auf
# ---------------------------------------------------------------------------

class CFATrainingSimulator(MaintenanceSimulator):
    """
    Erweitert MaintenanceSimulator um Aufzeichnung der täglichen Feature-Vektoren
    für die multivariate CFA-Regression.

    Pro Tag wird Φ(s_t) = Σ_{k ∈ remaining} φ(k) berechnet:
        φ(k) = [power_kW, age_years, is_DC, recovery_curve(dsm), mean_dist_to_others]
    """

    N_FEATURES = 4

    def __init__(self, *args, **kwargs) -> None:
        # stations_df ist das 4. Argument (policy, selector, coords, stations_df, mats, config)
        df = args[3] if len(args) > 3 else kwargs.get("stations_df")
        super().__init__(*args, **kwargs)
        self.training_records: list[dict] = []

        # Stationsfeatures vorberechnen
        date_col, type_col = "Inbetriebnahmedatum", "Art der Ladeeinrichtung"
        ref = pd.Timestamp("2026-01-01")
        self._node_to_age: dict[int, float] = {}
        self._node_to_is_dc: dict[int, float] = {}
        if df is not None:
            for i, (_, row) in enumerate(df.iterrows()):
                nid = i + 1
                if date_col in df.columns and pd.notna(row.get(date_col)):
                    age = max(0.0, (ref - pd.Timestamp(row[date_col])).days / 365.25)
                else:
                    age = 5.0
                self._node_to_age[nid] = age
                self._node_to_is_dc[nid] = float(
                    row.get(type_col, "") == "Schnellladeeinrichtung"
                )

        # Mittlere Distanz zu allen anderen Stationen
        from src.planning.clustering import _approx_km
        coords = self.all_coords
        n = len(coords)
        self._node_to_mean_dist: dict[int, float] = {
            i: float(np.mean([
                _approx_km(coords[i], coords[j]) for j in range(1, n) if j != i
            ]))
            for i in range(1, n)
        } if coords is not None and n > 2 else {}

    def _phi(self, node_idx: int, dsm: float) -> np.ndarray:
        fail_cfg = self.config.get("failure_simulation", {})
        recovery_days = float(fail_cfg.get("recovery_days", 365))
        initial_factor = float(fail_cfg.get("initial_factor", 0.1))
        dsm_c = min(dsm, recovery_days)
        recovery_curve = initial_factor + (1.0 - initial_factor) * dsm_c / recovery_days
        return np.array([
            self.node_to_power.get(node_idx, 22.0),
            self._node_to_age.get(node_idx, 5.0),
            recovery_curve,
            self._node_to_mean_dist.get(node_idx, 5.0),
        ])

    def set_station_stats(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Setzt Skalierungsparameter (Einzelstationsebene) für Training und Inferenz."""
        self._mu_station = mu
        self._sigma_station = sigma

    def _run_day(self, day, remaining, team_states, carryover_tasks, day_disruptions):
        # Φ_scaled(s_t) = Σ_{k ∈ remaining} (φ(k) - μ_station) / σ_station
        mu = getattr(self, "_mu_station", np.zeros(self.N_FEATURES))
        sigma = getattr(self, "_sigma_station", np.ones(self.N_FEATURES))
        phi_sum = np.zeros(self.N_FEATURES)
        for idx in remaining:
            node_idx = idx + 1
            dsm = float(self._days_since_maintenance[node_idx])
            phi = self._phi(node_idx, dsm)
            phi_sum += (phi - mu) / np.maximum(sigma, 1e-8)
        self.training_records.append({"day": day, "feature": phi_sum})
        return super()._run_day(day, remaining, team_states, carryover_tasks, day_disruptions)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def fit_theta(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Multivariate OLS: cost_to_go ≈ θᵀ × X + intercept.

    X enthält bereits pro-Station standardisierte und summierte Features
    (Skalierung erfolgt im Simulator vor dem Summieren).

    Returns
    -------
    (theta_vector, intercept, r2)
    """
    A = np.column_stack([X, np.ones(len(X))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    theta_vec     = coeffs[:-1]
    intercept_val = float(coeffs[-1])

    y_pred = X @ theta_vec + intercept_val
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return theta_vec, intercept_val, r2


# ---------------------------------------------------------------------------
# Eine Trainingsrunde
# ---------------------------------------------------------------------------

def run_round(
    round_idx: int,
    cfg: dict,
    coords: np.ndarray,
    df_base: pd.DataFrame,
    mats: dict,
    seeds: list[int],
    max_days: int,
    theta_prev: np.ndarray | None,
    station_mu: np.ndarray = None,
    station_sigma: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Führt N Simulationen durch und gibt Trainingsdaten (X, y) zurück.

    round_idx == 1  → Myopic-Policy (Bootstrap)
    round_idx >= 2  → CFA-Policy mit theta_prev (Vektor)
    X: (n_samples, N_FEATURES) — Φ(s_t) = Σ_k φ(k) pro Tag
    y: (n_samples,)            — cost_to_go ab Tag t
    """
    X_all: list[np.ndarray] = []
    y_all: list[float] = []
    use_value_based = cfg["planning"].get("value_based_zone_selection", False)

    label = "Myopic" if round_idx == 1 else f"CFA(θ={theta_prev})"
    print(f"\n  Runde {round_idx}: {label}\n")

    for run_i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"    Lauf {run_i}/{len(seeds)} (Seed {seed})...", end=" ", flush=True)

        run_cfg = {**cfg, "project": {**cfg.get("project", {}), "seed": seed}}

        clusterer = ZoneClusterer(
            n_zones=run_cfg["planning"]["n_zones"],
            random_state=seed,
        )
        clusterer.fit(coords[1:], (run_cfg["depot"]["lat"], run_cfg["depot"]["lon"]))

        selector = DailyZoneSelector(clusterer, run_cfg, coords)

        if round_idx == 1:
            solver = VRPSolver(mats, run_cfg, all_coords=coords)
            policy = MyopicPolicy(solver, coords, mats, run_cfg)
        else:
            policy = CFAModel(
                mats, run_cfg,
                all_coords=coords,
                stations_df=df_base,
                theta_override=theta_prev,
            )
            if use_value_based:
                selector.value_fn = policy._value

        sim = CFATrainingSimulator(policy, selector, coords, df_base, mats, run_cfg)
        sim.set_station_stats(station_mu, station_sigma)
        result = sim.run(max_days=max_days)

        day_costs = [dr.total_cost_eur for dr in result.day_results]
        n_days    = len(sim.training_records)

        for i in range(n_days):
            phi_sum    = sim.training_records[i]["feature"]  # np.ndarray (N_FEATURES,)
            cost_to_go = sum(day_costs[i:])
            if np.any(phi_sum > 0):
                X_all.append(phi_sum)
                y_all.append(cost_to_go)

        elapsed = time.time() - t0
        days    = result.days_to_complete or "?"
        print(f"fertig ({days} Tage, {result.total_cost_eur:,.0f} €, {elapsed:.0f}s)")

    return np.array(X_all), np.array(y_all)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CFA-Training (iteratives Policy Iteration)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Anzahl Trainingsläufe pro Runde (Standard: 20)")
    parser.add_argument("--rounds",   type=int, default=3,
                        help="Anzahl Iterationsrunden (Standard: 3; 1 = nur Myopic-Bootstrap)")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Tage pro Lauf (Standard: 365)")
    parser.add_argument("--verbose",  action="store_true",
                        help="OR-Tools-Logging aktivieren")
    parser.add_argument("--out",      type=str, default="data/training/cfa/theta.json",
                        help="Ausgabepfad für θ (Standard: data/training/cfa/theta.json)")
    parser.add_argument("--fresh",    action="store_true",
                        help="Checkpoints ignorieren und von Runde 1 neu starten")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("failure_simulation", {}).get("mode", "csv") != "stochastic":
        print("FEHLER: CFA-Training erfordert failure_simulation.mode = stochastic.")
        sys.exit(1)

    print("Lade Stationsdaten...")
    df_base = load_stations(cfg)
    coords  = np.array(get_coordinates(df_base, cfg))
    mats    = load_traffic_matrices(cfg)
    print(f"  {len(df_base)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    seeds = list(range(1, args.runs + 1))
    t0_total = time.time()

    # --- Checkpoint laden (--resume) ---
    theta: float | None = None
    intercept_val: float = 0.0
    history: list[dict] = []
    start_round = 1

    existing = sorted(ckpt_dir.glob("round_*.json")) if not args.fresh else []
    if existing:
        latest = existing[-1]
        with open(latest) as f:
            ckpt = json.load(f)
        theta         = np.array(ckpt["theta"], dtype=float)
        intercept_val = float(ckpt["intercept"])
        history       = ckpt.get("history", [])
        start_round   = ckpt["round"] + 1
        print(f"  Checkpoint gefunden: {latest.name}  "
              f"(θ={theta}, Runde {ckpt['round']} abgeschlossen)")
        if start_round > args.rounds:
            print(f"  Alle {args.rounds} Runden bereits abgeschlossen. Nichts zu tun.")
            sys.exit(0)
    elif args.fresh:
        print("  --fresh: starte von Runde 1 (Checkpoints ignoriert).")

    feature_names = ["power_kW", "age_years", "recovery_curve", "mean_dist_km"]
    print(f"\nIteratives CFA-Training: Runde {start_round}–{args.rounds}, {args.runs} Läufe/Runde")
    print(f"Features: {feature_names}")

    # Stationsstatistiken auf Einzelstationsebene berechnen (dsm=182 als Mittelwert)
    print("  Berechne Stationsstatistiken...")
    _tmp_sim = CFATrainingSimulator(
        MyopicPolicy(VRPSolver(mats, cfg, all_coords=coords), coords, mats, cfg),
        None, coords, df_base, mats, cfg,
    )
    fail_cfg = cfg.get("failure_simulation", {})
    rec_days = float(fail_cfg.get("recovery_days", 365))
    init_fac = float(fail_cfg.get("initial_factor", 0.1))
    rep_dsm  = rec_days / 2  # repräsentativer dsm-Wert (Jahresmitte)
    all_phi  = np.array([
        _tmp_sim._phi(i + 1, rep_dsm) for i in range(len(df_base))
    ])
    station_mu    = all_phi.mean(axis=0)
    station_sigma = all_phi.std(axis=0)
    station_sigma = np.where(station_sigma < 1e-8, 1.0, station_sigma)
    print(f"  μ_station = {station_mu}")
    print(f"  σ_station = {station_sigma}")

    for round_idx in range(start_round, args.rounds + 1):
        X, y = run_round(
            round_idx, cfg, coords, df_base, mats, seeds, args.max_days, theta,
            station_mu=station_mu, station_sigma=station_sigma,
        )

        print(f"\n  Regression auf {len(X)} Datenpunkten, {X.shape[1]} Features...")
        theta_new, intercept_new, r2 = fit_theta(X, y)

        print(f"    R²        = {r2:.4f}")
        print(f"    intercept = {intercept_new:,.2f} EUR")
        for name, t in zip(feature_names, theta_new):
            print(f"    θ[{name}] = {t:+.6e}")

        history.append({
            "round": round_idx, "theta": theta_new.tolist(),
            "intercept": intercept_new, "r2": r2,
        })
        theta = theta_new
        intercept_val = intercept_new

        ckpt_path = ckpt_dir / f"round_{round_idx}.json"
        with open(ckpt_path, "w") as f:
            json.dump({
                "round": round_idx, "theta": theta.tolist(),
                "intercept": intercept_val, "r2": r2,
                "feature_means": station_mu.tolist(),
                "feature_stds": station_sigma.tolist(),
                "history": history,
            }, f, indent=2)
        print(f"    Checkpoint: {ckpt_path}")

    print(f"\n  Trainingszeit gesamt: {time.time() - t0_total:.0f}s")

    payload = {
        "theta":          theta.tolist(),
        "feature_names":  feature_names,
        "feature_means":  station_mu.tolist(),
        "feature_stds":   station_sigma.tolist(),
        "intercept":      intercept_val,
        "r2":             history[-1]["r2"],
        "n_runs":         args.runs,
        "n_rounds":       args.rounds,
        "n_datapoints":   int(len(X)),
        "history":        history,
        "cost_to_go_mean": float(np.mean(y)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nθ gespeichert: {out_path}")


if __name__ == "__main__":
    main()
