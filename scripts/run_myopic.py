"""
Startet die Myopic-Simulation für 100 Tage.

Ausführen:
    .venv/bin/python3 scripts/run_myopic.py
    .venv/bin/python3 scripts/run_myopic.py --days 10 --log-day 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Projektverzeichnis zum Suchpfad hinzufügen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_stations, get_coordinates, load_traffic_matrices
from src.planning.clustering import ZoneClusterer
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import VRPSolver
from src.models.myopic import MyopicModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Myopic-Simulation")
    parser.add_argument("--max-days", type=int, default=365,
                        help="Maximale Simulationstage (Sicherheitsgrenze, Standard: 365)")
    parser.add_argument("--log-day", type=int, default=None,
                        help="Stunden-Log für diesen Tag auf der Konsole ausgeben")
    parser.add_argument("--output", type=str, default="logs/myopic_simulation.log",
                        help="Ausgabedatei für das vollständige Protokoll")
    parser.add_argument("--verbose", action="store_true", help="OR-Tools Logging aktivieren")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # --- Konfiguration & Daten laden ---
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Lade Stationsdaten...")
    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))
    mats = load_traffic_matrices(cfg)
    print(f"  {len(df)} Stationen, {len(mats)} Stundenmatrizen geladen.")

    # --- Clustering ---
    print("Clustering...")
    clusterer = ZoneClusterer(
        n_zones=cfg["planning"]["n_zones"],
        random_state=cfg["project"]["seed"],
    )
    clusterer.fit(coords[1:], (cfg["depot"]["lat"], cfg["depot"]["lon"]))

    # --- Solver & Selector ---
    solver = VRPSolver(mats, cfg, all_coords=coords)
    selector = DailyZoneSelector(clusterer, cfg, coords)

    # --- Myopic Modell ---
    model = MyopicModel(solver, selector, coords, df, mats, cfg)

    # --- Stördaten laden ---
    mal_df = pd.read_csv("data/malfunction.csv")
    print(f"  {len(mal_df)} Störereignisse geladen.")

    # --- Simulation ---
    print(f"\nStarte Simulation (max. {args.max_days} Tage, stoppt automatisch)...\n")
    result = model.run(mal_df, max_days=args.max_days)

    # --- Ergebnis ---
    model.print_summary(result)

    if args.log_day is not None:
        model.print_day_log(result, args.log_day)

    # --- Log-Datei speichern ---
    model.write_log(result, args.output)


if __name__ == "__main__":
    main()
