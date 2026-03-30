"""
Berechnet die Fahrzeit-/Distanzmatrix für alle Ladesäulen via lokaler OSRM-Instanz.

Voraussetzung: OSRM Docker-Container läuft auf localhost:5000.
               Siehe scripts/setup_osrm.sh für Setup-Anleitung.

Ausführung (aus dem Projektordner):
    python scripts/build_travel_matrix.py

Ausgabe:
    data/distance_matrices/travel_times_duration.npy  — Fahrzeiten in Sekunden
    data/distance_matrices/travel_times_distance.npy  — Distanzen in Metern
"""
from __future__ import annotations

import sys
from pathlib import Path

# Projektroot zum Python-Pfad hinzufügen (damit src.* importierbar ist)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.loader import load_config, load_stations, get_coordinates
from src.api.osrm import fetch_travel_matrix, check_osrm_connection


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")

    osrm_url     = config["osrm"]["base_url"]
    osrm_timeout = config["osrm"]["timeout"]
    cache_prefix = PROJECT_ROOT / config["data"]["distance_matrix_path"]

    # 1. OSRM-Verbindung prüfen
    print(f"[Setup] Prüfe OSRM-Verbindung zu {osrm_url} …")
    if not check_osrm_connection(osrm_url):
        print(
            "\n[FEHLER] OSRM ist nicht erreichbar!\n"
            "Starte zuerst den Docker-Container:\n\n"
            "  docker run -t -i -p 5000:5000 \\\n"
            "    -v \"$(pwd)/data/osrm:/data\" osrm/osrm-backend \\\n"
            "    osrm-routed --algorithm mld /data/bavaria.osrm\n\n"
            "Oder führe scripts/setup_osrm.sh aus (einmaliges Preprocessing)."
        )
        sys.exit(1)
    print("[Setup] OSRM erreichbar.\n")

    # 2. Stationen laden
    print("[Daten] Lade Ladesäulen …")
    df = load_stations(config)
    coords = get_coordinates(df, config)
    n = len(coords)
    print(f"[Daten] {n} Koordinaten geladen (1 Depot + {n - 1} Stationen).\n")

    # 3. Matrix berechnen (ein einziger OSRM-Request)
    duration_matrix, distance_matrix = fetch_travel_matrix(
        coords=coords,
        osrm_base_url=osrm_url,
        cache_path=cache_prefix,
        timeout=osrm_timeout,
    )

    # 4. Kurze Statistik ausgeben
    _print_stats(duration_matrix, distance_matrix, n)


def _print_stats(duration: np.ndarray, distance: np.ndarray, n: int) -> None:
    # Nur Nicht-Diagonal-Einträge für Statistik
    mask = ~np.eye(n, dtype=bool)
    dur_flat  = duration[mask]
    dist_flat = distance[mask]

    print("\n=== Matrix-Statistik ===")
    print(f"  Größe              : {n}×{n} = {n * n:,} Einträge")
    print(f"  Fahrzeit  — min    : {dur_flat.min() / 60:.1f} min")
    print(f"  Fahrzeit  — median : {np.median(dur_flat) / 60:.1f} min")
    print(f"  Fahrzeit  — max    : {dur_flat.max() / 60:.1f} min")
    print(f"  Distanz   — min    : {dist_flat.min() / 1000:.2f} km")
    print(f"  Distanz   — median : {np.median(dist_flat) / 1000:.2f} km")
    print(f"  Distanz   — max    : {dist_flat.max() / 1000:.2f} km")
    print("========================\n")


if __name__ == "__main__":
    main()
