"""
Baut stündliche Fahrzeit-Matrizen mit Stau-Zuschlägen.

Ablauf:
  1. Basis-Fahrzeit-Matrix laden  (travel_times_duration.npy)
  2. Stau-CSV laden               (data/stau_verzögerungen.csv)
  3. Routen-Geometrien von OSRM abrufen und cachen (route_geometries.pkl)
  4. Für jede Stunde: Stau-Zuschläge auf passende Routen addieren
  5. Matrizen speichern:          data/distance_matrices/traffic_matrix_{h}uhr.npy

Erwartetes CSV-Format (aus fetch_traffic_delays.py):
  id, ort, strecke, start_lat, start_lon, ziel_lat, ziel_lon,
  Standardzeit, 8 Uhr, 9 Uhr, ..., 17 Uhr
  (Zeiten in Sekunden)

Verwendung:
    python scripts/build_traffic_matrix.py
    python scripts/build_traffic_matrix.py --hours 8 12 17   # nur bestimmte Stunden
    python scripts/build_traffic_matrix.py --tolerance 75    # Toleranz in Metern
    python scripts/build_traffic_matrix.py --no-cache        # Geometrien neu abrufen
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.osrm import fetch_route_geometry, check_osrm_connection
from src.data.loader import load_config, load_stations, get_coordinates

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_TRAFFIC_CSV = PROJECT_ROOT / "data" / "traffic_data.csv"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "distance_matrices"
DEFAULT_TOLERANCE_M = 10.0   # Meter Toleranz für Punkt-auf-Strecke-Prüfung
DEFAULT_WORKERS     = 8      # Parallele OSRM-Requests
TRAFFIC_HOURS       = list(range(8, 18))   # 8, 9, ..., 17


# ---------------------------------------------------------------------------
# Geometrie-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _point_to_segment_distance_m(
    px: float, py: float,   # Punkt (lon, lat)
    ax: float, ay: float,   # Segment-Start (lon, lat)
    bx: float, by: float,   # Segment-Ende  (lon, lat)
) -> float:
    """
    Nächster Abstand (in Metern) vom Punkt P zum Liniensegment AB.
    Verwendet planare Näherung — ausreichend für ~50-m-Toleranz in Würzburg.
    """
    # Skalierungsfaktoren lon→m, lat→m bei ~49.8°N
    lat_ref = _deg2rad((ay + by) / 2)
    mx = math.cos(lat_ref) * 111_320.0   # m pro Längengrad
    my = 110_540.0                        # m pro Breitengrad

    # Auf Meter-Koordinaten umrechnen
    px_m, py_m = px * mx, py * my
    ax_m, ay_m = ax * mx, ay * my
    bx_m, by_m = bx * mx, by * my

    dx, dy = bx_m - ax_m, by_m - ay_m
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px_m - ax_m, py_m - ay_m)

    t = max(0.0, min(1.0, ((px_m - ax_m) * dx + (py_m - ay_m) * dy) / len_sq))
    nearest_x = ax_m + t * dx
    nearest_y = ay_m + t * dy
    return math.hypot(px_m - nearest_x, py_m - nearest_y)


def _find_nearest_segment(
    point: tuple[float, float],          # (lon, lat)
    polyline: list[tuple[float, float]], # [(lon, lat), ...]
    tolerance_m: float,
) -> tuple[int | None, float]:
    """
    Findet den Index des nächsten Segments auf der Polyline.

    Returns
    -------
    (segment_index, distance_m)
    segment_index ist None wenn kein Segment innerhalb der Toleranz liegt.
    """
    px, py = point
    best_idx = None
    best_dist = float("inf")

    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        d = _point_to_segment_distance_m(px, py, ax, ay, bx, by)
        if d < best_dist:
            best_dist = d
            best_idx = i

    if best_dist > tolerance_m:
        return None, best_dist
    return best_idx, best_dist


def node_on_route_directed(
    node_start_lon: float, node_start_lat: float,
    node_end_lon: float,   node_end_lat: float,
    polyline: list[tuple[float, float]],
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> bool:
    """
    Prüft ob der Verkehrsknoten (start→end) auf der Route liegt UND
    in der richtigen Richtung durchfahren wird.

    Die Start-Koordinate des Knotens muss auf der Route VOR der
    Ziel-Koordinate des Knotens liegen (Segment-Index start < end).
    Damit werden Fahrten aus der Gegenrichtung korrekt ausgeschlossen.

    Parameters
    ----------
    node_start_lon/lat : Koordinaten des Stau-Anfangs (Einfahrt in Staubereich).
    node_end_lon/lat   : Koordinaten des Stau-Endes   (Ausfahrt aus Staubereich).
    polyline           : OSRM-Routen-Geometrie als [(lon, lat), ...].
    tolerance_m        : Maximaler Abstand in Metern für Punkt-auf-Strecke.
    """
    start_idx, _ = _find_nearest_segment(
        (node_start_lon, node_start_lat), polyline, tolerance_m
    )
    end_idx, _ = _find_nearest_segment(
        (node_end_lon, node_end_lat), polyline, tolerance_m
    )

    if start_idx is None or end_idx is None:
        return False

    # start_idx < end_idx → Knoten liegt in Fahrtrichtung auf der Route
    return start_idx < end_idx


# ---------------------------------------------------------------------------
# Stau-Matrizen bauen (streaming — keine Geometrien im RAM halten)
# ---------------------------------------------------------------------------

def _compute_pair(
    i: int,
    j: int,
    coords: list[tuple[float, float]],
    node_data: list[tuple[float, float, float, float]],
    delays: list[dict[int, float]],
    hours: list[int],
    tolerance_m: float,
    osrm_url: str,
) -> tuple[int, int, dict[int, float]]:
    """Berechnet Stau-Zuschläge für ein einzelnes Routenpaar (i→j)."""
    polyline = fetch_route_geometry(start=coords[i], end=coords[j], osrm_base_url=osrm_url)
    additions = {h: 0.0 for h in hours}
    if polyline and len(polyline) >= 2:
        for k, (slon, slat, elon, elat) in enumerate(node_data):
            if node_on_route_directed(slon, slat, elon, elat, polyline, tolerance_m):
                for h in hours:
                    additions[h] += delays[k][h]
    return i, j, additions


def build_traffic_matrices(
    base_duration: np.ndarray,
    coords: list[tuple[float, float]],
    traffic_df: pd.DataFrame,
    osrm_url: str,
    hours: list[int],
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    num_workers: int = DEFAULT_WORKERS,
    progress_path: Path | None = None,
    apply_mean_to_unmatched: bool = False,
) -> dict[int, np.ndarray]:
    """
    Erstellt für jede Stunde eine modifizierte Fahrzeit-Matrix.

    Geometrien werden parallel von OSRM abgerufen (num_workers Threads) und
    sofort verworfen — kein RAM-Anstieg durch gecachte Geometrien.

    Fortschritt wird in progress_path als .npy-Zwischendatei gesichert
    (alle 5.000 Routen) um bei einem Abbruch weitermachen zu können.

    apply_mean_to_unmatched: Wenn True, erhalten Routen ohne Stau-Treffer
    den stündlichen Mittelwert aller gematchten Routen als Zuschlag.
    """
    n = base_duration.shape[0]

    # Stau-Verzögerungen vorberechnen: delay[k][h] = extra Sekunden
    delays: list[dict[int, float]] = []
    for _, row in traffic_df.iterrows():
        node_delays: dict[int, float] = {}
        for h in hours:
            col = f"{h}_uhr_stau"
            val = row.get(col)
            node_delays[h] = max(0.0, float(val)) if pd.notna(val) else 0.0
        delays.append(node_delays)

    # Knoten-Koordinaten als einfache Liste vorhalten (thread-sicher, read-only)
    node_data = [
        (float(r["start_lon"]), float(r["start_lat"]), float(r["ziel_lon"]), float(r["ziel_lat"]))
        for _, r in traffic_df.iterrows()
    ]

    # Fortschritt laden falls vorhanden
    start_k = 0
    traffic_matrices = {h: base_duration.copy() for h in hours}
    # Separate Matrix zum Tracking der Route-Zuschläge (für Mittelwertberechnung)
    additions_matrices = {h: np.zeros((n, n), dtype=np.float64) for h in hours}
    if progress_path and progress_path.exists():
        saved = np.load(progress_path, allow_pickle=True).item()
        traffic_matrices = saved["matrices"]
        additions_matrices = saved.get("additions_matrices", additions_matrices)
        start_k = saved["next_k"]
        print(f"[Matrix] Fortschritt geladen — starte bei Route {start_k:,}")

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    total = len(pairs)
    pairs = pairs[start_k:]

    print(f"[Matrix] {num_workers} parallele Threads …")

    iterator = tqdm(total=total, desc="Stau-Matrix", initial=start_k) if _HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _compute_pair, i, j, coords, node_data, delays, hours, tolerance_m, osrm_url
            ): step
            for step, (i, j) in enumerate(pairs)
        }

        for step, future in enumerate(as_completed(futures)):
            i, j, additions = future.result()
            for h in hours:
                traffic_matrices[h][i, j] += additions[h]
                additions_matrices[h][i, j] = additions[h]

            if iterator:
                iterator.update(1)

            # Zwischenspeichern alle 5.000 Routen (Absturz-Schutz)
            if progress_path and (step + 1) % 5_000 == 0:
                np.save(progress_path, {
                    "matrices": traffic_matrices,
                    "additions_matrices": additions_matrices,
                    "next_k": start_k + step + 1,
                })

    if iterator:
        iterator.close()

    # Fortschrittsdatei aufräumen
    if progress_path and progress_path.exists():
        progress_path.unlink()

    # Durchschnittlichen Stau auf ungematchte Routen anwenden
    if apply_mean_to_unmatched:
        off_diag = ~np.eye(n, dtype=bool)
        for h in hours:
            matched_mask = (additions_matrices[h] > 0) & off_diag
            n_matched = matched_mask.sum()
            if n_matched == 0:
                print(f"  {h:2d} Uhr: keine gematchten Routen — Mittelwert-Fallback übersprungen.")
                continue
            mean_delay = additions_matrices[h][matched_mask].mean()
            unmatched_mask = (additions_matrices[h] == 0) & off_diag
            traffic_matrices[h][unmatched_mask] += mean_delay
            print(
                f"  {h:2d} Uhr: {n_matched:,} gematchte / {unmatched_mask.sum():,} ungematchte Routen"
                f" — Ø Zuschlag {mean_delay:.1f}s auf ungematchte addiert."
            )

    return traffic_matrices


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main(
    hours: list[int] = TRAFFIC_HOURS,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    num_workers: int = DEFAULT_WORKERS,
    traffic_csv: Path = DEFAULT_TRAFFIC_CSV,
) -> None:
    config   = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    osrm_url = config["osrm"]["base_url"]
    apply_mean = config.get("traffic", {}).get("apply_mean_delay_to_unmatched", False)

    # 1. OSRM prüfen
    print(f"[Setup] Prüfe OSRM-Verbindung zu {osrm_url} …")
    if not check_osrm_connection(osrm_url):
        print("[FEHLER] OSRM nicht erreichbar. Starte: ./scripts/start_osrm.sh")
        sys.exit(1)
    print("[Setup] OSRM erreichbar.\n")

    # 2. Koordinaten laden (Depot + Stationen)
    df = load_stations(config)
    coords = get_coordinates(df, config)
    n = len(coords)
    print(f"[Daten] {n} Koordinaten (1 Depot + {n - 1} Stationen).")

    # 3. Basis-Matrix laden
    base_dur_path = PROJECT_ROOT / "data" / "distance_matrices" / "travel_times_duration.npy"
    if not base_dur_path.exists():
        print(f"[FEHLER] Basis-Matrix nicht gefunden: {base_dur_path}")
        print("         Führe zuerst aus: python scripts/build_travel_matrix.py")
        sys.exit(1)
    base_duration = np.load(base_dur_path)
    print(f"[Daten] Basis-Matrix geladen: {base_duration.shape}.\n")

    # 4. Stau-CSV laden
    if not traffic_csv.exists():
        print(f"[FEHLER] Stau-CSV nicht gefunden: {traffic_csv}")
        print("         Führe zuerst aus: python scripts/fetch_traffic_delays.py")
        sys.exit(1)
    traffic_df = pd.read_csv(traffic_csv)
    required = {"start_lat", "start_lon", "ziel_lat", "ziel_lon", "standard_zeit"}
    missing = required - set(traffic_df.columns)
    if missing:
        print(f"[FEHLER] Fehlende Spalten in der CSV: {missing}")
        sys.exit(1)
    print(f"[Daten] {len(traffic_df)} Verkehrsknoten geladen.\n")

    # 5. Stau-Matrizen bauen (Geometrien werden on-the-fly abgerufen)
    progress_path = DEFAULT_OUTPUT_DIR / "traffic_matrix_progress.npy"
    print(f"\n[Matrix] Berechne Stau-Matrizen für {hours} Uhr …")
    if apply_mean:
        print("[Matrix] Modus: Ø-Stau auf ungematchte Routen aktiv (traffic.apply_mean_delay_to_unmatched=true)")
    traffic_matrices = build_traffic_matrices(
        base_duration=base_duration,
        coords=coords,
        traffic_df=traffic_df,
        osrm_url=osrm_url,
        hours=hours,
        tolerance_m=tolerance_m,
        num_workers=num_workers,
        progress_path=progress_path,
        apply_mean_to_unmatched=apply_mean,
    )

    # 7. Speichern
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h, matrix in traffic_matrices.items():
        out_path = DEFAULT_OUTPUT_DIR / f"traffic_matrix_{h}uhr.npy"
        np.save(out_path, matrix)
        avg_delay = (matrix - base_duration)[~np.eye(n, dtype=bool)].mean()
        print(f"  {h:2d} Uhr → {out_path.name}  (Ø Stau-Zuschlag: {avg_delay:.1f}s)")

    print("\n[Fertig] Alle Matrizen gespeichert.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stündliche Fahrzeit-Matrizen mit Stau-Zuschlägen berechnen."
    )
    parser.add_argument(
        "--hours", nargs="+", type=int, default=TRAFFIC_HOURS,
        help=f"Stunden für die Berechnung (Standard: {TRAFFIC_HOURS})",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_M,
        help=f"Toleranz in Metern (Standard: {DEFAULT_TOLERANCE_M})",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_TRAFFIC_CSV,
        help=f"Pfad zur Stau-CSV (Standard: {DEFAULT_TRAFFIC_CSV})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Anzahl paralleler Threads (Standard: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    main(
        hours=args.hours,
        tolerance_m=args.tolerance,
        num_workers=args.workers,
        traffic_csv=args.csv,
    )
