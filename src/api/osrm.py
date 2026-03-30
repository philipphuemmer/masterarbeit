"""
OSRM (Open Source Routing Machine) Travel Time Matrix.

Berechnet die vollständige n×n Fahrzeit- und Distanzmatrix über eine
lokale OSRM-Instanz (Docker) mittels der Table API — ein einziger Request
für alle Stationen, kein Rate-Limiting.

Setup (einmalig):
    Siehe Skript scripts/setup_osrm.sh oder die README für Docker-Befehle.

Verwendung:
    python -m scripts.build_travel_matrix
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests

# tqdm ist optional — fällt sanft zurück, falls nicht installiert
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def fetch_travel_matrix(
    coords: list[tuple[float, float]],
    osrm_base_url: str = "http://localhost:5000",
    cache_path: str | Path | None = None,
    timeout: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet die vollständige n×n Fahrzeit- und Distanzmatrix via OSRM.

    Sendet **einen einzigen** Request an die OSRM Table API — keine Batches,
    kein Rate-Limiting.  Diagonal (i==i) wird explizit auf 0 gesetzt.

    Parameters
    ----------
    coords : Liste von (lat, lon)-Tupeln. Index 0 = Depot.
    osrm_base_url : URL der laufenden OSRM-Instanz.
    cache_path : Pfad-Präfix für .npy-Cache-Dateien (ohne Erweiterung).
                 Wenn angegeben, wird bei existierendem Cache sofort geladen.
    timeout : HTTP-Timeout in Sekunden.

    Returns
    -------
    (duration_matrix, distance_matrix)
        duration_matrix : float32, Fahrzeit in Sekunden.
        distance_matrix : float32, Distanz in Metern.
    """
    # Cache prüfen
    if cache_path is not None:
        cache_path = Path(cache_path)
        dur_path  = cache_path.parent / (cache_path.stem + "_duration.npy")
        dist_path = cache_path.parent / (cache_path.stem + "_distance.npy")
        if dur_path.exists() and dist_path.exists():
            print(f"[OSRM] Lade gecachte Matrix von {cache_path.parent}/")
            return np.load(dur_path), np.load(dist_path)

    n = len(coords)
    print(f"[OSRM] Berechne {n}×{n} Matrix ({n * n:,} Einträge) …")

    # Koordinaten-String für OSRM: "lon,lat;lon,lat;…"
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = (
        f"{osrm_base_url.rstrip('/')}/table/v1/driving/{coords_str}"
        "?annotations=duration,distance"
    )

    t0 = time.time()
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Keine Verbindung zu OSRM ({osrm_base_url}). "
            "Läuft der Docker-Container? Siehe README: docker run …"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"OSRM hat nach {timeout}s nicht geantwortet. "
            "Versuche timeout zu erhöhen oder OSRM-Container neu zu starten."
        )

    data = response.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM Fehler: {data.get('code')} — {data.get('message')}")

    elapsed = time.time() - t0
    print(f"[OSRM] Antwort erhalten in {elapsed:.1f}s")

    # Rohdaten extrahieren
    raw_durations  = data.get("durations")   # n×n Liste (Sekunden, float oder null)
    raw_distances  = data.get("distances")   # n×n Liste (Meter, float oder null)

    duration_matrix = _parse_matrix(raw_durations, n, coords, fallback_speed_kmh=30.0, metric="duration")
    distance_matrix = _parse_matrix(raw_distances, n, coords, fallback_speed_kmh=30.0, metric="distance")

    # Diagonal explizit 0 (OSRM gibt manchmal kleine Werte für i==i)
    np.fill_diagonal(duration_matrix, 0.0)
    np.fill_diagonal(distance_matrix, 0.0)

    # Cache speichern
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dur_path,  duration_matrix)
        np.save(dist_path, distance_matrix)
        print(f"[OSRM] Matrix gespeichert unter {cache_path.parent}/")

    return duration_matrix, distance_matrix


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _parse_matrix(
    raw: list[list[float | None]] | None,
    n: int,
    coords: list[tuple[float, float]],
    fallback_speed_kmh: float,
    metric: str,
) -> np.ndarray:
    """
    Wandelt die OSRM-Rohantwort in ein numpy float32-Array um.
    Fehlende Werte (null) werden durch Haversine-Schätzungen ersetzt.
    """
    matrix = np.zeros((n, n), dtype=np.float32)

    if raw is None:
        # OSRM hat diese Annotation nicht geliefert — alles per Haversine
        print(f"[OSRM] Warnung: '{metric}' nicht in Antwort, nutze Haversine-Fallback.")
        for i in range(n):
            for j in range(n):
                matrix[i, j] = _haversine_fallback(coords[i], coords[j], fallback_speed_kmh, metric)
        return matrix

    fallback_count = 0
    for i, row in enumerate(raw):
        for j, val in enumerate(row):
            if val is None:
                matrix[i, j] = _haversine_fallback(coords[i], coords[j], fallback_speed_kmh, metric)
                fallback_count += 1
            else:
                matrix[i, j] = float(val)

    if fallback_count > 0:
        print(f"[OSRM] Warnung: {fallback_count} fehlende Werte durch Haversine ersetzt.")

    return matrix


def _haversine_meters(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Luftliniendistanz in Metern zwischen zwei (lat, lon)-Punkten."""
    R = 6_371_000.0
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))


def _haversine_fallback(
    p1: tuple[float, float],
    p2: tuple[float, float],
    avg_speed_kmh: float,
    metric: str,
) -> float:
    """Gibt Haversine-Distanz (m) oder -Fahrzeit (s) zurück."""
    meters = _haversine_meters(p1, p2)
    if metric == "distance":
        return meters
    return meters / (avg_speed_kmh * 1000 / 3600)


def check_osrm_connection(base_url: str = "http://localhost:5000") -> bool:
    """Prüft, ob OSRM erreichbar ist. Gibt True/False zurück."""
    try:
        # Minimaler Test-Request mit einem Punkt
        r = requests.get(f"{base_url}/route/v1/driving/9.95,49.79", timeout=5)
        return r.status_code < 500
    except requests.exceptions.RequestException:
        return False


# Persistente Session für Verbindungs-Wiederverwendung (deutlich schneller bei vielen Requests)
_session = requests.Session()


def fetch_route_geometry(
    start: tuple[float, float],
    end: tuple[float, float],
    osrm_base_url: str = "http://localhost:5000",
    timeout: int = 10,
) -> list[tuple[float, float]] | None:
    """
    Gibt die vollständige Routen-Geometrie zwischen zwei Punkten zurück.

    Parameters
    ----------
    start : (lat, lon) des Startpunkts.
    end   : (lat, lon) des Zielpunkts.

    Returns
    -------
    Liste von (lon, lat)-Tupeln entlang der Route, oder None bei Fehler.
    """
    start_lat, start_lon = start
    end_lat, end_lon = end
    url = (
        f"{osrm_base_url.rstrip('/')}/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        "?overview=full&geometries=geojson"
    )
    try:
        resp = _session.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        coords = data["routes"][0]["geometry"]["coordinates"]
        return [(c[0], c[1]) for c in coords]  # (lon, lat)
    except requests.exceptions.RequestException:
        return None
