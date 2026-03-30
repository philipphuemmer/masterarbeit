"""
Google Maps Distance Matrix API Wrapper.

Fragt Fahrzeit-/Distanzmatrizen ab und cacht sie als NumPy-Arrays,
damit API-Kosten minimiert werden.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import googlemaps
from dotenv import load_dotenv

load_dotenv()


def _get_client() -> googlemaps.Client:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY nicht gesetzt. Bitte .env aus .env.example erstellen."
        )
    return googlemaps.Client(key=api_key)


def fetch_distance_matrix(
    coords: list[tuple[float, float]],
    mode: str = "driving",
    cache_path: str | Path | None = None,
    chunk_size: int = 10,
    sleep_between_requests: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet die vollständige n×n Fahrzeit- und Distanzmatrix für alle Koordinaten.

    Parameters
    ----------
    coords : Liste von (lat, lon)-Tupeln. Index 0 = Depot.
    mode : "driving" | "walking" | "bicycling" | "transit"
    cache_path : Wenn angegeben, wird Matrix als .npy gespeichert/geladen.
    chunk_size : Anzahl Zeilen/Spalten pro API-Aufruf (max 10×10 = 100 Elemente).
    sleep_between_requests : Pause zwischen API-Calls (Rate Limiting).

    Returns
    -------
    (duration_matrix, distance_matrix) : Fahrzeit in Sekunden, Distanz in Metern.
    """
    n = len(coords)

    if cache_path is not None:
        cache_path = Path(cache_path)
        dur_path = cache_path.parent / (cache_path.stem + "_duration.npy")
        dist_path = cache_path.parent / (cache_path.stem + "_distance.npy")
        if dur_path.exists() and dist_path.exists():
            print(f"Lade gecachte Distanzmatrix von {cache_path.parent}")
            return np.load(dur_path), np.load(dist_path)

    client = _get_client()
    duration_matrix = np.zeros((n, n), dtype=np.float32)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    total_calls = ((n + chunk_size - 1) // chunk_size) ** 2
    call_count = 0

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        origins = coords[i_start:i_end]

        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            destinations = coords[j_start:j_end]

            result = client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                units="metric",
            )

            for r_idx, row in enumerate(result["rows"]):
                for c_idx, element in enumerate(row["elements"]):
                    i = i_start + r_idx
                    j = j_start + c_idx
                    if element["status"] == "OK":
                        duration_matrix[i, j] = element["duration"]["value"]
                        distance_matrix[i, j] = element["distance"]["value"]
                    else:
                        # Fallback: Haversine-Schätzung (~ 30 km/h Stadtgeschwindigkeit)
                        duration_matrix[i, j] = _haversine_seconds(coords[i], coords[j])
                        distance_matrix[i, j] = _haversine_meters(coords[i], coords[j])

            call_count += 1
            print(f"API-Calls: {call_count}/{total_calls}", end="\r")
            time.sleep(sleep_between_requests)

    print()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dur_path, duration_matrix)
        np.save(dist_path, distance_matrix)
        print(f"Matrix gecacht unter {cache_path.parent}")

    return duration_matrix, distance_matrix


def _haversine_meters(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Luftlinie in Metern zwischen zwei (lat, lon)-Punkten."""
    R = 6_371_000.0
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))


def _haversine_seconds(
    p1: tuple[float, float],
    p2: tuple[float, float],
    avg_speed_kmh: float = 30.0,
) -> float:
    """Geschätzte Fahrzeit in Sekunden (Luftlinie + Stadtgeschwindigkeit)."""
    meters = _haversine_meters(p1, p2)
    return float(meters / (avg_speed_kmh * 1000 / 3600))


def build_haversine_matrix(
    coords: list[tuple[float, float]],
    avg_speed_kmh: float = 30.0,
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut eine rein Haversine-basierte Matrix (kein API-Call).
    Nützlich für Tests und Entwicklung ohne API-Key.
    """
    n = len(coords)
    duration_matrix = np.zeros((n, n), dtype=np.float32)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            dist = _haversine_meters(coords[i], coords[j])
            distance_matrix[i, j] = dist
            duration_matrix[i, j] = dist / (avg_speed_kmh * 1000 / 3600)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path.parent / (cache_path.stem + "_duration.npy"), duration_matrix)
        np.save(cache_path.parent / (cache_path.stem + "_distance.npy"), distance_matrix)

    return duration_matrix, distance_matrix
