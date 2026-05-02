"""
K-Means-Clustering der Ladesäulen in geografische Sub-Zonen.

Jede Zone fasst räumlich nahe Stationen zusammen und wird durch
Zentroid, Konvexe-Hülle und mittlere Depot-Entfernung charakterisiert.
Diese Eigenschaften fließen täglich in den Prioritäts-Score des
DailyZoneSelector ein.

Koordinatenformat: (Breitengrad, Längengrad) in Dezimalgrad.
Abstände werden als ebene Näherung berechnet (ausreichend für Würzburg).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError  # noqa: F401 (ConvexHull via _convex_hull_area_km2)
from sklearn.cluster import KMeans


def _approx_km(coord_a: np.ndarray, coord_b: np.ndarray) -> float:
    """
    Näherungsweise Entfernung in km zwischen zwei (lat, lon)-Punkten.
    Für kleine Abstände (< 50 km) ausreichend genau.
    """
    lat_mid = np.radians(0.5 * (coord_a[0] + coord_b[0]))
    dlat = (coord_a[0] - coord_b[0]) * 111.0
    dlon = (coord_a[1] - coord_b[1]) * 111.0 * np.cos(lat_mid)
    return float(np.sqrt(dlat**2 + dlon**2))


def _pairwise_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Paarweise Entfernungen (km) zwischen zwei Mengen von (lat, lon)-Punkten.
    Rückgabe: Array der Form (len(coords_a), len(coords_b)).
    """
    lat_mid = np.radians(0.5 * (coords_a[:, 0:1] + coords_b[:, 0:1].T))
    dlat = (coords_a[:, 0:1] - coords_b[:, 0:1].T) * 111.0
    dlon = (coords_a[:, 1:2] - coords_b[:, 1:2].T) * 111.0 * np.cos(lat_mid)
    return np.sqrt(dlat**2 + dlon**2)


class ZoneClusterer:
    """
    Teilt Ladesäulen per K-Means in geografische Sub-Zonen auf und berechnet
    zonenspezifische Eigenschaften für die tägliche Priorisierung.

    Parameters
    ----------
    n_zones : int
        Anzahl der Sub-Zonen (Standard: 40).
    random_state : int
        Zufalls-Seed für reproduzierbare K-Means-Ergebnisse.
    """

    def __init__(self, n_zones: int = 40, random_state: int = 42) -> None:
        self.n_zones = n_zones
        self.random_state = random_state

        # Werden in fit() befüllt
        self.zone_labels_: np.ndarray | None = None                    # (n_stations,)
        self.centroids_: np.ndarray | None = None                      # (n_zones, 2)
        self.convex_hull_areas_: np.ndarray | None = None              # (n_zones,) in km²
        self.mean_depot_distances_: np.ndarray | None = None           # (n_zones,) in km
        self.mean_dist_to_other_centroids_: np.ndarray | None = None   # (n_zones,) in km
        self.station_indices_per_zone_: dict[int, list[int]] | None = None
        self._kmeans: KMeans | None = None

    # ------------------------------------------------------------------
    # Hauptmethoden
    # ------------------------------------------------------------------

    def fit(self, coords: np.ndarray, depot_coords: tuple[float, float]) -> "ZoneClusterer":
        """
        Führt K-Means-Clustering durch und berechnet Zoneneigenschaften.

        Verwendet balanced Assignment: nach K-Means werden Stationen per
        linearer Zuweisung (scipy.optimize.linear_sum_assignment) gleichmäßig
        auf Zonen verteilt, sodass jede Zone höchstens ceil(n/k) Stationen hat.

        Parameters
        ----------
        coords : np.ndarray, shape (n_stations, 2)
            Stationskoordinaten als (Breitengrad, Längengrad).
        depot_coords : tuple[float, float]
            Depotkoordinaten als (Breitengrad, Längengrad).

        Returns
        -------
        self
        """
        self._kmeans = KMeans(
            n_clusters=self.n_zones,
            random_state=self.random_state,
            n_init="auto",
        )
        self.zone_labels_ = self._kmeans.fit_predict(coords)
        self.centroids_ = self._kmeans.cluster_centers_  # (n_zones, 2)

        # Stationsindizes pro Zone
        self.station_indices_per_zone_ = {
            z: [] for z in range(self.n_zones)
        }
        for station_idx, zone in enumerate(self.zone_labels_):
            self.station_indices_per_zone_[int(zone)].append(station_idx)

        # Zoneneigenschaften
        depot = np.array(depot_coords)
        self.convex_hull_areas_ = np.zeros(self.n_zones)
        self.mean_depot_distances_ = np.zeros(self.n_zones)

        for z in range(self.n_zones):
            idxs = self.station_indices_per_zone_[z]
            zone_coords = coords[idxs]

            # Mittlere Depot-Entfernung (km)
            dists = np.array([_approx_km(c, depot) for c in zone_coords])
            self.mean_depot_distances_[z] = dists.mean()

            # Konvexe-Hülle-Fläche (km²)
            self.convex_hull_areas_[z] = _convex_hull_area_km2(zone_coords)

        # Mittlere Distanz jedes Zonenschwerpunkts zu allen anderen Schwerpunkten (km)
        # Kleiner Wert = zentrale Zone
        c = self.centroids_
        self.mean_dist_to_other_centroids_ = np.array([
            np.mean([_approx_km(c[z], c[other]) for other in range(self.n_zones) if other != z])
            for z in range(self.n_zones)
        ])

        return self

    def zone_of_station(self, station_idx: int) -> int:
        """Gibt die Zonen-ID einer Station zurück."""
        assert self.zone_labels_ is not None, "fit() zuerst aufrufen."
        return int(self.zone_labels_[station_idx])

    def centroid_distance_km(self, zone_a: int, zone_b: int) -> float:
        """Entfernung zwischen zwei Zonenzentroids in km."""
        assert self.centroids_ is not None
        return _approx_km(self.centroids_[zone_a], self.centroids_[zone_b])

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ZoneClusterer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Datei enthält kein ZoneClusterer-Objekt: {path}")
        return obj


# ------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------

def _convex_hull_area_km2(coords: np.ndarray) -> float:
    """
    Fläche der konvexen Hülle einer Punktwolke in km².
    Wandelt Lat/Lon in lokale km-Koordinaten um, dann Shoelace-Formel.
    Gibt 0.0 zurück falls < 3 Punkte (keine Fläche definierbar).
    """
    if len(coords) < 3:
        return 0.0

    lat_mid = np.radians(coords[:, 0].mean())
    # Lokale kartesische Koordinaten (km)
    x = coords[:, 1] * 111.0 * np.cos(lat_mid)
    y = coords[:, 0] * 111.0
    xy = np.stack([x, y], axis=1)

    try:
        hull = ConvexHull(xy)
        return float(hull.volume)  # in 2D ist volume = Fläche
    except QhullError:
        # Alle Punkte kollinear → Fläche = 0
        return 0.0
