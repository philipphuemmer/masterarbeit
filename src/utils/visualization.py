"""
Visualisierungshelfer für Ladesäulen und Wartungsrouten.
"""
from __future__ import annotations

import folium
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_stations_map(
    df: pd.DataFrame,
    lat_col: str = "Breitengrad",
    lon_col: str = "Längengrad",
    depot: tuple[float, float] | None = (49.7913, 9.9534),
    highlight_indices: list[int] | None = None,
) -> folium.Map:
    """
    Erstellt eine interaktive Folium-Karte aller Ladesäulen.

    Parameters
    ----------
    highlight_indices : Stationsindizes (0-basiert ohne Depot), die rot markiert werden.
    """
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    highlight_set = set(highlight_indices or [])

    for idx, row in df.iterrows():
        color = "red" if idx in highlight_set else "blue"
        popup = (
            f"<b>ID:</b> {row.get('Ladeeinrichtungs-ID', idx)}<br>"
            f"<b>Betreiber:</b> {row.get('Betreiber', '-')}<br>"
            f"<b>Leistung:</b> {row.get('Nennleistung Ladeeinrichtung [kW]', '-')} kW<br>"
            f"<b>Ladepunkte:</b> {row.get('Anzahl Ladepunkte', '-')}<br>"
            f"<b>Straße:</b> {row.get('Straße', '-')} {row.get('Hausnummer', '')}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(m)

    if depot is not None:
        folium.Marker(
            location=list(depot),
            popup="Depot (Stadtwerke Würzburg)",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

    return m


def plot_route(
    coords: list[tuple[float, float]],
    route: list[int],
    depot_idx: int = 0,
) -> folium.Map:
    """
    Zeichnet eine Wartungsroute auf einer Folium-Karte.

    Parameters
    ----------
    coords : Liste aller (lat, lon)-Punkte (Index 0 = Depot).
    route : Besuchsreihenfolge als Indizes in coords.
    """
    center = list(coords[depot_idx])
    m = folium.Map(location=center, zoom_start=13)

    # Stationen
    for i, (lat, lon) in enumerate(coords):
        color = "green" if i == depot_idx else "blue"
        icon_name = "home" if i == depot_idx else "info-sign"
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=color, icon=icon_name),
            popup=f"{'Depot' if i == depot_idx else f'Station {i}'}",
        ).add_to(m)

    # Route als Linie
    route_coords = [list(coords[i]) for i in route]
    folium.PolyLine(route_coords, color="red", weight=3, opacity=0.8).add_to(m)

    return m


def plot_distance_matrix(matrix: np.ndarray, title: str = "Distanzmatrix [m]") -> None:
    """Heatmap der Distanzmatrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("Ziel-Index")
    ax.set_ylabel("Start-Index")
    plt.tight_layout()
    plt.show()
