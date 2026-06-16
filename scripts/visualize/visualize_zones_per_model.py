"""
Visualisiert für jedes Modell (Myopic, Myopic+, CFA-Future, DB-Simple) und jede
Zonen-Auswahl-Variante (centrality, value_based, rollout) die Zonen als
Flächen, eingefärbt nach dem durchschnittlichen Tag, an dem die jeweilige Zone
über alle Monte-Carlo-Runs abgearbeitet wurde.

Die Zonen selbst stammen aus dem K-Means-Clustering (ZoneClusterer) und sind
für gleiches n_zones + Seed deterministisch identisch über alle Varianten.

Pro (Modell, Variante) werden erzeugt:
  - eine interaktive HTML-Karte unter data/processed/zone_maps/<label>.html
  - eine statische PNG-Grafik unter data/figures/zone_maps/<label>.png

Ausführen:
    .venv/bin/python3 scripts/visualize/visualize_zones_per_model.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import branca.colormap as cm
import folium
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull, Voronoi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_stations, get_coordinates
from src.planning.clustering import ZoneClusterer

# ── Konfiguration ──────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DEPOT = (cfg["depot"]["lat"], cfg["depot"]["lon"])
CLUSTER_SEED = cfg["project"]["seed"]

OUT_DIR = Path("data/processed/zone_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("data/figures/zone_maps")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (Modell, Variante) -> Verzeichnis mit run_*.json
RUN_DIRS: dict[tuple[str, str], Path] = {
    ("myopic", "centrality"): Path("logs/myopic_centrality/json"),
    ("myopic", "value_based"): Path("logs/myopic_value_based/json"),
    ("myopic", "rollout"): Path("logs/myopic_rollout/json"),

    ("myopic_plus", "centrality"): Path("logs/myopic_plus_centrality/json"),
    ("myopic_plus", "value_based"): Path("logs/myopic_plus_value_based/json"),
    ("myopic_plus", "rollout"): Path("logs/myopic_plus_rollout/json"),

    ("cfa_future", "centrality"): Path("logs/cfa_future_centrality/json"),
    ("cfa_future", "value_based"): Path("logs/cfa_future_value_based/json"),
    ("cfa_future", "rollout"): Path("logs/cfa_future_rollout/json"),

    ("db_simple", "centrality"): Path("logs/db_simple/json"),
    ("db_simple", "value_based"): Path("logs/db_simple_value_based/json"),
    ("db_simple", "rollout"): Path("logs/db_simple_rollout/json"),
}


def _collect_visit_data(json_dir: Path) -> tuple[Counter, Counter, int | None]:
    """Sammelt pro Station (node_idx) Besuchsanzahl und Summe der Tage im 8-Uhr-Tagesplan."""
    counts: Counter = Counter()
    day_sums: Counter = Counter()
    n_zones: int | None = None

    run_files = sorted(json_dir.glob("run_*.json"))
    for f in run_files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if n_zones is None:
            n_zones = data.get("meta", {}).get("model_params", {}).get("n_zones")

        for entry in data.get("hourly", []):
            if entry.get("hour") != 8:
                continue
            day = entry.get("day")
            for team_plan in entry.get("initial_plan", []):
                for stop in team_plan.get("route", []):
                    node_idx = stop.get("node_idx")
                    if node_idx is not None:
                        counts[node_idx] += 1
                        day_sums[node_idx] += day

    return counts, day_sums, n_zones


def _zone_stats(counts: Counter, day_sums: Counter, clusterer: ZoneClusterer,
                n_zones: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregiert pro Zone: Gesamt-Besuche und durchschnittlicher Bearbeitungstag."""
    zone_visits = np.zeros(n_zones)
    zone_day_sum = np.zeros(n_zones)

    for node_idx, c in counts.items():
        station_idx = node_idx - 1
        if 0 <= station_idx < len(clusterer.zone_labels_):
            z = clusterer.zone_labels_[station_idx]
            zone_visits[z] += c
            zone_day_sum[z] += day_sums[node_idx]

    with np.errstate(invalid="ignore", divide="ignore"):
        zone_avg_day = np.where(zone_visits > 0, zone_day_sum / zone_visits, np.nan)

    return zone_visits, zone_avg_day


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> list[np.ndarray]:
    """
    Erweitert die (teils unbeschränkten) Voronoi-Regionen von scipy zu endlichen
    Polygonen, indem unendliche Kanten weit nach außen verlängert werden.
    Gibt pro Eingabepunkt (in Originalreihenfolge) ein Polygon (Array von (x,y)) zurück.
    Standard-Rezept (Pauli Virtanen, scipy-Mailingliste).
    """
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    center = vor.points.mean(axis=0)
    new_vertices = vor.vertices.tolist()

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    polygons = []
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            polygons.append(np.asarray(new_vertices)[vertices])
            continue

        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # endliche Kante

            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.asarray(new_region)[np.argsort(angles)]

        polygons.append(np.asarray(new_vertices)[new_region])

    return polygons


def _clip_polygon(poly: np.ndarray, edges: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
    """Schneidet ein konvexes Polygon mit einem konvexen Polygon (Sutherland-Hodgman).

    `edges` muss die Kanten des Clip-Polygons in Gegenuhrzeigersinn enthalten.
    """

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(map(tuple, poly))
    for a, b in edges:
        if not output:
            break
        input_list = output
        output = []
        for i in range(len(input_list)):
            cur = input_list[i]
            prev = input_list[i - 1]
            cur_in = inside(cur, a, b)
            prev_in = inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, a, b))
    return np.array(output) if output else poly


def _compute_zone_polygons(clusterer: ZoneClusterer, station_coords: np.ndarray) -> list[np.ndarray]:
    """Voronoi-Tessellation der Zonenzentroide (lon, lat), geclippt auf die konvexe Hülle der Stationen."""
    centroids_lonlat = clusterer.centroids_[:, ::-1]
    vor = Voronoi(centroids_lonlat)
    raw_polygons = _voronoi_finite_polygons_2d(vor)

    stations_lonlat = station_coords[:, ::-1]
    hull = ConvexHull(stations_lonlat)
    hull_pts = stations_lonlat[hull.vertices]  # scipy: Gegenuhrzeigersinn in 2D

    # Hülle leicht nach außen vergrößern, damit Randstationen nicht exakt auf der Grenze liegen
    center = hull_pts.mean(axis=0)
    hull_pts = center + (hull_pts - center) * 1.05

    n = len(hull_pts)
    edges = [(tuple(hull_pts[i]), tuple(hull_pts[(i + 1) % n])) for i in range(n)]

    return [_clip_polygon(p, edges) for p in raw_polygons]


def _build_map(label: str, zone_visits: np.ndarray, zone_avg_day: np.ndarray,
               zone_polygons: list[np.ndarray], n_zones: int,
               max_day: int) -> folium.Map:
    colormap = cm.LinearColormap(
        ["#1a9850", "#fee08b", "#d73027"],
        vmin=1, vmax=max_day,
        caption=f"Durchschnittlicher Bearbeitungstag ({label})",
    )

    m = folium.Map(location=list(DEPOT), zoom_start=12, tiles="CartoDB positron")
    colormap.add_to(m)

    for z in range(n_zones):
        poly_lonlat = zone_polygons[z]
        locations = [[float(lat), float(lon)] for lon, lat in poly_lonlat]

        avg_day = zone_avg_day[z]
        color = colormap(avg_day) if not np.isnan(avg_day) else "#cccccc"
        tooltip = (
            f"Zone {z}<br>"
            f"Gesamt-Besuche: {int(zone_visits[z])}<br>"
            f"Ø Bearbeitungstag: {avg_day:.1f}" if not np.isnan(avg_day)
            else f"Zone {z}<br>nie angefahren"
        )

        folium.Polygon(
            locations=locations,
            color="#666666",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.75 if not np.isnan(avg_day) else 0.2,
            tooltip=tooltip,
        ).add_to(m)

    folium.Marker(
        location=list(DEPOT),
        tooltip="<b>Depot</b> – WVV Betriebshof Sanderau",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:1.9">
      <b>{label}</b><br>
      Fläche = Zone (konvexe Hülle der Stationen)<br>
      Farbe = Ø Tag der Bearbeitung (grün = früh, rot = spät)<br>
      Grau = nie im 8-Uhr-Plan enthalten
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m


def _build_static_figure(label: str, zone_visits: np.ndarray, zone_avg_day: np.ndarray,
                          zone_polygons: list[np.ndarray], n_zones: int,
                          max_day: int) -> plt.Figure:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = plt.Normalize(vmin=1, vmax=max_day)

    colors = []
    for z in range(n_zones):
        avg_day = zone_avg_day[z]
        colors.append(cmap(norm(avg_day)) if not np.isnan(avg_day) else (0.85, 0.85, 0.85, 1.0))

    fig, ax = plt.subplots(figsize=(8, 8))
    pc = PolyCollection(zone_polygons, facecolors=colors, edgecolors="#666666", linewidths=0.3)
    ax.add_collection(pc)

    ax.scatter(DEPOT[1], DEPOT[0], color="black", marker="^", s=120,
               label="Depot", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Ø Bearbeitungstag (über alle Runs)")

    ax.set_title(label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.autoscale_view()
    fig.tight_layout()
    return fig


def main() -> None:
    print("Lade Stationsdaten...")
    df = load_stations(cfg)
    coords = np.array(get_coordinates(df, cfg))

    clusterer_cache: dict[int, ZoneClusterer] = {}
    polygons_cache: dict[int, list[np.ndarray]] = {}

    # ── Pass 1: alle Daten einlesen, globales Max für die Farbskala bestimmen ──
    results: dict[str, tuple[np.ndarray, np.ndarray, list[np.ndarray], int, int]] = {}
    global_max_day = 0.0

    for (model, variant), json_dir in RUN_DIRS.items():
        label = f"{model}_{variant}"
        if not json_dir.exists() or not any(json_dir.glob("run_*.json")):
            print(f"[skip] {label}: kein Verzeichnis/keine run_*.json unter {json_dir}")
            continue

        counts, day_sums, n_zones = _collect_visit_data(json_dir)
        if not counts:
            print(f"[skip] {label}: keine 8-Uhr-Tagespläne gefunden")
            continue
        if n_zones is None:
            n_zones = cfg["planning"]["n_zones"]

        if n_zones not in clusterer_cache:
            clusterer = ZoneClusterer(n_zones=n_zones, random_state=CLUSTER_SEED)
            clusterer.fit(coords[1:], DEPOT)
            clusterer_cache[n_zones] = clusterer
            polygons_cache[n_zones] = _compute_zone_polygons(clusterer, coords[1:])
        clusterer = clusterer_cache[n_zones]
        zone_polygons = polygons_cache[n_zones]

        zone_visits, zone_avg_day = _zone_stats(counts, day_sums, clusterer, n_zones)
        global_max_day = max(global_max_day, float(np.nanmax(zone_avg_day)))
        results[label] = (zone_visits, zone_avg_day, zone_polygons, n_zones, len(counts))

    max_day = int(np.ceil(global_max_day))
    print(f"\nGemeinsame Farbskala: 1 - {max_day} (max. Ø-Bearbeitungstag über alle Varianten)")

    # ── Pass 2: Karten/Grafiken mit gemeinsamer Skala erzeugen ──
    for label, (zone_visits, zone_avg_day, zone_polygons, n_zones, n_stations) in results.items():
        m = _build_map(label, zone_visits, zone_avg_day, zone_polygons, n_zones, max_day)
        out_path = OUT_DIR / f"{label}.html"
        m.save(str(out_path))

        fig = _build_static_figure(label, zone_visits, zone_avg_day, zone_polygons, n_zones, max_day)
        fig_path = FIG_DIR / f"{label}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        print(f"[ok]   {label}: {n_stations} Stationen, {n_zones} Zonen -> {out_path}, {fig_path}")

    print(f"\nFertig. Interaktive Karten: {OUT_DIR.resolve()}")
    print(f"Statische Grafiken (für Thesis): {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
