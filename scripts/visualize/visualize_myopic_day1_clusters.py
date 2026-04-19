"""
Visualisiert die Cluster (Zonen) für Tag 1 von Myopic:
- Alle Stationen aller 60 Zonen (klein, grau)
- Zone 14 (Team 0, blau) und Zone 58 (Team 1, rot) hervorgehoben
- Tatsächlich gewählte Stops inkl. Nearest-Neighbor-Expansion
- Zonenzentroids als Sterne
- Depot
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import folium
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.loader import load_stations, get_coordinates
from src.planning.clustering import ZoneClusterer, _convex_hull_area_km2

df = load_stations(cfg)
coords = np.array(get_coordinates(df, cfg))

DEPOT_LAT = cfg["depot"]["lat"]
DEPOT_LON = cfg["depot"]["lon"]

clusterer = ZoneClusterer(n_zones=cfg["planning"]["n_zones"], random_state=1)
clusterer.fit(coords[1:], (DEPOT_LAT, DEPOT_LON))

def node_coords(node_idx):
    c = coords[node_idx]
    return (float(c[0]), float(c[1]))

def station_coords(station_idx):
    c = coords[station_idx + 1]
    return (float(c[0]), float(c[1]))

# ── Daten ──────────────────────────────────────────────────────────────────────
ZONE_T0 = 14   # Team 0 Startzone
ZONE_T1 = 58   # Team 1 Startzone

# Selector-Output (vor OR-Tools)
selector_t0 = [1, 2, 3, 6, 7, 8, 21, 22, 37, 38]
selector_t1 = [32, 57, 58, 59, 68, 74, 75, 136, 137, 153]

# OR-Tools Finalplan (aus Log)
final_t0 = [8, 7, 6, 22, 21, 2, 1, 3, 153, 136, 137]
final_t1 = [38, 37, 32, 57, 58, 59, 74, 75, 68]

zone_all_t0 = clusterer.station_indices_per_zone_[ZONE_T0]  # station_indices (0-based)
zone_all_t1 = clusterer.station_indices_per_zone_[ZONE_T1]

COLORS = {0: "#1a6faf", 1: "#c0392b"}

# ── Karte ──────────────────────────────────────────────────────────────────────
m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=13, tiles="CartoDB positron")

fg_all    = folium.FeatureGroup(name="Alle anderen Zonen (grau)", show=True)
fg_zone   = folium.FeatureGroup(name="Startzonen Zone 14 + 58", show=True)
fg_sel    = folium.FeatureGroup(name="Selector-Output (vor OR-Tools)", show=True)
fg_final  = folium.FeatureGroup(name="OR-Tools Finalplan", show=True)
fg_hull   = folium.FeatureGroup(name="Konvexe Hüllen der Startzonen", show=True)

# Alle Stationen aller anderen Zonen (klein, grau)
selected_nodes = set(final_t0 + final_t1)
for station_idx in range(len(df)):
    node_idx = station_idx + 1
    zone = clusterer.zone_of_station(station_idx)
    if zone in (ZONE_T0, ZONE_T1):
        continue
    if node_idx in selected_nodes:
        continue
    lat, lon = station_coords(station_idx)
    folium.CircleMarker(
        location=[lat, lon], radius=3,
        color="#aaaaaa", fill=True, fill_color="#cccccc", fill_opacity=0.5,
        tooltip=f"Node {node_idx} | Zone {zone}",
    ).add_to(fg_all)

# Alle Stationen in Zone 14 und 58 (mittelgroß, Zonenfarbe, nicht selektiert)
for station_idx in zone_all_t0:
    node_idx = station_idx + 1
    lat, lon = station_coords(station_idx)
    in_sel = node_idx in selector_t0
    folium.CircleMarker(
        location=[lat, lon], radius=7,
        color=COLORS[0], fill=True, fill_color=COLORS[0],
        fill_opacity=0.25 if not in_sel else 0.0,
        weight=1.5,
        tooltip=f"Node {node_idx} | Zone {ZONE_T0} (Team 0) {'✓ selektiert' if in_sel else '– nicht selektiert'}",
    ).add_to(fg_zone)

for station_idx in zone_all_t1:
    node_idx = station_idx + 1
    lat, lon = station_coords(station_idx)
    in_sel = node_idx in selector_t1
    folium.CircleMarker(
        location=[lat, lon], radius=7,
        color=COLORS[1], fill=True, fill_color=COLORS[1],
        fill_opacity=0.25 if not in_sel else 0.0,
        weight=1.5,
        tooltip=f"Node {node_idx} | Zone {ZONE_T1} (Team 1) {'✓ selektiert' if in_sel else '– nicht selektiert'}",
    ).add_to(fg_zone)

# Konvexe Hüllen der Startzonen
for zone_id, team_id in [(ZONE_T0, 0), (ZONE_T1, 1)]:
    zone_stations = clusterer.station_indices_per_zone_[zone_id]
    zone_coords_raw = coords[1:][zone_stations]
    if len(zone_coords_raw) >= 3:
        try:
            hull = ConvexHull(zone_coords_raw)
            hull_pts = zone_coords_raw[hull.vertices].tolist()
            hull_pts.append(hull_pts[0])  # schließen
            folium.Polygon(
                locations=[[p[0], p[1]] for p in hull_pts],
                color=COLORS[team_id], weight=2,
                fill=True, fill_color=COLORS[team_id], fill_opacity=0.06,
                tooltip=f"Zone {zone_id} (Team {team_id}): {len(zone_stations)} Stationen",
            ).add_to(fg_hull)
        except Exception:
            pass

# Zonenzentroids
for zone_id, team_id in [(ZONE_T0, 0), (ZONE_T1, 1)]:
    c = clusterer.centroids_[zone_id]
    folium.Marker(
        location=[float(c[0]), float(c[1])],
        tooltip=f"Zentroid Zone {zone_id} (Team {team_id})",
        icon=folium.Icon(color="blue" if team_id == 0 else "red", icon="star", prefix="fa"),
    ).add_to(fg_zone)

# Selector-Output (vor OR-Tools): gefüllt, mittelgross
for rank, node in enumerate(selector_t0, 1):
    lat, lon = node_coords(node)
    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color=COLORS[0], fill=True, fill_color=COLORS[0], fill_opacity=0.7,
        tooltip=f"Team 0 | Selector #{rank} | Node {node}",
    ).add_to(fg_sel)

for rank, node in enumerate(selector_t1, 1):
    lat, lon = node_coords(node)
    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color=COLORS[1], fill=True, fill_color=COLORS[1], fill_opacity=0.7,
        tooltip=f"Team 1 | Selector #{rank} | Node {node}",
    ).add_to(fg_sel)

# OR-Tools Finalplan mit Route
for team_id, stops in [(0, final_t0), (1, final_t1)]:
    color = COLORS[team_id]
    full = [0] + stops + [0]
    latlons = [node_coords(n) if n != 0 else (DEPOT_LAT, DEPOT_LON) for n in full]
    folium.PolyLine(
        latlons, color=color, weight=3.0, opacity=0.9,
        tooltip=f"Team {team_id} Route",
    ).add_to(fg_final)
    for rank, node in enumerate(stops, 1):
        lat, lon = node_coords(node)
        folium.Marker(
            location=[lat, lon],
            tooltip=f"Team {team_id} | Stop {rank} | Node {node}",
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:white;'
                     f'background:{color};border-radius:50%;width:20px;height:20px;'
                     f'display:flex;align-items:center;justify-content:center;'
                     f'border:1px solid white">{rank}</div>',
                icon_size=(20, 20), icon_anchor=(10, 10),
            ),
        ).add_to(fg_final)

# Depot
folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot (WVV Betriebshof)",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

fg_all.add_to(m)
fg_hull.add_to(m)
fg_zone.add_to(m)
fg_sel.add_to(m)
fg_final.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:2.0">
  <b>Myopic – Tag 1: Cluster-Logik</b><br>
  <span style="color:#aaa">&#9632;</span> Alle anderen Zonen (57 Zonen, ~377 Stationen)<br>
  <span style="color:#1a6faf">&#9632;</span> Zone 14 – Team 0 Startzone (24 Stationen total)<br>
  <span style="color:#c0392b">&#9632;</span> Zone 58 – Team 1 Startzone (12 Stationen total)<br>
  <hr style="margin:4px 0">
  Transparent = in Zone, nicht selektiert<br>
  Halb-gefüllt = Selector-Output (vor OR-Tools)<br>
  Nummeriert = OR-Tools Finalplan (Reihenfolge)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out = "data/processed/myopic_day1_clusters.html"
m.save(out)
print(f"Karte gespeichert: {out}")
