"""
Visualisiert den Initialplan von Tag 1 als interaktive Karte.
Team 0 = Blau, Team 1 = Rot.
Störungen = Orange (eingebaut) / Schwarz (Carryover).
"""
import sys
import yaml
import pandas as pd
import folium

sys.path.insert(0, ".")

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.loader import load_stations, get_coordinates

df = load_stations(cfg)
coords_list = get_coordinates(df, cfg)  # list of (lat, lon): [depot, s1, s2, ...]
import numpy as np
coords = np.array(coords_list)

DEPOT_LAT = cfg["depot"]["lat"]
DEPOT_LON = cfg["depot"]["lon"]

# node_idx → (lat, lon)
def node_coords(node_idx):
    c = coords[node_idx]  # [lat, lon]
    return (float(c[0]), float(c[1]))

# ── Routen aus dem Log ────────────────────────────────────────────────────────
team0_initial = [162,161,160,159,150,133,176,177,182,178,180,181,188,187,186,185]
team1_initial = [28,73,49,183,200,191,194,195,199,218,215,217,216,214,34,35]

# Disruptions: (node_idx, label, handled_by_team)
disruptions = [
    (157, "Typ 1\n11:00\n22kW",  0),
    (340, "Typ 2\n11:00\n250kW", 0),
    (296, "Typ 1\n12:00\n22kW",  1),
    (328, "Typ 1\n14:00\n250kW", 1),
    (18,  "Typ 2\n14:00\n98kW",  0),
]

# ── Karte ─────────────────────────────────────────────────────────────────────
m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=13, tiles="CartoDB positron")

COLORS = {0: "#1a6faf", 1: "#c0392b"}  # Blau / Rot
TEAM_NAMES = {0: "Team 0", 1: "Team 1"}

def add_route(team_id, stops):
    color = COLORS[team_id]
    name = TEAM_NAMES[team_id]
    full_route = [0] + stops + [0]  # Depot → Stops → Depot

    # Linie
    latlons = [node_coords(n) if n != 0 else (DEPOT_LAT, DEPOT_LON)
               for n in full_route]
    folium.PolyLine(
        latlons, color=color, weight=2.5, opacity=0.8,
        tooltip=name,
    ).add_to(m)

    # Stops
    for rank, node in enumerate(stops, start=1):
        lat, lon = node_coords(node)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{name} | Stop {rank} | Node {node}",
        ).add_to(m)

add_route(0, team0_initial)
add_route(1, team1_initial)

# Depot
folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot (WVV Betriebshof)",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# Störungen
for node, label, team in disruptions:
    lat, lon = node_coords(node)
    folium.Marker(
        location=[lat, lon],
        tooltip=f"Störung | {label.replace(chr(10), ' | ')} → {TEAM_NAMES[team]}",
        icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
    ).add_to(m)

# Legende
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:1.8">
  <b>Initialplan Tag 1</b><br>
  <span style="color:#1a6faf">&#9632;</span> Team 0 (16 Stops, 08:09–16:26)<br>
  <span style="color:#c0392b">&#9632;</span> Team 1 (16 Stops, 08:01–16:39)<br>
  <span style="color:orange">&#9632;</span> Störung (5 gesamt, alle eingebaut)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out = "data/processed/day1_initialplan.html"
m.save(out)
print(f"Karte gespeichert: {out}")
