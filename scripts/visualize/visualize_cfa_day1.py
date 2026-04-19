"""
Visualisiert Initialplan + Tagesverlauf von CFA Tag 1 als interaktive Karte.
Daten direkt aus logs/cfa/run_1.json gelesen.

Team 0 = Blau, Team 1 = Rot.
Störungen = Orange (eingebaut) / Schwarz (Carryover).
"""
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import folium

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.loader import load_stations, get_coordinates

df = load_stations(cfg)
coords = np.array(get_coordinates(df, cfg))

DEPOT_LAT = cfg["depot"]["lat"]
DEPOT_LON = cfg["depot"]["lon"]

def node_coords(node_idx):
    c = coords[node_idx]
    return (float(c[0]), float(c[1]))

# ── Routen aus Log ────────────────────────────────────────────────────────────
team0_initial = [37, 38, 3, 6, 1, 21, 2, 22, 153, 136, 137]
team1_initial = [8, 32, 68, 59, 7, 58, 57, 74, 75]

# Finalrouten nach allen Replans (aus letztem bekannten Zustand)
# Team 0: 11:00 → Node 248 eingebaut, Endroute: [248] (restliche erledigter)
# Team 1: mehrfach umgeplant, letzte Route: [76, 7, 6, 22, 1, 2, 21]
team0_final = [37, 38, 3, 6, 1, 21, 2, 22, 153, 136, 137, 248]
team1_final = [8, 32, 68, 59, 268, 76, 7, 6, 22, 1, 2, 21]

# Störungen: (node_idx, stunde, typ, kw, status, team)
disruptions = [
    (268, "09:00", "Typ 2", 300, "eingebaut", 1),
    (76,  "11:00", "Typ 1",  22, "eingebaut", 1),
    (38,  "12:00", "Typ 1",  22, "eingebaut", 0),
    (248, "13:00", "Typ 2",  90, "eingebaut", 0),
    (74,  "15:00", "Typ 1",  22, "Carryover", None),
    (370, "15:00", "Typ 2",  11, "Carryover", None),
]

# ── Karte ─────────────────────────────────────────────────────────────────────
m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=13, tiles="CartoDB positron")

COLORS = {0: "#1a6faf", 1: "#c0392b"}
TEAM_NAMES = {0: "Team 0", 1: "Team 1"}

fg_initial = folium.FeatureGroup(name="Initialplan (gestrichelt)", show=True)
fg_final   = folium.FeatureGroup(name="Tagesverlauf (final)", show=True)
fg_disrupt = folium.FeatureGroup(name="Störungen", show=True)

def add_route(fg, team_id, stops, dashed=False, label_suffix=""):
    color = COLORS[team_id]
    name = TEAM_NAMES[team_id]
    full_route = [0] + stops + [0]
    latlons = [node_coords(n) if n != 0 else (DEPOT_LAT, DEPOT_LON) for n in full_route]
    folium.PolyLine(
        latlons, color=color,
        weight=2.0 if dashed else 3.0,
        opacity=0.4 if dashed else 0.85,
        dash_array="8 4" if dashed else None,
        tooltip=f"{name} {label_suffix}",
    ).add_to(fg)

    for rank, node in enumerate(stops, start=1):
        lat, lon = node_coords(node)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if not dashed else 4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85 if not dashed else 0.3,
            tooltip=f"{name} | Stop {rank} | Node {node} {label_suffix}",
        ).add_to(fg)

# Initialplan gestrichelt
add_route(fg_initial, 0, team0_initial, dashed=True, label_suffix="(Initialplan)")
add_route(fg_initial, 1, team1_initial, dashed=True, label_suffix="(Initialplan)")

# Finalrouten solid
add_route(fg_final, 0, team0_final, dashed=False, label_suffix="(final)")
add_route(fg_final, 1, team1_final, dashed=False, label_suffix="(final)")

# Depot
folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot (WVV Betriebshof)",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# Störungen
for node, stunde, typ, kw, status, team in disruptions:
    lat, lon = node_coords(node)
    color = "orange" if status == "eingebaut" else "black"
    team_str = f"→ {TEAM_NAMES[team]}" if team is not None else "→ Carryover"
    folium.Marker(
        location=[lat, lon],
        tooltip=f"Störung {stunde} | {typ} | {kw} kW | {status} {team_str}",
        icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
    ).add_to(fg_disrupt)

fg_initial.add_to(m)
fg_final.add_to(m)
fg_disrupt.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# Legende
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:2.0">
  <b>CFA – Tag 1</b><br>
  <span style="color:#1a6faf">&#9632;</span> Team 0 (11 Stops initial, +Node 248 Störung)<br>
  <span style="color:#c0392b">&#9632;</span> Team 1 (9 Stops initial, +3 Störungen)<br>
  <span style="color:orange">&#9632;</span> Störung eingebaut (4 von 6)<br>
  <span style="color:black">&#9632;</span> Carryover (2: Node 74, 370 um 15:00)<br>
  <hr style="margin:4px 0">
  Gestrichelt = Initialplan &nbsp;|&nbsp; Durchgezogen = Tagesverlauf
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out = "data/processed/cfa_day1.html"
m.save(out)
print(f"Karte gespeichert: {out}")
