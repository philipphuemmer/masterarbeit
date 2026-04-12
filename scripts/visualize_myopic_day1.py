"""
Visualisiert Initialplan + Tagesverlauf von Myopic Tag 1 als interaktive Karte.
Daten direkt aus logs/myopic/run_1.json gelesen.

Team 0 = Blau, Team 1 = Rot.
Störungen = Orange (eingebaut) / Schwarz (Carryover).
Ausgebaute Routine-Stops = Hellgrau.
"""
import sys
import yaml
import numpy as np
import folium

sys.path.insert(0, ".")

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

# ── Routen aus Log ─────────────────────────────────────────────────────────────
team0_initial = [8, 7, 6, 22, 21, 2, 1, 3, 153, 136, 137]
team1_initial = [38, 37, 32, 57, 58, 59, 74, 75, 68]

# Stops die durch Störungen ausgebaut wurden (Carryover)
# [68, 75] ausgebaut für Node 76 @ 11:00
# [74]     ausgebaut für Node 38 @ 12:00  (74 war schon raus, 74 erneut?)
# [137,136] ausgebaut für Node 248 @ 13:00
# [59, 58] ausgebaut für Node 74 @ 15:00
dropped = {
    68:  ("Team 1", "ausgebaut 11:00 für Node 76"),
    75:  ("Team 1", "ausgebaut 11:00 für Node 76"),
    74:  ("Team 1", "ausgebaut 12:00 für Node 38"),
    136: ("Team 0", "ausgebaut 13:00 für Node 248"),
    137: ("Team 0", "ausgebaut 13:00 für Node 248"),
    59:  ("Team 1", "ausgebaut 15:00 für Node 74"),
    58:  ("Team 1", "ausgebaut 15:00 für Node 74"),
}

# Finalrouten (Initialplan minus drops plus Störungen in Reihenfolge)
# Team 0: [8,7,6,22,21,2,1,3,153] + [248(Störung)]  (136,137 ausgebaut)
# Team 1: [38,37,32,57] + [268(Störung)] + [76(Störung)] + [38(Störung)] + [74(Störung)]
#         (68,75,74,59,58 ausgebaut)
team0_final = [8, 7, 6, 22, 21, 2, 1, 3, 153, 248]
team1_final = [38, 37, 32, 57, 268, 76, 38, 74]

# Störungen: (node_idx, stunde, typ, kw, status, team, ankunft, dropped_nodes)
disruptions = [
    (268, "09:00", "Typ 2", 300, "eingebaut", 1, "09:43", []),
    (76,  "11:00", "Typ 1",  22, "eingebaut", 1, "11:26", [68, 75]),
    (38,  "12:00", "Typ 1",  22, "eingebaut", 1, "12:36", [74]),
    (248, "13:00", "Typ 2",  90, "eingebaut", 0, "13:33", [136, 137]),
    (74,  "15:00", "Typ 1",  22, "eingebaut", 1, "15:13", [59, 58]),
    (370, "15:00", "Typ 2",  11, "Carryover", None, "-",  []),
]

# ── Karte ──────────────────────────────────────────────────────────────────────
m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=13, tiles="CartoDB positron")

COLORS = {0: "#1a6faf", 1: "#c0392b"}
TEAM_NAMES = {0: "Team 0", 1: "Team 1"}

fg_initial = folium.FeatureGroup(name="Initialplan (gestrichelt)", show=True)
fg_final   = folium.FeatureGroup(name="Tagesverlauf (final)", show=True)
fg_dropped = folium.FeatureGroup(name="Ausgebaute Routine-Stops", show=True)
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

add_route(fg_initial, 0, team0_initial, dashed=True, label_suffix="(Initialplan)")
add_route(fg_initial, 1, team1_initial, dashed=True, label_suffix="(Initialplan)")
add_route(fg_final,   0, team0_final,   dashed=False, label_suffix="(final)")
add_route(fg_final,   1, team1_final,   dashed=False, label_suffix="(final)")

# Ausgebaute Stops (grau, mit X)
for node, (team_name, reason) in dropped.items():
    lat, lon = node_coords(node)
    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color="#888888",
        fill=True,
        fill_color="#cccccc",
        fill_opacity=0.8,
        tooltip=f"Ausgebaut | Node {node} | {team_name} | {reason}",
    ).add_to(fg_dropped)

# Depot
folium.Marker(
    location=[DEPOT_LAT, DEPOT_LON],
    tooltip="Depot (WVV Betriebshof)",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# Störungen
for node, stunde, typ, kw, status, team, ankunft, _ in disruptions:
    lat, lon = node_coords(node)
    color = "orange" if status == "eingebaut" else "black"
    team_str = f"→ {TEAM_NAMES[team]}, Ankunft {ankunft}" if team is not None else "→ Carryover"
    folium.Marker(
        location=[lat, lon],
        tooltip=f"Störung {stunde} | {typ} | {kw} kW | {status} {team_str}",
        icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
    ).add_to(fg_disrupt)

fg_initial.add_to(m)
fg_final.add_to(m)
fg_dropped.add_to(m)
fg_disrupt.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:2.0">
  <b>Myopic – Tag 1</b><br>
  <span style="color:#1a6faf">&#9632;</span> Team 0 (11 Stops initial, 2 ausgebaut, +Node 248)<br>
  <span style="color:#c0392b">&#9632;</span> Team 1 (9 Stops initial, 5 ausgebaut, +4 Störungen)<br>
  <span style="color:#888">&#9632;</span> Ausgebaute Routine-Stops (7 total)<br>
  <span style="color:orange">&#9632;</span> Störung eingebaut (5 von 6)<br>
  <span style="color:black">&#9632;</span> Carryover (1: Node 370 um 15:00)<br>
  <hr style="margin:4px 0">
  Gestrichelt = Initialplan &nbsp;|&nbsp; Durchgezogen = Tagesverlauf
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out = "data/processed/myopic_day1.html"
m.save(out)
print(f"Karte gespeichert: {out}")
