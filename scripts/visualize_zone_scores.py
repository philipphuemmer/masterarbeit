"""
Visualisiert die Zonenscores und die Startzonenentscheidung für Tag 1.

Erzeugt zwei HTML-Karten:
  data/processed/zone_scores_day1.html   – alle 60 Zonen mit Score-Heatmap
  data/processed/zone_decision_day1.html – Entscheidungspfad: Top-20, Team-Zuweisung, Expansion

Ausführen:
    .venv/bin/python3 scripts/visualize_zone_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import folium
from folium.plugins import FloatImage
import branca.colormap as cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_stations, get_coordinates
from src.planning.clustering import ZoneClusterer, _approx_km
from src.planning.selector import DailyZoneSelector
from src.planning.vrp_solver import TeamState


# ── Konfiguration & Daten ─────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

df = load_stations(cfg)
coords = np.array(get_coordinates(df, cfg))   # (n_stations+1, 2), Index 0 = Depot
DEPOT = (cfg["depot"]["lat"], cfg["depot"]["lon"])
n_stations = len(df)

print(f"Stationen: {n_stations}, Depot: {DEPOT}")

# ── Clustering ────────────────────────────────────────────────────────────────
print("Clustering (K-Means, 60 Zonen)...")
clusterer = ZoneClusterer(
    n_zones=cfg["planning"]["n_zones"],
    random_state=cfg["project"]["seed"],
)
clusterer.fit(coords[1:], DEPOT)

# ── Score berechnen (exakt wie DailyZoneSelector._score_zones) ────────────────
all_zones = list(range(cfg["planning"]["n_zones"]))
dists = clusterer.mean_depot_distances_[all_zones]
areas = clusterer.convex_hull_areas_[all_zones]

def norm(arr):
    span = arr.max() - arr.min()
    return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)

w_depot = cfg["planning"]["priority_weights"]["depot_distance"]
w_area  = cfg["planning"]["priority_weights"]["convex_hull_area"]
scores_raw = w_depot * norm(dists) + w_area * norm(areas)

# Rang: Index 0 = höchster Score
sorted_idx = np.argsort(scores_raw)[::-1]   # zone-ids absteigend nach Score
zone_rank  = {int(z): int(r) for r, z in enumerate(sorted_idx)}  # zone → Rang (0-basiert)

n_top = cfg["planning"]["n_top_candidates"]   # 20
top_zones = list(sorted_idx[:n_top])

print(f"\nTop-{n_top} Zonen (nach Score absteigend):")
for rank, z in enumerate(top_zones):
    print(f"  Rang {rank+1:2d}: Zone {z:2d} | Score={scores_raw[z]:.3f} | "
          f"Depot-Dist={dists[z]:.2f} km | Area={areas[z]:.4f} km²  | "
          f"Stationen: {len(clusterer.station_indices_per_zone_[z])}")

# ── Startzonenzuweisung (exakt wie _assign_starting_zones) ────────────────────
team_states = [
    TeamState(team_id=0, current_node=0, current_time=0),
    TeamState(team_id=1, current_node=0, current_time=0),
]
min_sep = cfg["planning"]["min_team_separation_km"]

result_zones: dict[int, int] = {}
used_zones: list[int] = []

for state in team_states:
    team_coord = coords[state.current_node]
    best_zone, best_dist = None, np.inf
    reject_log = []
    for z in top_zones:
        if z in used_zones:
            continue
        d_to_team = _approx_km(team_coord, clusterer.centroids_[z])
        too_close = any(
            _approx_km(clusterer.centroids_[z], clusterer.centroids_[uz]) < min_sep
            for uz in used_zones
        )
        if too_close:
            reject_log.append((z, d_to_team, "zu_nah"))
            continue
        if d_to_team < best_dist:
            best_dist = d_to_team
            best_zone = z
    if best_zone is None:
        for z in top_zones:
            if z not in used_zones:
                best_zone = z
                break
        print(f"  Team {state.team_id}: Fallback (kein Zone mit ≥{min_sep}km Abstand gefunden)!")
    result_zones[state.team_id] = best_zone
    used_zones.append(best_zone)
    print(f"\nTeam {state.team_id}: Startzone {best_zone} "
          f"(Rang {zone_rank[best_zone]+1}, Score={scores_raw[best_zone]:.3f}, "
          f"Zentroid-Dist vom Depot: {_approx_km(team_coord, clusterer.centroids_[best_zone]):.2f} km)")
    if reject_log:
        print(f"  Abgelehnte Zonen (zu nah zu Team 0): {[z for z,_,_ in reject_log]}")

# Abstand zwischen den beiden Startzonen
z0, z1 = result_zones[0], result_zones[1]
sep_km = _approx_km(clusterer.centroids_[z0], clusterer.centroids_[z1])
print(f"\nAbstand Startzone Team 0 ↔ Team 1: {sep_km:.2f} km (Mindest: {min_sep} km)")

# ── Stationsauswahl (Nearest-Neighbor-Expansion) ──────────────────────────────
selector = DailyZoneSelector(clusterer, cfg, coords)
remaining = list(range(n_stations))
assignment = selector.select_for_day(remaining, team_states, carryover_tasks=[])

print(f"\nStationen Team 0: {len(assignment.team_tasks[0])} → "
      f"{[t.node_idx for t in assignment.team_tasks[0]]}")
print(f"Stationen Team 1: {len(assignment.team_tasks[1])} → "
      f"{[t.node_idx for t in assignment.team_tasks[1]]}")


# ══════════════════════════════════════════════════════════════════════════════
# KARTE 1: Alle 60 Zonen mit Score-Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\nErstelle Karte 1: Zone-Score-Heatmap...")

colormap = cm.LinearColormap(
    ["#d7e8f7", "#4a90d9", "#1a3a6b"],
    vmin=0, vmax=1,
    caption="Prioritätsscore (0 = niedrig, 1 = hoch)",
)

m1 = folium.Map(location=list(DEPOT), zoom_start=13, tiles="CartoDB positron")
colormap.add_to(m1)

station_coords = coords[1:]  # (n_stations, 2)

for z in range(cfg["planning"]["n_zones"]):
    idxs = clusterer.station_indices_per_zone_[z]
    zone_score = scores_raw[z]
    color = colormap(zone_score)
    rank = zone_rank[z] + 1  # 1-basiert

    # Stationen der Zone
    for s in idxs:
        lat, lon = float(station_coords[s][0]), float(station_coords[s][1])
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0.5,
            tooltip=(
                f"Zone {z} | Rang {rank} | Score {zone_score:.3f}<br>"
                f"Depot-Dist: {dists[z]:.2f} km | Area: {areas[z]:.4f} km²<br>"
                f"Stationen in Zone: {len(idxs)}"
            ),
        ).add_to(m1)

    # Zentroid-Marker
    c = clusterer.centroids_[z]
    folium.CircleMarker(
        location=[float(c[0]), float(c[1])],
        radius=9,
        color="white",
        fill=True,
        fill_color=color,
        fill_opacity=1.0,
        weight=1.5,
        tooltip=(
            f"<b>Zone {z} – Zentroid</b><br>"
            f"Rang {rank} / {cfg['planning']['n_zones']}<br>"
            f"Score: {zone_score:.3f}<br>"
            f"  ↳ {w_depot}×norm(Depot-Dist={dists[z]:.2f}km) = {w_depot*norm(dists)[z]:.3f}<br>"
            f"  ↳ {w_area}×norm(Area={areas[z]:.4f}km²) = {w_area*norm(areas)[z]:.3f}<br>"
            f"Stationen: {len(idxs)}"
        ),
    ).add_to(m1)

# Depot
folium.Marker(
    location=list(DEPOT),
    tooltip="<b>Depot</b> – WVV Betriebshof Sanderau",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m1)

legend1 = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:1.9">
  <b>Prioritätsscore aller {cfg['planning']['n_zones']} Zonen – Tag 1</b><br>
  Score = {w_depot}×norm(Depot-Entfernung) + {w_area}×norm(Konvexe-Hülle-Fläche)<br>
  <span style="color:#1a3a6b">&#9632;</span> Hoher Score (weit + groß)<br>
  <span style="color:#4a90d9">&#9632;</span> Mittlerer Score<br>
  <span style="color:#d7e8f7;border:1px solid #ccc">&#9632;</span> Niedriger Score (nah + klein)<br>
  Großer Kreis = Zonenzentreid (Score-Farbe), Hover für Details
</div>
"""
m1.get_root().html.add_child(folium.Element(legend1))

out1 = "data/processed/zone_scores_day1.html"
m1.save(out1)
print(f"  → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# KARTE 2: Entscheidungspfad – Top-20, Team-Zonen, Expansion
# ══════════════════════════════════════════════════════════════════════════════
print("Erstelle Karte 2: Entscheidungspfad Tag 1...")

m2 = folium.Map(location=list(DEPOT), zoom_start=13, tiles="CartoDB positron")

TEAM_COLORS = {0: "#1a6faf", 1: "#c0392b"}

# Alle nicht-Top-20 Zonen grau (ausgegraut)
for z in all_zones:
    if z in top_zones:
        continue
    idxs = clusterer.station_indices_per_zone_[z]
    for s in idxs:
        lat, lon = float(station_coords[s][0]), float(station_coords[s][1])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#cccccc",
            fill=True,
            fill_color="#cccccc",
            fill_opacity=0.4,
            weight=0.3,
            tooltip=f"Zone {z} | Rang {zone_rank[z]+1} | Score {scores_raw[z]:.3f} | nicht in Top-{n_top}",
        ).add_to(m2)
    c = clusterer.centroids_[z]
    folium.CircleMarker(
        location=[float(c[0]), float(c[1])],
        radius=6,
        color="#bbbbbb",
        fill=True,
        fill_color="#dddddd",
        fill_opacity=0.6,
        weight=1,
        tooltip=f"Zone {z} | Rang {zone_rank[z]+1} | Score {scores_raw[z]:.3f} | nicht in Top-{n_top}",
    ).add_to(m2)

# Top-20 Zonen: grün umrandet, Rang als Label
for rank_0, z in enumerate(top_zones):
    idxs = clusterer.station_indices_per_zone_[z]
    rank = rank_0 + 1

    # Ist das eine Startzone?
    team_id = None
    for tid, sz in result_zones.items():
        if sz == z:
            team_id = tid
            break

    if team_id is not None:
        fill_col  = TEAM_COLORS[team_id]
        border_col = "white"
        label_text = f"T{team_id} – Rang {rank}"
        radius_c   = 13
    else:
        fill_col   = "#27ae60"
        border_col = "#1e8449"
        label_text = f"Rang {rank}"
        radius_c   = 9

    # Stationen
    for s in idxs:
        lat, lon = float(station_coords[s][0]), float(station_coords[s][1])
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=fill_col if team_id is not None else "#27ae60",
            fill=True,
            fill_color=fill_col if team_id is not None else "#27ae60",
            fill_opacity=0.7,
            weight=0.5,
            tooltip=(
                f"Zone {z} | Top-{n_top} Rang {rank} | Score {scores_raw[z]:.3f}<br>"
                f"Depot-Dist: {dists[z]:.2f} km | Area: {areas[z]:.4f} km²"
            ),
        ).add_to(m2)

    # Zentroid
    c = clusterer.centroids_[z]
    sep_note = ""
    if team_id == 1:
        sep_note = f"<br>Abstand zu Team 0-Zone: {sep_km:.2f} km (≥{min_sep} km ✓)"
    elif team_id == 0:
        sep_note = f"<br>Nächste Zone vom Depot ({_approx_km(coords[0], clusterer.centroids_[z]):.2f} km)"

    folium.CircleMarker(
        location=[float(c[0]), float(c[1])],
        radius=radius_c,
        color=border_col,
        fill=True,
        fill_color=fill_col,
        fill_opacity=1.0,
        weight=2,
        tooltip=(
            f"<b>Zone {z} – {'STARTZONE Team '+str(team_id) if team_id is not None else 'Top-'+str(n_top)}</b><br>"
            f"Score: {scores_raw[z]:.3f} (Rang {rank}/{cfg['planning']['n_zones']})<br>"
            f"  {w_depot}×norm(Dist) = {w_depot*norm(dists)[z]:.3f}<br>"
            f"  {w_area}×norm(Area) = {w_area*norm(areas)[z]:.3f}<br>"
            f"Depot-Dist: {dists[z]:.2f} km | Area: {areas[z]:.4f} km²<br>"
            f"Stationen in Zone: {len(idxs)}{sep_note}"
        ),
    ).add_to(m2)

    # Rang-Label auf dem Zentroid
    folium.Marker(
        location=[float(c[0]), float(c[1])],
        icon=folium.DivIcon(
            html=f'<div style="font-size:8px;font-weight:bold;color:white;'
                 f'text-align:center;line-height:1.1;margin-top:-4px">{label_text}</div>',
            icon_size=(60, 20),
            icon_anchor=(30, 10),
        ),
    ).add_to(m2)

# Tatsächlich zugewiesene Stationen pro Team (nach Expansion)
for tid in [0, 1]:
    tasks = assignment.team_tasks[tid]
    color = TEAM_COLORS[tid]
    for t in tasks:
        node = t.node_idx  # 1-basiert
        lat, lon = float(coords[node][0]), float(coords[node][1])
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1.5,
            tooltip=f"Team {tid} | Node {node} | {t.task_type}",
        ).add_to(m2)

# Linie: Depot → Zentroid Team 0, Depot → Zentroid Team 1
for tid, z in result_zones.items():
    c = clusterer.centroids_[z]
    folium.PolyLine(
        [list(DEPOT), [float(c[0]), float(c[1])]],
        color=TEAM_COLORS[tid],
        weight=2,
        dash_array="6 4",
        opacity=0.7,
        tooltip=f"Depot → Startzone Team {tid} ({_approx_km(coords[0], c):.2f} km)",
    ).add_to(m2)

# Doppelpfeil zwischen den Startzonen
c0 = clusterer.centroids_[result_zones[0]]
c1 = clusterer.centroids_[result_zones[1]]
folium.PolyLine(
    [[float(c0[0]), float(c0[1])], [float(c1[0]), float(c1[1])]],
    color="#7d3c98",
    weight=2,
    dash_array="3 6",
    opacity=0.8,
    tooltip=f"Abstand Startzonen: {sep_km:.2f} km (Minimum: {min_sep} km)",
).add_to(m2)

# Depot
folium.Marker(
    location=list(DEPOT),
    tooltip="<b>Depot</b> – WVV Betriebshof Sanderau<br>Beide Teams starten hier",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m2)

legend2 = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-size:13px;line-height:1.9;max-width:320px">
  <b>Startzonenentscheidung Tag 1</b><br>
  <span style="color:#27ae60">&#9632;</span> Top-{n_top} Zonen (Kandidaten)<br>
  <span style="color:{TEAM_COLORS[0]}">&#9632;</span> Team 0 – Startzone Zone {result_zones[0]}
    (Rang {zone_rank[result_zones[0]]+1}, Score {scores_raw[result_zones[0]]:.3f})<br>
  <span style="color:{TEAM_COLORS[1]}">&#9632;</span> Team 1 – Startzone Zone {result_zones[1]}
    (Rang {zone_rank[result_zones[1]]+1}, Score {scores_raw[result_zones[1]]:.3f})<br>
  <span style="color:#cccccc">&#9632;</span> Restliche Zonen (nicht in Top-{n_top})<br>
  &#8212; &#8212; Verbindung Depot → Startzone<br>
  <span style="color:#7d3c98">&#8226; &#8226;</span> Abstand Startzonen: {sep_km:.2f} km<br>
  <br><small>Große Kreise = Zonenzentroids (Hover für Score-Details)<br>
  Kleine Kreise = tatsächlich zugewiesene Stationen</small>
</div>
"""
m2.get_root().html.add_child(folium.Element(legend2))

out2 = "data/processed/zone_decision_day1.html"
m2.save(out2)
print(f"  → {out2}")
print("\nFertig!")
