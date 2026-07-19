"""
Karte der Ladesäulenverteilung in Würzburg.
Ausgabe: notebooks/images/stations_map.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.data.loader import load_config, load_stations

config = load_config()
df = load_stations(config)

lat_col = config["data"]["lat_col"]
lon_col = config["data"]["lon_col"]
depot_lat = config["depot"]["lat"]
depot_lon = config["depot"]["lon"]

# GeoDataFrame erstellen (WGS84 → Web Mercator für contextily)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

depot_gdf = gpd.GeoDataFrame(
    [{"geometry": gpd.points_from_xy([depot_lon], [depot_lat])[0]}],
    crs="EPSG:4326",
).to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Stationen zuerst (setzt den Axes-Extent für contextily)
ax.scatter(
    gdf.geometry.x,
    gdf.geometry.y,
    color="#e63946",
    s=32,
    alpha=0.85,
    linewidths=0.4,
    edgecolors="white",
    zorder=3,
)

# Depot
ax.scatter(
    depot_gdf.geometry.x,
    depot_gdf.geometry.y,
    marker="s",
    s=130,
    color="#1a1a2e",
    edgecolors="white",
    linewidths=1.2,
    zorder=5,
)

# Zoom auf Stadtgebiet (leichter Puffer um die Stationen)
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
pad_x = (bounds[2] - bounds[0]) * 0.08
pad_y = (bounds[3] - bounds[1]) * 0.08
ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

# Basemap mit höherem Zoom für mehr Stadtdetail
ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.VoyagerNoLabels,
    zoom=13,
    attribution_size=7,
)

# Legende
depot_patch = mpatches.Patch(color="#1a1a2e", label="Depot (WVV Betriebshof)")
station_patch = mpatches.Patch(color="#e63946", label=f"Charging Station (n={len(df)})")
ax.legend(
    handles=[depot_patch, station_patch],
    loc="lower left",
    fontsize=18,
    framealpha=0.9,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.01, 0.05),
)

ax.set_axis_off()

plt.tight_layout()

out_path = Path("notebooks/images/stations_map.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}  ({len(df)} stations)")
