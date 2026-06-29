"""
Generates thesis figures for Chapter 7.2 (Impact of Zone Selection on Cost).

Figures saved to thesis/figures/:
  - fig72_grouped_boxplot.pdf  — Grouped boxplot: total cost by model × zone-selection
  - fig72_stacked_bar.pdf      — Stacked bar: mean cost components per variant

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter72.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = [
    ("Myopic",  "centrality",  "logs/ergebnisse/myopic_centrality/json"),
    ("Myopic+", "centrality",  "logs/ergebnisse/myopic_plus_centrality/json"),
    ("CFA",     "centrality",  "logs/ergebnisse/cfa_future_centrality/json"),
    ("DB",      "centrality",  "logs/ergebnisse/db_simple_centrality/json"),
    ("Myopic",  "value-based", "logs/ergebnisse/myopic_value_based/json"),
    ("Myopic+", "value-based", "logs/ergebnisse/myopic_plus_value_based/json"),
    ("CFA",     "value-based", "logs/ergebnisse/cfa_future_value_based/json"),
    ("DB",      "value-based", "logs/ergebnisse/db_simple_value_based/json"),
]

MODEL_ORDER = ["Myopic", "Myopic+", "CFA", "DB"]
ZONE_PALETTE = {"centrality": "#aec6e8", "value-based": "#1f4e8c"}

OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_runs(model: str, zone: str, json_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(json_dir).glob("run_*.json")):
        d = json.load(open(f))
        s = d["summary"]
        rows.append({
            "model":             model,
            "zone_selection":    zone,
            "total_cost_eur":    s["total_cost_eur"],
            "wage_cost_eur":     s["wage_cost_eur"],
            "fuel_cost_eur":     s["fuel_cost_eur"],
            "downtime_cost_eur": s["downtime_cost_eur"],
        })
    return pd.DataFrame(rows)


df = pd.concat([load_runs(*v) for v in VARIANTS], ignore_index=True)

# ---------------------------------------------------------------------------
# Figure 1 — Grouped Boxplot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="model", y="total_cost_eur",
    hue="zone_selection",
    order=MODEL_ORDER,
    hue_order=["centrality", "value-based"],
    palette=ZONE_PALETTE,
    width=0.55,
    flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
    linewidth=0.8,
    ax=ax,
)

ax.set_xlabel("")
ax.set_ylabel("Total Cost (€)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(labelsize=10)
ax.grid(axis="y", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Centrality", "Value-based"], title="Zone Selection",
          fontsize=9, title_fontsize=9, loc="upper right")

sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig72_grouped_boxplot.pdf", dpi=300)
fig.savefig(OUT_DIR / "fig72_grouped_boxplot.png", dpi=200)
plt.close(fig)
print("Saved fig72_grouped_boxplot")

# ---------------------------------------------------------------------------
# Figure 2 — Stacked Bar: mean cost components
# ---------------------------------------------------------------------------

# Build ordered label list: Myopic-c, Myopic-vb, Myopic+-c, ...
records = []
for model in MODEL_ORDER:
    for zone in ["centrality", "value-based"]:
        grp = df[(df["model"] == model) & (df["zone_selection"] == zone)]
        records.append({
            "label":    f"{model}\n({zone})",
            "model":    model,
            "zone":     zone,
            "wage":     grp["wage_cost_eur"].mean(),
            "fuel":     grp["fuel_cost_eur"].mean(),
            "downtime": grp["downtime_cost_eur"].mean(),
        })
summary = pd.DataFrame(records)

x = np.arange(len(summary))
width = 0.6

COLORS = {"wage": "#4C72B0", "fuel": "#55A868", "downtime": "#C44E52"}

fig, ax = plt.subplots(figsize=(11, 5))

bars_wage     = ax.bar(x, summary["wage"],     width, label="Labor",    color=COLORS["wage"])
bars_fuel     = ax.bar(x, summary["fuel"],     width, label="Travel",   color=COLORS["fuel"],
                       bottom=summary["wage"])
bars_downtime = ax.bar(x, summary["downtime"], width, label="Downtime", color=COLORS["downtime"],
                       bottom=summary["wage"] + summary["fuel"])

# Shade value-based bars slightly darker via edgecolor
for i, (_, row) in enumerate(summary.iterrows()):
    if row["zone"] == "value-based":
        for bars in [bars_wage, bars_fuel, bars_downtime]:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(1.2)

ax.set_xticks(x)
ax.set_xticklabels(summary["label"], fontsize=8.5)
ax.set_ylabel("Mean Cost (€)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.tick_params(labelsize=9)
ax.grid(axis="y", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")

# Vertical separators between model groups
for i in [1.5, 3.5, 5.5]:
    ax.axvline(i, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig72_stacked_bar.pdf", dpi=300)
fig.savefig(OUT_DIR / "fig72_stacked_bar.png", dpi=200)
plt.close(fig)
print("Saved fig72_stacked_bar")
print(f"\nAll figures saved to {OUT_DIR.resolve()}")
