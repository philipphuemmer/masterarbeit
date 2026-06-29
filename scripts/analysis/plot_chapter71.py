"""
Generates thesis figures for Chapter 7.1 (Policy Mechanism Comparison).
Uses centrality-based zone selection only to isolate the policy effect.

Figures saved to thesis/figures/:
  - fig71_boxplot.pdf   — Boxplot: total cost by policy (centrality)
  - fig71_scatter.pdf   — Scatter: same-day rate vs. total cost (centrality)

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter71.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = [
    ("Myopic",  "logs/ergebnisse/myopic_centrality/json"),
    ("Myopic+", "logs/ergebnisse/myopic_plus_centrality/json"),
    ("CFA",     "logs/ergebnisse/cfa_future_centrality/json"),
    ("DB",      "logs/ergebnisse/db_simple_centrality/json"),
]

MODEL_ORDER  = ["Myopic", "Myopic+", "CFA", "DB"]
MODEL_COLORS = {
    "Myopic":  "#8c8c8c",
    "Myopic+": "#4C72B0",
    "CFA":     "#DD8452",
    "DB":      "#55A868",
}

OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_runs(model: str, json_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(json_dir).glob("run_*.json")):
        d = json.load(open(f))
        s = d["summary"]
        rows.append({
            "model":          model,
            "total_cost_eur": s["total_cost_eur"],
            "same_day_rate":  s["same_day_rate"] * 100,
        })
    return pd.DataFrame(rows)


df = pd.concat([load_runs(*v) for v in VARIANTS], ignore_index=True)

# ---------------------------------------------------------------------------
# Figure 1 — Boxplot (centrality only)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

sns.boxplot(
    data=df,
    x="model", y="total_cost_eur",
    hue="model",
    order=MODEL_ORDER,
    hue_order=MODEL_ORDER,
    palette=MODEL_COLORS,
    width=0.5,
    flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
    linewidth=0.8,
    legend=False,
    ax=ax,
)

ax.set_xlabel("")
ax.set_ylabel("Total Cost (€)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(labelsize=10)
ax.grid(axis="y", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig71_boxplot.pdf", dpi=300)
fig.savefig(OUT_DIR / "fig71_boxplot.png", dpi=200)
plt.close(fig)
print("Saved fig71_boxplot")

# ---------------------------------------------------------------------------
# Figure 2 — Scatter: Same-Day-Rate vs. Total Cost (centrality only)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 5))

for model, grp in df.groupby("model"):
    ax.scatter(
        grp["same_day_rate"], grp["total_cost_eur"],
        color=MODEL_COLORS[model],
        alpha=0.3,
        s=10,
        linewidths=0,
        label=model,
    )

ax.set_xlabel("Same-Day Rate (%)", fontsize=11)
ax.set_ylabel("Total Cost (€)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(labelsize=10)
ax.grid(linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(title="Policy", fontsize=9, title_fontsize=9, loc="upper right")
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig71_scatter.pdf", dpi=300)
fig.savefig(OUT_DIR / "fig71_scatter.png", dpi=200)
plt.close(fig)
print("Saved fig71_scatter")
print(f"\nAll figures saved to {OUT_DIR.resolve()}")
