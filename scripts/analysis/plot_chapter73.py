"""
Generates thesis figures for Chapter 7.3 (VFA Rollout: Performance and Limitations).

Figures saved to thesis/figures/:
  - fig73_delta_boxplot.pdf   — Paired delta cost: base vs. base+rollout per policy
  - fig73_overrides_hist.pdf  — Histogram of replan overrides per run (all 4 rollout variants)

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter73.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAIRS = [
    ("Myopic",  "logs/ergebnisse/myopic_value_based/json",       "logs/ergebnisse/myopic_rollout/json"),
    ("Myopic+", "logs/ergebnisse/myopic_plus_value_based/json",  "logs/ergebnisse/myopic_plus_rollout/json"),
    ("CFA",     "logs/ergebnisse/cfa_future_value_based/json",   "logs/ergebnisse/cfa_future_rollout/json"),
    ("DB",      "logs/ergebnisse/db_simple_value_based/json",    "logs/ergebnisse/db_simple_rollout/json"),
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

def load_costs(json_dir: str) -> dict[int, dict]:
    data = {}
    for f in sorted(Path(json_dir).glob("run_*.json")):
        seed = int(f.stem.split("_")[1])
        d = json.load(open(f))
        s = d["summary"]
        rh = d.get("rolling_horizon_meta", {})
        data[seed] = {
            "total_cost_eur": s["total_cost_eur"],
            "replan_overrides": rh.get("replan_overrides", rh.get("rh_overrides", 0)),
        }
    return data


delta_rows = []
override_rows = []

for model, base_dir, rollout_dir in PAIRS:
    base    = load_costs(base_dir)
    rollout = load_costs(rollout_dir)
    common  = sorted(set(base) & set(rollout))
    for seed in common:
        delta = base[seed]["total_cost_eur"] - rollout[seed]["total_cost_eur"]
        delta_rows.append({"model": model, "delta": delta})
        override_rows.append({
            "model": model,
            "replan_overrides": rollout[seed]["replan_overrides"],
        })

df_delta     = pd.DataFrame(delta_rows)
df_overrides = pd.DataFrame(override_rows)

# ---------------------------------------------------------------------------
# Figure 1 — Boxplot Δ-Kosten (gepaart)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

sns.boxplot(
    data=df_delta,
    x="model", y="delta",
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

ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
ax.set_xlabel("")
ax.set_ylabel("Cost Reduction via Rollout (€)\n(base − rollout, per run)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.tick_params(labelsize=10)
ax.grid(axis="y", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig73_delta_boxplot.pdf", dpi=300)
fig.savefig(OUT_DIR / "fig73_delta_boxplot.png", dpi=200)
plt.close(fig)
print("Saved fig73_delta_boxplot")

# ---------------------------------------------------------------------------
# Figure 2 — Histogram Replan-Overrides
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)

max_ov = int(df_overrides["replan_overrides"].max())
bins   = np.arange(0, max_ov + 2) - 0.5

for ax, model in zip(axes, MODEL_ORDER):
    vals = df_overrides[df_overrides["model"] == model]["replan_overrides"]
    mean_ov = vals.mean()
    ax.hist(vals, bins=bins, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    ax.axvline(mean_ov, color="black", linewidth=1.2, linestyle="--",
               label=f"mean = {mean_ov:.2f}")
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("Replan Overrides", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, handlelength=1)
    ax.set_xlim(-0.5, max_ov + 0.5)
    sns.despine(ax=ax)

axes[0].set_ylabel("Number of Runs", fontsize=10)
fig.suptitle("Distribution of Replan Overrides per Run (500 runs each)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig73_overrides_hist.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig73_overrides_hist.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig73_overrides_hist")
print(f"\nAll figures saved to {OUT_DIR.resolve()}")
