"""
Generates thesis figure for Chapter 7.4 (Statistical Validation).

Figure saved to thesis/figures/:
  - fig74_cohens_d.pdf  — Horizontal bar chart of Cohen's d for key comparisons

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter74.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Key comparisons (from results/analysis/chapter74/paired_tests.csv)
# grouped by type: zone-selection, policy-value-based, rollout, policy-centrality
# ---------------------------------------------------------------------------

COMPARISONS = [
    # (label, cohens_d, group)
    # Zone selection effects (large)
    ("DB: centrality → value-based",      1.488, "zone"),
    ("CFA: centrality → value-based",     1.497, "zone"),
    ("Myopic+: centrality → value-based", 1.370, "zone"),
    ("Myopic: centrality → value-based",  1.288, "zone"),
    # Policy differences under value-based (small)
    ("Myopic vs DB (value-based)",         0.365, "policy_vb"),
    ("Myopic vs CFA (value-based)",        0.308, "policy_vb"),
    ("Myopic+ vs DB (value-based)",        0.199, "policy_vb"),
    ("Myopic+ vs CFA (value-based)",       0.158, "policy_vb"),
    ("Myopic vs Myopic+ (value-based)",    0.174, "policy_vb"),
    ("CFA vs DB (value-based)",            0.051, "policy_vb"),
    # Rollout effects (negligible)
    ("CFA: value-based → +Rollout",        0.080, "rollout"),
    ("DB: value-based → +Rollout",         0.050, "rollout"),
    ("Myopic+: value-based → +Rollout",    0.024, "rollout"),
    ("Myopic: value-based → +Rollout",     0.006, "rollout"),
    # Policy differences under centrality (negligible)
    ("Myopic vs Myopic+ (centrality)",     0.068, "policy_c"),
    ("Myopic vs CFA (centrality)",         0.060, "policy_c"),
    ("Myopic vs DB (centrality)",          0.055, "policy_c"),
    ("Myopic+ vs CFA (centrality)",        0.013, "policy_c"),
    ("Myopic+ vs DB (centrality)",         0.010, "policy_c"),
    ("CFA vs DB (centrality)",             0.001, "policy_c"),
]

GROUP_COLORS = {
    "zone":      "#C44E52",
    "policy_vb": "#4C72B0",
    "rollout":   "#55A868",
    "policy_c":  "#8c8c8c",
}

GROUP_LABELS = {
    "zone":      "Zone selection effect",
    "policy_vb": "Policy diff. (value-based)",
    "rollout":   "Rollout effect",
    "policy_c":  "Policy diff. (centrality)",
}

labels  = [c[0] for c in COMPARISONS]
values  = [c[1] for c in COMPARISONS]
groups  = [c[2] for c in COMPARISONS]
colors  = [GROUP_COLORS[g] for g in groups]

fig, ax = plt.subplots(figsize=(9, 6.5))

y = np.arange(len(labels))
bars = ax.barh(y, values, color=colors, height=0.65, edgecolor="none")

# Reference lines
for x_ref, style, label in [
    (0.20, "--", "$d = 0.20$ (small)"),
    (0.50, "-.", "$d = 0.50$ (medium)"),
    (0.80, ":",  "$d = 0.80$ (large)"),
]:
    ax.axvline(x_ref, color="black", linewidth=0.9, linestyle=style, alpha=0.6, label=label)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Cohen's $d$ (absolute value)", fontsize=10)
ax.set_xlim(0, 1.65)
ax.tick_params(labelsize=9)
ax.grid(axis="x", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# Legend: groups
group_patches = [mpatches.Patch(color=c, label=l) for g, (c, l) in
                 {g: (GROUP_COLORS[g], GROUP_LABELS[g]) for g in ["zone", "policy_vb", "rollout", "policy_c"]}.items()]
ref_handles, ref_labels = ax.get_legend_handles_labels()
ax.legend(handles=group_patches + ref_handles,
          labels=[p.get_label() for p in group_patches] + ref_labels,
          fontsize=8, loc="lower right", ncol=1)

import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

import matplotlib
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig74_cohens_d.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig74_cohens_d.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved fig74_cohens_d to {OUT_DIR.resolve()}")
