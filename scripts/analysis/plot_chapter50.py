"""
Generates thesis figure for Chapter 5 (Solution Methodology) overview.

Figure saved to thesis/figures/:
  - fig50_policy_hierarchy.pdf  — Hierarchy diagram: shared zone selection layer,
                                   the four base policies (Myopic -> Myopic+ -> CFA -> DB),
                                   and the VFA rollout layer wrapping any of them.

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter50.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Use Computer Modern fonts (same as LaTeX) for math rendering
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']

OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors — reuse the same policy palette as fig71-fig74
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "Myopic":  "#8c8c8c",
    "Myopic+": "#4C72B0",
    "CFA":     "#DD8452",
    "DB":      "#55A868",
}
ZONE_COLOR = "#c9c2ab"
VFA_COLOR  = "#3B3B58"

# Each entry: (display_name, score_formula, drop_formula, note)
BOXES = [
    ("Myopic",
     r"$\mathrm{score} = \dfrac{1}{d_{\mathrm{cur},i}}$",
     r"$\mathrm{drop} = \dfrac{1}{\Delta t_i}$",
     "greedy, no\neconomic weighting"),
    ("Myopic+",
     r"$\mathrm{score} = \dfrac{p_i}{d_{\mathrm{cur},i}}$",
     r"$\mathrm{drop} = \dfrac{p_i}{\Delta t_i}$",
     "+ power-weighted\nurgency"),
    ("CFA",
     r"$\mathrm{score} = \dfrac{\tilde{C}_i + s}{d_{\mathrm{cur},i}}$",
     r"$\mathrm{drop} = \tilde{C}_i - \dfrac{c_L}{60}\,\Delta t_i$",
     "+ learned station\n" r"value $\tilde{C}$"),
    ("Dynamic Balance",
     r"$\mathrm{score} = \dfrac{\tilde{C}_i + s}{d_{\mathrm{cur},i}^{2\delta}}$",
     r"$\mathrm{drop} = \tilde{C}_i - \dfrac{2\delta\, c_L}{60}\,\Delta t_i$",
     "+ state-dependent\n" r"balance $\delta$"),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.1))
ax.set_xlim(0.25, 11.15)
ax.set_ylim(0.35, 6.95)
ax.axis("off")

BOX_W, BOX_H = 2.35, 2.60
X0, GAP = 0.55, 0.30
Y_MID = 2.30

# Fixed y-positions for formula rows (absolute, same for all boxes)
Y_NAME  = Y_MID + BOX_H - 0.22   # policy name
Y_SCORE = Y_MID + BOX_H - 0.62   # score formula
Y_DROP  = Y_MID + BOX_H - 1.37   # drop formula
Y_NOTE  = Y_MID + 0.34           # italic note (va="bottom")

box_centers = []
for i, (name, score_str, drop_str, note) in enumerate(BOXES):
    x = X0 + i * (BOX_W + GAP)
    color = MODEL_COLORS[name if name != "Dynamic Balance" else "DB"]
    box = FancyBboxPatch(
        (x, Y_MID), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.16,
        zorder=2,
    )
    ax.add_patch(box)
    cx = x + BOX_W / 2
    box_centers.append(cx)

    ax.text(cx, Y_NAME, name, ha="center", va="top",
            fontsize=11.0, fontweight="bold", color=color, zorder=3)
    ax.text(cx, Y_SCORE, score_str, ha="center", va="top",
            fontsize=9.5, color="#2a2a2a", zorder=3)
    ax.text(cx, Y_DROP, drop_str, ha="center", va="top",
            fontsize=9.5, color="#2a2a2a", zorder=3)
    ax.text(cx, Y_NOTE, note, ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#444444", zorder=3,
            linespacing=1.45, multialignment="center")

# Arrows "extends" between the four base-policy boxes
for i in range(len(BOXES) - 1):
    x_from = X0 + i * (BOX_W + GAP) + BOX_W
    x_to = x_from + GAP
    arr = FancyArrowPatch(
        (x_from + 0.03, Y_MID + BOX_H / 2), (x_to - 0.03, Y_MID + BOX_H / 2),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
        color="#555555", zorder=4,
    )
    ax.add_patch(arr)

# ---------------------------------------------------------------------------
# Bottom layer — shared zone selection
# ---------------------------------------------------------------------------

zone_y, zone_h = 0.50, 1.10
zone_x0 = X0
zone_x1 = X0 + 3 * (BOX_W + GAP) + BOX_W
zone_box = FancyBboxPatch(
    (zone_x0, zone_y), zone_x1 - zone_x0, zone_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=1.3, edgecolor=ZONE_COLOR, facecolor=ZONE_COLOR, alpha=0.35, zorder=1,
)
ax.add_patch(zone_box)
ax.text((zone_x0 + zone_x1) / 2, zone_y + zone_h / 2 + 0.18,
        "Shared Zone Selection Layer",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#4a4530")
ax.text((zone_x0 + zone_x1) / 2, zone_y + zone_h / 2 - 0.26,
        "provides each team's daily station pool via K-Means zone clustering and zone scoring",
        ha="center", va="center", fontsize=9.0, color="#5a5540")

for cx in box_centers:
    arr = FancyArrowPatch(
        (cx, zone_y + zone_h + 0.03), (cx, Y_MID - 0.03),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
        color="#8a8468", zorder=1,
    )
    ax.add_patch(arr)

# ---------------------------------------------------------------------------
# Top layer — VFA rollout
# ---------------------------------------------------------------------------

vfa_y, vfa_h = 5.60, 1.20
vfa_box = FancyBboxPatch(
    (zone_x0, vfa_y), zone_x1 - zone_x0, vfa_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=1.6, edgecolor=VFA_COLOR, facecolor=VFA_COLOR, alpha=0.90,
    zorder=2,
)
ax.add_patch(vfa_box)
ax.text((zone_x0 + zone_x1) / 2, vfa_y + vfa_h / 2 + 0.22,
        "Value Function Approximation — Rollout Layer",
        ha="center", va="center", fontsize=11.0, fontweight="bold", color="white")
ax.text((zone_x0 + zone_x1) / 2, vfa_y + vfa_h / 2 - 0.28,
        "improves drop decisions by replacing greedy selection\n"
        "with forward-looking Monte-Carlo rollout evaluation",
        ha="center", va="center", fontsize=9.0, color="#e8e8ee", linespacing=1.5)

for cx in box_centers:
    arr = FancyArrowPatch(
        (cx, vfa_y - 0.03), (cx, Y_MID + BOX_H + 0.03),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
        color=VFA_COLOR, zorder=1,
    )
    ax.add_patch(arr)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig50_policy_hierarchy.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig50_policy_hierarchy.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig50_policy_hierarchy")
