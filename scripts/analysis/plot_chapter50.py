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

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

BOXES = [
    ("Myopic",
     r"score $= 1\,/\,d_{\mathrm{cur},i}$" "\n" r"drop $= 1\,/\,\Delta t_i$",
     "greedy, no economic weighting"),
    ("Myopic+",
     r"score $= p_i\,/\,d_{\mathrm{cur},i}$" "\n" r"drop $= p_i\,/\,\Delta t_i$",
     "+ power-weighted urgency"),
    ("CFA",
     r"score $= (\tilde{C}_i+s)/d_{\mathrm{cur},i}$" "\n" r"drop $= \tilde{C}_i - \frac{c_L}{60}\Delta t_i$",
     "+ learned station value " r"$\tilde{C}$"),
    ("Dynamic Balance",
     r"score $= (\tilde{C}_i+s)/d_{\mathrm{cur},i}^{2\delta}$" "\n" r"drop $= \tilde{C}_i - 2\delta\frac{c_L}{60}\Delta t_i$",
     "+ state-dependent balance " r"$\delta$"),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.4)
ax.axis("off")

BOX_W, BOX_H = 2.35, 2.05
X0, GAP = 0.55, 0.30
Y_MID = 2.55

box_centers = []
for i, (name, formula, note) in enumerate(BOXES):
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
    ax.text(cx, Y_MID + BOX_H - 0.30, name, ha="center", va="top",
             fontsize=11.5, fontweight="bold", color=color, zorder=3)
    ax.text(cx, Y_MID + BOX_H - 0.72, formula, ha="center", va="top",
             fontsize=8.3, color="#2a2a2a", zorder=3, linespacing=1.6)
    ax.text(cx, Y_MID + 0.22, note, ha="center", va="bottom",
             fontsize=8.3, style="italic", color="#444444", zorder=3, wrap=True)

# Arrows "extends" between the four base-policy boxes
for i in range(len(BOXES) - 1):
    x_from = X0 + i * (BOX_W + GAP) + BOX_W
    x_to = x_from + GAP
    arr = FancyArrowPatch(
        (x_from + 0.03, Y_MID + BOX_H / 2), (x_to - 0.03, Y_MID + BOX_H / 2),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
        color="#555555", zorder=1,
    )
    ax.add_patch(arr)

# ---------------------------------------------------------------------------
# Bottom layer — shared zone selection
# ---------------------------------------------------------------------------

zone_y, zone_h = 0.35, 1.35
zone_box = FancyBboxPatch(
    (X0, zone_y), box_centers[-1] - box_centers[0] + BOX_W - 0.0 + (X0 - X0), zone_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=1.3, edgecolor=ZONE_COLOR, facecolor=ZONE_COLOR, alpha=0.35, zorder=1,
)
# recompute width precisely: from left edge of box0 to right edge of box3
zone_x0 = X0
zone_x1 = X0 + 3 * (BOX_W + GAP) + BOX_W
zone_box.set_bounds(zone_x0, zone_y, zone_x1 - zone_x0, zone_h)
ax.add_patch(zone_box)
ax.text((zone_x0 + zone_x1) / 2, zone_y + zone_h / 2 + 0.16,
         "Shared Zone Selection Layer",
         ha="center", va="center", fontsize=10.5, fontweight="bold", color="#4a4530")
ax.text((zone_x0 + zone_x1) / 2, zone_y + zone_h / 2 - 0.32,
         "K-Means zones, scored centrality-based or value-based (policy-specific station value)",
         ha="center", va="center", fontsize=8.2, color="#5a5540")

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

vfa_y, vfa_h = 5.65, 1.35
vfa_box = FancyBboxPatch(
    (zone_x0, vfa_y), zone_x1 - zone_x0, vfa_h,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=1.6, edgecolor=VFA_COLOR, facecolor=VFA_COLOR, alpha=0.90,
    linestyle="--", zorder=2,
)
ax.add_patch(vfa_box)
ax.text((zone_x0 + zone_x1) / 2, vfa_y + vfa_h / 2 + 0.20,
         "Value Function Approximation — Rollout Layer",
         ha="center", va="center", fontsize=10.8, fontweight="bold", color="white")
ax.text((zone_x0 + zone_x1) / 2, vfa_y + vfa_h / 2 - 0.28,
         "wraps any base policy above; overrides its greedy drop decision via short-horizon\n"
         "Monte-Carlo rollout when a candidate wins ≥70% of scenarios",
         ha="center", va="center", fontsize=8.0, color="#e8e8ee", linespacing=1.4)

for cx in box_centers:
    arr = FancyArrowPatch(
        (cx, vfa_y - 0.03), (cx, Y_MID + BOX_H + 0.03),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
        color=VFA_COLOR, linestyle=":", zorder=1,
    )
    ax.add_patch(arr)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig50_policy_hierarchy.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig50_policy_hierarchy.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig50_policy_hierarchy")
