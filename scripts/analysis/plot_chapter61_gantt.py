"""
Generates thesis figure for Chapter 6.1 (Simulation Design).

Figure saved to thesis/figures/:
  - fig61_example_day_gantt.pdf — Real executed daily timeline for both teams
    (CFA-Future, value-based zone selection, seed 1, day 1), showing routine
    stops, inserted disruptions, travel, and the lunch break, drawn from the
    actual simulation log.

Source data: logs/ergebnisse/cfa_future_value_based/json/run_1.json (day 1,
'executed_plan' at hour 16). This is real model output, not illustrative data.

Run from repo root:
    .venv/bin/python3 scripts/analysis/plot_chapter61_gantt.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = Path("logs/ergebnisse/cfa_future_value_based/json/run_1.json")
OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COLORS = {
    "routine": "#4C72B0",
    "Typ 1":   "#DD8452",
    "Typ 2":   "#B33F3F",
}
TRAVEL_COLOR = "#c9c2ab"
LUNCH_COLOR = "#E0B84B"
DEPOT_COLOR = "#222222"

WORKDAY_START_MIN = 8 * 60  # 08:00 reference (t = 0)


def to_min(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m) - WORKDAY_START_MIN


def main() -> None:
    data = json.load(open(SRC))
    day1_hourly = [h for h in data["hourly"] if h["day"] == 1]
    h16 = next(h for h in day1_hourly if h["hour"] == 16)
    executed = h16["executed_plan"]

    day1_summary = next(d for d in data["days"] if d["day"] == 1)
    n_planned = day1_summary["n_routine_tasks"]
    n_completed = day1_summary["n_routine_completed"]
    n_carryover_routine = n_planned - n_completed
    n_disruptions = day1_summary["disruptions_handled"] + day1_summary["disruptions_carryover"]

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    team_rows = {0: 1, 1: 0}  # team 0 drawn on top row
    row_h = 0.62

    for team in executed:
        tid = team["team_id"]
        row = team_rows[tid]
        route = team["route"]

        cursor = 0.0  # minutes since 08:00
        for stop in route:
            arr = to_min(stop["arrival_time"])
            dep = to_min(stop["departure_time"])
            if arr > cursor:
                ax.broken_barh([(cursor, arr - cursor)], (row - row_h / 2, row_h),
                                facecolors=TRAVEL_COLOR, edgecolors="none", zorder=2)

            color = TASK_COLORS.get(stop["task_type"], "#999999")
            service_min = stop["service_min"]
            gap = dep - arr
            # The simulator folds the 60-min lunch break into whichever stop's
            # dwell time spans 12:00-13:00, without emitting a separate lunch
            # record. Detect and draw it as its own segment for readability.
            has_lunch = (gap - service_min) >= 50
            service_end = (arr + service_min) if has_lunch else dep

            ax.broken_barh([(arr, service_end - arr)], (row - row_h / 2, row_h),
                            facecolors=color, edgecolors="white", linewidth=0.6, zorder=3)
            if has_lunch:
                ax.broken_barh([(service_end, dep - service_end)], (row - row_h / 2, row_h),
                                facecolors=LUNCH_COLOR, edgecolors="white", linewidth=0.6, zorder=3)
                if (dep - service_end) >= 35:
                    ax.text((service_end + dep) / 2, row, "Lunch", ha="center", va="center",
                            fontsize=6.3, color="#3a3a3a", zorder=4)

            label = f"St. {stop['node_idx']}"
            if (service_end - arr) >= 35:
                ax.text((arr + service_end) / 2, row, label, ha="center", va="center",
                        fontsize=6.6, color="white", zorder=4)
            cursor = dep

        depot_return = to_min(team["depot_return_time"])
        if depot_return > cursor:
            ax.broken_barh([(cursor, depot_return - cursor)], (row - row_h / 2, row_h),
                            facecolors=TRAVEL_COLOR, edgecolors="none", zorder=2)
        ax.plot(depot_return, row, marker="s", color=DEPOT_COLOR, markersize=5, zorder=5)

    for tid, row in team_rows.items():
        ax.plot(0, row, marker="s", color=DEPOT_COLOR, markersize=5, zorder=5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Team 2", "Team 1"], fontsize=10)
    ax.set_ylim(-0.55, 1.55)

    x_max = 9 * 60 + 15
    xticks = list(range(0, x_max, 60))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{8 + t // 60:02d}:00" for t in xticks], fontsize=9)
    ax.set_xlim(-8, x_max)
    ax.set_xlabel("Time of day", fontsize=10)

    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    legend_elems = [
        Patch(facecolor=TASK_COLORS["routine"], label="Routine maintenance"),
        Patch(facecolor=TASK_COLORS["Typ 1"], label="Disruption (Typ 1)"),
        Patch(facecolor=TASK_COLORS["Typ 2"], label="Disruption (Typ 2)"),
        Patch(facecolor=LUNCH_COLOR, label="Lunch break"),
        Patch(facecolor=TRAVEL_COLOR, label="Travel"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=DEPOT_COLOR,
                   markersize=6, label="Depot"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=6, fontsize=8.0, frameon=False)

    ax.set_title(
        f"Example Simulated Day (CFA-Future, value-based, seed 1, day 1) — "
        f"{n_disruptions} disruptions inserted, {n_carryover_routine} routine stops carried over",
        fontsize=9.8,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig61_example_day_gantt.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig61_example_day_gantt.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig61_example_day_gantt")


if __name__ == "__main__":
    main()
