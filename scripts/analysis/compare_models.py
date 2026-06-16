"""
Vergleicht mehrere Monte-Carlo-Ergebnisse (Modelle / Zone-Selection-Varianten / Rollout)
anhand der gespeicherten run_<seed>.json-Dateien.

Liest pro Lauf die `summary`- und `rolling_horizon_meta`-Felder, aggregiert pro Variante
(MW, SD, Min, Max) und erzeugt:
  - eine kombinierte CSV mit allen Einzelläufen
  - eine Zusammenfassungstabelle (CSV + Konsole)
  - Boxplots (Gesamtkosten, Same-Day-Rate)
  - Stacked-Bar der Kostenkomponenten
  - Scatter Same-Day-Rate vs. Gesamtkosten
  - Bar der Rollout-Overrides (falls vorhanden)
  - gepaarte Signifikanztests (t-Test + Wilcoxon) für Varianten mit gemeinsamen Seeds

Beispiel:
    .venv/bin/python3 scripts/analysis/compare_models.py \\
        --dirs "Myopic=logs/myopic/json" \\
               "Myopic+ (centrality)=logs/myopic_plus_centrality/json" \\
               "CFA-Future (value-based)=logs/cfa_future_value_based_500/json" \\
        --out results/analysis/zone_selection_comparison
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def _load_runs(label: str, json_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(json_dir.glob("run_*.json")):
        seed = int(f.stem.split("_")[1])
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        s = data["summary"]
        rh_meta = data.get("rolling_horizon_meta", {})
        rows.append({
            "label": label,
            "seed": seed,
            "total_cost_eur": s["total_cost_eur"],
            "wage_cost_eur": s["wage_cost_eur"],
            "fuel_cost_eur": s["fuel_cost_eur"],
            "downtime_cost_eur": s["downtime_cost_eur"],
            "same_day_rate": s["same_day_rate"] * 100,
            "total_disruptions": s["total_disruptions"],
            "total_carryover": s["total_carryover"],
            "days_simulated": s["days_simulated"],
            "initial_overrides": rh_meta.get("initial_overrides", 0),
            "replan_overrides": rh_meta.get("replan_overrides", rh_meta.get("rh_overrides", 0)),
        })
    if not rows:
        raise ValueError(f"Keine run_*.json in {json_dir} gefunden.")
    return pd.DataFrame(rows)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("label").agg(
        n=("seed", "count"),
        gesamtkosten_mw=("total_cost_eur", "mean"),
        gesamtkosten_sd=("total_cost_eur", "std"),
        lohnkosten_mw=("wage_cost_eur", "mean"),
        fahrtkosten_mw=("fuel_cost_eur", "mean"),
        ausfallkosten_mw=("downtime_cost_eur", "mean"),
        same_day_rate_mw=("same_day_rate", "mean"),
        carryover_mw=("total_carryover", "mean"),
        replan_overrides_mw=("replan_overrides", "mean"),
    ).round(2)
    return agg


def _paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(df["label"].unique(), 2):
        da = df[df["label"] == a].set_index("seed")["total_cost_eur"]
        db = df[df["label"] == b].set_index("seed")["total_cost_eur"]
        common = da.index.intersection(db.index)
        if len(common) < 2:
            continue
        xa, xb = da.loc[common], db.loc[common]
        diff = xa - xb
        t_stat, t_p = stats.ttest_rel(xa, xb)
        try:
            w_stat, w_p = stats.wilcoxon(xa, xb)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        d = diff.mean() / diff.std() if diff.std() > 0 else float("nan")
        rows.append({
            "A": a, "B": b, "n_common_seeds": len(common),
            "mean_diff_A_minus_B": round(diff.mean(), 2),
            "cohens_d": round(d, 3),
            "t_test_p": round(t_p, 4),
            "wilcoxon_p": round(w_p, 4),
        })
    return pd.DataFrame(rows)


def _plots(df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    order = list(df["label"].unique())

    # Boxplot Gesamtkosten
    plt.figure(figsize=(max(6, 1.5 * len(order)), 5))
    sns.boxplot(data=df, x="label", y="total_cost_eur", order=order)
    plt.ylabel("Gesamtkosten (€)")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_gesamtkosten.png", dpi=150)
    plt.close()

    # Boxplot Same-Day-Rate
    plt.figure(figsize=(max(6, 1.5 * len(order)), 5))
    sns.boxplot(data=df, x="label", y="same_day_rate", order=order)
    plt.ylabel("Same-Day-Rate (%)")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_same_day_rate.png", dpi=150)
    plt.close()

    # Stacked Bar Kostenkomponenten
    means = df.groupby("label")[["wage_cost_eur", "fuel_cost_eur", "downtime_cost_eur"]].mean().loc[order]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(order)), 5))
    means.plot(kind="bar", stacked=True, ax=ax,
               color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Kosten (€)")
    ax.set_xlabel("")
    ax.legend(["Lohnkosten", "Fahrtkosten", "Ausfallkosten"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_bar_kostenkomponenten.png", dpi=150)
    plt.close()

    # Scatter Same-Day-Rate vs. Gesamtkosten
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="same_day_rate", y="total_cost_eur", hue="label", alpha=0.6)
    plt.xlabel("Same-Day-Rate (%)")
    plt.ylabel("Gesamtkosten (€)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_same_day_vs_kosten.png", dpi=150)
    plt.close()

    # Rollout-Overrides (nur falls vorhanden)
    if df["replan_overrides"].sum() > 0:
        plt.figure(figsize=(max(6, 1.5 * len(order)), 5))
        sns.barplot(data=df, x="label", y="replan_overrides", order=order, errorbar="sd")
        plt.ylabel("Replan-Rollout-Overrides (Ø pro Lauf)")
        plt.xlabel("")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_rollout_overrides.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergleicht Monte-Carlo-Ergebnisse mehrerer Modelle/Varianten")
    parser.add_argument("--dirs", nargs="+", required=True,
                         help='Liste "Label=Pfad/zu/json" (z.B. "Myopic=logs/myopic/json")')
    parser.add_argument("--out", type=str, default="results/analysis/comparison",
                         help="Ausgabeordner für CSVs und Plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for spec in args.dirs:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f'Ungültiges --dirs Argument (erwartet "Label=Pfad"): {spec}')
        frames.append(_load_runs(label, Path(path)))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "runs_combined.csv", index=False)

    summary = _summary_table(df)
    summary.to_csv(out_dir / "summary.csv")
    print("\n=== Zusammenfassung ===")
    print(summary.to_string())

    paired = _paired_tests(df)
    if not paired.empty:
        paired.to_csv(out_dir / "paired_tests.csv", index=False)
        print("\n=== Gepaarte Tests (Gesamtkosten, gleiche Seeds) ===")
        print(paired.to_string(index=False))

    _plots(df, out_dir)
    print(f"\nPlots und Tabellen gespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
