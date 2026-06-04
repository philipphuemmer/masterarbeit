"""Gemeinsame analyse()-Funktion für alle Monte-Carlo-Scripts."""
from __future__ import annotations

import io

import numpy as np

from src.models.simulator import SimulationResult


def analyse(
    results: list[SimulationResult],
    seeds: list[int],
    cfg: dict | None = None,
    cost_params: dict | None = None,
    initial_overrides: list[int] | None = None,
    replan_overrides: list[int] | None = None,
    model_cfg_lines: list[str] | None = None,
) -> str:
    """Gibt aggregierte Statistiken über alle Läufe zurück.

    Parameters
    ----------
    model_cfg_lines:
        Optionale modellspezifische Zeilen die nach dem gemeinsamen
        Störungssimulations-Block in den SIMULATIONSPARAMETER-Abschnitt
        eingefügt werden.
    """
    total_costs   = np.array([r.total_cost_eur for r in results])
    op_costs      = np.array([sum(d.operational_cost_eur for d in r.day_results) for r in results])
    wage_costs    = np.array([sum(d.wage_cost_eur for d in r.day_results) for r in results])
    fuel_costs    = np.array([sum(d.fuel_cost_eur for d in r.day_results) for r in results])
    dt_costs      = np.array([sum(d.downtime_cost_eur for d in r.day_results) for r in results])
    days_done     = np.array([r.days_to_complete if r.days_to_complete is not None else np.nan
                              for r in results])
    same_day_rate = np.array([r.same_day_rate for r in results])
    total_disrupt = np.array([r.total_disruptions for r in results])
    carryovers    = np.array([r.total_carryover for r in results])

    buf = io.StringIO()

    def out(line: str = "") -> None:
        buf.write(line + "\n")

    sep = "=" * 70
    out(f"\n{sep}")
    out(f"  MONTE-CARLO-ANALYSE  –  {len(results)} Läufe (Seeds {seeds[0]}–{seeds[-1]})")
    out(sep)

    def row(label: str, arr: np.ndarray, unit: str = "") -> None:
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            out(f"  {label:<32}  (keine Daten)")
            return
        out(
            f"  {label:<32}  "
            f"MW {np.mean(finite):>10,.2f}  "
            f"SD {np.std(finite):>9,.2f}  "
            f"Min {np.min(finite):>10,.2f}  "
            f"Max {np.max(finite):>10,.2f}"
            + (f"  {unit}" if unit else "")
        )

    row("Gesamtkosten (€)",           total_costs,   "€")
    row("  Betriebskosten (€)",       op_costs,      "€")
    row("    Lohnkosten (€)",         wage_costs,    "€")
    row("    Fahrtkosten (€)",        fuel_costs,    "€")
    row("  Ausfallkosten (€)",        dt_costs,      "€")
    row("Simulationstage",            days_done)
    row("Same-Day-Rate",              same_day_rate * 100, "%")
    row("Gesamtstörungen",            total_disrupt)
    row("Gesamtcarryover",            carryovers)
    if replan_overrides is not None:
        row("  Replan-Rollout-Overrides", np.array(replan_overrides, dtype=float))
    if initial_overrides is not None:
        row("  Initialplan-Rollout-Overrides", np.array(initial_overrides, dtype=float))

    out(sep)

    rh_col = replan_overrides is not None
    out(f"\n  {'Seed':>5}  {'Tage':>5}  {'Gesamt (€)':>12}  "
        f"{'Lohn (€)':>10}  {'Fahrt (€)':>10}  {'Ausfall (€)':>11}  "
        f"{'Same-Day %':>10}  {'Störungen':>9}  {'Carryover':>9}"
        + (f"  {'Ov-R':>5}  {'Ov-I':>5}" if rh_col else "")
        + f"  {'Ø-Gesamt (€)':>12}")
    out(f"  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*9}"
        + (f"  {'-'*5}  {'-'*5}" if rh_col else "")
        + f"  {'-'*12}")

    cumulative_sum = 0.0
    for i, r in enumerate(results):
        wage = sum(d.wage_cost_eur for d in r.day_results)
        fuel = sum(d.fuel_cost_eur for d in r.day_results)
        dt   = sum(d.downtime_cost_eur for d in r.day_results)
        d_   = r.days_to_complete if r.days_to_complete is not None else "-"
        cumulative_sum += r.total_cost_eur
        cum_avg = cumulative_sum / (i + 1)
        rh_str = (
            f"  {replan_overrides[i]:>5}  {initial_overrides[i]:>5}"
            if rh_col else ""
        )
        out(
            f"  {seeds[i]:>5}  {str(d_):>5}  {r.total_cost_eur:>12,.2f}  "
            f"{wage:>10,.2f}  {fuel:>10,.2f}  {dt:>11,.2f}  "
            f"{r.same_day_rate * 100:>9.1f}%  "
            f"{r.total_disruptions:>9}  {r.total_carryover:>9}"
            + rh_str
            + f"  {cum_avg:>12,.2f}"
        )
    out()

    if cfg:
        pl = cfg.get("planning", {})
        mt = cfg.get("maintenance", {})
        fs = cfg.get("failure_simulation", {})
        pw = pl.get("priority_weights", {})

        out(f"\n{sep}")
        out("  SIMULATIONSPARAMETER")
        out(sep)

        out("\n  Zonenauswahl")
        out(f"    Anzahl Zonen             : {pl.get('n_zones', '–')}")
        out(f"    Top-Kandidaten           : {pl.get('n_top_candidates', '–')}")
        out(f"    Min. Teamabstand         : {pl.get('min_team_separation_km', '–')} km")
        out(f"    Max. Stationen/Team      : {pl.get('max_stations_per_team', '–')}")
        out(f"    Zeitpuffer Depot         : {pl.get('travel_reserve_min', '–')} min")
        out(f"    V̂-basierte Zonenauswahl  : {pl.get('zone_selection_mode', 'classic')}")
        out(f"    Team-Zuweisung           : {'Ja' if pl.get('use_team_assignment', True) else 'Nein'}")
        out(f"    Gewicht Depot-Entfernung : {pw.get('depot_distance', '–')}")
        out(f"    Gewicht Fläche           : {pw.get('convex_hull_area', '–')}")
        out(f"    Gewicht Zonenwert (V̂)    : {pw.get('zone_value', '–')}")

        out("\n  Solver (OR-Tools)")
        slm = mt.get("solver_limit_mode", "time")
        out(f"    Abbruchkriterium         : {slm}")
        if slm == "solution":
            out(f"    Lösungslimit initial     : {mt.get('solver_solution_limit_initial', '–')}")
            out(f"    Lösungslimit Replan      : {mt.get('solver_solution_limit_replan', '–')}")
        else:
            out(f"    Zeitlimit initial        : {mt.get('solver_time_limit_initial', '–')} s")
            out(f"    Zeitlimit Replan         : {mt.get('solver_time_limit_replan', '–')} s")
        out(f"    Makespan-Koeffizient     : {mt.get('global_span_cost_coefficient', 0)}")
        out(f"    Mittagspause             : {mt.get('lunch_duration_min', 0)} min")

        out("\n  Wartung")
        out(f"    Teams                    : {mt.get('n_teams', '–')}")
        sh = mt.get("workday_start_hour", 8)
        eh = mt.get("workday_end_hour", 16)
        out(f"    Arbeitstag               : {sh:02d}:00–{eh:02d}:00")
        out(f"    Mittlere Servicezeit     : {mt.get('mean_service_time', '–')} min")

        if cost_params:
            out("\n  Kosten")
            out(f"    Lohn                     : {cost_params.get('wage_eur_per_hour', '–'):.2f} €/h")
            out(f"    Fahrtkosten              : {cost_params.get('fuel_eur_per_km', '–'):.2f} €/km")
            out(f"    Ausfallkosten            : {cost_params.get('downtime_eur_per_kwh', '–'):.2f} €/kWh")

        out("\n  Störungssimulation")
        out(f"    Modus                    : {fs.get('mode', '–')}")
        if fs.get("mode") == "stochastic":
            out(f"    p(Typ-1)/h               : {fs.get('p1_per_hour', 0):.5f}")
            out(f"    p(Typ-2)/h               : {fs.get('p2_per_hour', 0):.5f}")
            out(f"    Erholungsdauer           : {fs.get('recovery_days', '–')} Tage")
            out(f"    Initialfaktor            : {fs.get('initial_factor', 0):.2f}")

        if model_cfg_lines:
            for line in model_cfg_lines:
                out(line)

        out("")

    return buf.getvalue()
