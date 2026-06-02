"""
Rekonstruiert run_N.json aus run_N.log + Overview-Log.
Erzeugt minimale JSONs, die mit _load_result_from_json und --resume kompatibel sind.

Verwendung:
    python scripts/reconstruct_json_from_logs.py --log-dir logs/cfa_future_win_rate_h_21_n_25_stochastic_travel_0_7
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

N_STATIONS = 397


def parse_overview_overrides(overview_path: Path) -> dict[int, int]:
    """Liest RH-Override-Zahlen pro Seed aus der Overview-Log."""
    overrides: dict[int, int] = {}
    if not overview_path.exists():
        return overrides
    in_table = False
    for line in overview_path.read_text(encoding="utf-8").splitlines():
        # Tabellenzeilen: "      1     36     33,287.95  ..."
        if re.match(r"\s+-----\s+-----", line):
            in_table = True
            continue
        if not in_table:
            continue
        # Leerzeile = Ende
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            seed = int(parts[0])
        except ValueError:
            continue
        # RH-Overr. ist das letzte Feld (Index -1 bei altem Format mit 1 Override-Spalte)
        try:
            override_val = int(parts[-1])
        except ValueError:
            override_val = 0
        overrides[seed] = override_val
    return overrides


def parse_run_log(log_path: Path) -> dict:
    """Parst eine run_N.log und gibt ein dict zurück, das dem JSON-Format entspricht."""
    text = log_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ---- Kopfzeilen ----
    days_simulated = None
    days_to_complete = None
    total_disruptions = None
    same_day_handled = None
    total_carryover = None
    total_cost = None
    op_cost = None
    wage_cost_total = None
    fuel_cost_total = None
    downtime_cost_total = None

    for line in lines:
        m = re.search(r"Tage simuliert\s*:\s*(\d+)", line)
        if m:
            days_simulated = int(m.group(1))

        m = re.search(r"Alle Stationen gewartet\s*:\s*Tag\s*(\d+)", line)
        if m:
            days_to_complete = int(m.group(1))

        m = re.search(r"Störungen gesamt\s*:\s*([\d,]+)", line)
        if m:
            total_disruptions = int(m.group(1).replace(",", ""))

        m = re.search(r"Gleichen Tag erledigt\s*:\s*([\d,]+)", line)
        if m:
            same_day_handled = int(m.group(1).replace(",", ""))

        m = re.search(r"Carryover\s*:\s*([\d,]+)", line)
        if m and total_carryover is None:
            total_carryover = int(m.group(1).replace(",", ""))

        m = re.search(r"Gesamtkosten\s*:\s*([\d,. ]+)EUR", line)
        if m:
            total_cost = float(m.group(1).replace(",", "").replace(" ", ""))

        m = re.search(r"Betriebskosten\s*:\s*([\d,. ]+)EUR", line)
        if m:
            op_cost = float(m.group(1).replace(",", "").replace(" ", ""))

        m = re.search(r"Lohnkosten\s*:\s*([\d,. ]+)EUR", line)
        if m:
            wage_cost_total = float(m.group(1).replace(",", "").replace(" ", ""))

        m = re.search(r"Fahrtkosten\s*:\s*([\d,. ]+)EUR", line)
        if m:
            fuel_cost_total = float(m.group(1).replace(",", "").replace(" ", ""))

        m = re.search(r"Ausfallkosten\s*:\s*([\d,. ]+)EUR", line)
        if m:
            downtime_cost_total = float(m.group(1).replace(",", "").replace(" ", ""))

    # ---- Tagstabelle ----
    # Format: "   1       20        12       4      2    560.00    11.36    196.41      767.77"
    # Nur Zeilen im TAGESÜBERSICHT-Block lesen (vor STUNDEN-PROTOKOLL)
    day_rows = []
    tages_block = text.split("TAGESÜBERSICHT", 1)[-1]
    tages_block = tages_block.split("STUNDEN-PROTOKOLL", 1)[0]
    for line in tages_block.splitlines():
        m = re.match(
            r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            line,
        )
        if m:
            day, geplant, erledigt, stoerungen, carry, lohn, fahrt, ausfall, _ = m.groups()
            day_rows.append({
                "day": int(day),
                "n_routine_tasks": int(geplant),
                "n_routine_completed": int(erledigt),
                "disruptions_handled": int(stoerungen),
                "disruptions_carryover": int(carry),
                "wage_cost_eur": float(lohn),
                "fuel_cost_eur": float(fahrt),
                "downtime_cost_eur": float(ausfall),
                "operational_cost_eur": float(lohn) + float(fahrt),
            })

    total_completed = sum(d["n_routine_completed"] for d in day_rows)
    remaining = 0 if days_to_complete is not None else max(0, N_STATIONS - total_completed)

    same_day_rate = (same_day_handled / total_disruptions) if total_disruptions else 0.0

    return {
        "meta": {
            "label": "CFA-FUTURE SIMULATION [Rolling Horizon]",
            "run_id": None,
            "timestamp": None,
            "model_params": {},
        },
        "summary": {
            "days_simulated": days_simulated,
            "days_to_complete": days_to_complete,
            "remaining_stations_at_end": remaining,
            "total_disruptions": total_disruptions,
            "same_day_handled": same_day_handled,
            "total_carryover": total_carryover,
            "same_day_rate": round(same_day_rate, 6),
            "total_cost_eur": total_cost,
            "operational_cost_eur": op_cost,
            "wage_cost_eur": wage_cost_total,
            "fuel_cost_eur": fuel_cost_total,
            "downtime_cost_eur": downtime_cost_total,
        },
        "days": day_rows,
        "hourly": [],
        "rolling_horizon_meta": {
            "rh_overrides": 0,  # wird aus Overview befüllt
            "initial_overrides": 0,
            "replan_overrides": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, help="z.B. logs/cfa_future_win_rate_h_21_n_25_stochastic_travel_0_7")
    parser.add_argument("--start", type=int, default=1, help="Erster Run (inklusiv)")
    parser.add_argument("--end", type=int, default=None, help="Letzter Run (inklusiv), Standard: alle")
    parser.add_argument("--dry-run", action="store_true", help="Nur parsen, nichts schreiben")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_subdir = log_dir / "log"
    json_dir = log_dir / "json"
    json_dir.mkdir(exist_ok=True)

    dir_name = log_dir.name
    overview_path = log_subdir / f"{dir_name}_overview.log"
    overrides = parse_overview_overrides(overview_path)
    if overrides:
        print(f"  {len(overrides)} Override-Einträge aus Overview geladen.")
    else:
        print("  Keine Override-Daten gefunden, setze alle auf 0.")

    log_files = sorted(log_subdir.glob("run_*.log"), key=lambda p: int(re.search(r"run_(\d+)", p.name).group(1)))

    reconstructed = 0
    skipped = 0
    errors = 0

    for log_path in log_files:
        m = re.search(r"run_(\d+)\.log$", log_path.name)
        if not m:
            continue
        seed = int(m.group(1))
        if seed < args.start:
            continue
        if args.end is not None and seed > args.end:
            continue

        out_path = json_dir / f"run_{seed}.json"
        if out_path.exists():
            print(f"  run_{seed}: bereits vorhanden, übersprungen.")
            skipped += 1
            continue

        try:
            data = parse_run_log(log_path)
            data["meta"]["run_id"] = seed

            ov = overrides.get(seed, 0)
            data["rolling_horizon_meta"]["rh_overrides"] = ov
            data["rolling_horizon_meta"]["replan_overrides"] = ov

            if not args.dry_run:
                out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  run_{seed}: {data['summary']['days_simulated']} Tage, "
                      f"{data['summary']['total_disruptions']} Störungen, "
                      f"{len(data['days'])} Tagzeilen → {out_path.name}")
            else:
                print(f"  [dry-run] run_{seed}: {data['summary']['days_simulated']} Tage, "
                      f"{len(data['days'])} Tagzeilen")
            reconstructed += 1
        except Exception as e:
            print(f"  run_{seed}: FEHLER – {e}")
            errors += 1

    print(f"\nFertig: {reconstructed} rekonstruiert, {skipped} übersprungen, {errors} Fehler.")


if __name__ == "__main__":
    main()
