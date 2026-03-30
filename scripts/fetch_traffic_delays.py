"""
Stau-Verzögerungen an Würzburger Verkehrsknotenpunkten via Google Distance Matrix API.

Liest eine Excel-Datei mit Knotenpunkten (start_lon, start_lat, ziel_lon, ziel_lat)
und fragt für jeden Abschnitt ab:
  - Standardzeit: 3 Uhr nachts (kein Verkehr)
  - Verkehrszeit: 8–17 Uhr stündlich (mit Echtzeit-/historischem Verkehr)

Speichert das Ergebnis als CSV mit den Spalten:
  id, ort, strecke, standard_zeit, 8_uhr, 8_uhr_stau, 9_uhr, 9_uhr_stau, ..., 17_uhr, 17_uhr_stau

Verwendung:
    python scripts/fetch_traffic_delays.py
    python scripts/fetch_traffic_delays.py --excel data/knotenpunkte.xlsx --date 2026-03-09

WICHTIG: Noch nicht final ausführen — Excel wird noch befüllt.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import googlemaps
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Referenz-Datum (überschreibbar per --date)
DEFAULT_DATE = "2026-04-15"

# Stunden für die Verkehrsabfrage (ganzzahlig, lokale Zeit)
TRAFFIC_HOURS = list(range(8, 18))  # 8, 9, ..., 17

# Pause zwischen API-Calls in Sekunden (Rate Limiting)
SLEEP_BETWEEN_CALLS = 0.3

# Standard-Pfad zur Excel-Datei (Windows-Pfad über WSL-Mount)
DEFAULT_EXCEL = "/mnt/c/Users/Anwender/Desktop/Masterarbeit/Stau_Koordinaten.xlsx"

# Ausgabe-CSV
DEFAULT_OUTPUT = "data/traffic_data.csv"


# ---------------------------------------------------------------------------
# Google Maps Client
# ---------------------------------------------------------------------------

def _get_client() -> googlemaps.Client:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY nicht gesetzt. Bitte .env aus .env.example erstellen."
        )
    return googlemaps.Client(key=api_key)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _local_datetime_to_utc_timestamp(date_str: str, hour: int) -> int:
    """
    Gibt Unix-Timestamp (UTC) für das gegebene Datum + Stunde zurück.
    Annahme: Lokale Zeitzone ist Europe/Berlin (CET/CEST).
    """
    import zoneinfo
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    local_dt = datetime(
        year=int(date_str[:4]),
        month=int(date_str[5:7]),
        day=int(date_str[8:10]),
        hour=hour,
        minute=0,
        second=0,
        tzinfo=tz,
    )
    return int(local_dt.timestamp())


def _query_duration(
    client: googlemaps.Client,
    origin: tuple[float, float],
    destination: tuple[float, float],
    departure_timestamp: int,
) -> tuple[float | None, float | None]:
    """
    Fragt die Fahrzeit zwischen zwei Koordinaten ab.

    Parameters
    ----------
    origin : (lat, lon)
    destination : (lat, lon)
    departure_timestamp : Unix-Timestamp (UTC)

    Returns
    -------
    (duration, duration_in_traffic) in Sekunden — beide Werte aus einem einzigen API-Call.
    """
    result = client.distance_matrix(
        origins=[origin],
        destinations=[destination],
        mode="driving",
        departure_time=departure_timestamp,
        traffic_model="best_guess",
        units="metric",
    )

    element = result["rows"][0]["elements"][0]
    if element["status"] != "OK":
        print(f"  [WARN] Status {element['status']} für {origin} -> {destination}")
        return None, None

    duration = element["duration"]["value"]
    duration_in_traffic = element.get("duration_in_traffic", {}).get("value")
    return duration, duration_in_traffic


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def fetch_traffic_delays(
    excel_path: str | Path = DEFAULT_EXCEL,
    date_str: str = DEFAULT_DATE,
    output_path: str | Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Liest Excel, fragt für jede Zeile Baseline + stündliche Verkehrszeiten ab,
    und gibt das Ergebnis als DataFrame zurück (+ speichert CSV).

    Parameters
    ----------
    excel_path : Pfad zur Excel-Datei mit den Knotenpunkten.
    date_str   : Datum im Format 'YYYY-MM-DD' (z.B. '2026-03-09').
    output_path: Pfad für die Ausgabe-CSV.
    dry_run    : Wenn True, werden API-Calls übersprungen (nur Gerüst testen).

    Returns
    -------
    DataFrame mit Standardzeit + stündlichen Fahrzeiten.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel-Datei nicht gefunden: {excel_path}\n"
            "Bitte Pfad mit --excel angeben."
        )

    print(f"[INFO] Lese Excel: {excel_path}")
    df = pd.read_excel(excel_path)

    # Pflichtfelder prüfen
    required_cols = {"id", "ort", "strecke", "start_lat", "start_lon", "ziel_lat", "ziel_lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in der Excel: {missing}")

    print(f"[INFO] {len(df)} Knotenpunkte geladen.")
    print(f"[INFO] Referenzdatum: {date_str}")

    if not dry_run:
        client = _get_client()
    else:
        client = None
        print("[INFO] DRY RUN — keine echten API-Calls.")

    # Zeitstempel vorberechnen
    traffic_ts = {
        hour: _local_datetime_to_utc_timestamp(date_str, hour)
        for hour in TRAFFIC_HOURS
    }

    # Ergebnis-Spalten vorbereiten
    result_cols = ["standard_zeit"]
    for h in TRAFFIC_HOURS:
        result_cols += [f"{h}_uhr", f"{h}_uhr_stau"]
    for col in result_cols:
        df[col] = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    try:
        for idx, row in df.iterrows():
            origin = (row["start_lon"], row["start_lat"])        # Excel-Spaltenköpfe sind vertauscht: start_lon enthält lat-Werte
            destination = (row["ziel_lon"], row["ziel_lat"])    # analog für Ziel

            print(f"[{idx + 1}/{total}] {row.get('ort', '?')} — {row.get('strecke', '?')}")

            standard_zeit = None

            # Stündliche Verkehrszeiten 8–17 Uhr
            for hour in TRAFFIC_HOURS:
                if not dry_run:
                    duration, duration_in_traffic = _query_duration(
                        client, origin, destination, traffic_ts[hour]
                    )
                    # standard_zeit einmalig aus dem ersten Call ziehen
                    if standard_zeit is None and duration is not None:
                        standard_zeit = duration
                        df.at[idx, "standard_zeit"] = duration
                    df.at[idx, f"{hour}_uhr"] = duration_in_traffic
                    if duration_in_traffic is not None and standard_zeit is not None:
                        df.at[idx, f"{hour}_uhr_stau"] = duration_in_traffic - standard_zeit
                    time.sleep(SLEEP_BETWEEN_CALLS)
                else:
                    if standard_zeit is None:
                        standard_zeit = 120
                        df.at[idx, "standard_zeit"] = 120
                    df.at[idx, f"{hour}_uhr"] = 150
                    df.at[idx, f"{hour}_uhr_stau"] = 30

            # Nach jeder Zeile zwischenspeichern — kein Datenverlust bei Abbruch
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    except Exception as e:
        print(f"\n[FEHLER] Abbruch bei Zeile {idx + 1}: {e}")
        print(f"[INFO] Bisherige Ergebnisse gespeichert: {output_path}")
        raise

    print(f"\n[INFO] Fertig. Gespeichert: {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stau-Verzögerungen an Würzburger Knotenpunkten abfragen."
    )
    parser.add_argument(
        "--excel",
        default=DEFAULT_EXCEL,
        help=f"Pfad zur Excel-Eingabedatei (Standard: {DEFAULT_EXCEL})",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help=f"Referenzdatum YYYY-MM-DD (Standard: {DEFAULT_DATE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Ausgabe-CSV-Pfad (Standard: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Überspringe echte API-Calls (zum Testen des Gerüsts)",
    )
    args = parser.parse_args()

    result_df = fetch_traffic_delays(
        excel_path=args.excel,
        date_str=args.date,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    print("\nVorschau:")
    print(result_df[["id", "ort", "strecke", "standard_zeit", "8_uhr", "8_uhr_stau", "17_uhr", "17_uhr_stau"]].to_string())
