"""
Datenlader für den Ladesäulen-Datensatz Würzburg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stations(config: dict | None = None) -> pd.DataFrame:
    """
    Lädt und bereinigt den Ladesäulen-Datensatz.

    Returns
    -------
    pd.DataFrame mit bereinigten Stationen.
    Jede Zeile = eine Ladeeinrichtung (Säule, ggf. mehrere Ladepunkte).
    """
    if config is None:
        config = load_config()

    raw_path = Path(config["data"]["raw_path"])
    sep = config["data"].get("separator", ";")
    lat_col = config["data"]["lat_col"]
    lon_col = config["data"]["lon_col"]
    id_col = config["data"]["id_col"]

    df = pd.read_csv(raw_path, sep=sep, encoding="utf-8", low_memory=False)

    # Dezimalkomma -> Dezimalpunkt (deutsches CSV-Format)
    for col in [lat_col, lon_col]:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".").astype(float)
        else:
            df[col] = df[col].astype(float)

    # Nur Stationen mit gültigen Koordinaten behalten
    df = df.dropna(subset=[lat_col, lon_col])

    # Nur aktive Stationen (Status = "In Betrieb")
    if "Status" in df.columns:
        df = df[df["Status"] == "In Betrieb"].copy()

    # Inbetriebnahmedatum parsen
    if "Inbetriebnahmedatum" in df.columns:
        df["Inbetriebnahmedatum"] = pd.to_datetime(
            df["Inbetriebnahmedatum"], format="%d-%m-%y", errors="coerce"
        )

    # Nennleistung numerisch
    if "Nennleistung Ladeeinrichtung [kW]" in df.columns:
        df["Nennleistung Ladeeinrichtung [kW]"] = pd.to_numeric(
            df["Nennleistung Ladeeinrichtung [kW]"], errors="coerce"
        )

    # Anzahl Ladepunkte numerisch
    if "Anzahl Ladepunkte" in df.columns:
        df["Anzahl Ladepunkte"] = pd.to_numeric(
            df["Anzahl Ladepunkte"], errors="coerce"
        ).astype("Int64")

    df = df.reset_index(drop=True)
    return df


def get_coordinates(df: pd.DataFrame, config: dict | None = None) -> list[tuple[float, float]]:
    """
    Gibt eine Liste von (lat, lon)-Tupeln zurück.
    Index 0 = Depot (aus config), Index 1..N = Ladesäulen.
    """
    if config is None:
        config = load_config()

    depot = config["depot"]
    coords: list[tuple[float, float]] = [(depot["lat"], depot["lon"])]

    lat_col = config["data"]["lat_col"]
    lon_col = config["data"]["lon_col"]
    coords += list(zip(df[lat_col], df[lon_col]))
    return coords


def load_traffic_matrices(config: dict | None = None) -> dict[int, np.ndarray]:
    """
    Lädt alle stündlichen Reisezeitmatrizen (traffic_matrix_Xuhr.npy).

    Returns
    -------
    dict[int, np.ndarray]
        Schlüssel = Stunde (8–17), Wert = Matrix in Sekunden.
        Index 0 in jeder Matrix = Depot, Index 1..N = Ladesäulen.
    """
    if config is None:
        config = load_config()

    matrix_dir = Path(config["data"]["distance_matrix_path"]).parent
    matrices: dict[int, np.ndarray] = {}

    for hour in range(8, 18):
        path = matrix_dir / f"traffic_matrix_{hour}uhr.npy"
        if path.exists():
            matrices[hour] = np.load(path)

    if not matrices:
        raise FileNotFoundError(
            f"Keine traffic_matrix_Xuhr.npy Dateien gefunden in {matrix_dir}"
        )

    return matrices


def get_failure_rate_factors(df: pd.DataFrame, beta: float = 0.03) -> dict[int, float]:
    """
    Berechnet stationsindividuelle Ausfallraten-Faktoren aus Ladetyp und Alter.

    Typ-Faktor: Schnelllader (DC) = 2× Normal (AC), normalisiert auf Mittelwert 1.0.
    Alters-Faktor: linear um Mittelwert zentriert → 1 + β × (Alter - Ø_Alter).
    Kombination: type_factor × age_factor, Mittelwert ≈ 1.0.

    Returns
    -------
    dict {station_index (0-basiert) → kombinierter Faktor}
    """
    n = len(df)

    # Typ-Faktor
    type_col = "Art der Ladeeinrichtung"
    if type_col in df.columns:
        n_normal = (df[type_col] == "Normalladeeinrichtung").sum()
        n_schnell = (df[type_col] == "Schnellladeeinrichtung").sum()
        k = 2.0
        alpha_normal = n / (n_normal + k * n_schnell)
        alpha_schnell = k * alpha_normal
        type_factors = np.where(
            df[type_col].values == "Schnellladeeinrichtung",
            alpha_schnell,
            alpha_normal,
        )
    else:
        type_factors = np.ones(n)

    # Alters-Faktor
    date_col = "Inbetriebnahmedatum"
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        ref = pd.Timestamp("2026-01-01")
        ages = ((ref - df[date_col]).dt.days / 365.25).clip(lower=0)
        ages = ages.fillna(ages.mean())
        mean_age = float(ages.mean())
        age_factors = (1.0 + beta * (ages - mean_age)).clip(lower=0.5, upper=2.0).values
    else:
        age_factors = np.ones(n)

    combined = type_factors * age_factors
    return {i: float(combined[i]) for i in range(n)}


def save_processed(df: pd.DataFrame, config: dict | None = None) -> None:
    if config is None:
        config = load_config()
    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} stations to {out_path}")
