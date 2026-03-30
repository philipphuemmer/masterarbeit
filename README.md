# Masterarbeit – Wartungsoptimierung E-Ladesäulen Würzburg

Optimierung von Wartungsrouten für E-Ladesäulen in Würzburg.
Basis: Datensatz aller Ladesäulen (397 Einträge) mit GPS-Koordinaten.
Fahrzeiten: Google Maps Distance Matrix API.

## Projektstruktur

```
masterarbeit/
├── configs/
│   └── config.yaml              # Zentrale Konfiguration (Pfade, Reward, API)
├── data/
│   ├── raw/                     # Rohdaten (versioniert)
│   │   └── charging_stations_wue.csv
│   ├── processed/               # Bereinigte Daten (nicht versioniert)
│   └── distance_matrices/       # Gecachte API-Matrizen (nicht versioniert)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_distance_matrix.ipynb
│   └── 03_environment_test.ipynb
├── src/
│   ├── api/
│   │   └── google_maps.py       # Distance Matrix API + Haversine-Fallback
│   ├── data/
│   │   └── loader.py            # CSV-Lader und Vorverarbeitung
│   ├── environment/
│   │   └── maintenance_env.py   # Gymnasium-Environment
│   ├── models/                  # Optimierungsmodelle (noch offen)
│   └── utils/
│       └── visualization.py     # Karten (Folium) und Plots
├── tests/
├── .env.example                 # API-Key-Template
├── .gitignore
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API-Key einrichten
cp .env.example .env
# .env editieren: GOOGLE_MAPS_API_KEY=<dein Key>
```

## Schnellstart

```python
from src.data.loader import load_config, load_stations, get_coordinates
from src.api.google_maps import build_haversine_matrix
from src.environment.maintenance_env import MaintenanceEnv

config = load_config()
df = load_stations(config)
coords = get_coordinates(df, config)           # Index 0 = Depot
dur_matrix, dist_matrix = build_haversine_matrix(coords[:31])

env = MaintenanceEnv(dur_matrix, config=config, render_mode='ansi')
obs, info = env.reset(seed=42)
```

## Offene Entscheidungen

| Frage | Optionen |
|---|---|
| Optimierungsansatz | Reinforcement Learning / OR-Tools / Metaheuristiken |
| RL-Policy | PPO, DQN, A2C (stable-baselines3) |
| Fehlermodell | Poisson-Prozess / datengetrieben |
| Distanzmatrix | Google Maps API / OSRM (Open Source) |
