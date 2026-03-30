# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis on **maintenance route optimization for EV charging stations in Würzburg**. The project optimizes daily routes for 2 maintenance teams servicing 397 charging stations, using OR-Tools VRP solving, K-Means geographic clustering, and a Gymnasium RL environment. Three strategy tiers are planned: Myopic (greedy, implemented), CFA (cost function approximation, skeleton), and VFA (value function approximation, skeleton).

**Key numbers**: 397 stations, 2 teams, 8:00–17:00 workday, 60 zones, 16 stations/team/day max.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GOOGLE_MAPS_API_KEY
```

OSRM (local Docker, required for travel matrices):
```bash
bash scripts/setup_osrm.sh   # one-time: downloads OSM data
bash scripts/start_osrm.sh   # start container on localhost:5000
```

## Common Commands

```bash
# Run myopic simulation (100 days)
python scripts/run_myopic.py
python scripts/run_myopic.py --max-days 10 --log-day 1 --verbose

# Build travel/traffic matrices (requires OSRM / Google Maps API)
python scripts/build_travel_matrix.py
python scripts/build_traffic_matrix.py

# Generate synthetic failure data
python scripts/malfunction_poisson.py

# Visualizations
python scripts/visualize_day1.py
python scripts/visualize_zone_scores.py
```

No formal test suite exists yet (tests/ is empty).

## Architecture

### Data Flow
1. `src/data/loader.py` — loads CSV, parses German decimal commas, filters operational stations, prepends depot (index 0) to coordinate list
2. `src/api/osrm.py` or `src/api/google_maps.py` — builds n×n travel matrices; OSRM is preferred (no batching needed, local Docker); Google Maps batches in 10×10 chunks; both cache as `.npy` files
3. Hourly traffic matrices (`traffic_matrix_8uhr.npy` … `traffic_matrix_17uhr.npy`) enable time-dependent routing

### Planning Pipeline
- `src/planning/clustering.py` — K-Means into 60 zones; each zone has centroid, convex hull area, mean depot distance
- `src/planning/selector.py` — ranks zones by `α·depot_dist + β·area` (both 0.5), assigns top zones to teams with ≥1 km separation, expands to 16 stations max, reserves 60 min for depot travel
- `src/planning/vrp_solver.py` — Google OR-Tools with time windows (0–540 min), makespan balancing, supports intra-day replanning; selects correct hourly traffic matrix per departure time

### Models
- `src/models/myopic.py` — greedy cheapest-insertion; handles hourly disruptions; computes operational cost (40 €/h + 0.30 €/km) and downtime cost (0.50 €/kWh × power_kW)
- `src/models/cfa.py`, `src/models/vfa.py` — skeleton only, not yet implemented
- `src/models/cost_params.py` — shared economic constants

### RL Environment (`src/environment/maintenance_env.py`)
Custom `gymnasium.Env`. State: team positions (indices), team times (normalized), binary visited/needs-maintenance vectors. Action: `MultiDiscrete([n, n])` — both teams choose next station simultaneously. Rewards configured in `configs/config.yaml` under `environment`.

### Configuration
All parameters live in `configs/config.yaml`. Key sections: `depot`, `planning` (zone count, weights, caps), `environment` (reward shaping), `maintenance` (teams, workday, service times, OR-Tools time limits), `google_maps`, `osrm`.

### Failure Simulation
`data/malfunction.csv` — 100 simulated days; Poisson arrivals (λ₁=3/9, λ₂=1/9 per hour for Typ 1/2); Typ 1 = 60 min on-site, Typ 2 = 30 min dismount + depot round-trip + 30 min remount.

## Important Notes

- Station coordinates use German locale (comma as decimal separator) — `loader.py` handles this
- Depot is always **index 0** in all matrices and coordinate lists
- Distance matrices and processed data are git-ignored; regenerate with the build scripts
- `stable-baselines3` (PPO, DQN, A2C) is available for RL training against the Gymnasium env
