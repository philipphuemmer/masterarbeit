# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis on **maintenance route optimization for EV charging stations in Würzburg**. Optimizes daily routes for 2 maintenance teams servicing 397 charging stations using OR-Tools VRP solving, K-Means geographic clustering, and a Gymnasium RL environment.

**Key numbers**: 397 stations, 2 teams, 8:00–16:00 workday, 40 zones (configurable), 20 stations/team/day max.

**Five policy tiers** (all fully implemented):
1. **Myopic** — greedy cheapest-insertion
2. **MyopicPlus** — OR-Tools with power-weighted soft-deadlines, skip-penalties
3. **CFA Light** — MyopicPlus with V̂-based drop decision + cheapest-insertion routing
4. **CFA** — OR-Tools with learned V̂(k) = θ × power × dsm for scheduling + drop ordering
5. **VFA** — OR-Tools with learned V̂(s) = θᵀφ(s) global state features for ΔV̂-based scheduling + drop ordering

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GOOGLE_MAPS_API_KEY
```

OSRM (local Docker, required for travel matrices):
```bash
bash scripts/setup/setup_osrm.sh   # one-time: downloads OSM data
bash scripts/setup/start_osrm.sh   # start container on localhost:5000
```

## Common Commands

```bash
# Single runs
python scripts/run/run_myopic.py [--max-days 10] [--log-day 1] [--verbose]
python scripts/run/run_myopic_plus.py
python scripts/run/run_cfa_light.py
python scripts/run/run_cfa.py    # requires: python scripts/train/train_cfa.py first
python scripts/run/run_vfa.py    # requires: python scripts/train/train_vfa.py first

# Training (CFA/VFA learn θ from Monte Carlo rollouts of the Myopic policy)
python scripts/train/train_cfa.py
python scripts/train/train_vfa.py

# Monte Carlo (N runs, aggregate analysis written to logs/<model>/log/<model>_overview.log)
python scripts/monte_carlo/run_mc_myopic.py --runs 30
python scripts/monte_carlo/run_mc_myopic_plus.py --runs 30
python scripts/monte_carlo/run_mc_cfa_light.py --runs 30
python scripts/monte_carlo/run_mc_cfa.py --runs 30
python scripts/monte_carlo/run_mc_vfa.py --runs 30

# Build travel/traffic matrices (requires OSRM or Google Maps API)
python scripts/setup/build_travel_matrix.py
python scripts/setup/build_traffic_matrix.py

# Generate synthetic failure data
python scripts/setup/malfunction_poisson.py

# Visualizations
python scripts/visualize/visualize_myopic_day1.py
python scripts/visualize/visualize_cfa_day1.py
python scripts/visualize/visualize_myopic_day1_clusters.py
python scripts/visualize/visualize_zone_scores.py
```

No formal test suite exists (tests/ is empty).

## Architecture

### Data Flow
1. `src/data/loader.py` — loads CSV, parses German decimal commas, filters operational stations, prepends depot (index 0) to coordinate list
2. `src/api/osrm.py` or `src/api/google_maps.py` — builds n×n travel matrices; OSRM preferred (no batching, local Docker); both cache as `.npy`
3. Hourly traffic matrices (`traffic_matrix_8uhr.npy` … `traffic_matrix_17uhr.npy`) enable time-dependent routing

### Planning Pipeline
- `src/planning/clustering.py` — K-Means into N zones; each zone has centroid, convex hull area, mean depot distance
- `src/planning/selector.py` — `DailyZoneSelector` ranks open zones by score, assigns one start zone per team (with ≥1 km separation), expands to station list via nearest-neighbor. Two scoring modes controlled by `planning.value_based_zone_selection` in config:
  - `false` (default): `w_depot × depot_dist + w_area × convex_hull_area`
  - `true` (CFA/VFA): `w_value × Σ V̂(station) + w_depot × depot_dist`; set `selector.value_fn = policy._value` (CFA) or `policy._station_value` (VFA) after construction
- `src/planning/vrp_solver.py` — OR-Tools with time windows (0–workday_minutes), makespan balancing, intra-day replanning; selects correct hourly traffic matrix per departure time; supports `extra_costs` dict (node_idx → penalty minutes) and `soft_deadline_min` / `deadline_penalty` on `MaintenanceTask`

### Models
- `src/models/myopic.py` — greedy cheapest-insertion with hourly disruption handling
- `src/models/myopic_plus.py` — OR-Tools initial plan with power-weighted soft-deadlines and AddDisjunction skip-penalties for routing
- `src/models/cfa_light.py` — MyopicPlus plan + V̂-based drop + cheapest-insertion routing (no OR-Tools replan)
- `src/models/cfa.py` — OR-Tools with `V̂(k) = θ × power_kW × days_since_maintenance`; `θ` loaded from `data/cfa/theta.json`; drops lowest-V̂ routine stops when infeasible
- `src/models/vfa.py` — OR-Tools with 6-feature global state `V̂(s) = θᵀφ(s)`; `ΔV̂(k) = V̂(s) − V̂(s\k)` as extra_costs; `θ` loaded from `data/vfa/theta.json`; `_station_value(node_idx, dsm)` is the lightweight per-station approximation used for zone scoring
- `src/models/cost_params.py` — shared economic constants (40 €/h wage, 0.30 €/km fuel, 0.50 €/kWh downtime)
- `src/models/simulator.py` — shared simulation loop (`MaintenanceSimulator`) used by all policies

### Simulation Loop (`MaintenanceSimulator`)
`sim.run(mal_df, max_days)` drives the full year:
- Tracks `remaining` stations (not yet serviced this year), `carryover_tasks` (unfinished disruptions), and `_days_since_maintenance[node_idx]` (stochastic mode only)
- Calls `selector.select_for_day(remaining, team_states, carryover, dsm_array=...)` → `DailyZoneSelector` returns task lists per team
- Passes task lists to `policy.create_initial_plan()` → `VRPSolver`
- Each hour: generates disruptions, calls `policy.handle_disruptions()`
- `dsm_array` (`_days_since_maintenance`) is passed to the selector so `value_fn` can access per-station urgency for V̂-based zone scoring

### Simulation Output (JSON)
`write_json()` writes four sections: `meta`, `summary` (scalar KPIs), `days` (per-day costs/tasks), `hourly` (per-(day,hour) disruptions and actions including full route at 08:00 and per-hour team status).

### Failure Simulation
Two modes via `failure_simulation.mode` in config:
- `csv` — load from `data/malfunction.csv` (100 days, Poisson arrivals)
- `stochastic` — probabilistic per station per hour with recovery curve: `p(t) = p_base × (initial_factor + (1−initial_factor) × t/recovery_days)`

### Configuration
All parameters in `configs/config.yaml`. Key sections:
- `planning`: `n_zones`, `max_stations_per_team`, `value_based_zone_selection`, `priority_weights` (depot_distance, convex_hull_area, zone_value)
- `maintenance`: `n_teams`, `workday_start/end_hour`, `mean_service_time`, OR-Tools time limits
- `cfa.alpha`: disruption deadline penalty scaling factor
- `failure_simulation`: mode, per-hour probabilities, recovery curve

## Important Notes

- Depot is always **index 0** in all matrices and coordinate lists; `node_idx = station_index + 1`
- Distance matrices and processed data are git-ignored; regenerate with the build scripts
- NumPy scalar types (`float32`, `int64`) are not JSON-serialisable — `write_json()` handles this via custom encoder; keep in mind when adding new logged fields
- `value_based_zone_selection` only has effect when `failure_simulation.mode: stochastic` (requires `_days_since_maintenance` tracking); in `csv` mode it silently falls back to classical scoring
- CFA/VFA `θ` is trained on the Myopic policy's rollouts — no retraining needed when changing zone scoring
