# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis on **maintenance route optimization for EV charging stations in Würzburg**. Optimizes daily routes for 2 maintenance teams servicing 397 charging stations using OR-Tools VRP solving, K-Means geographic clustering, and a Gymnasium RL environment.

**Key numbers**: 397 stations, 2 teams, 8:00–16:00 workday, 40 zones (configurable), 20 stations/team/day max.

**Active policy tiers** (all in `src/models/`):
1. **Myopic** — OR-Tools initial plan, greedy cheapest-insertion replan; no learned components
2. **MyopicPlus** — OR-Tools everywhere; depot-distance sorted soft-deadlines, penalty ∝ `power_kW`; AddDisjunction replan
3. **CFA** — OR-Tools everywhere; V̂/depot-distance sorted soft-deadlines, penalty ∝ V̂; manual V̂-drop loop replan; θ ∈ ℝ⁴ learned via OLS
4. **VFA** — OR-Tools with ΔV̂ as `extra_costs`; manual V̂-drop loop replan; θ learned via OLS
5. **DB** — greedy cheapest-insertion initial plan; OR-Tools replan; MLP α modulates soft-deadline penalties; trained via PPO
6. **CFA-DB** — CFA-style U(k) soft-deadlines + DB's MLP α for drop-score; trained via PPO

**Deprecated** (in `src/models/alt/`, scripts in `scripts/alt/`): CFA-Light, CFA-Real, DB-Alt.

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
python scripts/run/run_cfa.py    # requires: python scripts/train/train_cfa.py first
python scripts/run/run_vfa.py    # requires: python scripts/train/train_vfa.py first
python scripts/run/run_db.py     # requires: python scripts/train/train_db.py first
python scripts/run/run_cfa_db.py # requires: python scripts/train/train_cfa_db.py first

# Training
# CFA/VFA: learn θ from Monte Carlo rollouts via OLS
python scripts/train/train_cfa.py   # → data/training/cfa/theta.json
python scripts/train/train_vfa.py   # → data/training/vfa/theta.json
# DB/CFA-DB: learn MLP α via PPO (requires failure_simulation.mode: stochastic)
python scripts/train/train_db.py [--iterations 50 --rollouts 10 --max-days 200]
python scripts/train/train_cfa_db.py

# Monte Carlo (aggregate analysis → logs/<model>/log/<model>_overview.log)
python scripts/monte_carlo/run_mc_myopic.py --runs 30
python scripts/monte_carlo/run_mc_myopic_plus.py --runs 30
python scripts/monte_carlo/run_mc_cfa.py --runs 30
python scripts/monte_carlo/run_mc_vfa.py --runs 30
python scripts/monte_carlo/run_mc_db.py --runs 30
python scripts/monte_carlo/run_mc_cfa_db.py --runs 30

# Build travel/traffic matrices (requires OSRM or Google Maps API)
python scripts/setup/build_travel_matrix.py
python scripts/setup/build_traffic_matrix.py

# Generate synthetic failure data
python scripts/setup/malfunction_poisson.py
```

No formal test suite exists (tests/ is empty).

## Architecture

### Data Flow
1. `src/data/loader.py` — loads CSV, parses German decimal commas, filters operational stations, prepends depot (index 0) to coordinate list
2. `src/api/osrm.py` or `src/api/google_maps.py` — builds n×n travel matrices; OSRM preferred (local Docker); both cache as `.npy`
3. Hourly traffic matrices (`traffic_matrix_8uhr.npy` … `traffic_matrix_17uhr.npy`) enable time-dependent routing

### Planning Pipeline
- `src/planning/clustering.py` — K-Means into N zones; each zone has centroid, convex hull area, mean depot distance
- `src/planning/selector.py` — `DailyZoneSelector` ranks open zones by score, assigns one start zone per team (≥1 km separation), expands to station list via nearest-neighbor. Two scoring modes via `planning.value_based_zone_selection`:
  - `false` (default): `w_depot × depot_dist + w_area × convex_hull_area`
  - `true`: `w_value × Σ zone_value(station) + w_depot × depot_dist`; requires `selector.value_fn` to be set after construction (see run scripts)
- `src/planning/vrp_solver.py` — OR-Tools with time windows (0–workday_minutes), makespan balancing, intra-day replanning; selects correct hourly traffic matrix per departure time; supports `extra_costs` (node_idx → penalty minutes), `soft_deadline_min` / `deadline_penalty`, and `skip_penalty` (AddDisjunction) on `MaintenanceTask`

### Models

**Common interface:** all models implement `create_initial_plan(tasks)` + `handle_disruptions(disruptions, sim_routes, time_min, hour, log)`.

**Initialplan — carryover disruptions:** MyopicPlus, CFA, CFA-DB all set `soft_deadline_min=0` + penalty ∝ `power_kW` for carryover disruption tasks so OR-Tools schedules them early. Myopic and VFA do not (no deadline mechanism / extra_costs instead).

**Zone selection value functions** (set in run scripts when `value_based_zone_selection: true`):

| Model | `selector.value_fn` | Zone value logic |
|---|---|---|
| Myopic | `policy._zone_value` | `power × recovery_curve(dsm)` |
| MyopicPlus | `policy._zone_value` | `power × recovery_curve(dsm)` |
| CFA | `policy._value` | `θᵀ × φ_scaled(k)` (4 features) |
| VFA | `policy._station_value` | `θ[0]×power×dsm + θ[1]×failure_risk×power` |
| DB | `policy._station_value` | `power × dsm` |
| CFA-DB | `policy._station_value` | `power × dsm` |

**CFA details** (`src/models/cfa.py`):
- `φ(k) = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others]`
- `recovery_curve = initial_factor + (1 − initial_factor) × dsm / recovery_days`
- `C̃(drop k) = θᵀ × φ_scaled(k)` (features z-scored with training μ, σ)
- θ ∈ ℝ⁴ loaded from `data/training/cfa/theta.json`; always read `feature_means` and `feature_stds` from same file
- Deadline-Sortierung: `C̃/depot_dist` ratio; Replan: manual V̂-drop loop (ascending C̃)

**VFA details** (`src/models/vfa.py`):
- `φ(s) = [Σ(power×dsm), Σ(failure_risk×power), mean(dsm), max(power×dsm), n_remaining/n_stations, n_carryover]`
- `V̂(s) = θᵀφ(s) + intercept`; `ΔV̂(k) = V̂(s) − V̂(s\k)` used as OR-Tools `extra_costs`
- `_station_value(node_idx, dsm)` uses only f0, f1 (no global context needed for zone scoring)

**DB details** (`src/models/db.py`):
- `φ(S) = [n_remaining/n_stations, frac(dsm>90), mean(dsm)/365, Σ(power×dsm)/norm, max(power×dsm)/norm, mean(dist_depot)/30km, std(dist_depot)/30km, n_carryover/10]`
- `α = σ(MLP(φ(S)))` ∈ (0,1); `penalty = max(1, round((1−α) × MAX_ROUTINE_PENALTY))`
- Initial plan: greedy cheapest-insertion (no OR-Tools); Replan: OR-Tools

### Simulation Loop (`MaintenanceSimulator`)
`sim.run(mal_df, max_days)` drives the full year:
- Tracks `remaining` stations, `carryover_tasks`, and `_days_since_maintenance[node_idx]` (stochastic mode only)
- Calls `selector.select_for_day(remaining, team_states, carryover, dsm_array=...)` → task lists per team
- Passes task lists to `policy.create_initial_plan()` → `VRPSolver`
- Each hour: generates disruptions, calls `policy.handle_disruptions()`

### Simulation Output (JSON)
`write_json()` writes: `meta`, `summary` (scalar KPIs), `days` (per-day costs/tasks), `hourly` (per-(day,hour) disruptions and actions). NumPy scalar types (`float32`, `int64`) are not JSON-serialisable — handled via custom encoder; keep in mind when adding new logged fields.

### Failure Simulation
Two modes via `failure_simulation.mode`:
- `csv` — load from `data/malfunction.csv` (100 days, Poisson arrivals)
- `stochastic` — `p(t) = (p1_per_hour + p2_per_hour) × recovery_curve(t)`; `t` = days since last maintenance

### Configuration
All parameters in `configs/config.yaml`. Key sections:
- `planning`: `n_zones`, `max_stations_per_team`, `value_based_zone_selection`, `priority_weights`
- `maintenance`: `n_teams`, `workday_start/end_hour`, `mean_service_time`, OR-Tools time limits
- `failure_simulation`: `mode`, `p1_per_hour`, `p2_per_hour`, `recovery_days`, `initial_factor`

## Important Notes

- Depot is always **index 0** in all matrices and coordinate lists; `node_idx = station_index + 1`
- Distance matrices and processed data are git-ignored; regenerate with the build scripts
- `value_based_zone_selection` only has effect when `failure_simulation.mode: stochastic` (requires `_days_since_maintenance`); in `csv` mode it silently falls back to classical scoring
- DB/CFA-DB MLP α **requires `failure_simulation.mode: stochastic`** (csv mode makes state features trivial)
- CFA θ is trained on Myopic policy rollouts via OLS — no retraining needed when changing zone scoring only
- `VRPSolver._solve_teams_independently()` solves each team in isolation (1-vehicle model per team); OR-Tools drops from one team are **not** offered to the other team
