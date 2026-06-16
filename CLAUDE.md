# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis on **maintenance route optimization for EV charging stations in Würzburg**. Optimizes daily routes for 2 maintenance teams servicing 397 charging stations using greedy cheapest-insertion routing, K-Means geographic clustering, and a Gymnasium RL environment.

**Key numbers**: 397 stations, 2 teams, 8:00–16:00 workday, 40 zones (configurable), 20 stations/team/day max.

**Active base policies** (all in `src/models/`):
1. **Myopic** (`myopic.py`) — greedy cheapest-insertion everywhere; no learned components
2. **Myopic+** (`myopic_plus.py`) — greedy everywhere; depot-distance sorted soft-deadlines, penalty ∝ `power_kW`
3. **CFA-Future** (`cfa_future.py`) — greedy everywhere; C̃/depot-distance sorted soft-deadlines, penalty ∝ C̃; manual C̃-drop loop replan; θ ∈ ℝ⁴ learned via contrastive suffix-simulation regression (`train_cfa_future.py`)
4. **DB-Simple** (`db_simple.py`) — extends CFA-Future 1:1, but modulates the distance exponent in greedy scoring and the drop-score weighting via a state-dependent balance parameter δ ∈ [0,1] (`δ=0.5` ⇔ identical to CFA-Future); δ from a rule-based or learned model (`data/training/db_simple/model.pkl`)

**"VFA" = Rolling-Horizon rollout of the base policies** (`src/models/rolling_horizon.py`):
- `RollingHorizonRunner` + `PolicyAdapter` wrap **any** of the 4 base policies above (Myopic, Myopic+, CFA-Future, DB-Simple) without changing their initial-plan/replan logic.
- At each disruption-driven drop decision, the top-`k` drop candidates (ranked by the base policy's own `drop_score_fn`) are evaluated via short-horizon (`horizon_days`) Monte-Carlo rollouts; the candidate with the lowest expected horizon cost is chosen, overriding the base policy's greedy choice if beneficial.
- This rollout-based value approximation **is** the current VFA concept — activated via `rolling_horizon.enabled: true` in `configs/config.yaml`. Supported by `run_myopic_plus.py`, `run_cfa_future.py`, `run_db_simple.py`.

> **Note:** OR-Tools (`src/planning/vrp_solver.py`) is legacy code — no longer used by any active model but kept for reference.

> **Deprecated model classes** (superseded by CFA-Future / DB-Simple / Rolling-Horizon-"VFA"; moved to `src/models/alt/`, scripts in `scripts/alt/`): `cfa.py` (CFA, OLS-trained θ), `vfa.py` (Hybrid-VFA with separately learned global θ), `db.py` (DB, PPO-trained MLP α), `cfa_db.py` (CFA-DB), `db_base.py` (DB-Base, predecessor of DB-Simple).

**Deprecated** (in `src/models/alt/`, scripts in `scripts/alt/`): CFA-Light, CFA-Real, DB-Alt, CFA, CFA-DB, DB, Hybrid-VFA, DB-Base.

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
# Single runs — 4 active base policies
python scripts/run/run_myopic.py [--max-days 10] [--log-day 1] [--verbose]
python scripts/run/run_myopic_plus.py
python scripts/run/run_cfa_future.py # requires: python scripts/train/train_cfa_future.py first
python scripts/run/run_db_simple.py  # requires CFA-Future θ + optional data/training/db_simple/model.pkl

# "VFA" = Rolling-Horizon rollout on top of a base policy
# rolling_horizon.enabled: true in configs/config.yaml (use run_myopic_plus.py / run_cfa_future.py / run_db_simple.py)

# Training
python scripts/train/train_cfa_future.py  # → data/training/cfa_future/theta.json (contrastive suffix-simulation regression)

# Monte Carlo (aggregate analysis → logs/<model>/log/<model>_overview.log)
python scripts/monte_carlo/run_mc_myopic.py --runs 30
python scripts/monte_carlo/run_mc_myopic_plus.py --runs 30
python scripts/monte_carlo/run_mc_cfa_future.py --runs 30
python scripts/monte_carlo/run_mc_db_simple.py --runs 30

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
- `src/planning/vrp_solver.py` — **legacy, unused** OR-Tools solver (time windows, makespan balancing, `extra_costs`, `soft_deadline_min`/`deadline_penalty`, AddDisjunction); kept for reference but not called by any active model

### Models

**Common interface:** all models implement `create_initial_plan(tasks)` + `handle_disruptions(disruptions, sim_routes, time_min, hour, log)`. The Rolling-Horizon ("VFA") runner wraps this interface via `PolicyAdapter` without altering it.

**Initialplan — carryover disruptions:** Myopic+ and CFA-Future (and thus DB-Simple, which inherits from CFA-Future) set `soft_deadline_min=0` + penalty ∝ `power_kW` for carryover disruption tasks so they are prioritized early in greedy insertion. Myopic does not use this mechanism.

**Zone selection value functions** (set in run scripts when `value_based_zone_selection: true`):

| Model | `selector.value_fn` | Zone value logic |
|---|---|---|
| Myopic | `policy._zone_value` | `power × recovery_curve(dsm)` |
| Myopic+ | `policy._zone_value` | `power × recovery_curve(dsm)` |
| CFA-Future | `policy._value` | `θᵀ × φ_scaled(k)` (4 features) |
| DB-Simple | `policy._value` (inherited from CFA-Future) | `θᵀ × φ_scaled(k)` (4 features) |
| "VFA" (Rolling-Horizon) | `policy._local_value` | local C̃ term of the wrapped base policy (zone selection is unaffected by the rollout layer) |

**CFA-Future details** (`src/models/cfa_future.py`):
- `φ(k) = [power_kW, age_years, recovery_curve(dsm), mean_dist_to_others]`
- `recovery_curve = initial_factor + (1 − initial_factor) × dsm / recovery_days`
- `C̃(drop k) = θᵀ × φ_scaled(k)` (features z-scored with training μ, σ)
- θ ∈ ℝ⁴ loaded from `data/training/cfa_future/theta.json`; always read `feature_means` and `feature_stds` from same file; θ learned via contrastive suffix-simulation regression (label: `cost_drop_k − cost_serve_k`, discounted, H=30 days), not OLS on Myopic rollouts
- Deadline-Sortierung: `C̃/depot_dist` ratio; Replan: manual C̃-drop loop (ascending C̃)

**DB-Simple details** (`src/models/db_simple.py`):
- Extends `CFAFutureModel` 1:1 — inherits `_phi()`, `_value()`, θ-loading, and OR-Tools paths unchanged
- Initial plan (greedy): `score(k) = (C̃(k) + shift) / dist(cur, k)^(2δ)`
- Replan drop-score: `drop_score(k) = C̃(k) − (2δ) × wage_per_min × detour(k)`
- `δ = 0.5` ⇒ identical to CFA-Future; `δ < 0.5` weights C̃ (future value) more, `δ > 0.5` weights routing efficiency more
- δ from `DBSimpleBalanceModel` (`data/training/db_simple/model.pkl`, rule-based `rule_delta()` or learned RF, fallback `default_delta=0.5`)

**Rolling-Horizon ("VFA") details** (`src/models/rolling_horizon.py`):
- `RollingHorizonRunner` drives the simulation day-by-day; `PolicyAdapter(base_policy)` exposes `get_drop_score_fn()`, `get_value_fn()`, and `get_route_score_fn_for_tasks()` derived from the base policy's own `_drop_score_fn`/`_value`
- On each disruption replan, the top-`k` drop candidates (by base-policy `drop_score_fn`) are each rolled out for `horizon_days` over `n_scenarios` stochastic scenarios; the candidate with the lowest expected horizon cost is chosen
- Config: `rolling_horizon.{enabled, horizon_days, n_scenarios, top_k_candidates, enable_replan, enable_initial, top_k_initial, fallback_to_legacy_on_timeout, time_budget_sec}`
- `enabled: false` → behaves exactly like the unwrapped `MaintenanceSimulator`

### Simulation Loop (`MaintenanceSimulator`)
`sim.run(mal_df, max_days)` drives the full year:
- Tracks `remaining` stations, `carryover_tasks`, and `_days_since_maintenance[node_idx]` (stochastic mode only)
- Calls `selector.select_for_day(remaining, team_states, carryover, dsm_array=...)` → task lists per team
- Passes task lists to `policy.create_initial_plan()` → greedy cheapest-insertion
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
- `maintenance`: `n_teams`, `workday_start/end_hour`, `mean_service_time`
- `failure_simulation`: `mode`, `p1_per_hour`, `p2_per_hour`, `recovery_days`, `initial_factor`

## Important Notes

- Depot is always **index 0** in all matrices and coordinate lists; `node_idx = station_index + 1`
- Distance matrices and processed data are git-ignored; regenerate with the build scripts
- `value_based_zone_selection` only has effect when `failure_simulation.mode: stochastic` (requires `_days_since_maintenance`); in `csv` mode it silently falls back to classical scoring
- DB-Simple's δ-model **requires `failure_simulation.mode: stochastic`** (csv mode makes state features trivial)
- CFA-Future θ is trained via contrastive suffix-simulation regression (`train_cfa_future.py`) — no retraining needed when changing zone scoring only
- `VRPSolver` is legacy — active models use greedy cheapest-insertion directly; drops from one team are never offered to the other team
