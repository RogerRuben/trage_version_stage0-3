# Stage4 acceleration v2 — construction and execution notes

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation with finite equivalence checks
- Origin Date: 2026-10-05
- Verification Status: see `checks.json` (bounded scope, not full-day equivalence)
- Version Label: acceleration_v2

## Scope

This is an opt-in technical implementation of the two agreed acceleration work
packages. The scientific configuration, frozen M3, Train remaining-time model,
Stage3 profiles, source demand, patience, candidate policy and solver time limit
are not changed. Completed MYOPIC and acceleration-v1 products are preserved.
No new policy, profile, date, parameter search, or scientific experiment grid is
introduced. A v2 run is written to `accelerated_v2_full_day`, never over v1.

## Offline products

`stage4/output/runtime_acceleration_v2/assets/` contains:

| Product | Contents / use |
|---|---|
| `orders.npy`, `route_metrics.npy` | Compact read-only decision inputs, IDs, release times, predicted service time, C/M/A compatibility masks and existing descriptors |
| `simulation_truth.npy` | Separate realized progression times, attached only to physical FleetPy requests; not forecast or optimization inputs |
| `positions.npy` | Exact WGS84 catalog plus the existing rounded future-position identity and 34.25-degree future projection |
| `forecast/templates.npy`, `draw_index.npy`, `draws_*.npy` | Train-only sampling tape, original Poisson/choice/jitter draw order, stable request IDs, underlying passenger-acceptance uniform |
| `fleet.json`, `scenario_fleet.parquet`, `fleet_windows.npy`, `fleet_events.npy` | Exact preselected fleet, immutable availability policies, normalization accounting and exogenous start/end events |
| `manifest.json` | One-level source identities, product hashes and schema version for invalidating stale reusable inputs |

The draw tape is built in 64-epoch chunks and loads at most two mmap chunks.
Positions are not merged or rounded for raw routing. The CURRENT spatial index
still uses the mean latitude of the currently available vehicles, not a new
fixed-latitude neighbor table. FleetPy busy/free status, actual position, booked
conditional readiness, remaining patience and current graphs remain live.
No decision or Test31 future demand is precomputed.

## Runtime changes

1. Candidate queries reuse tree coordinates instead of projecting each neighbor.
2. An event calendar considers active/upcoming sessions rather than rescanning
   every historical session and reconstructing timestamps every epoch.
3. Known waiting-request future edges are built once for all three scenarios;
   pickup geometry and future-position projection have bounded caches.
4. Routing queries are submitted as a bounded sparse epoch queue. Each backend
   primitive remains an independent 1-source/1-target Valhalla query; results
   fill caches in original request order, including legacy rounded aliases and
   exact-RAM-hit precedence. No many-to-many routing matrix is introduced.
5. A route can be pre-pruned only if even zero pickup ETA cannot finish an
   empirical session. Top-K is not refilled. Rounded aliases needed by retained
   candidates are protected. These certified prunes have their own counter,
   not a fabricated patience/failure attribution.
6. The already-computed current critical/count/carry-over optimum is reused only
   for the exact same sparse current graph. FLOW recourse drops mathematically
   redundant per-vehicle rows. Fixed idle-state variables are substituted with
   1, row bounds are adjusted exactly and the full vector is reconstructed for
   integral recourse recovery. The original sparse input caps remain in force.

The raw route library is on-demand SQLite, capped at 512 MiB / 2,000,000 entries
plus a small bounded journal. RAM LRU remains 50,000 entries. Raw time/distance
are keyed by full precision, minute, routing mode, auto costing, engine config,
library version and frozen tile identity; beta is applied at read time. Failures
are not persisted. The tile identity uses the existing frozen content SHA plus
a current filename/size/mtime inventory; it does not claim to rehash all tile
bytes on every launch. This is a frozen-local-deployment cache, not a general
mutable network cache. No full Valhalla JSON or city-wide OD matrix is stored.

## Resources and accounting

CPU only; numerical-library and Arrow threads = 1; route workers = 2; worker
queue chunks = 256. Sparse routing blocks are at most 50,000 arc lookups, and
the process-group RSS bound remains 2 GiB. The configuration cannot override
science settings. No GPU, dense order-by-vehicle matrix or dense recourse graph.

When worker routing is enabled, the parent Actor is created lazily and normally
never created: only two worker Actors exist. RSS and private committed memory
are reported separately, because C++ allocator reservation can pressure the
Windows pagefile without appearing in the resident working set. Available
system RAM and disk are logged. A disk reserve below 512 MiB stops the new run
and preserves output rather than retrying or altering Windows paging settings.

Observed construction: 30,000 compact orders (28,367 common eligible), 67,176
positions, 1,697,301 draws, 4,444 fixtures; 68,496,326 bytes (65.3 MiB); 97.24 s,
285.75 MiB peak parent RSS. The core suite has 43 joint/legacy tests; the final
run also includes 16 existing FleetPy lifecycle / pickup-duration tests, with
`FLEETPY_ROOT` pointing to the pinned local checkout (59 total).
Current source/query checks are recorded in `checks.json`; they are not a claim
that the full-day acceleration run has already completed.

Backend routing wall time retains the v1 unit. Additional queue-pipeline and
disk-lookup wall timers are recorded separately (these timers overlap, do not
add them to total runtime). Raw backend primitive evaluations, logical arc
lookups, cache hits and deduplications have separate units. A cold-cache run
will still spend substantial time in Valhalla; benefit across subsequent runs
depends on their exact-query overlap. No 30-minute guarantee is asserted.

## Construction / checks / later execution

From this worktree, in the existing CPU-limited environment:

```powershell
python -m stage4.analysis.acceleration_assets
python -m pytest -q -p no:cacheprovider stage4/tests/test_acceleration_v2.py stage4/tests/test_acceleration.py stage4/tests/test_flexibility_model.py stage4/tests/test_flexibility_native.py stage4/tests/test_rolling_or_baseline.py stage4/tests/test_routing_determinism.py
python -m stage4.analysis.acceleration_v2_checks
python -m compileall -q stage4/dispatch stage4/fleetpy_adapter stage4/analysis
git diff --check
```

The final additional native checks use the existing
`test_dwell_native.py` / `test_fleetpy_native_shell.py` files. Set
`$env:FLEETPY_ROOT='D:/pycodes/didi_xian_raw/.external/FleetPy'` first. These are
tiny lifecycle fixtures, not replay experiments or additional policy conditions.

`assets_summary.json` reports actual construction time/bytes/RSS. `checks.json`
reports exact compact-input and route-policy comparison, selected deterministic
forecast epochs, and 12 real independent routing primitives plus cache replay.
These are engineering checks, not a new performance or scientific Gate.

Only AFTER construction and these finite checks are complete, the existing
authorized M lookahead run can be launched serially:

```powershell
./stage4/scripts/launch_accelerated_flexibility.ps1 -AccelerationConfig stage4/config/symmetric_flexibility_acceleration_v2.json
```

This reuses completed MYOPIC and keeps the previously authorized 6h
administrative limit / 10s per-epoch solve limit. No automatic retry or output
overwrite is permitted. It does not promise identical selected assignments at
ties: finite tests verify candidate/forecast/query semantics and lexicographic
objective equivalence, not a bitwise full-day physical trajectory.

## Deliberately not introduced

No assumed-speed straight-line routing cutoff, graph coarsening, larger time
step, smaller Top-K, weaker patience, offline live busy/free schedule, full-day
action precomputation, M3 refit, GPU, commercial solver, or Benders rewrite.
Those would require a different scope or additional scientific qualification.
