# Efficient routing and repositioning protocol

This R0.5c protocol supersedes the broader R0.5b experiment grid. The canonical 41 scenarios remain frozen and are not overwritten.

## Routing mode

The frozen 60-arc/5,000-arc audit is reused because it strictly contains the new minimum evidence. The R0.5c reporting subset contains 20 fixed arcs: one known divergent Q50 arc, seven ordinary arcs, six patience-boundary arcs, and six peak-period arcs. Ordinary arcs use three repeats per deterministic mode; the known divergent arc uses ten. Both `SINGLE_SOURCE_MATRIX` and `SCALAR_ROUTE` have zero failures and zero within-mode ETA range. The known multi-source contrast is retained as evidence of the original batch-context issue.

The frozen R0.5c mode is `SCALAR_ROUTE`. This selection is not based on agreement with canonical values: both deterministic modes are exact on the sample, while scalar routing had higher measured throughput (approximately 925 versus 820 arcs/s). Routing remains arc-level over the existing sparse candidate set, CPU-only, with no ETA rounding or patience change.

## Authorized simulations

Exactly five full-day runs are authorized, sequentially:

1. `DET_Q50_CONTROL_A`
2. `DET_Q50_CONTROL_B`
3. `DET_Q50_REPOS`
4. `DET_Q75_CONTROL`
5. `DET_Q75_REPOS`

Q50 control A becomes the scientific control only if A/B are exactly equal. Q25, all-AV, and Gamma runs are not authorized. The existing `TRAIN_TOD_DEMAND_BALANCE` policy and Train-only reference remain frozen.

## Idle semantics

Baseline idle random walk is `NONE`. Availability requires FleetPy `IDLE` state and an empty assigned route. Assigned passenger pickup travel is represented by FleetPy `REPOSITION`, but this is empty pickup travel rather than proactive roaming. Passenger completion updates the current location to the drop-off, after which the vehicle remains there while idle. A proactive manager is instantiated only when `repositioning_enabled=true`.

## Resource guard

All runs are single-process and CPU-only. Candidate selection stays sparse; no vehicle-by-order dense matrix is constructed. A 10,800-second per-scenario guard implements the taskbook's 2.5–3 hour cost-effectiveness boundary.
