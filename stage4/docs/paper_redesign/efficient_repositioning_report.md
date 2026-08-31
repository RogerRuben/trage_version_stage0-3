# Efficient repositioning robustness report

## Decision

`REPOSITIONING_ROBUSTNESS_NOT_COST_EFFECTIVE`

Recommendation: `STOP_NOT_COST_EFFECTIVE`.

The first authorized full-day run, `DET_Q50_CONTROL_A`, was stopped without a scientific result after 9,400 seconds (156.7 minutes). It had not written `summary.json` or the request/assignment products. A conservative five-run projection based only on elapsed lower-bound cost is 13.06 hours, above the protocol's 10-hour hard reconsideration point and outside its 5–8 hour target. Q50 B, Q50 repositioning, Q75 control, and Q75 repositioning were not launched.

The partial Q50 A directory is not treated as an experiment result. No A/B repeatability claim and no repositioning-effect claim are made.

## Routing

- Reporting sample: 20 frozen arcs.
- Ordinary/boundary/peak repeats: three per deterministic mode.
- First known divergent arc: ten repeats per deterministic mode.
- Failures: zero.
- Maximum within-mode ETA range: 0 seconds.
- Frozen mode: `SCALAR_ROUTE` because both modes were exact and scalar had higher micro-spike throughput (approximately 925 versus 820 arcs/s).
- Full-day reality: the micro-spike did not capture the dominant FleetPy/Python state-advancement cost, so it materially underestimated end-to-end runtime.

## Spatial representativeness

Full Test31 versus the 30,000-order replay is closely aligned. All-day TVD is 0.054, Spearman spatial-share correlation is 0.965, and top-10% hotspot Jaccard overlap is 0.889. Evening values are 0.078, 0.947, and 0.545. Fleet-start distributions remain positively aligned without an obvious displacement outside the demand footprint.

`KEEP_CURRENT_FLEET_RECONSTRUCTION = YES`.

## Idle movement

Baseline idle random walk is `NONE`. Vehicles remain at their current position while FleetPy state is idle and no route is assigned. Passenger completion moves the current position to the drop-off. FleetPy `REPOSITION` before pickup is assigned empty pickup travel; proactive movement exists only when the frozen enhancement switch is enabled.

## Experiment status

| Run | Status |
|---|---|
| DET_Q50_CONTROL_A | STOPPED_NOT_COST_EFFECTIVE; no valid scientific output |
| DET_Q50_CONTROL_B | NOT_RUN_COST_STOP |
| DET_Q50_REPOS | NOT_RUN_COST_STOP |
| DET_Q75_CONTROL | NOT_RUN_COST_STOP |
| DET_Q75_REPOS | NOT_RUN_COST_STOP |
| Q25 | NOT RUN; unauthorized by R0.5c |
| all-AV | NOT RUN; unauthorized by R0.5c |

Q50 A/B exactness, paired effects, and a SUPPORTS/QUALIFIES/CHANGES classification are therefore `NOT IDENTIFIED`.

No Gamma frontier was launched.
