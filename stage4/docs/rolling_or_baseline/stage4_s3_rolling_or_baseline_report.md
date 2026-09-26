# Stage4 S3 Rolling OR Baseline

Recommendation: `GO_ROLLING_OR_BASELINE`

This is one bounded qA=0.25 scientific-interface and computational benchmark, not a policy comparison.

## Result

- Requests/matched/completed/expired: 1458/1215/1215/243
- HV/AV assignments: 1072/143
- Requested/achieved qA: 0.250000/0.250531
- HV vehicle-hour error: 0.872%

## Queue

- First-window match rate: 0.775720
- Carry-over entry/recovery: 0.224280/0.256881
- Critical count/recovery: 244/0.004098

## Computation

- Runtime: 43.404s
- Spatial/Top-K/valid pairs: 559624/76441/5406
- Peak Top-K/valid OR arcs per epoch: 915/105
- Memory design: cKDTree + Top-K 20 + CSR sparse constraints; no order-by-fleet dense matrix and no per-vehicle tick trace.
- GPU usage: none (CPU-only SciPy/HiGHS and Valhalla).
- Matrix batches/uncached arcs/cache hits: 3655/54071/22370
- Matrix failed arcs/fallback success/fallback failure/final failed arcs: 0/0/0/0
- Matrix/final arc failure rates: 0.000000/0.000000
- Solver p50/p95/max: 0.003765/0.016364/0.021611s

## Matrix failure vs single route cat-eye

- Sampled matrix failures: 0
- Single route success/failure: 0/0
- Single route success rate: N/A

## Matrix-route closure

- Original 9eed065 observation: 1,860 matrix-failed arcs from approximately 55,828 uncached arcs (3.332%).
- Exact production-adapter reproduction: 0 matrix-failed arcs.
- Requested/available failed-arc sample: 100/0.
- The original failure population was not reproducible, so no empirical 100-failure equivalence rate is claimed.
- Production policy now retries only failed matrix cells with an identical single route; an arc is deleted only if both calls fail.
- Closure: `CLOSED_WITH_NON_REPRODUCTION_AND_FAILED_CELL_ROUTE_FALLBACK`

## Interpretation limits

- Stage3 suitability gates passenger service routes only; AV pickup legs are checked only for Valhalla routability.
- HV supply units are effective service-session templates, not a physical fleet count.
- Passengers use the S3 ALL_ACCEPT_AV baseline.
