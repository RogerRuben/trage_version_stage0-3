# Stage4 S3 Rolling OR Baseline

Recommendation: `GO_ROLLING_OR_BASELINE`

This is one bounded qA=0.25 scientific-interface and computational benchmark, not a policy comparison.

## Result

- Requests/matched/completed/expired: 1458/1212/1212/246
- HV/AV assignments: 1067/145
- Requested/achieved qA: 0.250000/0.250531
- HV vehicle-hour error: 0.872%

## Queue

- First-window match rate: 0.755830
- Carry-over entry/recovery: 0.244170/0.308989
- Critical count/recovery: 247/0.004049

## Computation

- Runtime: 142.516s
- Spatial/Top-K/valid pairs: 581740/78447/5324
- Peak Top-K/valid OR arcs per epoch: 940/133
- Memory design: cKDTree + Top-K 20 + CSR sparse constraints; no order-by-fleet dense matrix and no per-vehicle tick trace.
- GPU usage: none (CPU-only SciPy/HiGHS and Valhalla).
- Routing queries/cache hits/failures: 3739/22619/1860
- Solver p50/p95/max: 0.004325/0.019366/0.134063s

## Interpretation limits

- Stage3 suitability gates passenger service routes only; AV pickup legs are checked only for Valhalla routability.
- HV supply units are effective service-session templates, not a physical fleet count.
- Passengers use the S3 ALL_ACCEPT_AV baseline.
