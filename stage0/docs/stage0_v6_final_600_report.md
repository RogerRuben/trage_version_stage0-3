# Stage 0 v6 Final Fixed-600 Report

## Result

- Overall status: **PASS**
- Reconciled orders: **600/600**
- Processing exceptions: **0**
- Cold/hot semantic equality: **True**
- Time conservation failures: **0**
- Distance conservation failures: **0**
- Duplicate interval allocations: **0**
- Non-direct observed-time violations: **0**
- Traversal duplicate-distance violations: **0**

## Four-axis quality

- GPS: `{"clean": 291, "local_outlier": 213, "sparse_or_ineligible": 92, "unresolved_gap": 4}`
- Route: `{"route_pass": 431, "route_fail": 136, "route_partial": 19, "route_uncertain": 14}`
- Dynamic: `{"dynamic_partial": 503, "dynamic_unusable": 92, "dynamic_strict": 5}`
- Canonical: `{"chain_resolved": 490, "unmapped": 110}`

## Failed specified-case regressions

- None

## Review queues

- Route images: **169**; triggered only by `route_fail`, `route_partial`, or `route_uncertain`.
- GPS diagnostics: **54**.
- Dynamic and `chain_resolved` states do not independently trigger route images.
