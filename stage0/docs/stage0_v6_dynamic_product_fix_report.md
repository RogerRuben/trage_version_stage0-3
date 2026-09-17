# Stage 0 v6 dynamic product fix - fixed-600 report

## Evidence and semantic boundary

- Frozen sample SHA-256: `160241c6f54f6c8083144a7c9ff052072e222f10ab4373fb9aca1b951c69746b`.
- Valhalla 3.8.2 remains the matcher. No candidate generator, HMM/Viterbi, Pareto search, boundary repair, restriction router, tile logic, canonical mapper, v5 code, sample, seed, or matching parameter was changed.
- Route usability and dynamic link-time usability are now independent outputs.
- Dynamic thresholds are initial engineering thresholds, not empirically optimized values.

## Acceptance summary

| Check | Result |
|---|---|
| 600/600 accounting | PASS |
| Processing exceptions | PASS: 0 |
| Cold/hot field-level equality | PASS |
| Time conservation failures | PASS: 0 |
| Timestamp anchor failure orders | PASS: 0 |
| Inferred-edge observed-time violations | PASS: 0 |
| Unresolved duplicate allocations | PASS: 0 |
| Preprocess boundaries materialized | PASS: 14 rows |
| Bucket manifests | PASS: 3 buckets |

## Route layer

| Metric | Before fix | After fix (hot) |
|---|---:|---:|
| Successful route orders | 599 | 599 |
| Strict Core | 543 | 441 |
| Analysis Set | 43 | 140 |
| Rejected | 14 | 19 |
| Formal eligible | 586 (97.67%) | 581 (96.83%) |
| Mean matched point share | 99.68% | 99.68% |
| Mean matched route interval share | 98.63% (old semantics) | 92.60% |
| Mean inferred distance share | 4.05% | 4.05% |
| Mean preprocess break count | not reported | 0.0233 |
| Mean canonical mapping share | 99.83% | 99.83% |

Cold/hot matched points were identical: **True**. Cold/hot normalized/canonical route parts were identical: **True**. This is the direct evidence that the Valhalla route result remained stable while downstream measurement semantics changed.

Route/resolved-GPS ratio P50/P90/P99: 1.0037/1.0210/1.1566. Route/raw-GPS ratio is retained separately as a diagnostic.

## Dynamic measurement layer

| Metric | Hot result |
|---|---:|
| Dynamic Strict | 1 |
| Dynamic Partial | 564 |
| Dynamic Unusable | 35 |
| Dynamic Partial or better | 565 (94.17%) |
| Route-eligible but dynamic-unusable | 31 |
| Mean direct-observed interval time share | 12.33% |
| Mean direct-observed distance share | 18.32% |
| Mean interval-supported time share | 10.04% |
| Mean engine-allocated time share | 0.00% |
| Mean unresolved time share | 77.63% |
| Mean timed traversal share | 29.75% |
| Mean valid timed traversals/order | 26.532 |

Unknown dynamic values are `NaN`, never zero. Engine allocation remains disabled; parsed Valhalla cumulative elapsed time is converted to per-edge increments but is not written into `observed_travel_time_s`.

## Unresolved-time causes

- `engine_interpolated_endpoint`: 196885.000 s (50.07%)
- `same_valhalla_edge_not_unique_canonical_edge`: 135626.000 s (34.49%)
- `missing_edge_index`: 31512.000 s (8.01%)
- `inferred_path_between_gps_anchors`: 13627.000 s (3.47%)
- `preprocess_time_gap`: 13208.000 s (3.36%)
- `unmatched_endpoint`: 2290.000 s (0.58%)
- `valhalla_route_discontinuity`: 55.000 s (0.01%)

Multi-edge continuous intervals are retained as `interval_supported` records with start/end timestamps and route distance, but are not assigned to individual link travel times. Intervals containing engine-interpolated edges are wholly `unresolved`.

## Performance and streaming

| Metric | Cold | Hot | Requirement |
|---|---:|---:|---:|
| Total wall | 194.150 s | 195.915 s | n/a |
| Wall/order | 0.324 s | 0.327 s | <= 0.400 s |
| Pure Valhalla/order | 0.0154 s | 0.0158 s | <= 0.050 s |
| Parsing | 12.854 s | 13.174 s | n/a |
| Canonical mapping | 122.934 s | 122.105 s | n/a |
| Product build | 27.886 s | 29.223 s | n/a |
| Quality evaluation | 3.997 s | 4.182 s | n/a |
| Parquet write | 1.005 s | 1.034 s | n/a |
| Peak RSS | 498.9 MB | 547.5 MB | <= 1024 MB |
| Maximum bucket RSS | 498.9 MB | 547.5 MB | n/a |

Hot order latency P50/P90/P99: 296.5/544.6/796.6 ms.

The run wrote one 200-order bucket for each of the three dates. Each product used a temporary Parquet followed by atomic replacement, and each bucket emitted its own manifest before in-memory product frames were released. This removes the previous all-600-product retention pattern and is structurally suitable for 6,000 orders, subject to a Gate 1 run rather than an unmeasured guarantee.

## Required conclusions

1. **Valhalla route stability:** yes. Cold/hot matched-point and route-part products are field-identical, with 0 processing exceptions.
2. **Route-layer usability:** 581/600 (96.83%) formal eligible after corrected route/resolved-GPS and break semantics.
3. **Dynamic-measurement usability:** 565/600 (94.17%) are `dynamic_partial` or `dynamic_strict`.
4. **Static-only orders:** 31 route-eligible orders are `dynamic_unusable` and must not create dynamic link labels.
5. **Main unresolved causes:** listed above by conserved interval duration; inferred paths, interpolated endpoints, unmatched/discontinuous intervals, and preprocess boundaries are kept separate.
6. **Cross-inferred-edge allocation:** no remaining violation was detected (0).
7. **6,000-order execution risk:** the 200-order atomic bucket design avoids linear retention of product DataFrames; measured peak RSS was 547.5 MB and every bucket manifest passed. A 6,000-order run is expected to be memory-safe but has not yet been executed.
8. **Gate 1 recommendation:** do not start automatically. Engineering acceptance must pass, the 100-order human audit must still show no systematic route error, and the measured dynamic-partial-or-better coverage must be accepted by the downstream Stage 1 owner.

No real route-accuracy claim is made from internal quality tiers.
