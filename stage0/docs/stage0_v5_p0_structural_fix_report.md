# Stage 0 v5 P0 structural fix report

## Scope and benchmark contract

This report covers the P0 structural corrections made after the first Gate 1 run. It is not a
replacement Gate 1 result and does not unlock Gate 2. The fixed benchmark contains 200 complete
orders from each of `20161010`, `20161014`, and `20161016`, runs with one worker and 32 buckets,
and uses the same stable sampling seed (`20261009`) in both versions. The sorted `(date,
order_id)` list contains 600 orders and has SHA-256
`160241c6f54f6c8083144a7c9ff052072e222f10ab4373fb9aca1b951c69746b` in both runs.

Baseline is commit `7af23e7cb0d9950087fba70e967fdb2b02b98246`. The P0 result was produced from the
uncommitted review working tree based on that commit; its eventual commit SHA must be recorded
after review and commit.

## Corrected structural defects

1. The 1,389,094-row movement table is converted once per process to a compact edge-state
   index. `build_movements` receives that index and does not build a pandas MultiIndex per order.
2. Dense SciPy all-node distance rows and whole-window source preloading were removed. HMM
   transitions now resolve same-edge and direct legal movements without search, and perform one
   bounded, target-terminating search per unresolved source for each adjacent point pair.
3. A shared directed OSM node is now the default evidence of a legal level transition. Bridge,
   tunnel, ramp, and suspicious attribute transitions are classified in
   `level_transition_type`; attribute changes alone no longer block a movement.
4. HMM transitions and final edge-aware reconstruction use the same legal movement router.
5. Grade-separation, parallel-road, and main/auxiliary ambiguity require close candidates;
   grade separation additionally requires spatially and heading-plausible alternatives.
6. Same-edge projection reversal up to 10 m is treated as GPS projection jitter with a penalty,
   not an impossible transition. The provisional 10 m value is the P95 (9.977 m, n=1,067) of
   low-displacement development case traces; it must be re-estimated on the complete Train data
   and frozen on Validation before Test.
7. A failed local HMM window is retried with wider anchors. One failed window no longer upgrades
   the entire order; `full_order_min_windows=4` now applies to unresolved local windows. Initial
   full-order HMM is driven only by the unexpanded raw ambiguity share.
8. Reconstruction uses bounded A* on compact arrays with a 50,000-pair LRU rather than unbounded
   NetworkX shortest paths.
9. Rejected point traces are retained by stable, per-reason quotas rather than in full.
10. Direction, topology, level, and restriction violations are independently computed.
11. Per-order performance output records candidate, ambiguity, local/full HMM, transition search,
    reconstruction, movement, quality and output I/O time, plus routing calls, expanded nodes,
    cache hits/misses, local windows, and the full-order trigger reason.

## Fixed 600-order comparison

| Metric | Baseline | P0 | Change |
|---|---:|---:|---:|
| Wall-clock runtime, one worker | 1,538.96 s | 284.20 s | 5.42x faster |
| Wall-clock seconds/order | 2.565 | 0.474 | -81.5% |
| Peak RSS | 3,596.85 MB | 2,781.47 MB | -22.7% |
| strict_core | 192 | 314 | +122 |
| analysis_set | 40 | 71 | +31 |
| rejected | 368 | 215 | -153 |
| full_order_hmm share | 35.0% | 24.33% | -10.67 pp |
| local_hmm share | 40.17% | 48.83% | +8.66 pp |
| geometric_fallback share | 24.50% | 26.50% | +2.00 pp |
| topology-gap events | 1,049 | 372 | -64.5% |
| mean inferred-distance share | 24.34% | 26.89% | +2.55 pp |
| retained point rows | 121,941 | 17,212 | -85.9% |
| output size (including network) | 78.98 MB | 72.73 MB | -7.9% |

The quality-class change is evidence that the old graph semantics rejected valid shared-node
level transitions; it is not presented as statistical model improvement. Geometric fallback and
inferred-distance share did not improve and remain explicit risks for the next Gate 1 run.

## P0 per-order performance profile

The P0 performance table contains one row per order. Values below are mean / P95:

| Stage | Mean ms | P95 ms |
|---|---:|---:|
| Candidate generation | 137.28 | 324.59 |
| Ambiguity detection | 23.96 | 57.72 |
| Local HMM | 25.02 | 96.69 |
| Full HMM | 18.60 | 87.48 |
| Transition search (included in HMM) | 33.34 | 103.39 |
| Matching total | 212.49 | 500.42 |
| Reconstruction | 124.53 | 558.37 |
| Movement construction | 3.20 | 6.72 |
| Quality evaluation | 4.67 | 6.49 |
| Output I/O allocation | 5.33 | 8.89 |

Across 600 orders, bounded movement routing made 171,937 source searches and expanded 5,522,922
edge states. Median expanded states per order were 5,170. Full-order triggers were 233 raw-high-
ambiguity orders and six orders with at least four failed local windows. There were 107 failed
local windows in total; isolated failures remained local/fallback and did not force a whole-order
HMM.

## Corrected diagnostics

The P0 run records 372 topology-gap events in 70 orders and 736 independent direction events in
169 orders. These values are no longer aliases. Level and explicit restriction violation counts
are both zero in this fixed sample. The current direction diagnostic uses selected-edge heading
and severe same-edge reverse motion; it still requires case review before any quality-threshold
freeze.

## Gate status and next action

All 600 sampled orders are accounted for; time- and distance-conservation failures are zero. This
benchmark passes its structural/performance purpose, but **Gate 1 remains HOLD** because the
corrected implementation has not yet rerun all 2,000 orders/day and the jitter/quality thresholds
have not been frozen using the full Train/Validation contract. The next data run, after code
review, is exactly the three-date Gate 1 command in `stage0_v5_reproduction_commands.md` using 32
buckets. No 6,000-order run was started as part of this P0 repair.
