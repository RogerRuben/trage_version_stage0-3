# Stage 0 v5 P1 correctness and performance report

## Benchmark contract

P1 uses the same 600 complete orders as P0: 200 each from `20161010`, `20161014`, and
`20161016`, stable seed `20261009`, one process, and 32 buckets. The sorted `(date, order_id)`
sample SHA-256 is
`160241c6f54f6c8083144a7c9ff052072e222f10ab4373fb9aca1b951c69746b`.
No 6,000-order Gate 1 run was started. Benchmark artifacts remain under ignored
`stage0/work_v5/benchmark_p1_*` directories.

The P0 report previously presented the final `full_order_hmm` mode share as though it were the
execution share. That was incorrect. P0 attempted full-order HMM for 239/600 orders (39.83%),
succeeded for 146 (24.33%), and failed for 93 (15.50%).

## Implemented corrections

1. HMM transition routing and final route recovery are separate APIs. HMM computes only physical
   distances; it never constructs or caches candidate-pair paths. A concrete path is recovered
   only after Viterbi selects a transition.
2. Positive path caches are cutoff independent. Negative caches record the exact exhaustively
   searched cutoff, so a failed 301 m query cannot incorrectly suppress a later 499 m query.
3. Selected transition paths are passed directly to reconstruction. Reconstruction separately
   reports bridge requests and actual new searches.
4. Candidate generation uses one process-level CRS transformer, one batched radius query per
   order, vectorized Shapely projection/distance/heading operations, and exact-distance filtering
   before candidate truncation. Radius recall uses `radius + maximum_sampling_spacing`, rather
   than stopping after the first ten edges among sixty sample points.
5. Low-motion points inherit the nearest reliable moving heading for scoring but are excluded
   from the full-order ambiguity denominator and from hard direction evidence.
6. Parallel relations are reduced to connected components with union-find. Confirmed
   `exact_duplicate` and `semantic_equivalent` edges retain their canonical identities for audit
   but share one candidate-state alias.
7. Selected-path routing includes edge `routing_penalty`, `candidate_penalty`, access/service
   effects already encoded by the canonical network, and movement penalties. HMM distance remains
   physical metres.
8. Direction hard failure now requires reliable displacement and a persistent conflict. Isolated
   low-speed heading noise remains a warning.
9. Case traces contain selected edge, rank, projection distance, ambiguity reason, HMM mode, and
   selected route parts including inferred edges. They are no longer raw GPS-only exports.
10. Thread-based bucket execution is disabled; scale-out is explicitly process-sharded. This P1
    benchmark remains single-process so concurrency is not part of the result.
11. Physical-distance Dijkstra frontiers are reused for repeated sources within one order and
    cleared at the next order boundary. This lowered source searches without retaining unbounded
    cross-order search trees.

## Corrected HMM and routing accounting

| Metric | P0 | P1 |
|---|---:|---:|
| Full-HMM attempt count/share | 239 / 39.83% | 192 / 32.00% |
| Full-HMM success count/share | 146 / 24.33% | 133 / 22.17% |
| Full-HMM failure count/share | 93 / 15.50% | 59 / 9.83% |
| Raw-ambiguity triggers | 233 | 185 |
| Multiple-local-failure triggers | 6 | 7 |
| Local-HMM attempt count | not separately recorded | 410 |
| Local window failures | 107 | 102 |
| Geometric fallback count/share | 159 / 26.50% | 120 / 20.00% |

P1 performed 124,461 physical-distance source searches and 5,994 selected-path searches, and
expanded 14,656,679 edge states. Before order-local source-frontier reuse, the same implementation
made 139,577 source searches and expanded 17,840,363 states; reuse reduced these by 10.8% and
17.8%, respectively. This is still too expensive: path construction is no longer
multiplied across unused HMM candidate pairs, but the distance-only search remains the dominant
unresolved routing cost.

For final selected transitions, 9,332 concrete paths were computed before reconstruction and
reused there. Reconstruction issued 251 additional bridge requests, all answered by the shared
positive/negative caches in the clean hot run; actual new reconstruction path searches were zero.
The selected-path precompute share was 97.38%. A bridge request is not reported as a path search.

## Cold and hot measurements

These are end-to-end observations, not a claimed matcher speedup:

| Metric | Cold run | Clean hot run |
|---|---:|---:|
| Total wall time | 363.97 s | 309.90 s |
| Seconds/order | 0.6066 | 0.5165 |
| Initialization | 63.92 s | 20.25 s |
| Candidate-index initialization | 46.54 s (cache miss) | 0.83 s (cache hit) |
| Pure compute | 321.19 ms/order | 318.83 ms/order |
| Output + diagnostic I/O | 7.30 ms/order | 7.70 ms/order |
| Unprofiled wall time | 99.41 s | 89.49 s |
| Peak RSS | 2,936.38 MB | 2,808.39 MB |

The hot run was 54.06 seconds faster, largely consistent with avoiding the 46.54-second candidate
index build. Pure compute differed by less than 1%, so no matcher speedup is attributed to the
warm cache itself. The remaining 89.49--99.41 seconds means the current profiler still misses
bucket-level DataFrame assembly, performance-table writes,
manifest/finalization, garbage collection, and operating-system scheduling/I/O effects. No pure
algorithm speedup is inferred from these wall times.

P1 candidate generation fell from the P0 mean of 137.28 ms/order to 80.85 ms/order in both the
final cold and hot runs. The clean-hot mean/P95 stage profile was:

| Stage | Mean ms | P95 ms |
|---|---:|---:|
| Candidate generation | 80.85 | 304.91 |
| Ambiguity detection | 19.45 | 46.54 |
| Local HMM | 61.73 | 223.57 |
| Full HMM | 38.53 | 228.44 |
| Transition distance search (inside HMM) | 84.15 | 258.07 |
| Selected-path search | 50.24 | 252.91 |
| Matching total | 257.17 | 733.86 |
| Route bridge search during reconstruction | 0.005 | 0.042 |
| Route-parts construction | 11.06 | 40.54 |
| Traversal construction | 40.27 | 100.17 |
| Movement construction | 3.68 | 7.80 |
| Quality evaluation | 6.65 | 10.48 |

`transition distance search` is included inside local/full HMM time; it must not be added again
when computing pure end-to-end time.

## Quality and continuity diagnostics

All 600 orders are accounted for; time- and distance-conservation failures are zero.

| Metric | P0 | P1 |
|---|---:|---:|
| strict_core | 314 | 398 |
| analysis_set | 71 | 103 |
| rejected | 215 | 99 |
| topology-gap events | 372 | 251 |
| direction-violation events | 736 | 160 |
| layer/restriction violations | 0 / 0 | 0 / 0 |
| mean inferred-distance share | 26.89% | 25.91% |

Of the 251 topology gaps, 232 occurred in geometric fallback, 19 in local-HMM results, and zero in
successful full-HMM or fast deterministic results. This localizes the remaining continuity defect
instead of labelling every missing movement as direction, topology, and layer failure at once.

## Remaining risks and gate status

P1 fixes the identified cache correctness bug, false low-speed ambiguity, duplicate candidate
states, candidate recall truncation, path/distance API mixing, route-cost omissions, and
unreviewable case traces. It does **not** establish acceptable full-scale performance:

- 32.0% of orders still attempt a full-order HMM;
- physical-distance routing still expands 14.66 million edge states for 600 orders;
- fallback remains 20.0%, and fallback accounts for most topology gaps;
- mean inferred-distance share remains 25.91%;
- hot-run wall time has large unprofiled variance;
- the 3 m reliable-heading threshold and 10 m same-edge jitter tolerance remain development
  parameters that require Train estimation and Validation freeze before Test.

Therefore Gate 1 and the 6,000-order run remain **HOLD**. The next performance work should profile
bucket assembly/finalization explicitly and reduce physical-distance source searches, then repeat
this exact 600-order sample before any scale-up.
