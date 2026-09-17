# Stage 0 v5 vs v6 Valhalla fixed-600 report

## Evidence boundary

- **Measured here:** v6 environment/tile build, one real-order smoke test, the exact fixed-600 cold and hot runs, product accounting, quality metrics, performance, and cold/hot determinism.
- **Task-supplied baseline:** the v5 comparison values below are the baseline stated in the Stage 0 v6 task. The local v5 worktree contains several historical snapshots with different metrics, so those snapshots are not silently substituted for the task baseline.
- **Not yet verified:** human route correctness. `Strict Core` is an internal quality tier, not a real accuracy estimate.

## Reproducibility

- Sample SHA-256: `160241c6f54f6c8083144a7c9ff052072e222f10ab4373fb9aca1b951c69746b` (expected value matched).
- Cold/hot product equality: **PASS**.
- Python / Valhalla: 3.12.13 / 3.8.2.
- Tiles were reused for both runs; the cold run did not rebuild the graph.

## Coverage

| Metric | v5 baseline | v6 hot |
|---|---:|---:|
| Accounted orders | 600/600 | 600/600 |
| Complete point match | not reported | 535 |
| Partial point match | not reported | 65 |
| No valid route | 142 | 1 |
| Orders with valid subtrace | not reported | 600 |
| Successful reconstruction | 458 | 599 |
| Strict Core | 320 | 543 |
| Analysis Set | 28 | 43 |
| Rejected | 252 | 14 |
| Formal analysis eligible | 348 (58.00%) | 586 (97.67%) |

## Route products

| Metric | v5 baseline | v6 hot |
|---|---:|---:|
| Mean route parts/order | not reported | 73.845 |
| Mean matched point share | not reported | 99.68% |
| Mean matched interval share | not reported | 98.63% |
| Mean inferred distance share | 10.88% | 4.05% |
| Mean unresolved time share | not reported | 6.67% |
| Mean canonical edge mapping share | not reported | 99.83% |

The v6 output includes `matched_points`, directed `route_parts`, continuous `link_traversals`, `turn_movements`, and `unresolved_intervals`. Inferred edges receive no observed dynamic time.

## Quality distributions

| Distribution | P50 | P90 | P99 |
|---|---:|---:|---:|
| OD endpoint error (m) | 5.062 | 12.264 | 58.157 |
| Snap distance (m, all matched points) | 3.010 | 8.805 | 38.217 |
| Route/GPS distance ratio | 1.0036 | 1.0202 | 1.1566 |
| Discontinuity count | 0 | 0 | 0 |
| Unmatched point share | 0.00% | 0.17% | 7.03% |

## Performance

| Metric | v5 baseline | v6 cold | v6 hot |
|---|---:|---:|---:|
| Total wall clock | 1187.3 s cold | 204.984 s | 198.065 s |
| Total wall/order | not reported | 0.342 s | 0.330 s |
| Pure matching/order | 1.818 s | 0.0146 s | 0.0136 s |
| Parsing/order | not reported | 0.0175 s | 0.0178 s |
| Product build/order | not reported | 0.1026 s | 0.1007 s |
| Peak RSS | 3034 MB | 574.5 MB | 610.7 MB |
| Processing exceptions | 0 | 0 | 0 |

Hot total order latency P50/P90/P99 was 294.2/578.5/876.6 ms. Hot pure matching latency P50/P90/P99 was 11.4/25.9/46.1 ms.

## Acceptance checks

| Check | Result |
|---|---|
| Fixed 600 accounting | PASS: 600/600 |
| Processing exceptions near zero | PASS: 0 |
| Human correctness not below v5 | PENDING: 100-case review pack generated, labels not completed |
| Formal eligible rate at least 58% | PASS: 97.67% |
| Inferred distance not above 10.88% | PASS: 4.05% |
| Hot order time materially below 1.818 s | PASS: 0.330 s wall, 0.0136 s pure match |
| Stage 1 traversal conversion | PASS at prototype product layer: link traversal and movement Parquet products generated; the Stage 1 consumer was not run in this round |
| No custom HMM/Pareto/restriction router in v6 | PASS |

## Architecture decision

Measured engineering evidence supports Valhalla replacing v5 candidate generation, KD-tree candidate indexing, local/full HMM, Viterbi retry, boundary repair, custom transition routing, Pareto search, and Exact failed-order review as the primary matcher.

Keep the v5 coordinate interpretation, raw archive/sample governance, fixed-sample hashing, position-aware distance semantics, traversal instance separation, observed/inferred provenance, unresolved intervals, dynamic-time rule, Parquet/manifest accounting, retention, and manual-review tooling. Their implementation should be adapted to normalized Valhalla output rather than HMM state objects.

## Gate recommendation

**Do not start the 6,000-order Gate 1 yet.** The automated feasibility checks pass, but the required 100-case human comparison is still unlabeled. If that review shows v6 correctness is not below v5, the measured coverage, inferred-distance, speed, memory, and product-conversion results support proceeding to Gate 1 without further matcher algorithm development.

No claim of true map-matching accuracy is made from the internal Strict Core share.
