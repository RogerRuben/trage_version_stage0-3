# Matching-capacity funnel and Top-K

Ten existing ODD_Q50_M_P70_REFERENCE states were evaluated at 07:30, 08:30, 12:00, 13:00, 17:00, 17:30, 18:00, 18:30, 21:00 and 23:00. Every physical-state hash matches the prior registry. This is a frozen-state diagnostic at q_A=.50, M, p=.70 under canonical zero-lead / 300-second timing, not a fleet-share sweep or daily service estimate.

## Edge volume versus simultaneous AV capacity

Counts below sum over ten states. U and M are state-order and state-matching totals, not unique daily orders. Loss percentages divide summed losses by the preceding summed count.

| Gate | E | U | M | E loss | U loss | M loss |
|---|---:|---:|---:|---:|---:|---:|
| Spatial | 95,180 | 720 | 718 | — | — | — |
| Passenger | 65,444 | 493 | 493 | 31.24% | 31.53% | 31.34% |
| Structural | 31,153 | 238 | 238 | 52.40% | 51.72% | 51.72% |
| Evidence | 15,775 | 124 | 124 | 49.36% | 47.90% | 47.90% |
| Shared Top-K | 1,371 | 124 | 124 | 91.31% | 0% | 0% |
| Route returned | 1,371 | 124 | 124 | 0% | 0% | 0% |
| Pickup within patience | 48 | 12 | 12 | 96.50% | 90.32% | 90.32% |
| Solver eligible | 48 | 12 | 12 | 0% | 0% | 0% |

Passenger, structural and evidence gates remove 225, 255 and 114 units of maximum matching; patience removes another 112. Substantive gate attrition translates to matchable-order attrition in this sample. Top-K removes 14,404 edges without losing U or M: edge attrition alone is not a capacity measure.

The sequence preserves production semantics, with shared HV/AV Top-K and route return independently reported before patience. Matching uses exactly the reported sparse graph. K20 G5 AV identities match the actual production candidate builder in all ten states. Gamma is applied in later optimization and is not a per-arc gate; assignment competition is not counted as a feasibility stage.

## Top-K sensitivity

**K20_CAPACITY_STABLE in all ten tested states.**

| State | M(K10) | M(K20) | M(K40) | M(K80) |
|---|---:|---:|---:|---:|
| 07:30 | 2 | 2 | 2 | 2 |
| 08:30 | 1 | 1 | 1 | 1 |
| 12:00 | 4 | 4 | 4 | 4 |
| 13:00 | 1 | 1 | 1 | 1 |
| 17:00 | 0 | 0 | 0 | 0 |
| 17:30 | 1 | 1 | 1 | 1 |
| 18:00 | 0 | 0 | 0 | 0 |
| 18:30 | 0 | 0 | 0 | 0 |
| 21:00 | 2 | 2 | 2 | 2 |
| 23:00 | 1 | 1 | 1 | 1 |

Final solver-eligible E totals are 38/48/50/55 for K10/20/40/80; U and M total 12 at every K. K20 -> K80 adds seven eligible edges without raising matching capacity. This is bounded to ten states, including four evening states, and does not prove universal K20 optimality.

All K values share physical inputs and deterministic per-arc routing; only K changes. Canonical K remains 20. B plus C/Top-K completed in 132.54 seconds, 30,779 routed arcs, zero failures, CPU only. Caches are released per state and matching matrices remain sparse.

## Scope

These results support substantive capacity loss under the zero-lead proxy; timing robustness remains unestablished until frozen Train-only RT parameters are recovered and transferred. Ten q_A=.50 snapshots do not identify the full-day magnitude of the fleet-share effect.

Deadline-level ever-eligible order share: NOT AVAILABLE FROM FROZEN LOGS used here. Aggregate counts and selected assignments do not enumerate unselected eligible options throughout every order's waiting history.

Outputs are mechanism_validity/snapshot_gate_matching.csv, topk_sensitivity.csv, and matching_capacity_summary.csv. The matching_capacity_funnel directory exposes the same three tables under the section-11 filenames, without separate computation.
