# Stage 0 noded-network v3 diagnostic

## Status

`xian_2017_core_noded_v3` is an exploratory topology candidate. It is not yet a
canonical Stage 0 product. Formal Stage 1--4 work remains on hold until the full
date chain and a versioned manual route-truth sample pass.

## Root cause

The former `xian_2017_core` graph retained many same-level road crossings as
geometric intersections without a shared graph node. Its 18,332 nodes were split
across 6,102 weakly connected components; the largest component contained only
18.40% of nodes. That fragmentation made legal short transitions appear
unreachable and was the dominant cause of low directed route continuity.

The v3 builder splits same-level LineStrings at interior intersections, preserves
declared bridge/tunnel/layer separation, and clusters near-identical endpoints.
It does not silently connect grade-separated crossings or merge overlapping
road geometries.

## Network comparison

| Metric | Original | Noded v3 |
|---|---:|---:|
| Source/noded links | 12,707 | 29,375 |
| Weak components | 6,102 | 46 |
| Strong components | 8,694 | 524 |
| Largest weak-component node share | 18.40% | 98.96% |
| Same-level intersection pairs noded | - | 21,152 |
| Grade-separated pairs left disconnected | - | 4,139 |
| Overlap pairs left for review | - | 6 |

## Fixed-sample rematch comparison

The same 1,000 orders from 2016-10-23 were rematched with one worker and bounded
BLAS/OpenMP threads. Old-network geometric fallback identifiers were rejected
when they did not belong to the new network version.

| Metric | Original v2 network | Noded v3 network |
|---|---:|---:|
| Input orders | 1,000 | 1,000 |
| Successfully reconstructed orders | 1,000 | 984 |
| Explicit failed-match orders | 0 | 16 |
| Strict Core orders | 119 | 872 |
| Strict Core share of input | 11.9% | 87.2% |
| Strict Core share of reconstructed | 11.9% | 88.62% |
| Reconstructed but rejected orders | 881 | 112 |
| Mean directed gaps/reconstructed order | 3.832 | 0.0976 |
| Reconstructed orders with at least one gap | 827 | 84 |
| Geometric-fallback orders | 94 | 0 |

The 75.3 percentage-point input-level Core improvement is diagnostic evidence that topology
noding, not downstream risk or dispatch logic, is the primary continuity issue.
The 16 failed orders are retained in an explicit failed-order product because no
complete HMM sequence and no same-network fallback was available. The remaining
rejected routes are not repaired automatically. The Extended set
is empty under the current bounded directed-bridge criteria because none of the
remaining gaps has a qualifying bridge in the noded graph.

## Promotion blockers

1. Rematch and classify the complete frozen fit/train/validation/test date chain.
2. Review a versioned manual route-truth sample, including grade-separated and
   overlapping-road exceptions.
3. Audit time and distance conservation after noded-link reconstruction.
4. Freeze Core/Extended coverage and the route-quality thresholds.
5. Publish a canonical network and Stage 0 manifest; downstream stages must read
   that manifest explicitly.

Until those conditions pass, this result is `DIAGNOSTIC_PASS`, not a canonical
Stage 0 acceptance.
