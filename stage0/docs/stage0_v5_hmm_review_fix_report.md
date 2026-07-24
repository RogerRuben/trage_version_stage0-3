# Stage 0 v5 HMM review corrections and fixed-600 rerun

## Scope

This report records corrections made against remote baseline
`dbc17444770b6d380f4f7c179c560dccec4b1a93`. It uses the frozen 600-order
development sample (200 orders each from `20161010`, `20161014`, and
`20161016`; seed `20261009`) with sample SHA-256
`160241c6f54f6c8083144a7c9ff052072e222f10ab4373fb9aca1b951c69746b`.
No Gate 1 or Gate 2 run was started.

## Review-item disposition

1. **HMM attempt accounting:** summaries now report full/local attempt,
   success, failure, failure conditional on attempt, trigger reason, failed
   local-window count, and pre-validation-to-final-mode cross-tabs. Final
   `matching_mode` is not used as a proxy for computational execution.
2. **Failed-order audit semantics:** position-aware auditing is N/A for a
   match failure, not an artificial error. Inferred-distance statistics
   exclude failed reconstructions and are reported separately for successful,
   strict-core, and analysis-eligible orders.
3. **Candidate semantics:** projection distance, reliable-heading evidence,
   and edge priors are independent emission terms. Low-motion points provide
   no direction evidence and are excluded from the full-order ambiguity
   denominator. Candidate complexity probes more than the top edge, and
   under-minimum/no-candidate states trigger explicit radius expansion.
4. **HMM transition semantics:** the duplicate speed-gap term was removed.
   Same-edge jitter, physical network distance, and generalized routing
   penalties remain separate. Direct movements include the target edge's
   configured routing/access cost rather than an incomplete extra penalty.
5. **Transition search:** candidate matrices use one constrained multi-target
   search per source edge. The generalized-cost label search stops once all
   requested targets have settled their minimum-cost physically feasible
   labels. Cache watermarks distinguish exact-cutoff results from complete
   frontiers; positive and negative cache entries cannot suppress a later,
   larger valid search.
6. **Local HMM anchors:** local solves include fixed geometric anchors outside
   the ambiguous interior. Successful windows are retained independently.
   Failed windows are expanded locally, and boundary transitions are repaired
   before any whole-order escalation.
7. **Fallback identity:** successful local patches plus unresolved windows are
   named `partial_local_hmm_fallback`; repaired cases are named
   `partial_local_hmm_repaired`. A selected sequence with an unresolved
   transition becomes `failed_no_continuous_route`, while preserving the
   attempted edge sequence and transition-level reason for audit.
8. **Bounded retry:** one local retry tier is capped at 1,200 m and only
   considers the top three candidates. It records initial/final cutoff and
   retry use. It does not force a 6 km bridge.
9. **HMM/output consistency:** the selected transition path and its physical
   distance are persisted and reused by reconstruction. Identity and distance
   mismatches are audited independently. A 1 mm numerical tolerance covers
   only float32 routing-array roundoff observed at approximately
   `1e-6`--`1e-5` m.
10. **Operational observability:** the runner supports `--log-file`; per-bucket
    progress logs include mode, quality, recent matching time, and RSS.
    Reports include performance percentiles, path searches, expanded states,
    failure reasons/status, candidate expansion, and failed point-index
    distributions.

An explicit missing/break state has **not** been introduced into Viterbi.
Orders with no continuous legal selected sequence remain rejected with
auditable provisional states. Adding a break state would change the formal
route product semantics and requires a separate method decision; it is not
represented here as completed.

## Fixed-600 measured result

The final clean-hot result is:

| Metric | Result |
|---|---:|
| Input/output accounting | 600 / 600 |
| Processing exceptions | 0 |
| Strict core / analysis set / rejected | 262 / 33 / 305 |
| No-continuous-route orders | 200 (33.33%) |
| Full-HMM attempts / successes / failures | 86 / 51 / 35 |
| Full-HMM attempt / failure share | 14.33% / 5.83% |
| Local-HMM attempt share | 85.17% |
| Fallback share | 8.00% |
| HMM/output path distance mismatches | 0 |
| Actual invalid position events | 1 |
| Successful-route inferred share, P50 / P90 / P99 | 7.54% / 31.70% / 64.73% |
| Pure compute | 1,055.38 ms/order |
| Total hot wall time | 740.72 s |
| Peak RSS | 3,040.14 MB |
| Path searches/order, P50 / P90 / P99 | 365 / 1,048.2 / 1,941.96 |
| Expanded states/order, P50 / P90 / P99 | 20,817.5 / 143,106.3 / 557,591.34 |

Failure reporting does not interpret `topology_gap_count=0` as continuity
success. There are 200 orders rejected before a formal route could be emitted.
The failed-transition audit records 699 no-path and 4 endpoint-cutoff events;
701 raw pairs have missing topology.

## Cold/hot determinism

The cold and clean-hot products were compared after stable sorting with
absolute numeric tolerance `1e-6`:

| Product | Rows | Mismatched rows |
|---|---:|---:|
| order_base | 600 | 0 |
| route_parts | 26,389 | 0 |
| link_traversals | 26,389 | 0 |
| turn_movements | 25,989 | 0 |
| route_quality | 600 | 0 |

The comparison result is
`stage0/output_v5/reports/fixed600_review_cold_hot_equality.json`.

## Gate decision

The fixed sample does **not** meet the exit contract:

- pure compute exceeds 0.25 s/order;
- hot wall time exceeds 0.35 s/order;
- full-HMM failure share exceeds 5%;
- one-third of orders still have no continuous legal route;
- the required completed manual review is unavailable.

Therefore Gate 1 and Gate 2 remain blocked. The largest performance hotspot is
transition ambiguity plus local-HMM path search; the largest correctness risk
is candidate/state continuity for the 200 failed routes, especially
partial-local patches whose remaining transitions have no legal path.
