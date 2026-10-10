# Stage 2 v4 methodology

Stage 2 v4 predicts future route-token pressure conditional on information
available strictly before an order's dispatch/departure time. It is isolated
from the legacy `stage1/output/prediction_split` and Stage 2 v3/oracle-route
products.

## Frozen upstream

- Stage 1 release: `stage1-v3-final`
- Stage 1 commit: `8bfdec5a043e3ecbdd6473ce49c735db7732e493`
- Supervised unit: `(order_id, traversal_id)`
- Route token unit: `(order_id, route_sequence)`
- Directed identity: `observed_directed_edge_uid`

Every command validates the frozen Stage 1 identity and fails closed on a
release, model, config, schema, bucket, or Stage 0 identity mismatch.

## Experimental tracks

1. `revealed_route_proxy_predispatch` is the main research track. The completed
   map-matched route is treated as a revealed assigned-route proxy, dynamic
   inputs stop at departure, and route-token time is estimated causally. This
   track may feed a Stage 3 prototype, but is not a deployable route generator.
2. `oracle_timing_upper_bound` substitutes actual token entry time only for the
   timing diagnostic. It cannot feed Stage 3 or deployment claims.
3. `planned_route_deployable` is a reserved interface for a future route that
   genuinely exists at decision time. It is not produced in v4.

All tracks set `fully_deployable = false` in this release.

## Temporal split

- Train: 20161009–20161024
- Validation-model: 20161025–20161026
- Calibration: 20161027
- Test: 20161031

Test is read only after model selection and calibration are frozen.

## Causality

`decision_time = order_base.departure_time`. A historical event is available
only when:

```text
observation_window_end_time < decision_time
```

Equality, the current order, incomplete same-bin events, and future events are
excluded. Estimated entry time selects forecast horizon and target time-of-day;
it never expands the available information set.

## Route and labels

The complete route skeleton comes from `route_sequence_context` and labels are
left-joined from `traversal_labels`. Unlabelled tokens remain in the sequence.
The physical traversal identity and oracle timing audit come from
`link_traversals`. Preflight records one-to-one or span alignment explicitly.

Formal targets are the four LCS components, reconstructed/equal-weight LCS
baseline, and RTS. IIS, PMIS, and dynamic GNS are excluded. Every component has
an independent mask; unavailable numeric targets remain NaN and unavailable
tails remain nullable boolean NA.

## Timing and model

Estimated token travel time uses strictly historical pace with this fallback:

1. directed edge × estimated time-bin × weekday type;
2. directed edge;
3. highway × estimated time-bin × weekday type;
4. highway;
5. global or static-speed fallback.

Two passes are used. The first uses departure-time bins; the second selects
profiles using first-pass entry estimates.

RC-MSTNet v4 uses Train-only vocabularies and normalization, continuous
overlapping route chunks, padding masks before and after local convolution, a
route Transformer, independent masked component losses, a two-part stop head,
reconstructed LCS consistency, RTS raw regression, and calibrated LCS/RTS tail
heads.

Percentiles are derived only with the frozen Stage 1 Train CDF. LCS order
aggregation uses estimated travel time, while RTS aggregation uses static route
length. No realized direct-observation duration or distance is used as a
decision-time weight.
