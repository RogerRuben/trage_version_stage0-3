# Stage 1 to Stage 2 contract

## Primary unit

Stage 2 treats `(order_id, traversal_id)` as the primary supervised unit.
Order-level labels are auxiliary summaries and descriptive targets.

## Allowed targets

LCS should preferentially retain the component vector:

```text
crawl_time_share
stop_time_share
speed_cv_bounded
acceleration_rms_bounded
```

`lcs_raw` and `lcs_pct` may also be modeled, but `lcs_raw` is an equal-weight
baseline rather than a unique true target.

RTS targets may include:

```text
rts_raw
rts_pct
reference_level_used
reference_sample_size
rts_measurement_available
```

Raw values represent absolute realized intensity. Percentiles represent
conditional position under the frozen Train distribution. They are not
interchangeable.

## Required masks

Stage 2 must honor `lcs_available`, `rts_available`,
`rts_measurement_available`, nullable tail fields, sequence masks, and
unavailable reasons. Missing labels must not be zero-filled. Unmapped lineage
parts remain masked, and evaluation-unseen edges use the frozen fallback path.

## Forbidden prediction-time inputs

At the decision time for an order, Stage 2 must not use:

- direct intervals observed after that decision;
- true arrival time;
- posterior realized speed or travel time;
- any realized Stage 1 target from the same order;
- Validation or Test observations to fit a catalog, support, reference, CDF,
  normalization, imputer, or feature vocabulary;
- numeric zero as a replacement for an unavailable label.

## Provenance

Every Stage 2 run must bind to the Stage 1 release manifest, model ID, config
SHA, code content SHA, output schema version, split/date contract, and the
frozen Stage 0 release identity.
