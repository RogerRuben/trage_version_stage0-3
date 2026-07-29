# Stage 1 prediction labels

## Current Stage 0 v6 candidate

`stage1_label_schema_v3` is the only Stage 1 label candidate intended to consume
the Valhalla-based `stage1/input_v1` product.

- Machine-readable contract:
  [`config/stage1_label_schema_v3.json`](config/stage1_label_schema_v3.json)
- Semantics and acceptance boundary:
  [`docs/stage1_label_schema_v3.md`](docs/stage1_label_schema_v3.md)
- Static code and reasonableness review:
  [`docs/stage1_v3_static_review.md`](docs/stage1_v3_static_review.md)

V3 is a **review candidate**, not a frozen scientific release. It uses only
`link_interval_observations` rows satisfying the direct-observation predicate
for dynamic labels.

Its initial scope is intentionally narrow:

- LCS and RTS are review candidates.
- GNS is a proposed, separately hashed external static edge extension; the
  current dynamic pipeline keeps it unavailable and excludes it from the core.
- IIS is unavailable because Stage 0 v6 does not provide reliable movement
  delay.
- PMIS is unavailable until current-canonical POI exposure is frozen.
- No core composite is emitted.

The review-candidate aggregation tail threshold is 0.90. RTS cohort fallback
is fixed to edge/time/weekday, edge/peak, edge, highway/time/weekday, highway,
then global.

Dynamic edge identity follows actual travel direction through
`observed_directed_edge_uid`; the original `canonical_edge_uid` remains
lineage. F-to-R observations create real reverse graph edges with actual
endpoints and inherited static attributes. OSM one-way disagreement is audited
but does not delete labels.

The v3 split is fixed:

- train: 20161009 through 20161024;
- validation: 20161025 through 20161027;
- test: 20161031 only.

References and normalization objects are fitted on train dates only.
Train RTS references use leave-one-out application; validation and test only
apply the full frozen train objects.

Input acceptance reconciles both `interval_measurements` and
`turn_movements`, and requires content plus schema hashes for every required
Stage 0 v6 product. Resume is permitted only for a PASS bucket whose declared
input, freeze, model, source-content, schema, row-count, and output hashes all
still match. Existing partial/mismatched targets fail closed and require
explicit operator inspection.

The implementation lives in `stage1/v3/` and is exposed through:

```text
python -m stage1.v3.cli --config stage1/config/stage1_label_schema_v3.json fit ...
python -m stage1.v3.cli --config stage1/config/stage1_label_schema_v3.json transform ...
python -m stage1.v3.cli --config stage1/config/stage1_label_schema_v3.json verify ...
python -m stage1.v3.cli --config stage1/config/stage1_label_schema_v3.json preflight ...
```

Fit and transform require the Stage 0 freeze manifest. The executable v3 source
tree is content-hashed automatically; `--stage1-code-sha` is only an optional
expected-value assertion. The checked-in config is `review_candidate`, so fit
and transform refuse to run without the explicit engineering-only
`--allow-review-candidate` override.

## Legacy v1/v2

The v1 and v2 code, schema, manifests, reports, and the 7+1+1
`split_config.json` describe earlier Stage 0/FMM engineering experiments. They
remain in the repository for reproducibility, but are **legacy for Stage 0 v6**.

In particular, the old builders expect fields such as `link_id`, `link_seq`,
`travel_time_sec`, and pre-aggregated movement/kinematic measurements. Those
semantics do not match `stage1/input_v1`, whose canonical IDs and dynamic truth
are carried by `canonical_edge_uid`, `traversal_id`, and direct GPS interval
observations.

Do not run `build_stage1_labels.py` or `build_stage1_labels_v2.py` directly on
Stage 0 v6 outputs. The v3 implementation must pass its engineering audit and
be separately frozen before label production begins.

Legacy outputs under `stage1/output/prediction_split/` and canonical smoke
artifacts are not evidence for v3 input compatibility or v3 scientific
validity.
