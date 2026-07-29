# Stage 1 label schema v3

## Status and boundary

`stage1_label_schema_v3` is the Stage 1 candidate for the frozen Stage 0 v6
Valhalla product `stage1/input_v1`. It is an engineering review candidate, not
a scientifically validated or production-executable release.

The checked-in configuration has `status = review_candidate`. Fit and transform
therefore fail closed unless a caller supplies the explicit
`--allow-review-candidate` engineering override. Production use requires a
separately reviewed configuration marked `frozen_for_execution`; every
threshold section must also be marked frozen.

V3 makes five deliberately narrow decisions:

- only valid `direct_observed` GPS intervals are dynamic truth;
- LCS and RTS are retrospective review-candidate labels;
- GNS is only a proposed, separately frozen static extension and is unavailable
  in the current dynamic pipeline;
- IIS and PMIS are unavailable;
- no numeric cross-dimension composite is emitted.

The machine-readable contract is
[`stage1/config/stage1_label_schema_v3.json`](../config/stage1_label_schema_v3.json).

## Frozen input and temporal split

Stage 1 reads `stage1/input_v1` without modifying it. Eleven Stage 0 products
must exist in every bucket and are content- and schema-hashed. Ten are loaded
and semantically validated. `route_segments` is lineage-only in v3:
`segment_route_distance_m` is explicitly deprecated for features because it is
a component distance, not a local segment distance. It is never summed or used
as segment length. Local route distance must be clipped again from the
corresponding route sequence.

| Split | Dates | Reference use |
|---|---|---|
| Train | 20161009–20161024 | Fit references and CDFs |
| Validation | 20161025–20161027 | Apply frozen train objects only |
| Test | 20161031 | Apply frozen train objects only |

The split is exact and disjoint. All expected dates must be present before
fitting. An `order_id` may occur in only one bucket across all splits. Calendar
features use `Asia/Shanghai`.

## Keys and direct evidence

The logical keys are:

| Product entity | Key |
|---|---|
| Order | `(split, date, order_id)` |
| Route part | `(split, date, order_id, route_sequence)` |
| Traversal | `(split, date, order_id, traversal_id)` |
| GPS interval | `(split, date, order_id, gps_interval_id)` |
| Movement | `(split, date, order_id, movement_sequence)` |

Dynamic supervision comes only from `link_interval_observations` rows satisfying
all of:

```text
measurement_source == "direct_observed"
label_valid is the boolean true
canonical_edge_uid is not null
observed_directed_edge_uid is not null
observed_direction is F or R
interval_start_time and interval_end_time are finite
interval_end_time > interval_start_time
observed_travel_time_s > 0
observed_distance_m >= 0
observed_speed_mps is finite and non-negative
```

The following identities are checked:

```text
observed_travel_time_s
    == interval_end_time - interval_start_time

observed_speed_mps
    == observed_distance_m / observed_travel_time_s
```

`link_traversals.time_observation_valid`, Valhalla elapsed time, and engine
interpolation are never accepted as realized labels. Each direct GPS interval
belongs to exactly one traversal. Non-direct classes remain in
`interval_measurements`, carry provenance, and never receive direct observed
time.

`link_traversals` is visit-aware and route-ordered. A traversal row represents
one continuous physical access to a canonical edge. It is not a guaranteed
sensor measurement of exact link entry and exit. `observation_window_start_time`
and `observation_window_end_time` in `traversal_labels` are the envelope of the
direct GPS intervals assigned to that traversal. Exact direct supervision
remains one row per GPS interval in `interval_labels`.

## Actual directed edge identity

`canonical_edge_uid` is retained as Stage 0 physical lineage. Dynamic modeling,
cohort references, support counts, and the directed graph use
`observed_directed_edge_uid`.

Direction is joined through
`observation -> traversal -> route_part`. When a physical `:F` identity was
actually traversed in direction `R`, Stage 1 creates the corresponding `:R`
identity, preserves actual canonical from/to nodes, inherits length and road
attributes, and sets `synthetic_reverse_edge = true`. The edge is materialized
in `directed_edge_catalog.parquet`; it is not merely a renamed string.

`osm_direction_disagreement` records the old against-oneway evidence but never
removes a label. OSM one-way metadata is not final evidence of actual travel
direction.

Canonical gaps remain in `route_sequence_context` with
`route_lineage_status = unmapped_lineage_gap` and
`sequence_feature_mask = false`. They produce no link supervision and do not
reject their bucket or order. OD nodes are nullable `Int64` and use the fallback
order: order endpoint, first/last route canonical endpoint, first/last OSM
endpoint, then null.

`movement_context` keeps the original `from_edge_uid`/`to_edge_uid` as
lineage and separately maps both sides to
`observed_from_directed_edge_uid`/`observed_to_directed_edge_uid`. A movement
touching an unmapped route part is explicitly `movement_lineage_only` and is
not valid actual-direction topology.

## LCS review candidate

LCS summarizes stop-go and longitudinal-control variation inside one continuous
direct-evidence window of a traversal. Candidate eligibility requires:

- at least 3 direct intervals;
- at least 6 seconds of direct observed time;
- at least 10 metres of direct observed distance;
- no overlap and no internal gap longer than 6 seconds;
- at least 2 valid adjacent acceleration pairs;
- maximum speed at most 75 m/s;
- maximum absolute acceleration at most 8 m/s².

The four components are:

1. low-speed time share below 5 m/s;
2. stop time share at or below 1 m/s;
3. bounded speed coefficient of variation, `x / (x + 1)`;
4. bounded acceleration RMS, `x / (x + 1)`.

Their configured weights are each 0.25 and must sum to one. All four must be
available. A long internal gap makes both the LCS traversal label and its RTS
pace unavailable; disconnected evidence is not blended.

## RTS review candidate

RTS compares direct observed seconds per metre with a train-fitted reference:

```text
excess_time_ratio =
    max(observed_sec_per_m / reference_sec_per_m - 1, 0)

rts_raw = excess_time_ratio / (1 + excess_time_ratio)
```

References use 4,096 fixed bins over `[0.01, 10]` seconds per metre, the median,
and this fallback order:

1. edge × 30-minute bin × weekday type;
2. edge × peak/off-peak;
3. edge;
4. highway × 30-minute bin × weekday type;
5. highway;
6. global.

Non-global cohorts require 100 samples. The global fallback is permitted when
it is non-empty; otherwise RTS stays unavailable. Peak windows in local time are
07:00–09:30 and 17:00–19:30.

For train rows, the reference quantile is leave-one-out: a row cannot establish
or alter the reference used to label itself. Validation and test use the full,
frozen train reference. Raw LCS/RTS CDFs use 1,000 fixed bins; train percentile
normalization is the ordinary full-train empirical self-rank, while validation
and test only apply that frozen train CDF. This distinction is recorded in the
model manifest.

RTS and its reference fit reject a direct window containing a speed above
75 m/s, matching the LCS physical-speed guard.

Edge and edge-hour observation support is fitted from Train only, with each GPS
interval counted in its own start hour. Low support uses road class x hour,
spatial neighbor, then global/hour. Connected-component `upper_region_id` is
reported only as a graph-degeneracy audit and is never a model fallback.
Validation and test never contribute to support counts.

Supported values at exactly 0 or 1 receive their occupied-bin empirical
midrank, rather than being forced to artificial CDF tails.

## Unavailable dimensions and composite

The current pipeline always writes GNS, IIS, and PMIS as unavailable with NA
numeric values.

GNS may later be built as a separately versioned `edge_static_features`
extension keyed by `canonical_edge_uid`. Such an extension must freeze its
canonical network and configuration, prove that an edge has one value across
all partitions, and aggregate over the complete `route_parts.length_m`
exposure. No join or order aggregation for that extension is enabled here, and
GNS is excluded from the v3 core identity.

IIS retains movement identity and provenance in `movement_context`, but Stage 0
does not provide reliable movement delay. PMIS waits for a current-canonical
POI exposure product. Missing exposure is not zero exposure.

`core_composite_status` is the literal `disabled`. No numeric field whose name
starts with `core_composite_` is allowed. Orders retain independent dimension
availability, coverage, reasons, and composition signatures.

## Output products

Each output bucket contains:

- `interval_labels.parquet`;
- `traversal_labels.parquet`;
- `route_sequence_context.parquet`;
- `movement_context.parquet`;
- `order_labels.parquet`;
- `order_label_quality.parquet`;
- `manifest.json`.

The exact required columns and primary keys are shared by the executable schema
constants and the JSON contract. Empty buckets retain those logical columns;
they are not written as zero-column products.

The model directory contains three sparse histogram Parquet files,
`directed_edge_catalog.parquet`, `support_counts.parquet`,
`histogram_metadata.json`, and `model_manifest.json`. The output root contains
`stage1_v3_summary.json`. `verify --report ...` writes the requested audit JSON.
There is no separate all-in-one `stage1_v3_manifest.json`.

## Identity, resume, and audit

The Stage 0 freeze manifest must be `FROZEN`, describe `stage1_input_v1`, and
report PASS coverage and verification. Production recorded a dirty commit
state, so Stage 1 also freezes the independently verified final release tuple:
tag `stage0-v6-final`, commit
`729275d81ec5dc224ac0967a6e600457764607b8`, and source content hash
`a5e482f4a0d2b607`. The bucket code identity must contain that content hash.

All eleven input Parquet files, their physical schemas, each bucket manifest,
and row counts are hashed. A PASS global preflight additionally validates the
complete directed catalog and records hashes for the three products used by
fit. Fit requires that report, verifies it once, then uses a product-pruned
loader for route parts, traversals, and direct observations. The fitted model
binds the preflight manifest and the complete train, validation, and test input
identities, even though its graph catalog, support, references, and
normalization use Train only. Transform and audit reject any byte-level input
change.

Validation/Test directed edges absent from Train are deterministically derived
at transform time, receive zero Train edge support, and enter the frozen
road-class, adjacent-node, or global-hour fallback. They never mutate the
training graph catalog or its audit-only connected-component regions.

Stage 1 hashes its executable v3 source files directly; a caller-provided
`--stage1-code-sha` is only an optional expected-value assertion and cannot
replace the computed content identity.

Resume is fail-closed. A bucket is skipped only when partition identity,
Stage 0 freeze identity, model/config/code identities, exact input hashes,
output product set, file hashes, physical schemas, required columns, and row
counts all match. An existing incomplete or mismatched target is not deleted or
silently recomputed; an operator must inspect and remove it explicitly. New
buckets are published by atomic sibling-directory rename.

Verification checks exact input/output key reconciliation, partition coverage,
foreign identities, direct-only values, two-way missingness, LCS/RTS formulas,
reference metadata, disabled dimensions, disabled composite fields, root
summary totals, and absence of stale extra buckets. PASS means engineering
conformance only.

## Commands

The CLI is:

```text
python -m stage1.v3.cli --config <config> fit \
  --input <stage1/input_v1> \
  --model-root <model_dir> \
  --stage0-freeze-manifest <freeze.json> \
  --validated-preflight <stage1_v3_preflight.json>

python -m stage1.v3.cli --config <config> transform \
  --input <stage1/input_v1> \
  --model-root <model_dir> \
  --output <output_dir> \
  --stage0-freeze-manifest <freeze.json>

python -m stage1.v3.cli --config <config> verify \
  --input <stage1/input_v1> \
  --model-root <model_dir> \
  --output <output_dir> \
  --stage0-freeze-manifest <freeze.json> \
  --report <stage1_v3_audit.json>

python -m stage1.v3.cli --config <config> preflight \
  --input <stage1/input_v1> \
  --report <stage1_v3_preflight.json>
```

The checked-in review-candidate config blocks the first two commands unless the
explicit engineering override is supplied. That override is not a scientific
release.

## Known upstream limitation

The current Stage 0 producer and contract retain a dormant engine interval
category whose source naming is not fully closed
(`engine_interpolated` versus the historical
`engine_allocated_only_time_s` field). The adapter explicitly maps and
reconciles that category for conservation auditing, but it never admits it as
direct dynamic supervision. This review does not modify the active Stage 0
producer.
