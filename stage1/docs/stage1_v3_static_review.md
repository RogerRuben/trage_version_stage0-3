# Stage 1 v3 static review

Date: 2026-07-29

## Review boundary and current evidence

This review covers the Stage 1 code path that consumes the immutable Stage 0 v6
product at `stage1/input_v1`. The implementation remains an engineering review
candidate, not a claim of scientific validity or model usefulness. It now has
unit-test and real-bucket loader evidence; the global preflight result is
reported separately.

The completed read-only global preflight passed with 220,000 unique orders and
18,604,112 direct observations. It measured 16,894,460 actual-F and 1,709,652
actual-R observations, exactly 711,914 UID/direction mismatches converted to
synthetic reverse identities, 789,379 OSM direction disagreements retained,
and 956 unmapped traversals with zero direct labels. All OD endpoints were
resolved by the nullable fallback chain. The machine-readable evidence is
`stage1_v3_preflight.json`.

## Effective product interpretation

The Stage 0 input contains two related but different dynamic products:

- `link_interval_observations` is the direct-label table. One row is one valid
  GPS interval assigned to one traversal. Its start and end timestamps are the
  actual observation boundaries used for direct supervision.
- `link_traversals` is the visit-aware route table. One row is one continuous
  physical visit to a canonical edge, ordered by `route_sequence`. It is not a
  guarantee of exact link entry and exit time.

Stage 1 preserves this distinction:

- `interval_labels` retains the direct GPS interval timestamps and observed
  time, distance, and speed;
- `traversal_labels` aggregates intervals for a single traversal and records
  the minimum and maximum direct-observation timestamps as an evidence window;
- `route_parts` and `link_traversals` provide the ordered canonical route and
  visit identity;
- `order_labels` contains order-level summaries, not a replacement for the
  interval or traversal tables.

Therefore the stored form is route-ordered and visit-aware, but the traversal
window must not be interpreted as exact entry/exit time for the full link.

## Static findings addressed

### Direct evidence and accounting

Only rows with `measurement_source == direct_observed` and a strict boolean
`label_valid == true` enter dynamic labels. Timestamp duration, distance, and
speed identities are checked. Integer or string lookalikes for boolean true are
rejected.

The adapter reconciles interval classification, traversal totals, route
distance, order dynamic totals, duplicate allocation counters, and the Stage 0
conservation flags. Non-direct and unresolved intervals cannot acquire observed
time in Stage 1.

### Leakage and reproducibility

Reference pace and normalization distributions are fitted from train dates
only. Train RTS reference application is leave-one-out; validation and test use
the complete frozen train reference. The fitted model binds the exact content,
schema, and row-count identities of all input buckets. Transform rejects any
input drift.

The executable v3 source tree is content-hashed. A caller-supplied code identity
is only an expected-value assertion. Resume checks exact input, model, config,
code, freeze, product, file, schema, column, and row-count identities.

The final Stage 0 release identity is recorded as tag `stage0-v6-final`, commit
`729275d81ec5dc224ac0967a6e600457764607b8`, and source content hash
`a5e482f4a0d2b607`. This resolves the production manifest's dirty-worktree
record by checking executable content equivalence rather than pretending the
record was clean.

### Direction identity, nullable input, and lineage gaps

Dynamic labels now use `observed_directed_edge_uid`, derived through
observation-to-traversal-to-route-part joins. F-to-R observations become actual
`:R` identities. Existing `:R` observations are retained even when the legacy
against-oneway flag is true. Synthetic reverse identities are materialized as
graph edges with actual endpoints and inherited static attributes.

OD and canonical node columns are normalized to nullable `Int64`, independent
of physical Parquet dtype. Unmapped route parts and traversals remain as masked
lineage gaps and cannot generate direct link labels.

### Support and segment-distance safety

Edge and edge-hour support counts are fitted from Train only. Low-support
samples route to road-class/hour, spatial-neighbor, upper-region, then
global/hour support. `route_segments.segment_route_distance_m` is declared
lineage-only and deprecated for features; local distance must be reconstructed
from route sequence.

### LCS continuity defect

The reviewed implementation originally referenced LCS component weights from
the wrong function scope. A continuous eligible traversal would have raised a
runtime name error. Weight construction and validation are now shared by the
interval and traversal builders, and a regression test describes the eligible
continuous-window case.

Internal direct-observation gaps longer than the configured limit make both LCS
and RTS pace unavailable. Disconnected observations are not blended into one
continuous traversal label.

### Output schema and audit strength

Empty buckets retain declared logical columns. Output columns are now an exact
contract rather than a minimum subset. Undeclared fields and missing fields both
fail validation and resume.

The audit checks source/output key reconciliation, direct-only provenance,
strict boolean types, formulas, missingness, frozen reference and CDF identity,
disabled dimensions, order aggregation regenerated from source products, root
summary totals, and stale extra buckets.

Legacy tolerance override sections were removed from the executable contract.
All active absolute and relative tolerances are declared in the v3 configuration
or frozen in the v3 code path.

## Reasonableness assessment

The post-review engineering structure is reasonable for producing two
independent retrospective candidates:

- LCS: traversal-local stop-go and longitudinal variation;
- RTS: traversal pace relative to a train-fitted cohort reference.

It is also reasonable to keep GNS, IIS, and PMIS unavailable. The current input
does not provide a frozen static-edge feature extension, reliable movement
delay, or current-canonical POI exposure. Treating their absence as zero would
create false labels. No numeric cross-dimension composite is emitted.

The following claims are intentionally not made:

- the LCS thresholds or weights are scientifically optimal;
- RTS is stable across bin counts, cohort support thresholds, or temporal
  regimes;
- either label improves a Stage 1 prediction model;
- an observation-window boundary is a physical link boundary time;
- full-scale fit, transform, and audit throughput or peak memory is acceptable.

Those require execution, distribution reports, sensitivity analysis, repeated
traversal reliability checks, and modeling validation.

## Current release blockers

- The checked-in configuration is `review_candidate`; fit and transform block
  unless an explicit engineering override is supplied.
- Model and output manifests remain `scientific_status = NOT_VALIDATED`.
- The checked-in support and label thresholds remain review candidates.
- Full fit/transform throughput has not been accepted as a scientific or
  production-performance gate.

The code should be reviewed as a candidate before any execution authorization.
