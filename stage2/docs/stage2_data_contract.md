# Stage2 data contract

This document freezes the first Stage2 link-level prediction contract for the Xi'an DiDi compact split experiment.

Current authoritative split is train `20161009-20161015`, validation
`20161016`, and test `20161017`.

Current split:

```text
Train:      20161009–20161015
Validation: 20161016
Test:       20161017
Matcher:    local_topology_fmm
```

The primary Stage2 table is currently an oracle-route upper-bound table, not a
deployable pre-dispatch table:

```text
stage2/output/link_dataset/{train,validation,test}.parquet
```

Each row represents:

```text
order_id × link_id × link_seq
```

## Field classes

Every field belongs to one or more of the following classes.

```text
A. model input
B. label only
C. audit/context descriptor only
D. forbidden leakage column
E. mask / validity flag
```

The key modeling rule is:

```text
Pre-dispatch available features may enter deployable Stage2 models. The current
actual route and actual link entry time are oracle-only. Post-trip realized
trajectory behavior must not enter model inputs. See
`stage2_decision_time_contract.md` for the authoritative availability rules.
```

## Primary keys and split fields

| Field | Class | Use |
|---|---|---|
| `order_id` | C | Order key; grouping, audit, aggregation. Not a model input. |
| `driver_id` | C/D | Audit only. Do not use as a model input in the first baseline. |
| `date` | C | Split/date audit. Do not use raw date as a model input in the first baseline. |
| `link_id` | C | Link key; may be used for historical profile baseline, not directly as a LightGBM feature in the first baseline. |
| `link_seq` | A/C | Route sequence position; allowed when a planned/estimated route is available. |
| `split` | C | Not stored inside the parquet; implied by file name. |

## Pre-dispatch available model-input features

These fields are allowed as first-version Stage2 model inputs.

| Field | Class | Rationale |
|---|---|---|
| `time_bin` | B in current table / A after replacement | Currently based on actual link `enter_time`; deployable data must use estimated entry time. |
| `hour` | B in current table / A after replacement | Currently based on actual link `enter_time`; deployable data must use estimated entry time. |
| `weekday_type` | A | Calendar context. |
| `peak_offpeak` | A | Temporal congestion context. |
| `is_weekend` | A | Calendar context. |
| `road_class` | A | Static road semantics. |
| `link_length_m` | A | Static link geometry. |
| `curvature_deg_per_km_link` | A | Static link geometry. |
| `minor_road` | A/C | Road-class context; not a core GNS monotonicity anchor. |
| `endpoint_degree` | A/C | Local topology context. |
| `link_fragmentation` | A | Static route/link structure. |
| `area_grid` | A | Spatial context. |
| `gns_pct_link` | A/C | Static geometry/navigation stress context; not a primary Stage2 target in this version. |
| `activity_intensity_index` | A/C | POI exposure/context descriptor; not a realized PMIS monotonicity anchor. |
| `poi_density_100m_*` | A/C | Link-level POI exposure. |
| `route_link_count` | B in current table / A after replacement | Currently reconstructed from the completed route; deployable data must use the planned route. |
| `position_ratio` | B in current table / A after replacement | Currently reconstructed from the completed route. |
| `distance_to_destination_ratio` | B in current table / A after replacement | Currently reconstructed from the completed route. |

If route geometry is available before dispatch, route-position features are valid pre-dispatch features. If a later experiment removes precomputed routes, these fields must be reclassified.

## Post-trip realized labels

These fields are labels or label-derived values and must not be model inputs.

| Field | Class | Meaning |
|---|---|---|
| `target_lcs_pct` | B | LCS link percentile target. |
| `target_iis_pct` | B | IIS link percentile target; missing is meaningful and must not be filled with zero. |
| `target_rts_pct` | B | RTS link percentile target. |
| `target_pmis_pct` | B | PMIS link percentile target. |
| `target_high_lcs_90` | B | Binary top-tail LCS target. |
| `target_high_iis_90` | B | Binary top-tail IIS target; valid only where `iis_valid = true`. |
| `target_high_rts_90` | B | Binary top-tail RTS target. |
| `target_high_pmis_90` | B | Binary top-tail PMIS target. |

## Audit descriptors and forbidden leakage columns

These fields may be used for label audit, diagnostics, or error analysis. They must not enter the first Stage2 model feature set.

| Field | Class | Reason |
|---|---|---|
| `matcher_version` | C | Pipeline provenance. |
| `traversal_quality` | C/D | Post-matching quality; use for filtering/audit, not prediction. |
| `observed_or_inferred` | C/D | Post-matching route evidence type. |
| `low_quality_flag` | C/D | Post-matching quality flag. |
| `travel_time_sec` | D | Realized post-trip travel time. |
| `observed_distance_m` | D | Realized/matched traversal output. |
| `reference_travel_time_sec` | D | Used in RTS construction; excluded from first baseline features to avoid target leakage. |
| `excess_time_ratio` | D | Realized RTS primitive. |
| `tail_delay_ratio` | D | Realized RTS primitive. |
| `low_speed_ratio_on_poi_link` | D | Realized PMIS primitive. |
| `stop_time_on_poi_link` | D | Realized PMIS primitive. |
| `delay_on_poi_link` | D | Realized PMIS primitive. |
| `*_cohort_level_used` | C/D | Label normalization provenance; do not use as model input. |
| `*_cohort_sample_size` | C/D | Label normalization provenance; do not use as model input. |

## Validity masks

| Field | Class | Meaning |
|---|---|---|
| `lcs_valid` | E | `target_lcs_pct` is observed. |
| `iis_valid` | E | `target_iis_pct` is observed. IIS missing is not zero. |
| `rts_valid` | E | `target_rts_pct` is observed. |
| `pmis_valid` | E | `target_pmis_pct` is observed. |

Each single-target model must filter by its own validity mask.

## Target definitions

The first Stage2 baseline predicts four link-level continuous targets:

```text
target_lcs_pct
target_iis_pct
target_rts_pct
target_pmis_pct
```

The first binary tail labels are:

```text
target_high_lcs_90
target_high_iis_90
target_high_rts_90
target_high_pmis_90
```

`gns_pct_link` is retained as a static geometry/navigation context feature. It is not the first main prediction target.

## Link-to-order aggregation principle

Order-level Stage3 candidates must preserve multiple aggregation views instead of collapsing immediately to one score:

```text
mean exposure
weighted exposure
max exposure
tail exposure
persistence
route-position exposure
pickup/dropoff-side exposure
intersection-cluster exposure
```

Recommended candidate names:

```text
LCS_score_candidate
IIS_score_candidate
RTS_score_candidate
PMIS_score_candidate
composite_ODD_stress_candidate
uncertainty_candidate
```

Avoid stronger names such as `safety_risk`, `AV_safety`, or `AV_difficulty` until Stage3/Stage4 vehicle-specific transformations are explicitly defined.
