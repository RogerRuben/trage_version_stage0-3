# Stage 1 canonical contract

## Research task

Stage 1 derives five retrospective, technology-neutral operational-stress
measurements from audited Stage 0 observations: LCS, IIS, GNS, RTS, and PMIS.

## Inputs

- Canonical Stage 0 link traversals, movements, OD, and quality flags.
- A versioned `stage1_label_schema_v2` definition.
- Fit-date manifest for all empirical distributions and normalization objects.

## Outputs

- Link/movement measurements with validity and observability flags.
- Order-level components and documented aggregation statistics.
- Normalization/CDF objects with support counts and extrapolation policy.
- Label manifest with schema version, fit dates, and source lineage.

## Allowed information

Retrospective trip observations may be used to create evaluation/training labels.
The resulting realized values may only be used as targets or evaluation-only fields
in later stages.

## Forbidden information

- Filling missing IIS severity with zero.
- Reusing realized Stage 1 values as Stage 2/3/4 dispatch features.
- Approximate global medians/quantiles without a declared error bound.
- Empty-bin CDF values without an explicit backoff and support count.
- Silent double weighting of LCS/RTS components inside PMIS/composites.

## Acceptance rules

For every dimension, schema v2 freezes physical meaning, formula, observation
window, weights, missingness, normalization, and order aggregation. Audits report:

- exact or proven-consistent quantile computation;
- CDF monotonicity, tail extrapolation, empty-bin backoff, and sample support;
- cross-dimension overlap/double-weight diagnostics;
- comparability under missing modalities;
- IIS applicability and intersection influence-zone validity.

Version v2 is written alongside v1 and never overwrites it.

## Dimension definitions frozen for v2

- **LCS**: observed stop/go and longitudinal control stress from valid link
  traversal kinematics.
- **IIS**: conditional intersection-interaction stress measured only inside the
  declared upstream influence area; applicability and severity are separate.
- **GNS**: geometry/network structural stress from frozen road attributes.
- **RTS**: realized excess travel-time stress relative to a fit-period cohort
  reference.
- **PMIS**: POI/activity interaction feature. In v2 it is excluded from the
  equal-weight core composite to prevent LCS/RTS double weighting.

All are HV operational-stress proxies, not AV crash, disengagement, takeover, or
safety ground truth.

## Time convention

Stage 1 values are produced after `T_complete` from retrospective observations.
They may be model targets/evaluation fields only.

## Missing and fallback rules

Validity is dimension-specific. IIS non-applicability differs from unavailable
severity. Cohort fallback records requested and fallback support separately.
Core composite comparison requires the fixed LCS/GNS/RTS mask; other masks have a
distinct `composition_signature`.

## Version and downstream consumers

Schema version: `stage1_label_schema_v2`. Consumers are Stage 2 training targets,
Stage 3 evaluation targets, and mathematical audits. Stage 4 cannot read realized
Stage 1 labels.
