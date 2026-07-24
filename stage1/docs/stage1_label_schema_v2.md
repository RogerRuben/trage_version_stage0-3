# Stage 1 label schema v2

V2 replaces the exploratory v1 fitting and composition semantics. It does not
overwrite v1 artifacts.

- Reference quantiles use fixed global histogram edges and merge integer counts.
  The result is invariant to input partition count and processing order at the
  declared bin resolution; partition medians are never averaged.
- Cohort normalization evaluates a monotone empirical CDF over ordered observed
  support. Empty internal bins are interpolated, values below/above support map
  to the lower/upper tail, and missing raw values remain missing. There is no
  `fillna(0.5)` rule.
- LCS, GNS and RTS form the equal-weight core composite. IIS remains a
  conditional movement-level modality and PMIS remains a
  separately reported activity-behavior interaction descriptor; both are excluded
  from that composite, preventing its LCS/RTS inputs from being counted twice.
- Missing modalities are not zero stress. Every order contains per-dimension
  availability, a JSON `dimension_mask`, `valid_dimension_count`, and an explicit
  `composition_signature` so differently observed orders are not silently treated
  as equivalent.
- IIS uses the Stage 0 upstream 75 m intersection influence-zone allocation.

The schema is frozen in `stage1/config/stage1_label_schema_v2.json`.
