# Stage1 label v1/v2 smoke comparison

This is a 1,000-order-per-day engineering comparison, not a formal label-validation study.

- All four dates have 1,000 overlapping orders.
- The largest semantic change is GNS (mean v2-v1 approximately -0.24; near-zero rank correlation).
- LCS, RTS, and PMIS retain moderate rank association but are not interchangeable with v1.
- IIS remains a conditional observed label; missing values are never replaced with zero.
- Core composite v2 uses LCS/GNS/RTS. PMIS is excluded from the equal-weight core to prevent duplicate interaction weighting.

Machine-readable evidence: `docs/pipeline_rebaseline/stage1_v1_v2_comparison.csv`.
