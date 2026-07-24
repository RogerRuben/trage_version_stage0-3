# Stage 0 v4 Core failure analysis

Status: diagnostic complete; canonical promotion remains **HOLD** pending independent review.

## Scope

The analysis covers the fixed 1,000-order diagnostics for 2016-10-19, 2016-10-20,
2016-10-22, and 2016-10-23 after the connector direction correction. Of 4,000 input
orders, 3,938 reconstructed successfully. The complete per-date counts are in
`stage0_v4_core_failure_breakdown.csv`.

## Main result

The frozen v4 thresholds yield 589 Strict Core candidates (14.96% of reconstructed
orders) and 643 preliminary Analysis Set candidates (16.33%). The difference of 54
orders is attributable only to soft quality flags.

| Condition | Type | Failed | Rate | Unique failures | Joint failures |
|---|---:|---:|---:|---:|---:|
| U-turn | hard diagnostic | 3,086 | 78.36% | 1,427 | 1,659 |
| Unreasonable detour | hard diagnostic | 1,519 | 38.57% | 63 | 1,456 |
| Direction continuity | hard diagnostic | 660 | 16.76% | 25 | 635 |
| Repeated-link share | soft | 280 | 7.11% | 4 | 276 |
| Route-length ratio | soft | 252 | 6.40% | 45 | 207 |
| Match confidence | soft | 161 | 4.09% | 2 | 159 |
| Interpolated-distance share | soft | 71 | 1.80% | 0 | 71 |
| Minimum route links | hard diagnostic | 64 | 1.63% | 19 | 45 |
| Projection, fallback, and OD endpoint bounds | mixed | 0 | 0.00% | 0 | 0 |

Core attrition is therefore not primarily caused by projection distance, fallback,
confidence, interpolation, or other soft conditions. It is dominated by structural
U-turn and detour detectors. These are automated diagnostic flags, not adjudicated
major errors. The v2 review pack deliberately oversamples 78 Strict Core and 72
boundary/rejected routes so that the detector precision can be checked without
changing the matcher or searching thresholds.

## Decision

No network v5 or threshold search is justified from this table alone. If independent
review shows that the U-turn/detour flags identify map-data ambiguity rather than
systematic route errors, the three-tier usage policy is retained and limitations are
documented. Stage 0 is reopened only if review finds systematic direction, grade,
cross-level, or severe detour errors under the registered gates.
