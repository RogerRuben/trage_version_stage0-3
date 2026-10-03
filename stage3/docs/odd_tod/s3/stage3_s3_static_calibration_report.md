# Stage 3 S3 Static Calibration Report

Unique Train-exposed complexes: **2,425** (support gate: 1,000; PASS).

Static caps use unique Train-exposed complexes, not demand-weighted encounter frequency. Every complex contributes once. The only baseline dimensions are A/M/D/L; member count, QA flags, confidence, bridge, tunnel, and layer are not capability caps.

`D_c` is the number of unique `valhalla_road_class` values on the complex's unique `INCOMING`/`OUTGOING` boundary edges. `INTERNAL` edges are explicitly excluded. The former S2B internal-edge diversity is retained only as `s2b_internal_road_class_diversity_qa` provenance and is not calibrated.

## Frozen thresholds (`higher`)

- `C`: `{"A_c": 4.0, "D_c": 2.0, "L_c": 10.0, "M_c": 9.0}`
- `M`: `{"A_c": 5.0, "D_c": 3.0, "L_c": 34.0, "M_c": 16.0}`
- `A`: `{"A_c": 9.0, "D_c": 3.0, "L_c": 73.0, "M_c": 25.0}`

Signal state distribution: `{"ROUNDABOUT": 6, "SIGNALIZED": 173, "UNKNOWN_CONTROL": 2246}`. Roundabout share: 0.247423%. Grade-separation-evidence share: 0.536082%.

Distribution summary: `{"A_c": {"count": 2425, "max": 16.0, "min": 2.0, "p25": 3.0, "p50": 3.0, "p75": 4.0, "p90": 5.0, "p975": 9.0}, "D_c": {"count": 2425, "max": 5.0, "min": 1.0, "p25": 1.0, "p50": 2.0, "p75": 2.0, "p90": 3.0, "p975": 3.0}, "L_c": {"count": 2425, "max": 456.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 10.0, "p90": 34.0, "p975": 73.0}, "M_c": {"count": 2425, "max": 64.0, "min": 1.0, "p25": 4.0, "p50": 9.0, "p75": 9.0, "p90": 16.0, "p975": 25.0}}`.

Advanced strict-tail support: `{"A_c": 56, "D_c": 23, "L_c": 60, "M_c": 52}`.
