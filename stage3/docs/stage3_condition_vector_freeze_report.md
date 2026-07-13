# Stage3 Condition Vector Freeze Report

This report freezes the Stage3-to-Stage4 condition vector used by the
ODD-constrained pricing and dynamic AV/HV dispatch experiments.

## Final semantic fixes

1. Core overall probability is trained against `core_overall_high_stress`,
   defined as `LCS_tail OR PMIS_tail OR RTS_tail`.
2. Extended overall probability is trained against
   `extended_overall_high_stress`, defined as
   `core_overall_high_stress OR IIS_tail`.
3. `extended_overall_high_stress_probability` is no longer constructed as
   `max(core_probability, IIS_tail_probability)`. It is exported from the
   Core+IIS Stage3 model trained on the extended label.
4. `lcs_expected`, `pmis_expected`, and `rts_expected` are exported from
   Stage3 continuous `pred_raw` outputs, not from route-level q90 features.
5. `decision_time` is the first estimated service-route link entry time. If
   that value is missing, the exporter falls back to `origin_timestamp` and
   records `decision_time_source`.
6. IIS has no standalone `iis_uncertainty` field in the current export. IIS
   uncertainty is represented by `iis_applicability`, `iis_availability`,
   `iis_coverage_quality`, and `modality_coverage_score`.

## Field status

| Field | Source | Probability? | Expected value? | Dispatch input? |
| --- | --- | ---: | ---: | ---: |
| `lcs_expected` | Core Stage3 `pred_raw` | No | Yes | Yes |
| `lcs_tail_probability` | Core Stage3 LCS tail head | Yes | No | Yes |
| `pmis_expected` | Core Stage3 `pred_raw` | No | Yes | Yes |
| `pmis_tail_probability` | Core Stage3 PMIS tail head | Yes | No | Yes |
| `rts_expected` | Core Stage3 `pred_raw` | No | Yes | Yes |
| `rts_tail_probability` | Core Stage3 RTS tail head | Yes | No | Yes |
| `core_overall_high_stress_probability` | Core Stage3 overall head | Yes | No | Yes |
| `extended_overall_high_stress_probability` | Core+IIS Stage3 extended head | Yes | No | Experimental |
| `iis_applicability` | Movement IIS aggregation | Yes-like score | No | Optional |
| `iis_severity` | Movement IIS severity aggregation | No | Severity proxy | Optional |
| `iis_tail_probability` | Movement IIS tail aggregation | Yes-like score | No | Optional |
| `overall_uncertainty` | LCS/PMIS/RTS uncertainty aggregate | No | No | Yes |

## Audit result

`stage3/scripts/audit_stage3_stage4_export.py` passed on
`stage3/output/stage4_inputs_final`:

- fold 1: 15,000 orders
- fold 2: 15,000 orders
- fold 3: 15,000 orders
- decision-time parse success: 100%
- origin/destination missing: 0
- duplicate order keys: 0
- forbidden realized/post-trip/future fields: 0

The Stage3 condition vector is frozen for the current Stage4 scenario
experiments. Further changes should be treated as a new schema version.
