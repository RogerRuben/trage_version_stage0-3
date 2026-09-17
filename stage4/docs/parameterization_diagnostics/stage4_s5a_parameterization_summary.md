# Stage4 S5A Parameterization Diagnostics

Recommendation: `GO_S5B_EXPERIMENTAL_DESIGN`

## Exposure population

Profile-M AV dispatch-ready orders with complete finite evidence: 9258.

| family | N | zero_share | mean | p50 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| static | 9258 | 0.115468 | 2.497456 | 1.264706 | 6.794118 | 9.441176 |
| dynamic | 9258 | 0.532296 | 0.149925 | 0.000000 | 0.520260 | 0.787577 |
| speed | 9258 | 0.998812 | 0.000439 | 0.000000 | 0.000000 | 0.000000 |

## Positive-only exposure

| family | positive_share | positive_mean | positive_p50 | positive_p90 |
| --- | --- | --- | --- | --- |
| static | 0.884532 | 2.823476 | 1.470588 | 8.500000 |
| dynamic | 0.467704 | 0.320555 | 0.192817 | 0.811896 |
| speed | 0.001188 | 0.369318 | 0.437500 | 0.437500 |

## Spearman correlation

| family | static | dynamic | speed |
| --- | --- | --- | --- |
| static | 1.000000 | 0.117812 | 0.000339 |
| dynamic | 0.117812 | 1.000000 | -0.008157 |
| speed | 0.000339 | -0.008157 | 1.000000 |

## Gamma reference regimes

Gamma denotes a cumulative reference-envelope exposure budget, not a safety or risk threshold.

| family | ZERO | MEAN | PATH | UNCONSTRAINED | PATH-MEAN | relative gap |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| static | 0 | 2.194904 | 2.756209 | null | 0.561305 | 0.255731 |
| dynamic | 0 | 0.141510 | 0.401175 | null | 0.259665 | 1.834954 |
| speed | 0 | 0.000000 | 0.000000 | null | 0.000000 | 0.000000 |

## Fleet vehicle-hour scenarios

q_A is active vehicle-hour share. HV session counts are effective service-session units, not a physical HV fleet count.

| requested_q_A | achieved_q_A | AV_vehicle_count | target_HV_vehicle_hours | achieved_HV_vehicle_hours | HV_vehicle_hour_error_pct | HV_template_support_sufficient | selected_HV_session_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.000000 | 0 | 14369.500000 | 12279.336389 | 14.545834 | False | 8435 |
| 0.100000 | 0.100212 | 60 | 12929.500000 | 12279.336389 | 5.028529 | False | 8435 |
| 0.250000 | 0.250531 | 150 | 10769.500000 | 10863.408889 | 0.871989 | True | 7395 |
| 0.500000 | 0.499391 | 299 | 7193.500000 | 7314.384722 | 1.680472 | True | 4939 |
| 0.750000 | 0.749922 | 449 | 3593.500000 | 3648.837500 | 1.539933 | True | 2467 |
| 1.000000 | 1.000452 | 599 | -6.500000 | 0.000000 | N/A | False | 0 |

Rows with HV_template_support_sufficient=false at positive targets saturate all 8,435 frozen effective HV sessions; no synthetic supply or optimizer was introduced.

The q_A=1 target is slightly negative when 24-hour AV count rounding overshoots H_base; no HV sessions are selected and the relative HV error is reported as N/A.

## Interpretation

No dispatch scenario, service-rate comparison, Gamma calibration, passenger-preference estimate, or cost-ratio experiment was run. S5A provides diagnostic evidence only.

Diagnostic runtime: 1.297s.
