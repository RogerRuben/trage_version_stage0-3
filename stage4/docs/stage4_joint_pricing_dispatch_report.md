# Stage4 Joint Pricing and Dispatch Report

## Scope

This stage implements ODD-constrained pricing-and-dispatch coordination for
mixed AV/HV ride-hailing fleets. Stage3 outputs are interpreted as
trajectory-informed operational stress and ODD-relevant condition vectors, not
as AV crash risk or empirical ADS failure probabilities.

## Stage3 semantic freeze

The Stage3 export now separates:

- `core_overall_high_stress_probability`: Core model trained on LCS/PMIS/RTS
  high-stress labels.
- `extended_overall_high_stress_probability`: Core+IIS model trained on
  extended labels that include IIS.
- `*_expected`: continuous Stage3 `pred_raw`, not q90.
- `decision_time`: first estimated route-link entry time with
  `origin_timestamp` fallback.

The Stage3 export audit passed for 45,000 orders across three folds.

## Capability mapping

AV capability mapping uses scenario-prior profiles. The final feasible shares
over the Stage4 input rows are:

| profile | feasible share |
| --- | ---: |
| intersection_sensitive_av | 0.609 |
| conservative_av | 0.626 |
| uncertainty_sensitive_av | 0.698 |
| moderate_av | 0.869 |
| mature_av | 0.955 |
| reference_hv | 1.000 |

The ODD margin is now `threshold - condition value` by dimension, combined with
an uncertainty margin. Missing-modality penalties enter AV feasibility rather
than only cost.

## Experiment run

The committed Stage4 result files use three folds, 1,500 orders per fold, and
69 scenarios. Full-fold inputs are available and the simulator supports
`--max-orders-per-fold 0`; the sample run is used here to keep repository
iteration time bounded.

## Strategy comparison

Mean across folds for strategy scenarios:

| Strategy | Match | Cancel | Platform profit | HV stress | AV share | AV ODD violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.886 | 0.114 | 36,217 | 0.369 | 0.520 | 0.131 |
| Nearest | 0.884 | 0.116 | 36,320 | 0.370 | 0.522 | 0.138 |
| GlobalMatch-MinPickup | 0.886 | 0.114 | 36,419 | 0.369 | 0.523 | 0.135 |
| GlobalMatch-MinOperatingCost | 0.886 | 0.114 | 44,995 | 0.375 | 0.958 | 0.134 |
| Cost-only | 0.886 | 0.114 | 45,001 | 0.379 | 0.954 | 0.135 |
| Simple Risk Penalty | 0.885 | 0.115 | 27,454 | 0.368 | 0.075 | 0.107 |
| ODD Gate Only | 0.883 | 0.117 | 34,473 | 0.380 | 0.439 | 0.000 |
| ODD-Gated Price-Aware | 0.877 | 0.123 | 41,634 | 0.433 | 0.824 | 0.000 |
| Three-Stakeholder Balanced | 0.877 | 0.123 | 41,839 | 0.437 | 0.834 | 0.000 |

ODD-constrained strategies eliminate AV hard ODD violations by construction.
They trade a small match-rate reduction for feasibility compliance and higher
AV assignment share under price-aware matching.

## Supply scenarios

For the base ODD-Gated Price-Aware mechanism:

| Supply | Match | Cancel | Platform profit |
| --- | ---: | ---: | ---: |
| abundant | 0.873 | 0.127 | 42,128 |
| moderate | 0.877 | 0.123 | 41,634 |
| tight | 0.879 | 0.121 | 41,367 |

The current reconstructed-supply sample does not produce a monotone supply
curve; this is a known limitation of using scenario reconstruction rather than
observed idle cruising traces.

## AV penetration

| AV share | Match | Cancel | Platform profit | HV net income | AV assignment share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.882 | 0.118 | 25,800 | 21,630 | 0.000 |
| 25% | 0.884 | 0.116 | 39,172 | 7,012 | 0.683 |
| 75% | 0.868 | 0.132 | 42,143 | 2,687 | 0.875 |

Higher AV penetration increases platform profit in this cost scenario but
reduces HV income exposure. This is a scenario result, not a welfare conclusion
about real platforms.

## ODD profiles

| ODD profile | Match | Cancel | Profit | HV income | AV share |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | 0.874 | 0.126 | 36,921 | 8,570 | 0.600 |
| intersection-sensitive | 0.875 | 0.125 | 36,455 | 9,367 | 0.582 |
| mature | 0.881 | 0.119 | 43,586 | 2,129 | 0.905 |
| uncertainty-sensitive | 0.877 | 0.123 | 38,343 | 7,157 | 0.665 |

ODD profile strictness changes the AV/HV feasible domain and therefore the
residual burden assigned to HVs.

## Pricing mechanisms

| Pricing | Match | Cancel | Mean fare | Passenger GC | Profit | HV income |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 uniform | 0.879 | 0.121 | 44.90 | 45.85 | 42,232 | 2,894 |
| P1 platform-funded | 0.878 | 0.122 | 44.92 | 45.83 | 40,596 | 3,685 |
| P2 passenger-funded | 0.872 | 0.128 | 46.96 | 47.92 | 43,833 | 3,854 |
| P4 AV discount + HV comp | 0.884 | 0.116 | 43.80 | 44.72 | 39,143 | 4,002 |
| P5 balanced | 0.862 | 0.138 | 43.88 | 44.82 | 38,785 | 3,844 |

Passenger-funded compensation raises fares and generalized cost. Platform-funded
and shared mechanisms shift burden between platform margin and HV compensation.

## Audits

All submitted audits passed:

- Stage3 export audit
- capability mapping audit
- matching feasibility audit
- pricing accounting audit
- scenario comparability audit
- dynamic state consistency audit

## Limitations

1. The committed result run uses a 1,500-order-per-fold scenario sample for
   runtime control; full inputs are available but require a longer foreground
   run.
2. Supply is reconstructed from observed service orders, not true idle vehicle
   traces.
3. AV capability profiles are scenario priors, not empirical ADS performance
   estimates.
4. ODD stress is an operational condition vector, not crash or disengagement
   risk.

## Next step

If a longer machine run is acceptable, execute the same simulator with
`--max-orders-per-fold 0` and regenerate `stage4/docs/results` and figures.
The mechanism code and audits are now in place for that full-fold run.
