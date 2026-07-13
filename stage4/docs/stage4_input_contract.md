# Stage4 Input Contract

Stage4 remains on hold until Stage3 has at least three strict rolling test folds,
fixed missing-modality behavior, validation-only calibration, and leakage-audited
prediction warehouses.

Each Stage4 candidate order row must contain only deployable predictions:

- `order_id`
- `decision_time`
- `origin_lon`, `origin_lat`
- `destination_lon`, `destination_lat`
- `route_id` or `matched_route_proxy_id`
- `route_proxy_type = observed_matched_service_route_proxy`
- `lcs_expected`, `lcs_tail_probability`, `lcs_uncertainty`
- `pmis_expected`, `pmis_tail_probability`, `pmis_uncertainty`
- `rts_expected`, `rts_tail_probability`, `rts_uncertainty`
- `core_overall_high_stress_probability`
- `extended_overall_high_stress_probability`
- `iis_applicability`, `iis_severity`, `iis_tail_probability`
- `iis_availability`, `iis_uncertainty`
- `overall_uncertainty`
- `modality_coverage_score`
- `route_prediction_confidence`

Forbidden fields:

- actual realized stress
- actual link entry time
- post-trip behavior features
- future traffic state
- Stage1 labels as model inputs
- Stage2 in-sample predictions

Stage4 readiness status is `READY_FOR_COUNTERFACTUAL_SIMULATION` only after the
Stage3 rolling and uncertainty gates pass. It is not a real deployment-readiness
claim.

## Capability mapping layer

Stage4 consumes a multi-dimensional, technology-neutral condition vector. It
must not reinterpret Stage3 outputs as AV crash risk, disengagement probability,
or real ADS safety failure probability.

Vehicle-specific behavior enters through explicit scenario profiles:

- `conservative_av`
- `moderate_av`
- `mature_av`
- `reference_hv`

These profiles define stress sensitivities, ODD thresholds, uncertainty
tolerance, and placeholder remote-assistance/fallback costs. They are scenario
parameters until real AV operational data are available for calibration.

IIS missingness must be treated as unknown intersection information, not zero
intersection stress. Capability mapping therefore uses availability-aware
aggregation and an explicit missing-modality uncertainty penalty.
