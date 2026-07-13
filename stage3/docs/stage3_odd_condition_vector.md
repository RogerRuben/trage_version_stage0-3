# Stage3 ODD-Relevant Condition Vector

Stage3 exports a technology-neutral, pre-dispatch order condition vector. It is not an AV crash-risk, disengagement-risk, or safety-failure probability.

## Semantics

The vector represents trajectory-informed operational stress:

- `lcs_expected`, `pmis_expected`, `rts_expected`: predicted continuous operational-stress levels.
- `lcs_tail_probability`, `pmis_tail_probability`, `rts_tail_probability`: probabilities of exceeding train-period high-stress thresholds.
- `lcs_uncertainty`, `pmis_uncertainty`, `rts_uncertainty`: predictive uncertainty proxies.
- `intersection_applicability`, `intersection_severity`, `intersection_tail_probability`: optional IIS movement-derived context.
- `composite_expected`: transparent aggregation of LCS/PMIS/RTS expected values.
- `core_overall_high_stress_probability`: Stage3 probability from a Core model trained on LCS OR PMIS OR RTS high stress. This is the main target.
- `extended_overall_high_stress_probability`: Stage3 probability from a Core+IIS model trained on core overall OR IIS tail. It is not constructed by taking the maximum of core and IIS scores.
- `overall_uncertainty`: predictive uncertainty proxy from Stage2/Stage3.
- `modality_coverage_score`: availability of required/optional prediction modalities.
- `route_prediction_confidence`: pre-dispatch route prediction confidence.

## Field boundary

- Stage3 inputs are held-out Stage2 predictions and pre-dispatch route context.
- Realized Stage1 labels are targets/evaluation-only.
- Actual post-trip behavior and future traffic state are forbidden as Stage3 inputs.
- `route_id` is an observed/matched service-route proxy, not a platform-generated multi-candidate route set.
- `decision_time` is the first estimated service-route link entry time, with OD origin timestamp as the nearest available order-start proxy when needed.

## Availability flags

Dimensions may be marked:

- `available`: production candidate from current data.
- `experimental`: useful but not yet stable enough as a required modality.
- `not_implemented_due_to_current_data_limit`: first-layer condition not currently supported by reliable data.

`geometry_complexity` is currently `not_implemented_due_to_current_data_limit`, rather than an AV-calibration item.
