# Stage3 ODD-Relevant Condition Vector

Stage3 exports a technology-neutral, pre-dispatch order condition vector. It is not an AV crash-risk, disengagement-risk, or safety-failure probability.

## Semantics

The vector represents trajectory-informed operational stress:

- `pred_stop_go_stress`: predicted stop-and-go / longitudinal control stress.
- `pred_poi_mediated_stress`: predicted POI-mediated interaction stress.
- `pred_reliability_stress`: predicted tail-delay / reliability stress.
- `pred_intersection_stress`: optional IIS movement-derived context.
- `pred_composite_operational_stress`: transparent composite of available predicted dimensions.
- `overall_high_stress_probability`: order-level high-stress probability from Stage3.
- `overall_uncertainty`: predictive uncertainty proxy from Stage2/Stage3.
- `modality_coverage_score`: availability of required/optional prediction modalities.
- `route_prediction_confidence`: pre-dispatch route prediction confidence.

## Field boundary

- Stage3 inputs are held-out Stage2 predictions and pre-dispatch route context.
- Realized Stage1 labels are targets/evaluation-only.
- Actual post-trip behavior and future traffic state are forbidden as Stage3 inputs.

## Availability flags

Dimensions may be marked:

- `available`: production candidate from current data.
- `experimental`: useful but not yet stable enough as a required modality.
- `future_av_calibration_required`: should be calibrated with real AV operational data before vehicle-specific claims.
