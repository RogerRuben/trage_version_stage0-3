# Stage 3 S4 v2 to Stage 4 Contract

Canonical input: `stage3/output/odd_tod/s4/test31_av_operational_suitability.parquet`.

- Key: `(date, order_id, profile_id)`; exactly one row for each Test31 order and C/M/A profile.
- `hard_state`: structural dispatch constraint only. `INFEASIBLE` means a known forbidden direction/maneuver/restriction; `UNKNOWN` means no known hard violation but critical evidence is missing.
- `rho_static`, `rho_dynamic`, `rho_speed`: nonnegative, unclipped envelope-utilization ratios.
- `rho_overall`: `max(rho_static, rho_dynamic, rho_speed)` with no weighted average. It is null when any required family is unevaluable.
- `static_vector` and `dynamic_12_ratios`/`dynamic_vector`: compact JSON objects of the component ratios.
- `reason_codes`: union of structural hard, critical unknown, and `SOFT_*_ENVELOPE_EXCEEDED` diagnostic codes.
- `passenger_acceptance_probability`: nullable Stage4 input placeholder; S4 v2 leaves every value null.
- `original_route`: always the frozen historical Test31 route marker.

Stage4 may combine hard state, continuous utilization, and passenger preference. S4 v2 does not perform dispatch, optimization, rerouting, fallback, or passenger modeling.

S5_AUTHORIZED = NO
