# Field availability contract

## Canonical event times

- `T_request`: latent/scenario request arrival.
- `T_decision`: information cutoff for dispatch prediction and assignment.
- `T_pickup`: passenger boarding / service-route start.
- `T_link_entry`: realized or predicted link-entry event.
- `T_complete`: observed trip completion.

## Rules

1. Every canonical field appears in `field_availability_registry.csv`.
2. Stage 2 dispatch features satisfy `availability_time <= T_decision` for all
   route links; future estimated link entry never changes that cutoff.
3. Stage 3 features originate from held-out Stage 2 dispatch predictions.
4. Stage 4 counterfactual inputs cannot originate at or after `T_complete`.
5. Test-population statistics cannot enter training, calibration, profile setting,
   or dispatch features.
6. Realized fields are restricted to target/evaluation or historical replay.
7. Undefined availability is a hard failure, not an implicit assumption.

Contract version: `field_availability_v2`.

