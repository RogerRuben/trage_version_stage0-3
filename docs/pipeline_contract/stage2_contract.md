# Stage 2 canonical contract

## Research task

Stage 2 predicts planned-route link conditions under an explicit decision-time
contract. Two modes are separate artifacts:

- `stage2_dispatch_prediction`: all route predictions are generated using only
  information available at dispatch time. This is the only Stage 4 input.
- `stage2_rolling_prediction`: en-route updates may use newly available state.
  This is an oracle/operational diagnostic and is never mixed with dispatch mode.

## Inputs

- Versioned planned/matched-route proxy known at the declared decision time.
- Stage 0 topology/route products and Stage 1 targets for training only.
- Strictly backward profiles and traffic state with availability timestamps.
- Fold manifest fixing train, validation, and test keys.

## Outputs

- Held-out link predictions for LCS, PMIS, RTS, and optional movement IIS.
- Predicted service-time distribution for Stage 4.
- Prediction cutoff, model-training cutoff, fallback level, support, fold, and mode.

## Allowed information

Dispatch mode may use OD, assigned-route proxy, calendar, and historical features
whose availability timestamp is no later than the order decision time.

## Forbidden information

- Actual link entry time or future route state in dispatch mode.
- Profiles containing the prediction date or future dates.
- In-sample predictions exported as held-out inputs.
- Test labels used for thresholding, calibration, scaling, or fallback selection.
- Mixing rolling predictions into a dispatch artifact.

## Acceptance rules

- `feature_timestamp <= availability_timestamp <= decision_time` for every feature.
- Prediction date is outside the training window.
- Route source, OD availability, fallback hierarchy, and support are non-null.
- Target and feature stores are key- and field-isolated.
- Dispatch and rolling artifacts have different manifest IDs and cannot be joined
  without an explicit diagnostic override.

## Time convention

`T_decision` is a single order-level cutoff applied to every planned-route link.
Estimated future link-entry time may be an output/position descriptor but cannot
advance feature availability beyond `T_decision` in dispatch mode.

## Missing and fallback rules

Every fallback records `requested_level_support_count`, `fallback_level`,
`fallback_support_count`, and `fallback_value_source`. An unseen requested link
keeps requested support zero even when a global value exists. Feature inclusion is
an exact whitelist plus role registry and exact-name blacklist.

## Version and downstream consumers

Products: `stage2_dispatch_prediction_v2` and `stage2_rolling_prediction_v2`.
Only the dispatch product is consumed by canonical Stage 3/4.
