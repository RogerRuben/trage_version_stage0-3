# Stage 2 v5.1 -> Stage 3 prototype contract

Stage 3 may consume only fields whose entry in manifest `field_eligibility` is `ELIGIBLE`, whose product `stability_status` and `stability_check_status` are both `PASS`, and whose file hashes match. The allowed per-field status vocabulary is `ELIGIBLE`, `EXPERIMENTAL`, and `BLOCKED`. The current admission is a read-only route-scenario prototype, not formal Stage 3 optimization.

## Eligible now

- pace P50/P90/P95;
- route service-time P50/P90/P95;
- availability and coverage;
- scenario/calibration/model provenance;
- externally supplied timeout-threshold comparisons.

## Experimental or blocked

- pace mean;
- route mean and standard deviation;
- route CVaR90/CVaR95;
- dispatch costs based on mean or CVaR;
- scenario samples as a formal optimization distribution.

Each manifest records `stability_check_id`, `stability_check_status`, `stability_status`, per-field eligibility, seed, model/checkpoint identities, input hash, calibration identity, and output hashes.

The reader fails closed when the manifest is missing, a hash differs, a requested field is not `ELIGIBLE`, the product stability status is not `PASS`, or the external threshold lacks non-label provenance. Existing 600/900/1200-second columns are diagnostics only and are not platform business constraints.

Prediction-source switching is explicit through `ScenarioSourceRegistry`. A reader configured for `tree`, `deep_p50`, or `deep_scenario` may only open a manifest declaring that source; unavailable sources and the non-deployable `oracle` upper bound fail rather than silently falling back.
