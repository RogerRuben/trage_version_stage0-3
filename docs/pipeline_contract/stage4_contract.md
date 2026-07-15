# Stage 4 canonical contract

## Research task

Stage 4 runs counterfactual mixed AV/HV dispatch using a frozen Stage 3 condition
vector. During pipeline rebaseline, the only allowed execution is a 500-1,000
order functional smoke: Safe GlobalMatch, O0, Stay, preassignment off,
replication 1. Smoke metrics are not research results.

## Inputs

- Canonical demand manifest.
- Demand-supply-decoupled supply manifest.
- Stage 3 canonical condition-vector manifest.
- Dispatch-time predicted service-time distribution.
- Exogenous, versioned AV capability scenario.
- Frozen operational zones, costs, random environment, and Safe/O0 configuration.

## Outputs

- Request, transition, vehicle-leg, event, plan, offer, and economy logs.
- Smoke summary and audits derived from those logs.
- Run manifest containing commit, config hash, input manifests, seed, status,
  canonical flag, and superseded run ID.

## Allowed information

Only fields available by the simulated decision time and exogenous scenario inputs.
Unknown condition-vector orders may form HV edges but never AV edges.

## Forbidden information

- Historical realized service duration in formal counterfactual mode.
- Test-day calibration of AV thresholds.
- Future demand or a driver's future/next observed position.
- Test-day realized supply used to generate counterfactual supply.
- Hard-coded or unaudited ODD PASS.
- Automatic discovery of input parquet files.
- Full-day O0-O3, Balanced, sensitivity, or extra-replication runs before the
  end-to-end canonical audit passes.

## Acceptance rules

- Exactly one declared input manifest per input role; hashes match.
- `completed + cancelled = demand universe`; state/event/leg/economy audits pass.
- AV ODD violations and unknown-condition AV assignments are zero.
- No forbidden field is present in the decision input schema.
- Smoke uses 500-1,000 orders and cannot be marked `formal_inference`.

## Time convention

Requests follow `T_request <= T_decision < T_pickup < T_complete`. Dispatch inputs
are frozen at each decision epoch. Historical replay has a separate config and
may read realized durations; counterfactual mode records zero realized-duration
reads.

## Missing and fallback rules

Unknown condition-vector orders have no AV edge; their HV edges still pass normal
time, radius, availability, and utility checks. Counterfactual service duration is
predicted distribution plus a pre-generated residual. No historical-duration
fallback is allowed.

## Version and downstream consumers

Contract version: `stage4_counterfactual_smoke_v2`. The smoke is consumed only by
the end-to-end audit, never by scientific comparison.
