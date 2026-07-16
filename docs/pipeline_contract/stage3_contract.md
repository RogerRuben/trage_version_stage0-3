# Stage 3 canonical contract

## Research task

Stage 3 calibrates held-out Stage 2 predictions, aggregates link predictions for
each declared candidate route, and applies pre-registered route-level ODD gates.
It does not collapse the result to a single grade.

## Inputs

- Only `stage2_dispatch_prediction` held-out predictions for formal Stage 4 use.
- Fold/date keys, prediction cutoffs, uncertainty, availability, and route position.
- Realized Stage 1 order aggregates as targets/evaluation-only fields, never features.

## Outputs

- LCS raw/tail/uncertainty.
- PMIS raw/tail/uncertainty.
- RTS raw/tail/uncertainty.
- IIS availability/applicability/conditional severity.
- Separately named calibrated overall probability and continuous severity outputs.
- Maximum-risk link, high-risk-link share, contiguous high-risk length, route
  confidence, route ODD feasibility, violation count/length, binding dimension,
  and remote-assistance requirement for each candidate route/profile.
- Model, calibration, cutoff, missing-modality, and lineage metadata.

## Allowed information

Held-out Stage 2 predictions and pre-dispatch route descriptors whose field roles
are declared in the output schema.

## Forbidden information

- Stage 1 realized labels as input features.
- Stage 2 in-sample predictions.
- Test-day calibration or scaler fitting.
- Treating missing IIS severity as zero.
- Naming q90 or another quantile as an expected value.
- Naming an uncalibrated score as a probability.
- Test-day tuning of an ODD threshold or route aggregation parameter.

## Acceptance rules

- Train/validation/test dates and keys are disjoint and temporally ordered.
- Calibration is validation-only and independently audited on test data.
- Overall probability and continuous severity remain semantically distinct.
- Missing modality policy is explicit and stable.
- Full-day and fold inference distributions are compared on common held-out keys;
  unexplained scale drift fails the artifact.
- Every Stage 4 field traces to a Stage 3 output manifest.

## Time convention

The vector is available at `T_decision` only if every input prediction cutoff is
no later than `T_decision`. Calibration objects are frozen before the test date.

## Missing and fallback rules

Missing modalities use masks and availability fields, never stress-value
imputation. Calibration fallback is selected on validation only and records method,
fit support, and version.

## Version and downstream consumers

Contract version: `stage3_condition_vector_v2`; the paper route extension is
`stage3_route_risk_v2`. Consumers are Stage 3.5 offline route generation and the
existing Stage 4 engineering smoke. Formal Stage 4 must read Stage 3.5, not an
unversioned Stage 3 directory.
