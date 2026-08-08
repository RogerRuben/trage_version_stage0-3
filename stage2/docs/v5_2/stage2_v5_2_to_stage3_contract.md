# Stage 2 v5.2 to Stage 3 contract

## Scope

Stage 3 reads predictions for the historical original service route. HV keeps
that route. Stage 3 may later evaluate AV original-route suitability and, only
if unsuitable, separately implement a constrained single fallback. Stage 2 does
not produce a fallback route.

There is **no route decision variable** in the downstream assignment interface.
Stage 4 may eventually receive only HV original route and AV original route,
one preselected fallback, or unavailable.

## Allowed inputs

- Original-route identity and ordered traversal identity.
- Travel-time P50.
- Crawl weighted mean/P90/high exposure/maximum consecutive high exposure.
- Stop weighted mean/P90/high exposure/maximum consecutive high exposure.
- Speed-CV weighted mean/P90/high exposure.
- Acceleration-RMS weighted mean/P90/high exposure.
- Static route-complexity fields, preserving NA where upstream fields do not exist.
- Micro-condition coverage, Train-only support, low-support share, and unseen-edge share.
- Prediction source, model ID/hash, transfer-model version, and feature cutoff time.
- `service_time_complete_flag` only when route pace-distance coverage reaches the
  frozen near-one threshold (`0.999`). Lower but admissible coverage is explicitly
  `partial_coverage_estimate`; it is never presented as complete service time.

The machine-readable allow mask is `STAGE3_ALLOWED_FIELDS` in
`stage2.v5_2.contracts`. Product schemas are independently namespaced as
`stage2_v5_2_micro_condition_tokens.2`,
`stage2_v5_2_original_route_micro_conditions.2`, and
`stage2_v5_2_static_route_complexity.2`.

The four core and Stage 3 deployable micro-condition families are crawl, stop,
speed-CV, and acceleration-RMS. RTS is retained only as a legacy/descriptive
diagnostic for comparison with frozen earlier releases. RTS fields, percentiles,
tails, and sensitivity summaries are excluded from Stage 3 assignment inputs and
must not influence AV route-suitability decisions.

## Evaluation-only fields

Target-availability flags, observed target values, truth, error, and absolute
error fields may exist in evaluation artifacts. They are listed by
`STAGE3_EVALUATION_ONLY_FIELDS` and must be removed before a Stage 3 serving or
assignment product is written.

## Forbidden inputs

Stage 3 must not read Stage 1 ground truth, actual future travel time, oracle
entry time, driver identity, evaluation truth, or any AV safety probability.
It must not infer missing static or dynamic fields as zero.

## Admission

This interface is structurally defined but not yet admitted. Admission requires
zero temporal leakage, completed micro products and manifests, performance-gate
PASS, completed rolling evaluation for the four core/deployable micro targets,
a separately reported RTS diagnostic, a documented
positive or negative transfer conclusion, stable pace P50, and final status
`READY_FOR_AV_ROUTE_SUITABILITY_STAGE`.

Historical scenario contracts remain available for appendix reproduction but
are not the v5.2 mainline Stage 3 contract.
