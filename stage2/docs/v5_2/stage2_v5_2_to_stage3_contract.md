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
- RTS time-weighted mean/P90/high exposure and distance-weighted sensitivity.
- Static route-complexity fields, preserving NA where upstream fields do not exist.
- Micro-condition coverage, Train-only support, low-support share, and unseen-edge share.
- Prediction source, model ID/hash, transfer-model version, and feature cutoff time.

## Forbidden inputs

Stage 3 must not read Stage 1 ground truth, actual future travel time, oracle
entry time, driver identity, evaluation truth, or any AV safety probability.
It must not infer missing static or dynamic fields as zero.

## Admission

This interface is structurally defined but not yet admitted. Admission requires
zero temporal leakage, completed micro products and manifests, performance-gate
PASS, completed rolling evaluation for all five micro targets, a documented
positive or negative transfer conclusion, stable pace P50, and final status
`READY_FOR_AV_ROUTE_SUITABILITY_STAGE`.

Historical scenario contracts remain available for appendix reproduction but
are not the v5.2 mainline Stage 3 contract.
