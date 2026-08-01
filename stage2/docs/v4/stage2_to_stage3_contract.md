# Stage 2 v4 → Stage 3 contract

Stage 3 may consume only records produced by the
`revealed_route_proxy_predispatch` track where `stage3_eligible_track = true`.

## Allowed products

- frozen route-token predictions and calibrated probabilities;
- order/route summaries built with decision-time weights;
- static route context;
- estimated entry time, uncertainty, and forecast horizon;
- provenance IDs for the Stage 2, Stage 1, and Stage 0 frozen releases.

## Required identity

- `stage2_v4_release_manifest.json` must be `ENGINEERING_PASS`;
- Stage 1 tag/commit/model/config IDs must match the release manifest;
- prediction, calibration, dataset, and tensor manifest hashes must match;
- `route_proxy_track` must equal `revealed_route_proxy_predispatch`;
- `fully_deployable` remains false.

## Prohibited inputs and claims

Stage 3 must not consume:

- `oracle_timing_upper_bound`;
- actual token entry/exit time or realized travel time;
- Stage 1 realized targets as features;
- IIS, PMIS, or dynamic GNS targets;
- uncalibrated tail scores as probabilities;
- legacy Stage 2 v3 products.

The outputs describe associative, route-conditioned dynamic pressure. They are
not causal effects, accident probabilities, AV failure probabilities, or final
AV/HV assignment decisions.

## Reserved deployable interface

`planned_route_deployable` is reserved for a future decision-time OD, routing
policy, and planned route. Stage 2 v4 defines the name but produces no records
for it.
