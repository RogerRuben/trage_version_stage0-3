# R1 Train-only AV repositioning robustness protocol

## Frozen scientific question

R1 asks whether spatial misplacement of idle AVs in the no-active-repositioning baseline explains the high-penetration service decline. It is a favorable spatial robustness test, not an optimal relocation method or deployable robotaxi policy.

## Pre-registration identity

- Base enhancement commit: `b91fd3e165d7064fb58cddaed90dba51ab13f97f`.
- Policy: `TRAIN_TOD_DEMAND_BALANCE`.
- Policy version: `stage4_repositioning_r1.1`.
- Enabled scenarios: `R1_Q25_M_P70_REPOS`, `R1_Q50_M_P70_REPOS`, `R1_Q75_M_P70_REPOS`, and `R1_BENCH_AV_M_REPOS`.
- Frozen anchors: `MAIN_Q25_M_P70`, `MAIN_Q50_M_P70`, `MAIN_Q75_M_P70`, and `BENCH_AV_M`.
- GPU: none. No dense order-by-vehicle or reposition matrix is constructed.

## Train-only demand reference

The reference reads only `stage1/input_v1/split=train/date=20161009` through `date=20161024`. For every core order with a non-null `start_node`, Unix departure time is converted to `Asia/Shanghai` and mapped to one of 96 fixed 15-minute clock-time bins. `start_node` is the existing Stage1 OSM node identity; WGS84 coordinates come from the frozen `map_data/xian_roadmap_update.osm.pbf` bound by the Stage0 freeze manifest.

Validation dates and Test31 are excluded. The derived parquet and manifest record exact dates, input-file-path digest, PBF SHA, row counts, node coverage, and their own SHA. Test31 requests, future demand density, realized Test31 traffic, future passenger acceptance, and future queue state never enter the reference.

## Frozen policy order

At every 15-minute boundary:

1. FleetPy advances existing vehicle routes and activates current requests.
2. The unchanged 30-second sparse rolling dispatcher runs normally.
3. Only AVs still idle after dispatch enter R1.
4. Each idle AV is assigned to its nearest positive-demand Train node for that clock bin solely to measure current distribution.
5. The idle count is converted to node quotas with deterministic largest remainder.
6. Only AVs above their current-node quota are surplus. Vehicles are ordered by `vehicle_id`; the first quota units remain.
7. Surplus AVs are assigned sequentially to the nearest deficit with remaining quota; ties use node identity.
8. Proposals are grouped by destination and routed through the existing sparse Valhalla adapter. This avoids a dense vehicle-by-node matrix.
9. A successful movement becomes one FleetPy `REPOSITION` leg. The AV is unavailable until FleetPy completes that leg; there is no mid-route diversion.
10. A routing failure leaves the AV at its current location and is logged. No teleportation is permitted.

Passenger-route Stage3 suitability is not applied to empty reposition routes. Their travel time and distance are Valhalla operational abstractions and are explicitly not ODD-certified.

## Frozen baseline and stop gate

The 30-second dispatch, 300-second patience, fleet construction, acceptance CRN, Profile M, unconstrained Gamma regime, Stage3 interface, candidate radii, shared Top-K, Valhalla pickup routing, solver, normalization, and horizon remain unchanged. The feature is default-off.

Before any enabled run, the modified code must rerun all four anchors with `repositioning_enabled=False`. The 19 frozen summary metrics, request-outcome fingerprint, and assignment fingerprint must reproduce exactly. Any mismatch stops R1.

## Outputs and interpretation

R1 preserves N0–N6 and 15-minute gates. Pairwise reporting covers service, matched/expired requests, pickup waits, queue pressure, assignment mix, N5/N0, N6/N0, structural/evidence/patience retention, reposition trip/time/distance burden, and the pre-specified 17:00–18:59 window.

Results will be classified descriptively as `SUPPORTS CURRENT STORY`, `QUALIFIES CURRENT STORY`, or `CHANGES CURRENT STORY`. No p-values, causal claims, optimality claims, deployment claims, or empty-route safety claims are authorized.
