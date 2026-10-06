# Replay absolute calibration v1 — fixed scope

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation and bounded execution
- Origin Date: 2026-10-06
- Verification Status: UNVERIFIED before execution
- Version Label: replay_calibration_v1

## Question and authorized endpoint

Explain the gap between historically completed trips and the current counterfactual replay. Complete historical-chain and ETA diagnostics, then one current-version all-HV reference. Preserve all existing mixed-fleet results and frozen inputs. This is not a new C/M/A experiment grid and not an absolute target of 100% service.

## Offline diagnostics, fixed before results

- Read all valid Test31 raw orders for predecessor construction; do not construct predecessor chains from the 30,000 sampled orders alone.
- Retain/report missing predecessors, overlapping histories, invalid coordinates and gaps over the existing 90-minute session break separately.
- For common research orders with a non-overlapping predecessor in the same inferred session, select at most 1,200 pairs by SHA256(seed=20261006, order_id). No outcome-dependent sampling.
- Query the existing single-source Valhalla matrix for those sparse previous-dropoff → next-pickup OD pairs. One routing process; no all-pairs matrix.
- Apply the frozen beta at the next historical boarding proxy, matching the canonical zero-lead decision clock. Compare raw and corrected ETA with (a) the observed inter-trip gap and (b) 300 seconds. The gap is an upper budget including unknown idle time, not an observed pickup-time label.
- A corrected ETA within the historic gap but over 300 seconds is a **timing/idle-hold conflict witness**, not proof that a specific canonical unserved order would be recovered.
- Report cohort groups, candidate-visit attrition, and day-clipped pickup+service vehicle-hours. Do not interpret repeated arc counts as disjoint order-level causes.
- Do not refit beta from the Test31 gaps; do not change request release or import realized future information into dispatch.

## One same-interface all-HV reference

Use the existing accelerated v4 SERVICE_PRESERVING_LOOKAHEAD runner and its exact 28,367 common orders. Change only q_A=0 and AV acceptance=1 (acceptance is inactive with no AVs). Reconstruct all 8,435 frozen HV session templates at the same baseline-normalized supply. Mixed reference has approximately 0.155% more total hours due to its frozen rounding; report this instead of silently forcing exact equality.

Keep request release=boarding proxy, patience=300 s, 30-s dispatch, Top-K=20, 2–8 km search, empirical HV windows, M3 checkpoint/predictions, Train-only forecast/remaining-time model, beta, no repositioning, no Gamma, no cost, 10-s solver limit and v4 routing/cache behavior unchanged. Use actual all-HV fixture windows, not cached mixed-fleet windows.

Output: `stage4/output/replay_calibration_v1/all_hv/SERVICE_PRESERVING_LOOKAHEAD`. This must not enter the original MYOPIC-vs-LOOKAHEAD analysis as if it were a policy ablation. Compare to the completed M/q=.5/P=.7 v4 run as a combined **fleet-composition/compatibility reference**, not an identified pure ODD effect. The legacy all-HV 78.89% result is context only.

## Execution and resources

Single native scenario, two bounded routing workers, CPU only; process-group RSS cap 2,048 MiB; sparse model limits unchanged; optional disk route cache at most 512 MiB; administrative timeout 10,800 s; no automatic retry or threshold search. Preserve partial outputs on failure.

Commands (environment: stage0-valhalla; one BLAS/OpenMP thread, GPU disabled):

```powershell
python -m stage4.analysis.replay_calibration_v1 --phase diagnostic
python -m stage4.analysis.symmetric_flexibility_full_day --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --policy SERVICE_PRESERVING_LOOKAHEAD --acceleration-config stage4/config/symmetric_flexibility_acceleration_v4.json --calibration-config stage4/config/replay_calibration_v1.json
python -m stage4.analysis.replay_calibration_v1 --phase report
```

## Interpretation and stop

Report actual execution times, resource peaks, baseline hashes, reference service/wait differences, and limits of historical-gap inference. Only one date/seed; neither historical completed-trip sampling nor this replay estimates the real platform's unconditional request acceptance rate. Stop after the diagnosis/report and commit/push; any timing/reposition/ETA redesign requires a separately declared experiment rather than tuning to 100%.
