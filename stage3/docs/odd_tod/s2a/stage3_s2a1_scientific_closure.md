# Stage 3 S2A.1 Scientific Closure

Status: `STAGE3_S2A1_SCIENTIFIC_CLOSURE_COMPLETE`. This closure does not re-export the network, retrain/reselect the speed model, change P85/MAP, or change the 60/80/120 caps.

## Anchor unit and 502 → 180 reconciliation

The validation population is **502 canonical-segment anchors**, not 502 directed identities. Their class distribution is `{"20.0": 61, "30.0": 11, "50.0": 54, "60.0": 78, "70.0": 298}`. Of these, 500 map to the full network and collapse onto 180 unique frozen Valhalla directed edges because multiple canonical split segments share one physical directed edge. Two anchors remain unmapped. Accordingly, the deployed full-network speed table contains 180 `KNOWN_STAGE0_OSM` rows. Validation remains canonical-segment weighted; deployment remains one row per `stage3_edge_uid`.

## What the cap validation identifies

- 60 km/h: 204 anchors at/below and 298 above; this binary boundary is empirically identified. Frozen B2 CV accuracy is `0.806773`.
- 80 km/h: 502 at/below and 0 above. Accuracy `1.0` is **not empirically identified / trivial under current anchor support**.
- 120 km/h: 502 at/below and 0 above. Accuracy `1.0` is **not empirically identified / trivial under current anchor support**.

The previously reported macro `0.935591` is only a mechanical average over three thresholds. It must not be presented as validation performance for three AV profiles. The identified-boundary result is the 60-km/h accuracy above.

## Frozen B0 versus B2 downstream impact

Known anchors are invariant. B2 applies only to the 4,898 `INFERRED_SPEED_AND_CLASS` edges; every other inferred edge retains B0 fallback.

| Cap | Full-network disagreements | Full-network rate | B2-applicable disagreements | Applicable rate |
|---:|---:|---:|---:|---:|
| 60 | 26 / 209454 | 0.012413% | 26 / 4898 | 0.530829% |
| 80 | 3 / 209454 | 0.001432% | 3 / 4898 | 0.061249% |
| 120 | 0 / 209454 | 0.000000% | 0 / 4898 | 0.000000% |

Thus MAP's small CV advantage has a very small effect on deployed compatibility classification and does not establish 80/120 profile validity.

## Historical reverse-direction closure

All 6,502 unmatched reverse identities are classified as `HISTORICAL_DIRECTION_OVERLAY` plus `AV_ROUTABILITY_VIOLATION`, not missing. Their historical observations remain usable for descriptive/history features, but they are excluded from AV routing. A mapped physical forward reference exists for 6,388; the remaining 114 retain the same overlay/violation semantics without a fabricated graph link.

## Scientific closure

The speed-domain proxy is retained primarily for the Conservative 60-km/h boundary. Stage3's principal separation of C/M/A is expected to come from intersection/movement structure and dynamic E/Q/C. S2B remains unauthorized pending review.
