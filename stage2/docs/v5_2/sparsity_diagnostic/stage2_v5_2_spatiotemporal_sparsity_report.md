# Stage 2 v5.2 spatiotemporal sparsity diagnostic

- Diagnostic classification: `DIAG-B`
- Conclusion: `STRUCTURED_TRANSFER_POSITIVE; SUPPORT_AWARE_GATE_NOT_SUPPORTED`
- Phase C correction: `PASS` / `FAIL`
- Prediction-level artifact availability: `PASS`
- Reinference/retraining: `NO / NO`
- Transfer-v2 / Phase D / Stage 3 authorized: `NO / NO / NO`

## Frozen scope

- Train-only support: `20161009-20161021`
- Diagnostic evaluation: `20161025-20161027`
- Existing time bin: Xi'an local 30-minute `estimated_time_bin` (0-47)
- Unit: unique physical traversal `(date, order_id, traversal_id)`
- RTS is excluded from the diagnostic main line.

## Table 1 — Spatial sparsity (aggregate)

| support_group | train_unique_edges | evaluation_unique_edges | evaluation_traversals | evaluation_share |
|---|---|---|---|---|
| unseen | 0 | 786 | 1027 | 0.000488999 |
| low | 6683 | 1708 | 2331 | 0.00110989 |
| medium | 11018 | 10038 | 292932 | 0.139478 |
| high | 5889 | 5889 | 1803919 | 0.858924 |

## Table 2 — Spatiotemporal sparsity (aggregate)

| support_group | train_unique_edge_time_cells | evaluation_unique_edge_time_cells | evaluation_traversals | evaluation_share |
|---|---|---|---|---|
| unseen | 0 | 22233 | 25003 | 0.011905 |
| low | 155944 | 37581 | 50087 | 0.0238486 |
| medium | 257389 | 198424 | 574903 | 0.273736 |
| high | 133694 | 133508 | 1450216 | 0.69051 |

## Table 6 — Spatial-high / temporal-sparse

| target | n | m1_mae | m3_mae | m4_mae | m3_vs_m1_relative_improvement | m4_vs_m3_relative_improvement |
|---|---|---|---|---|---|---|
| crawl | 4333 | 0.0780991 | 0.0716975 | 0.0715127 | 0.0819676 | 0.00257825 |
| stop | 4333 | 0.00385179 | 0.00382503 | 0.00394831 | 0.00694594 | -0.0322283 |
| speed_cv | 2800 | 0.0699371 | 0.0690975 | 0.0691426 | 0.0120053 | -0.000653628 |
| acceleration_rms | 1754 | 0.139155 | 0.138587 | 0.138656 | 0.00408288 | -0.000500881 |

## M1 cell-level support relationship

| target | support_dimension | cell_count | spearman_log1p_support_vs_cell_mae |
|---|---|---|---|
| crawl | spatial | 172690 | -0.136309 |
| crawl | spatiotemporal | 172690 | 0.0317763 |
| crawl | target_specific | 172690 | 0.0114017 |
| stop | spatial | 172690 | -0.114019 |
| stop | spatiotemporal | 172690 | 0.000834413 |
| stop | target_specific | 172690 | -0.236067 |
| speed_cv | spatial | 120066 | 0.0772828 |
| speed_cv | spatiotemporal | 120066 | 0.0593678 |
| speed_cv | target_specific | 120066 | 0.0308505 |
| acceleration_rms | spatial | 88880 | 0.108475 |
| acceleration_rms | spatiotemporal | 88880 | 0.0687963 |
| acceleration_rms | target_specific | 88880 | 0.0668551 |

## M3/M4 improvement by spatiotemporal support

| target | support_group | comparison | n | baseline_mae | candidate_mae | relative_improvement |
|---|---|---|---|---|---|---|
| crawl | unseen | M3_vs_M1 | 7945 | 0.232174 | 0.225024 | 0.0307957 |
| crawl | unseen | M4_vs_M1 | 7945 | 0.232174 | 0.22546 | 0.028918 |
| crawl | unseen | M4_vs_M3 | 7945 | 0.225024 | 0.22546 | -0.00193732 |
| crawl | low | M3_vs_M1 | 17602 | 0.196902 | 0.190672 | 0.031639 |
| crawl | low | M4_vs_M1 | 17602 | 0.196902 | 0.190767 | 0.0311562 |
| crawl | low | M4_vs_M3 | 17602 | 0.190672 | 0.190767 | -0.000498649 |
| crawl | medium | M3_vs_M1 | 201184 | 0.171608 | 0.168002 | 0.0210126 |
| crawl | medium | M4_vs_M1 | 201184 | 0.171608 | 0.167878 | 0.02174 |
| crawl | medium | M4_vs_M3 | 201184 | 0.168002 | 0.167878 | 0.000743069 |
| crawl | high | M3_vs_M1 | 553509 | 0.170377 | 0.168497 | 0.0110381 |
| crawl | high | M4_vs_M1 | 553509 | 0.170377 | 0.16838 | 0.0117198 |
| crawl | high | M4_vs_M3 | 553509 | 0.168497 | 0.16838 | 0.000689282 |
| stop | unseen | M3_vs_M1 | 7945 | 0.0366272 | 0.0389025 | -0.0621193 |
| stop | unseen | M4_vs_M1 | 7945 | 0.0366272 | 0.0395525 | -0.0798667 |
| stop | unseen | M4_vs_M3 | 7945 | 0.0389025 | 0.0395525 | -0.0167095 |
| stop | low | M3_vs_M1 | 17602 | 0.0159168 | 0.0162404 | -0.0203329 |
| stop | low | M4_vs_M1 | 17602 | 0.0159168 | 0.0162612 | -0.0216424 |
| stop | low | M4_vs_M3 | 17602 | 0.0162404 | 0.0162612 | -0.00128336 |
| stop | medium | M3_vs_M1 | 201184 | 0.00656752 | 0.00658755 | -0.00305041 |
| stop | medium | M4_vs_M1 | 201184 | 0.00656752 | 0.00658544 | -0.0027293 |
| stop | medium | M4_vs_M3 | 201184 | 0.00658755 | 0.00658544 | 0.000320137 |
| stop | high | M3_vs_M1 | 553509 | 0.00408181 | 0.00417784 | -0.023525 |
| stop | high | M4_vs_M1 | 553509 | 0.00408181 | 0.0042019 | -0.0294209 |
| stop | high | M4_vs_M3 | 553509 | 0.00417784 | 0.0042019 | -0.00576042 |
| speed_cv | unseen | M3_vs_M1 | 5302 | 0.0705919 | 0.0694624 | 0.0159999 |
| speed_cv | unseen | M4_vs_M1 | 5302 | 0.0705919 | 0.0695107 | 0.0153165 |
| speed_cv | unseen | M4_vs_M3 | 5302 | 0.0694624 | 0.0695107 | -0.000694589 |
| speed_cv | low | M3_vs_M1 | 12432 | 0.0653373 | 0.0648711 | 0.00713448 |
| speed_cv | low | M4_vs_M1 | 12432 | 0.0653373 | 0.0648484 | 0.00748302 |
| speed_cv | low | M4_vs_M3 | 12432 | 0.0648711 | 0.0648484 | 0.000351044 |
| speed_cv | medium | M3_vs_M1 | 138109 | 0.0631859 | 0.0628635 | 0.005102 |
| speed_cv | medium | M4_vs_M1 | 138109 | 0.0631859 | 0.0628648 | 0.0050811 |
| speed_cv | medium | M4_vs_M3 | 138109 | 0.0628635 | 0.0628648 | -2.10062e-05 |
| speed_cv | high | M3_vs_M1 | 374098 | 0.0628048 | 0.0625016 | 0.00482684 |
| speed_cv | high | M4_vs_M1 | 374098 | 0.0628048 | 0.0625085 | 0.00471824 |
| speed_cv | high | M4_vs_M3 | 374098 | 0.0625016 | 0.0625085 | -0.000109129 |
| acceleration_rms | unseen | M3_vs_M1 | 3578 | 0.118051 | 0.117562 | 0.0041486 |
| acceleration_rms | unseen | M4_vs_M1 | 3578 | 0.118051 | 0.117491 | 0.00474652 |
| acceleration_rms | unseen | M4_vs_M3 | 3578 | 0.117562 | 0.117491 | 0.000600414 |
| acceleration_rms | low | M3_vs_M1 | 8707 | 0.116811 | 0.116559 | 0.00215581 |
| acceleration_rms | low | M4_vs_M1 | 8707 | 0.116811 | 0.116545 | 0.00227372 |
| acceleration_rms | low | M4_vs_M3 | 8707 | 0.116559 | 0.116545 | 0.000118165 |
| acceleration_rms | medium | M3_vs_M1 | 95865 | 0.118911 | 0.118574 | 0.00283483 |
| acceleration_rms | medium | M4_vs_M1 | 95865 | 0.118911 | 0.11857 | 0.00286565 |
| acceleration_rms | medium | M4_vs_M3 | 95865 | 0.118574 | 0.11857 | 3.09066e-05 |
| acceleration_rms | high | M3_vs_M1 | 255767 | 0.119515 | 0.119077 | 0.0036667 |
| acceleration_rms | high | M4_vs_M1 | 255767 | 0.119515 | 0.11908 | 0.0036371 |
| acceleration_rms | high | M4_vs_M3 | 255767 | 0.119077 | 0.11908 | -2.97146e-05 |

## Structure representation diagnostic

- Unique edges: `23590`
- Unique signatures: `1477`
- Mean edges/signature: `38.033`
- Collision-signature share: `74.746%`
- Maximum edges sharing a signature: `2604`

## Diagnostic interpretation

Structured transfer has a repeatable positive signal in sparse edge-time cells, while M4 adds no stable increment over M3. Stop support-aware expansion and review M1 versus M3 for the engineering candidate.

This is descriptive evidence, not a causal estimate. Effect sizes, sample counts, cluster-bootstrap uncertainty, and per-day direction are all preserved in the bound CSV tables.

## Stop state

No training, inference, checkpoint selection, tau selection, M5/M6, rolling Phase D, 20161028-30 data, or Stage 0/1 rebuild was performed.
Part B remains user-gated. `TRANSFER_V2_AUTHORIZED=NO`, `PHASE_D_AUTHORIZED=NO`, `STAGE3_AUTHORIZED=NO`.
