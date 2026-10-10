# Stage 2 v4 Final Evaluation

- Engineering status: `PASS`
- Frozen Test date: `20161031`
- Orders: `30,000`
- Traversal/route-token predictions: `2,116,712`
- Test tuning violations: `0`

| Target | Count | MAE | RMSE | Pearson |
|---|---:|---:|---:|---:|
| acceleration_rms_bounded | 357,248 | 0.121123 | 0.149087 | 0.281620 |
| crawl_time_share | 777,417 | 0.156268 | 0.275394 | 0.506841 |
| lcs_pct | 318,221 | 0.252749 | 0.300830 | 0.145786 |
| lcs_raw | 318,221 | 0.061398 | 0.077450 | 0.310383 |
| rts_pct | 717,805 | 0.249004 | 0.281483 | 0.359997 |
| rts_raw | 717,805 | 0.091831 | 0.125862 | 0.472873 |
| speed_cv_bounded | 524,081 | 0.064236 | 0.080218 | 0.414036 |
| stop_time_share | 777,417 | 0.005213 | 0.045369 | 0.580426 |

| Tail | Variant | Brier | ROC AUC | ECE |
|---|---|---:|---:|---:|
| lcs | calibrated | 0.090149 | 0.634486 | 0.002886 |
| lcs | uncalibrated | 0.090134 | 0.635191 | 0.007676 |
| rts | calibrated | 0.076544 | 0.741355 | 0.009160 |
| rts | uncalibrated | 0.076679 | 0.741545 | 0.014522 |

| Interval | Coverage | Mean width |
|---|---:|---:|
| lcs | 0.906273 | 0.251669 |
| rts | 0.913814 | 0.301251 |

Subgroup, bootstrap, order aggregation, calibration, entry-time and baseline-component details are under `stage2/output_v4/reports/`.
