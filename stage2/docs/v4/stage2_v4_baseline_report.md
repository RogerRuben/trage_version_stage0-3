# Stage 2 v4 Baseline Report

- Engineering status: `PASS`
- Fit dates: `20161009, 20161010, 20161011, 20161012, 20161013, 20161014, 20161015, 20161016, 20161017, 20161018, 20161019, 20161020, 20161021, 20161022, 20161023, 20161024`
- Validation dates: `20161025, 20161026`
- Test rows read: `0`

| Target | Best validation variant | RMSE/Brier |
|---|---:|---:|
| acceleration_rms_bounded | tree | 0.146913 |
| crawl_time_share | tree | 0.278994 |
| lcs_raw | tree | 0.075109 |
| lcs_tail_event | tree | 0.096853 |
| rts_raw | tree | 0.135773 |
| rts_tail_event | tree | 0.099210 |
| speed_cv_bounded | tree | 0.079707 |
| stop_time_share | tree | 0.049648 |

Detailed component comparisons are in `stage2/output_v4/reports/ablation_results.csv`.
