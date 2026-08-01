# Stage 2 v5 legacy frozen benchmark

20161031 is a legacy frozen benchmark for v4/v5 comparability. It was not used for model selection and is not claimed as an untouched final test.

| Model | Direct pace MAE (s/m) |
|---|---:|
| strict_historical_profile | 0.030014544 |
| v4_static_entry_time | 0.030013534 |
| hist_gradient_boosting | 0.028492864 |
| rc_mstnet_v5_mean | 0.028135204 |
| rc_mstnet_v5_p50 | 0.027334769 |

The v5 mean improves on the tree baseline by 1.26% on 717,805 direct-pace traversals.

Frozen calibrated route coverage: P50 0.6099, P90 0.9461, P95 0.9685 across 30,000 routes.

Actual frozen RC-MSTNet v4/v5 auxiliary-target metrics are recorded separately on identical traversal rows; percentile-tail results are descriptive only.
