# Stage3 order-level predictive modeling

Stage3 consumes only held-out Stage2 predictions. Its current canonical
prototype uses Fold 1/2/3 test days as train/validation/test dates 17/18/19.

```text
stage2_prediction_warehouse -> order targets/features
                            -> rule baseline
                            -> order LightGBM
                            -> DeepSets / route attention
                            -> calibrated order stress vector
```

Realized Stage1 measurements appear only in `order_targets`; they are excluded
from the warehouse and order-feature tables. See
`stage2/docs/stage2_deep_v3_stage3_prototype_report.md` for results and limits.
