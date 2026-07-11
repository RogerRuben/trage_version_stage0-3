# Stage2 rolling/OOF protocol

Publication-grade comparison uses consecutive temporal folds rather than one
fixed test day. The default fold is seven train days, one validation day and one
test day. At least three folds are required before a deep residual branch can be
promoted.

Every fold must rebuild rolling profiles and lagged features using only
information available before each prediction timestamp. All compared models
must use the same order/link or movement keys and the same feature-availability
contract.

The ablation ladder is:

1. static tabular;
2. static plus rolling historical profiles;
3. plus lagged traffic state;
4. plus planned-route context;
5. plus target-specific topology residual;
6. RTS-specific propagation residual;
7. validation-selected hybrid fusion.

Report AUC, AP, rank correlations, top-k precision/recall/lift/NDCG, decile
behavior and order-level tail separation. Brier and ECE are valid only for a
calibrated tail-event probability. Confidence intervals must cluster by order,
link and day where supported.

Use `build_stage2_rolling_fold_manifest.py` to verify that enough consecutive
retained days exist before launching the model loop.

