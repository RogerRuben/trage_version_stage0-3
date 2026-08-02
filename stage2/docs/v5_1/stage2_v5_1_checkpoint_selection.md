# Stage 2 v5.1 checkpoint selection

Every candidate must first pass finite-output, monotonic-quantile, pace mean/ratio, and route-scenario smoke gates on merged unique traversals. Single-row error contribution is not used to choose among checkpoints because it can be driven by one extreme validation label, but it remains a fail-closed final Stage 3 admission check.

Stable candidates are ordered by validation P50 MAE, family-appropriate distribution loss (pinball for M3; NLL for parametric baselines), mean MAE, P90/P95 coverage error, then checkpoint ID. A lower multi-task batch loss cannot override a failed hard gate.

The frozen v5 checkpoint does not satisfy this selection contract and remains evidence only; the policy applies to the next v5.1 training run.
