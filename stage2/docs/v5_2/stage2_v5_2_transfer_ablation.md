# Stage 2 v5.2 transfer ablation

Status: **NOT RUN**.

The frozen matrix is M0 tree baseline, M1 RC-MSTNet v5.1, M2 structure-only,
M3 ID plus structure concatenation, M4 support-aware spatial transfer, M5 M4
plus temporal adapter, and optional M6 task pretraining. M5/M6 stop if M4 fails
its adoption rule.

No transfer method is adopted by this code-only commit. Spatial, temporal, and
task-transfer conclusions remain pending development/rolling backtesting.
HorizonGate remains a negative ablation; availability is diagnostic and IPW is
sensitivity-only.
