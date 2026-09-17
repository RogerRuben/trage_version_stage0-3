# Stage3 → action-space attribution: pre-experiment review

2026-09-18。只完成审计、数学说明、实验提案；**没有启动新实验**。

1. [A0 — 仓库审计](A0_repository_audit.md)
2. [A1 — 数学与归因口径](A1_math_memo.md)
3. [A2 — 待冻结实验提案](A2_experiment_proposal.md)

关键发现：主27场景 Gamma-off 与旧10状态 Gamma-on 不同；130条归档是 post-gate 输入，不能还原松门后的 ETA；reverse 原因经 fallback 转换后不能只看 final hard reasons；共享 Top-K 和重优化最优计数使 D4 不是 assignment freedom 的自动上界。

状态：A0_COMPLETE / A1_COMPLETE / A2_DRAFT_FOR_FREEZE / EXPERIMENT_NOT_RUN。

下一步由用户统一冻结协议，并解决既有 pickup ETA 覆盖与路线语义。未授权新增路由、训练、全天模拟、Stage3修改或Stage5。
