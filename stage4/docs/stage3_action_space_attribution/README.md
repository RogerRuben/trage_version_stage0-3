# Stage3 → action-space attribution: pre-experiment review

最新：用户冻结v1协议后，Q0 + Phase0已经完成，见 [阶段报告](Q0_Phase0_report.md) 和 [冻结协议v1](stage3_action_space_attribution_protocol_v1.md)。Q0 PASS；输入状态 READY_AFTER_BOUNDED_PICKUP_ETA_COMPLETION；独立缺失pickup keys=16261。未补路由，未运行最终50条件。

2026-09-18。只完成审计、数学说明、实验提案；**没有启动新实验**。

1. [A0 — 仓库审计](A0_repository_audit.md)
2. [A1 — 数学与归因口径](A1_math_memo.md)
3. [A2 — 待冻结实验提案](A2_experiment_proposal.md)

关键发现：主27场景 Gamma-off 与旧10状态 Gamma-on 不同；130条归档是 post-gate 输入，不能还原松门后的 ETA；reverse 原因经 fallback 转换后不能只看 final hard reasons；共享 Top-K 和重优化最优计数使 D4 不是 assignment freedom 的自动上界。

状态：A0_COMPLETE / A1_COMPLETE / A2_DRAFT_FOR_FREEZE / EXPERIMENT_NOT_RUN。

上述A0/A1/A2保留为冻结前草案，冲突处以v1协议为准。下一步由用户审查Q0/Phase0后授权，才补pickup ETA并运行最终50条件；训练、全天模拟、Stage3修改与Stage5均不在范围内。
