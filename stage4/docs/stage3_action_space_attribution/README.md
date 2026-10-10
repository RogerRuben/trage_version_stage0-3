# Stage3 → action-space attribution: pre-experiment review

## 当前状态：支线关闭，接受冻结接口边界

停止追加审计、Stage3修复、重新聚类、组合消融和Stage5扩展，回到论文收敛。
这不是宣告接口无误：端点/movement覆盖缺口保留，并已纳入主稿正文及附录。
41场景保持不变；没有估计修复后服务收益。
见 [收敛决定](../manuscript_v3/research_boundary_closure.md)。

已完成：[执行](execution_report_v1.md)、[D13](d13_report.md)、
[来源审计](provenance_audit_report.md)、[原生端点](native_endpoint_candidate_report.md)、
[影子对账](endpoint_shadow_reconciliation_report.md)、[55节点](native_node_complex_context_report.md)。

以下为历史冻结前状态，不再表示待执行任务；历史报告的修复建议不构成当前授权。

最新：用户冻结v1协议后，Q0 + Phase0已经完成，见 [阶段报告](Q0_Phase0_report.md) 和 [冻结协议v1](stage3_action_space_attribution_protocol_v1.md)。Q0 PASS；输入状态 READY_AFTER_BOUNDED_PICKUP_ETA_COMPLETION；独立缺失pickup keys=16261。未补路由，未运行最终50条件。

2026-09-18。只完成审计、数学说明、实验提案；**没有启动新实验**。

1. [A0 — 仓库审计](A0_repository_audit.md)
2. [A1 — 数学与归因口径](A1_math_memo.md)
3. [A2 — 待冻结实验提案](A2_experiment_proposal.md)

关键发现：主27场景 Gamma-off 与旧10状态 Gamma-on 不同；130条归档是 post-gate 输入，不能还原松门后的 ETA；reverse 原因经 fallback 转换后不能只看 final hard reasons；共享 Top-K 和重优化最优计数使 D4 不是 assignment freedom 的自动上界。

状态：A0_COMPLETE / A1_COMPLETE / A2_DRAFT_FOR_FREEZE / EXPERIMENT_NOT_RUN。

上述A0/A1/A2保留为冻结前草案，冲突处以v1协议为准。下一步由用户审查Q0/Phase0后授权，才补pickup ETA并运行最终50条件；训练、全天模拟、Stage3修改与Stage5均不在范围内。
