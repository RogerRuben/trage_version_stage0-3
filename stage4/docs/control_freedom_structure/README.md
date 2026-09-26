# Narrow structural check

最新状态：用户已授权十个冻结状态的有界确定性路由重建，真实 A1/A2 诊断现已完成。先读 [行动空间诊断报告](action_space_diagnostic_report.md)。此前输入缺口已关闭；下列初轮报告保留为历史记录。

阅读顺序：

1. [本轮结果：输入缺口与 joint-fleet 反例](structure_and_input_report.md)
2. [真实行动空间诊断任务书](../control_freedom_structure_agent_taskbook.md)
3. [八项合成核验脚本](../../tools/joint_fleet_structure_check.py)

结论：固定单车自身信息不足以表达本例的车队背景价值；简单上下文规则仍精确最优。真实 A1/A2 行动空间未计算，因为已检查文件没有完整带权 mixed candidate arcs。

不启动 Stage5，也不改 V3。最新实测：严格 pickup 下 A1 普通换车 4/10，但 HV/AV 服务指示变化 0/10；5% 预算下车型变化仅 1/10。完整重建 arcs 与 witness 已保存，不再需要从 selected-only 表推测替代解。
