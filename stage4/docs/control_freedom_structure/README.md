# Narrow structural check

阅读顺序：

1. [本轮结果：输入缺口与 joint-fleet 反例](structure_and_input_report.md)
2. [真实行动空间诊断任务书](../control_freedom_structure_agent_taskbook.md)
3. [八项合成核验脚本](../../tools/joint_fleet_structure_check.py)

结论：固定单车自身信息不足以表达本例的车队背景价值；简单上下文规则仍精确最优。真实 A1/A2 行动空间未计算，因为已检查文件没有完整带权 mixed candidate arcs。

本轮不启动 Stage5，也不改 V3。继续真实诊断需要找到完整既有弧归档，或由用户另行授权有界确定性路由重建；不得从 selected-only 表推测替代解。
