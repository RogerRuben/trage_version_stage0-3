# v5代码、历史输入与固定C运营对照

2026-10-11完成三项最小修改；随后按固定计划完成唯一一组C全天对照，旧v4结果保留。

- [实现、数学保证与检查报告](implementation_report.md)
- [历史目标来源与映射统计](preparation_summary.json)
- [800动作稀疏合成内核测速](kernel_check.json)
- [显式v5配置](../../../config/scheme_a_two_scale_city_v5.json)
- [运行前固定计划](execution_plan.md)
- [一次全天结果与解释](full_day_report.md)
- [完整聚合比较](full_day_comparison.json)
- [原生完整摘要](../two_scale_city_v5/chain_layout_baseline_summary.json)

模型仍是一个布局—派单—移动框架；现有Stage1节点、冻结M3产生的兼容历史模板和Stage3规则进入类型条件的目标分布。保持旧C规则、q10/P70、300秒接单、600秒独立接驾以及原资源上限。

v4默认行为、配置和已有结果保留。数学保证是已准入候选图上的主优先级保持和毫米空驶次级最优，不是全天最优。

本次实测总空驶下降15.66%、平均接驾缩短43.12秒，但少服务27单，C载客车时基本不变。只能说明组合方案的效率—服务取舍，不能声称全面优于基线、利润改善或类型调位的独立收益。唯一一组已完成；不追加三组、M/A、参数搜索或重跑。
