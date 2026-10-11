# v5调位模块：一次组件对照

唯一新增通用历史调位＋主最优面空驶优化一组；旧v4与完整v5直接复用。

- [固定运行计划](execution_plan.md)
- [结果与下一控制参考建议](report.md)
- [完整三版本聚合比较](component_comparison.json)
- [新增原生摘要](../two_scale_city_v5_common/chain_layout_baseline_summary.json)
- [独立配置](../../../config/scheme_a_two_scale_city_v5_common.json)
- [只读分析入口](../../../analysis/operations_v5_ablation_results.py)

新增组相对v4：净少8单、总空驶−16.37%、平均接驾−45.04秒。类型调位相对新增组：C多14单，但全队少19单、总空驶+307.43km。建议以通用＋次级优化作为下一轮统一布局比较的参考，不声称所有指标都支配或多日普适。

本轮仅完成这一组，不生成新布局或更多实验。原生产品在本地忽略目录，远端只提交代码和聚合报告。
