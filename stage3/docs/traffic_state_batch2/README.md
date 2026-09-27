# 交通状态第二批：测量支持、E/Q/C 重叠与 profile 草案

**计算完成；适配草案未接入生产。** 基线 commit：4062ea4。

- [结果报告](report.md)
- [C/M/A 适配假设草案](profile_adaptation_assumptions.md)
- [执行前范围](protocol.md)
- [测量恢复数字](measurement_recovery.json)
- [原动态上限家族交叉表](dimension_family_cross_table.json)
- [描述性相关矩阵](descriptive_correlations.json)
- [运行与来源摘要](summary.json)

![现有动态上限与状态暴露](eqc_state_overlap.png)

本地完整产物：`stage3/output/traffic_state_batch2/`。
包含 19 个 cell_support 日表和 3 个 Validation route_overlap 日表，合计约 55 MiB。
逐路线产品含订单 ID，保持 Git 忽略；本目录汇总不包含订单或驾驶员 ID。

运行方式：

```powershell
conda run -n stage0-valhalla python -m stage3.scripts.traffic_state_batch2
conda run -n stage0-valhalla python -m pytest stage3/tests_odd_tod/test_traffic_state_batch2.py -q -p no:cacheprovider
```

已运行完成，不要重复启动。运行器要求输出目录不存在，以避免覆盖；复现时使用
只改变 output_root/report_root 的配置副本。不删除当前产品，不重算 Stage1 标签。

范围：09–27 测量支持、25–27 解释性对照；没有 Test31、训练、推断、派单或 profile 修改。
