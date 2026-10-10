# 交通状态第一批：完成与阅读入口

状态：**第一批描述性分析完成；状态代理尚未验证为可部署交通分类。**

范围：09–27，信号盘点 → 支持度 → 候选状态 → 描述性验证。
没有修改 C/M/A、训练模型、读取 Test31、运行派单或修改论文。

建议顺序：

1. [结果与限制](report.md)：最重要的结论及下一步边界。
2. [信号和时间语义盘点](traffic_signal_inventory.md)。
3. [支持度与产物 schema](support_and_schema.md)。
4. [状态定义](traffic_state_definition.md)。
5. [执行前协议](protocol.md)、[机器摘要](summary.json)。

![状态及未知占比](state_support_overview.png)

[六组高支持路段时间序列图](illustrative_temporal_cases.png)用于查看同一路段的时变信号；
它们按支持量选取，不是随机样本，不是空间相邻路段的完整走廊验证。

## 可复现入口

代码：`stage3/scripts/traffic_state_batch1.py`。
配置：`stage3/config/traffic_state_batch1.json`。

```powershell
conda run -n stage0-valhalla python -m stage3.scripts.traffic_state_batch1
conda run -n stage0-valhalla python -m stage3.scripts.traffic_state_batch1 --summarize-only
conda run -n stage0-valhalla python -m pytest stage3/tests_odd_tod/test_traffic_state_batch1.py -q -p no:cacheprovider
```

第一条要求 output_root 尚不存在，防止覆盖已有结果；本次已经完成，不要重复运行。
复现应使用仅改变输出目录的配置副本，不删除当前产品。
第二条只读取现有日表、精确核对分类、补描述性汇总和重画图，不重拟合参考。

本地完整产品：`stage3/output/traffic_state_batch1/`，包括 19 个逐日日表、
Train 参考及示例；保持 Git 忽略。这里只提交不含订单/驾驶员 ID 的汇总、代码和图。
运行基线 commit：`298db48`。
