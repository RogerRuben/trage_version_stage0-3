# 第二批执行前范围

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: authorized descriptive analysis
- Date: 2026-09-27
- Status: protocol before execution

用户授权“下一批”，本批覆盖支持恢复、第 5 步交叉解释、第 6 步 profile 假设草案。
不实施草案、不改冻结 profile、不训练、不派单、不读 Test31，不改第一批阈值/状态。

## 已找到的数据路径

Stage1 output_v3 bucket 仍拒绝访问（含 ACL 只读查询），不修改权限。
但 input_v1 的 link_interval_observations 和 link_traversals 可读。
按原 direct_observed + label_valid 聚合计数/时长/距离/窗口，恢复测量支持；
不重新计算 Stage1 标签，组件有效性直接复用 hash-bound Stage2 路线表的 target masks。

测量恢复：09–27，按日期→bucket 处理，仅把汇总与 Validation 路线描述写新目录。
distance/allocated_distance 是 traversal 覆盖，distance/canonical_length 另列，
两者都不是整个 half-hour 的监测覆盖。0.5/0.8 仅用于描述分层，不作新筛选阈值。

## 两条比较线

1. 预测定义重叠：Validation 25–27，现有 M3 预测 pace、crawl+stop 与第一批 q20
   参考派生 proxy；不要求假造“当前有3个订单”。缺参考或无效预测保留 UNKNOWN。
   用原 P50 时间权重聚合预测状态占比/连续暴露，与已冻结同订单 12 个 E/Q/C
   及三个 profile 的 dynamic caps 比较。这只是同一预测信号上的定义重叠。
2. 观测事后解释：按完成时间匹配第一批原状态，使用恢复的直接观测时长聚合；
   是部分观测路线的描述，与原全路线 E/Q/C 不在同一分母上。
   标明同一订单参与了 cell 构造，不当独立真值或预测精度，不用于派单。

不读取 458 MB exact CDF 来重算 E/Q/C，复用已有 route descriptors。
统计包括所有预定相关系数（描述性 Spearman，无 p 值）、profile×状态占比、
“crawl/stop 已超限”和“CV/加速度独立超限”的交叉表；不将相关性宣称独立因果。
无支持、混合证据和观测范围都不能补零成畅通。

## 适配草案边界

保留结构性规则、速度域与动态交通的区别。
候选嵌套策略：C 的目标环境为非拥堵，M 扩展到拥堵，A 扩展到严重拥堵。
这是运营策略假设，不是 AV 能力事实；UNKNOWN/MIXED 对所有 profile 保留不确定。
正的暴露容忍预算本批不按结果拟合，严格零暴露版本只能作为待批准的对照假设。
不得同时把 crawl/stop E/Q/C 和同源状态暴露重复算成独立惩罚。

## 运行和停止

单进程 CPU、逐日窄列读取；RSS 采样、30 分钟硬超时，不自动重试失败分析。
命令：`python -m stage3.scripts.traffic_state_batch2`。
最小测试：区间支持重构、方向与缺值、连续暴露断点、动态上限嵌套。
报告完成后停止在草案，不运行接口部署/动态对照；提交并推送已授权 origin。
