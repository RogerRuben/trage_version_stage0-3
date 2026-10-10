# 支持度及产物说明

## 四种支持需分开

| 支持层 | 本批可提供 | 不能替代的证据 |
|---|---|---|
| 当前窗口 | order_count、joint_order_count、完成时刻跨度、最新证据 age | 完整窗口/完整 link 覆盖；车辆独立性 |
| 低 pace 参考 | reference_order_days、reference_days、q20/q50 | 真实 free-flow 标定 |
| 模型 | 冻结训练范围、预测 cache 来源及已有 support_group_code 的 schema | 新状态分类预测精度、当前实时观测支持 |
| 测量质量 | 原 history 来源、有效 pace、lcs_available 及组件非空 | 缺失的 interval/pair/distance coverage 计数 |

当前状态最低 3 个 joint-valid 订单身份、事件跨度 300 秒、最新证据 age<=600 秒；
参考最低 100 order-days、5 天。都是候选规则，支持充分不等于真值已验证。

## 日表

路径：`stage3/output/traffic_state_batch1/cells/date=YYYYMMDD.parquet`。
主键：`date + observed_directed_edge_uid + window_end`。

- `date`：源订单日期，已去掉 completion date 不一致的 351 条事件。
- `window_end`：UTC Unix 秒，解释为 Xi'an local 半小时窗口末端。
- `canonical_highway / road_class_count`：道路类别；实测冲突 cell 数为 0。
- `event_count / order_count`：所有完成事件及不同订单身份数。
- `*_order_mean / *_order_count`：先订单内均值、再订单间均值及非空订单数。
  CV/acceleration 的 count 是 presence，不能直接叫 target-valid。
- `joint_order_count / joint_pace / joint_low_speed_share / joint_stop_share`：
  来自相同 joint-valid 事件集合的订单等权汇总，避免分母人群错位。
- `joint_first / joint_last / joint_event_span_s / latest_joint_age_s`：
  joint 证据完成时刻的跨度与新鲜度；不是 GPS 覆盖持续时长。
- `reference_order_days / reference_days / reference_pace_q20 / reference_pace_q50`。
- `pace_ratio`：joint_pace/reference_pace_q20，不是标准 TTI。
- `state`：三个代理状态、MIXED_EVIDENCE 或 UNKNOWN。
- `role`：train_descriptive_in_sample 或 validation_descriptive。
- `source=COMPLETED_DIRECT_TRAVERSAL_EVENTS`。
- `decision_time_deployable=false`：本批产品不能直接作为在线接口使用。

`train_reference.parquet`：每方向 edge 一行，共 11,724 行；4,644 个 edge 达到参考
最低支持。`illustrative_cases.parquet`：6 组 Validation edge/day 的完整观测窗口。

## 分母

有观测 cell：2,066,052；非 UNKNOWN：465,006（22.507%）。
19 天观察到的方向 edge 并集：13,941。
在“这个并集 ×19天×48桶”内，共 12,714,192 个可能 cell，其中 10,648,140
没有事件。这里只计算数量，不物化 1,000 多万空行；缺测隐式为未知。
有观测占约 16.25%，非 UNKNOWN 约占该并集空间的 3.657%。
**这不是全路网覆盖率**；观察并集以外的道路没有进入分母。

order_cell_exposures 是订单在不同 cell 的重复暴露计数，events 是 traversal
事件数，都不能称“独立订单数”或车流量。不把样本量当交通流量，不估计密度。

## 输出规模与资源

逐日读取和释放对象，只跨日期保留紧凑数值 reference 池；无全路网稠密矩阵。
本次主分析 70.22 秒、峰值 RSS 546.69 MiB、GPU 0。
主数据保留在原有 Git-ignored 输出树；版本库仅提交汇总和代码。
