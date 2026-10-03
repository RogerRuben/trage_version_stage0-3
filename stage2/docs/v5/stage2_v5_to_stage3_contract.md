# Stage 2 v5 → Stage 3 随机服务状态契约

Stage 3 的职责是聚合路线条件随机服务状态并比较候选路线。它不做 AV 能力、失效或安全概率映射。

## 允许读取

- traversal pace/time 的 point、P50、P90、P95 与分布参数；
- route service-time scenarios、mean/std/P50/P90/P95、CVaR90/CVaR95；
- 使用外部给定 service-time threshold 计算的 timeout probability；
- RTS raw/pct、校准 tail probability；
- LCS crawl/stop/speed-CV/acceleration 组件，仅作辅助状态；
- label availability probability、history/support/coverage；
- prediction interval、entry-time uncertainty；
- model ID、scenario generator ID、seed、输入 hash、代码/config/split provenance。

## 禁止读取或解释

- 订单结束后才可得的真实 travel time 作为决策输入；
- oracle timing track；
- Stage 1 真实标签；
- 未校准概率；
- AV 安全、失效、事故、接管或 ODD 概率；
- 把 RTS 或 LCS 当成物理服务时间的替代量。

## 必要字段约束

```text
decision_time < every source event time cutoff
traversal_time > 0 when allocated_distance_m > 0
pace_p50 <= pace_p90 <= pace_p95
route_time_scenario == sum(traversal_time_scenario)
route_p50 <= route_p90 <= route_p95
timeout threshold provenance == external
scenario seed/model/input hash present
```

缺失预测保持 NaN 并带 availability/coverage，不得填 0。Stage 3 只能消费 release verification 中声明为 eligible 的 track。

## Stage 3 准入

准入状态由 v5 verification 的真实条件计算：`NOT_READY`、`READY_FOR_ROUTE_SCENARIO_PROTOTYPE` 或 `READY_FOR_STAGE3`。不得在 manifest 或报告中硬编码状态。

