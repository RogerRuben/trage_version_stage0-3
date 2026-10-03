# 六条件动态实验前置检查：状态边界问题

## Material Passport

- 日期：2026-09-26；起始基线：cf90224。
- 状态：STRICT_PRE_EPOCH_2_OF_2_PASS；NATIVE_CHECKPOINT_NOT_RESTORED；SIX_DYNAMIC_CONDITIONS_NOT_RUN。
- 目的：验证10:30初始化，不引入dwell、不修改正式41场景、不运行全天仿真。

## 1. 已确认的代码问题

`frozen_state_prediction_ablation._vehicle_state`将assignment_time≤t视为已经发生，排除当前epoch已派单车辆；`_waiting_requests`仅排除assignment_time<t订单。因此二者组合形成“派单前请求+派单后车辆”的混合时点，并非完整的派单前状态。

这会系统地排除canonical在该epoch选择的车辆，存在对当前决策结果的条件化。不能仅靠neutral identity通过、旧结果重现或参数单调性排除它。

新增`pre_decision_vehicle_state`使用严格<t。旧函数保留为历史复现接口并增加警告；旧报告数字不静默覆盖。边界测试覆盖当前派单车辆、此前仍忙车辆及恰好在t完成的车辆。

## 2. 两个实际epoch核对

时间固定为2016-10-31 10:30。重建相同完整fleet fixtures、等待请求、此前累计Gamma状态、M/p=.70规则；通过生产候选构造器截取弧，再使用原Gamma约束求解。路由使用既有单源确定matrix，不构建稠密订单×车辆矩阵。

| 指标 | q50原值 | q50旧恢复 | q50严格恢复 | q75原值 | q75旧恢复 | q75严格恢复 |
|---|---:|---:|---:|---:|---:|---:|
| 可用车辆 | 367 | 359 | 367 | 396 | 389 | 396 |
| 等待订单 | 64 | 64 | 64 | 95 | 95 | 95 |
| 有效候选弧 | 19 | 9 | 19 | 13 | 2 | 13 |
| 选中订单 | 8 | 4 | 8 | 7 | 2 | 7 |
| 选中pair对称差 | 0 | 12 | 0 | 0 | 9 | 0 |

严格恢复的pickup目标值也逐值相等：q50=1278.5862265889114s，q75=958.3237806225172s。候选弧身份原始全集未保存，因此19/13是数量对账；选中pair身份有assignment日志可核对且完全一致。该两epoch不足以证明所有状态或所有未来轨迹完全一致。

旧恢复检查25.84s、峰值297.8MiB；严格恢复21.73s、峰值304.9MiB，均0路由失败。结果分别保留在本目录 `dwell_checkpoint_legacy.json` 和 `dwell_checkpoint_strict.json`。

## 3. 对之前结果的影响

调用同一旧函数的分析包括frozen-state prediction ablation、mechanism_validity、rt_patience_sensitivity、lead_patience_factorial。相关结果只能作为旧混合状态的历史计算，不应继续支撑完整派单前容量、q排序或prediction决策收益结论。需要重新界定/核算，而不是只修图表标题。

这**不推翻正式41场景的native动态运行结果**：正式runner并不调用该分析状态恢复函数。它也不影响GPS时间语义、提前release特征截点检查或武汉源数据分析。

本次没有按结果改变Gamma、patience、q、模型或窗口，没有重算或覆写旧48格。新两epoch提供故障定位与严格边界验证，不是六条件实验结果。

## 4. 动态检查点仍缺什么

正式目录仅保存assignment、request outcomes、epoch与exposure等产品；`save_final_state`是空实现，没有native序列化checkpoint。`resume`功能是按场景跳过已完成任务，不是从中途恢复车辆。

10:30已有q50=247个未完成assignment（29个接驾、218个载客），q75=157个（18/139）。不能将这些车辆丢掉，或全部当作在终点等待到completion；必须恢复native腿、当前位置、剩余时间、request对象、状态回调、network注册指标、累计暴露及待请求状态。仅idle图核对通过还不等于动态checkpoint已恢复。

日志和当前线性native network足以作为恢复依据的一部分，但仍需构建与测试恢复器；不声称数据不可恢复，也不虚构已有checkpoint。若选择从午夜重放生成检查点，必须先确定运行预算；本轮没有自动扩大到前半日重跑。

## 5. 建议先后顺序与当前停止点

建议先用严格pre-decision状态修复并重算受影响的固定状态证据，保持旧产品作为lineage；再实现完整native checkpoint，执行0-overhead continuation验证；最后才做六条件动态对照。

原因是新增时序问题影响上一轮科学依据，继续叠加dwell会将问题混在一起。当前已完成两个epoch的定位和验证、提供显式pre-decision接口与测试，并暂停六条件实验。下一步需用户决定是否优先重算受影响分析；这属于对既定六条件任务顺序的实质调整。
