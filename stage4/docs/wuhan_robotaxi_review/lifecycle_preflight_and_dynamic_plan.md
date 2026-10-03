# 请求生命周期预审与有限动态对照提案

## Material Passport

- 日期：2026-09-25；代码基线：eb4f241。
- Origin Skill：academic-research-suite / experiment-agent / plan。
- 状态：READ_ONLY_PREFLIGHT_COMPLETE；动态实验：NOT_RUN。
- 核对脚本：`stage4/tools/audit_replay_lifecycle.py`；数字：`lifecycle_audit_summary.json`。
- 范围：数据时间语义、执行时间链、预测缓存时点与对照设计。不修改Stage0–3、冻结场景或论文。

## 1. 结论

1. **仿真没有额外叠加上下车停留，但不能证明GPS观察时长不含停留。** 所以新增60/120s只能暂称“额外周转开销”，不能称恢复真实缺失的上车时间。
2. **提前释放不能直接沿用旧M3缓存。** Low/Base/High下各有29,999单的输入历史时间戳不符合新释放截点。仅把缓存重命名或重跑同一输入不能修复。
3. 先考虑zero-lead下、额外pickup开销0/60/120s的小范围动态对照。它避开提前截点问题，但仍是回放环境下的机制对照，不是武汉参数标定或实时部署验证。

## 2. 当前实际时间链

| 环节 | 当前来源/行为 | 语义边界 |
|---|---|---|
| 原订单开始 | Stage0 `clean.timestamp.min()`写入Stage1 departure_time | 预处理GPS观察窗起点，不是认证呼单/车门事件 |
| 原订单结束 | `clean.timestamp.max()`写入arrival_time | GPS观察窗终点，不是认证下客结束 |
| 请求释放 | Stage4复制departure_time；native sim秒数round | canonical zero-lead代理，不还原潜在呼单 |
| 请求激活/派单 | native demand激活，30s rolling触发 | release与assignment可不同，记录activation lag |
| 车辆到达 | assignment_time + corrected pickup ETA | 当前pickup_time与boarding重合，因停留为0 |
| 开始载客 | pickup BOARDING腿duration=0 | 未单列乘客等候或车门服务时间 |
| 载客结束 | pickup_time + realized_service_time_s | 后者是原arrival−departure，不保证纯移动时长 |
| 下客处理 | dropoff BOARDING腿duration=0 | 无独立下客开销 |
| 再次可派 | native为IDLE、无assigned_route，且在availability窗口内 | 不等于无限期可用；还要等下一dispatch epoch |

载客realized时长用于assignment后的环境推进，predicted_service_time用于HV admission等决策判断。前者不能移入决策特征。当前S0 pickup beta是Test31聚合的回放交通校准，不是严格样本外ETA预测。

代码依据：`stage0/v6/stage1_production.py::_frames_for_accepted`、`stage4/replay_foundation.py::load_replay_orders`、`native_fleet_control.py::_assign/acknowledge_boarding/acknowledge_alighting/_available`及`test31_demand_adapter.py::load_all_test31_requests`。

## 3. 落盘时间守恒

检查q50 reference的18,113条assignment和q75的12,039条assignment，全部completed、关键时间无缺失：

- pickup−assignment−pickup_eta最大绝对误差约1e−9s。
- service_end−pickup−realized_service_time最大绝对误差约1e−9s。
- 超过1s的载客时间差异：0。

所以这两个运行产品没有证据表明仿真暗中再次加了boarding/dwell，也没有发现分钟/epoch截断造成额外载客时间。这个核对不证明其他所有场景或输入的事件含义。

**无法识别的部分**：GPS观察窗内端点附近停车，可能是乘客服务、交通控制、堵车、漂移或记录边界。现有order_base及assignment没有独立上下车事件，不能从低速片段唯一识别它们。武汉字段也不是与西安逐单可连接的事件。因此不能先从observed duration减60s再加60s，不能从中剪掉所有停车。

## 4. 提前释放的预测可用性

读取M3实际源表 `stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day=20161031.parquet` 的4个窄列，分批65,536行；共2,116,712 tokens、30,000单。

| release | 历史时间戳≥release的token | 至少一个此类token的订单 | 原decision晚于新release的token |
|---|---:|---:|---:|
| Zero | 0 | 0 | 0 |
| Low | 2,116,689 | 29,999 | 2,116,712 |
| Base | 2,116,689 | 29,999 | 2,116,712 |
| High | 2,116,689 | 29,999 | 2,116,712 |

`availability_timestamp`无缺失，`decision_time−feature_age_s`与其重构差异为0。Stage2 `_temporal_source`以该时间关系构造cutoff，S4 inference读取这些历史及时间特征。因此现有cache的“decision_time_only”只针对旧decision，而不自动适用于提前后的release。

这不等于证明每一个晚时间戳都改变了模型输出，也不是证明整个旧M3使用未来；zero-lead检查通过。它证明的是**不能把旧输入集合认证为新截点可用**。本次只检查时间一致性，未重新审计每个源特征的全部血缘。

已有48格对照保留冻结描述符且不推进车辆，因此其机械敏感性结论不因此失效；不可将其升级为早期预测或动态调度效果。

### 提前release动态版本的必要步骤

1. 确定新release及所选预测时点；所有历史event availability严格早于该时点，排除当前订单事后观测。
2. 重建history count/age、rolling windows、time-of-day、horizon、estimated entry/bin及依赖它们的profile查询，不能仅改decision_time。
3. 同一冻结M3、normalization、vocabulary重新inference，再用冻结CDF计算E/Q/C；无须重训，但必须新cache。
4. 原完整路线仍属于revealed-route proxy，提前知道它也是条件假设；没有OD候选生成与选择证据时，不能声称真实出发前路线完全已知。
5. 若声明实时部署验证，还需将决策ETA与Test31事后环境校准分离；否则明确为day-specific retrospective replay。

## 5. 建议先做的动态对照（待确认，不在本轮执行）

**RQ：在当前回放环境下，额外pickup服务开销如何通过车辆再次可用时间影响后续匹配？**

- q50/q75、M、p_A=.70；zero-lead、300s patience、30s epoch、原Gamma/成本/Top-K不变。
- 额外pickup开销0/60/120s；HV/AV同值，不凭武汉样本假设AV专有延迟；dropoff开销暂为0。
- 六个条件；建议观测11:00–12:00，10:30–11:00统一warmup。窗口是设计提案，不基于新结果挑选。
- 以相同canonical 10:30检查点初始化：必须恢复所有车辆位置、未完成接驾/载客腿、剩余时间、session边界、仍等待请求、CRN及累计exposure。不能只用先前fixed-state的idle车辆列表初始化动态仿真。
- 各条件独立推进warmup和观测；0s control先检查检查点状态一致及守恒。若严格复现不成立，先说明路由/时间语义差异，不在不同底座上作差。
- 观测期释放订单形成固定报告cohort；12:00后不引入新请求，但继续处理至deadline及已分配行程结束，避免末端截尾。统计观测期车辆忙时与cohort最终结果分开。

### 必须显式拆开的时间

```text
release → assignment → vehicle_arrival → service_start → service_end → next_available
                         + pickup overhead                + dropoff overhead
```

建议patience继续约束vehicle_arrival，保留原“车辆到达前最多等多久”的定义；另报request→service_start。若改为限制boarding完成，应另设实验轴，不与增加dwell混合。

计划service_end = vehicle_arrival + pickup_overhead + predicted_service_time + dropoff_overhead，HV session admission也使用该值；actual progression用realized_service_time替代预测项，但不作决策输入。若仍按旧预测结束时间放行HV，新增dwell会被错误地漏算。

此处overhead是**额外**开销：不从原GPS时长扣除任何停车，不承诺修复真实缺失时长。Gamma暴露仍对应原路线，是否包含站点开销必须沿用既有定义，不偷偷改变风险权重。

### 最小验收与资源

- 0开销的兼容行为、非零开销不能提前释放车辆、deadline指向arrival、HV admission含额外开销、时间守恒及期末cohort对账。
- 指标：cohort匹配/过期、request→arrival与service_start、有效可用车时、后续assignment数、HV跨session违例；不是要求服务改善。
- 单进程、CPU、稀疏候选；先工程短样本验证与运行预算评估，建议单条件30分钟上限、RSS2GiB警戒，不启动GPU或全天重定位。耗时超过预算则保留未完成状态，不以简化规则换取“完成”。

## 6. 当前交付与停止点

本轮只读检查耗时11.4秒，峰值工作集376.1 MiB，无路由、推理、仿真。产物已具备制定小范围对照的依据；**没有实现dwell或重新生成提前release预测**。

研究规划技能用于区分证据与设计：建议先确认上述“zero-lead + 额外开销”六条件设计；提前release实验保留为独立、需要特征重建的阶段。不能把两个问题同时改完后只汇报总服务差异。
