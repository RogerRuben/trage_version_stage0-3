# 场景证据盘点与武汉研究的校准适用性

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / plan
- Origin Date: 2026-09-25
- Verification Status: ANALYZED（本地代码与已有落盘报告核对；未重跑实验）
- Version Label: scenario_calibration_inventory_v1
- 基线：960451f；目的为场景设计，不修改论文、41个冻结场景或模型。

## 1. 判断

武汉研究值得用于改进**运行场景的结构和外部合理性检查**，但现有公开材料不足以标定西安的乘客耐心、AV接受率、在线供给或能力边界。建议分三层：

1. 西安历史数据估计可识别的交通/供给参数。
2. 武汉完成订单样本提供请求生命周期与停留时间的外部数量级参考。
3. 无法识别的行为参数明确作为敏感性轴，不伪装成实证估计。

不要以“复现武汉完成订单等待分布”为目标同时调整 lead、patience、车数和ETA；多个参数可补偿得到同样分布，且完成样本受到选择。

## 2. 已有证据，哪些不必再做

| 对象 | 已完成 | 仍不能回答 | 处理 |
|---|---|---|---|
| 全日基准 | 41/41完成，30,000单/场景，zero-lead、300s patience | 其他时间假设下的全日服务率 | 保留，不整体重跑 |
| 类型与availability | 继承availability后的neutral identity 10/10 PASS | AV全时在线是否贴近现实 | 不重复identity测试 |
| E/U/M门机制 | 10状态M合计718→493→238→124→124→12 | 混合系统全天总服务的分解 | 保留固定状态证据，不扩大其含义 |
| Top-K | K10/20/40/80在10状态最终M一致 | 所有时间、所有策略都不敏感 | 不继续扫K |
| 请求释放时间 | 4物理状态×4 lead变体，M合计8/26/25/40 | 独立patience效应、反事实车辆历史 | 缺口见下一节 |
| RT恢复 | gap分位420/748/1733/3693s，15项旧指纹通过 | 真正呼单时间、原始manifest | 不再恢复或重拟合旧RT |
| 供给空间重建 | Test31全量与样本TVD .054、相关 .965 | 在线未服务车辆、未来部署选址 | 保留，不按武汉活动图调车位 |
| 归一化 | S5A.1精确H_base=12,279.336389车时 | 物理车数/真实市场渗透率 | 使用修正版，不引用旧14,369.5作分母 |
| repositioning | 确定路由下首个全日控制运行9400s无有效结果，按成本停止 | 重定位有无效益 | 不能写无效；不自动重启 |
| prediction common evaluator | NOT_RUN | decision superiority | 当前校准不需启动此独立问题 |

核对的落盘产品：`stage4/output/paper_enhancement/mechanism_validity/{summary.json,matching_capacity_summary.csv,request_time_sensitivity_summary.json,request_time_sensitivity.csv}`。旧 `stage4/docs/results/request_time_sensitivity_summary.csv` 只有表头，不能当成旧ABM已完成敏感性实验。当前统计应引用后来的mechanism_validity产物。

部分旧文档是历史快照：`canonical_request_time_setting.md`末段仍写RT恢复blocked，已被恢复报告及闭环报告取代；旧S5A参数表也已被S5A.1取代。本次不改写历史记录。

## 3. 两个实质缺口，一个解释边界

### 3.1 现有“RT/patience”实验没有改变patience上限

`rt_patience_sensitivity.py`锁定配置300s；`frozen_state_prediction_ablation._waiting_requests`与`mechanism_validity.gate_graphs`也有300s常数。已有结果检验的是固定300s下，release变化导致等待队列及剩余耐心变化，不是lead×patience完整二维检验。

因此，若要改进运行场景，最优先补的是独立patience轴。不能只改配置JSON：waiting membership、deadline、critical标记、路由ETA过滤必须同步使用同一个参数，否则表面上改变耐心，实际仍有旧300s筛选。

### 3.2 上下车停留时间目前为零

`native_fleet_control.py`的pickup BOARDING及dropoff BOARDING腿均为`duration=0.0`；`replay_service_time_adapter.py`亦如此。载客腿使用冻结`realized_service_time_s`。

武汉样本到达起点→开始行程中位60s、P90=120s，提示零停留值得检验。但它不是纯车门操作测量，包含乘客到达/等待等；不能把整段一律当技术boarding time，也不能用它推断下客停留。

若增加停留，先核对西安轨迹起止是否已经包含驻留，避免重复分配时间。一次性加60s到汇总结果不足以验证全日效应，因为它会改变下一次availability和车辆位置历史。

### 3.3 ETA“校准”不是严格样本外预测

S0的当前校准使用Test31聚合时段ATA/Valhalla比值；`replay_foundation.py`已明确这是day-specific replay traffic calibration，不是严格out-of-sample decision-time predictor。不能拿较早的`pickup_eta_calibration_method.md`中Train-only circuity描述代替当前FleetPy实现。

这不是本轮新发现的未声明bug，但若升级为前瞻部署场景，应分离“仿真环境旅行时间”和“决策可用ETA”。单纯把release提前而不刷新信息截点，不能宣称因果可用的预测验证。

## 4. 武汉材料具体能校准什么

| 目标 | 公开证据 | 合理用途 | 不允许的移植 |
|---|---|---|---|
| request→boarding lead | 完成样本P50=360s，分钟精度 | 外部量级；与西安重建lead比较 | 逐单反推真实西安request；按事后等待做决策 |
| pickup patience | 完成单request→arrival P50=300s，45.3%超过300s | 提醒300s为有约束性的假设 | 把中位等待设为耐心中位数；拟合取消hazard |
| pickup dwell | 到达→开始P50=60s、P90=120s | 外部锚定的停留敏感性范围 | 当成精确车门时长或HV/AV差异 |
| availability | 完成订单中402辆车的活动记录 | 描述已服务活动 | 当在线小时或将无订单视为离线 |
| service domain | 活动网格与强度不同 | 设置“服务域”和“域内兼容性”两个不同概念 | 将零活动变为禁止通行，按Test31结果选有利区域 |
| 路线复杂度 | 距离加权OPCI均值差异 | 外部选择/暴露证据 | 调Gamma使西安均值等于武汉；设SNE安全阈值 |
| 乘客接受率 | 未获得与当前选择模型同定义的偏好数据 | 保留0.4/0.7/1.0情景标签 | 用完成率或Robotaxi订单份额估计p_A |

资料边界：单城、不同年份、单日处理完成样本；取消时间字段有歧义。补充源表有日级取消/成功汇总，但没有对应个体风险集与有效取消时刻，不能由汇总识别patience分布。

## 5. 建议最小实验：先机制，再动态场景

下述是待确认的设计提案，不是已经执行的实验或已经冻结的新参数。没有以预期服务率改善作为通过标准。

### A：lead × patience独立敏感性（优先）

- RQ：现有匹配容量损失和q排序，对耐心上限有多敏感？
- 复用4个q50/q75、12:00/17:30固定物理状态及4种既有lead。
- 建议patience为180/300/600s，共48格，其中300s的16格必须先复现已有E/U/M；180/600只是压力测试值，不是武汉估计。可先用48格避免扩展其他维度。
- 固定M profile、p_A=.70、种子、位置、历史assignment、routing clock；不训练模型。
- 记录各门E/U/M，尤其同时输出AV子图和mixed图，避免把AV机制当成总容量分解。不得将HV和AV匹配数简单相加。
- 同一固定候选集合上放宽patience的M应非递减；若等待队列随patience变化，必须另外报告cohort和分母变化，不能把不同队列差值当个体效应。报告retained cohort与新增等待cohort。
- 延长patience会纳入canonical已过期但未匹配订单；这是条件反事实，依然不能重建另一条车辆历史。
- 所有routing按实际稀疏候选并集缓存，每状态释放；CPU单进程，无GPU、无稠密矩阵。先确认路由服务可用，再设小型任务预算；若超预算停止并报告，不升级全天。
- 成功标准：基准复现、单位/状态对账、无隐藏300s常数，而不是某个方向的结果。

### B：请求生命周期的最小动态对照（有条件）

只有A完成且决定研究问题是“全日/动态服务后果”时，才补独立运行场景。结构必须区分release、assignment、vehicle arrival、boarding、dropoff、next availability。

建议先预注册一个有限时窗及warmup，固定q50/q75两档；分别改变lead、patience、pickup dwell，不一次全部改完。停留对照可用0/60/120s，明确为外部锚定情景。两类车辆先采用相同停留，避免把一般站点耗时与车辆类型混杂；AV特有差异需要另外证据。

若决策时间提前，动态特征必须在新release截点计算；不能继续把旧时刻描述符称为早期可用。若只做“固定描述符的机械环境敏感性”，必须明确降格。选取策略、时间窗和预算后方可执行，不偷偷恢复已因成本停止的全日重定位任务。

### 暂不补

不重复Top-K/identity/RT恢复；不全量重跑41场景；不同时新增重定位、充电、地理围栏、乘客choice model；不追求拟合武汉服务率；不继续SNE门阈值搜索。

## 6. 需要先选定的科学目标

最小A回答“机制对时间假设是否稳健”，B才开始回答“不同运行环境下总服务会怎样”。武汉研究最值得借鉴的是运行生命周期与服务域分层，不是一个万能校准数字。

建议先确认A的180/300/600s设计，再执行；B保留为A之后的独立决策。本轮完成盘点，没有启动新的路由、仿真或修改冻结参数。

## 7. 主要本地依据

- `stage4/docs/final_experiments/stage4_final_experiment_execution_summary.md`
- `stage4/docs/vehicle_hour_normalization_correction/stage4_s5a1_vehicle_hour_normalization_summary.md`
- `stage4/docs/paper_redesign/mechanism_validity_closure_report.md`
- `stage4/docs/paper_redesign/request_time_patience_sensitivity_report.md`
- `stage4/docs/paper_redesign/efficient_repositioning_report.md`
- `stage4/docs/replay_foundation/stage4_s0_replay_foundation_summary.md`
- `stage4/docs/wuhan_robotaxi_review/request_lifecycle/README.md`
- `stage4/docs/wuhan_robotaxi_review/spatial_activity/README.md`
- `stage4/config/experimental_design/acceptance_design.json`

解释纪律：区分完成样本与请求总体、按车/日相关性与独立样本、聚合与个体、相关与因果；没有p值、置信区间或择优调参。已有结果本轮只核对，不称重新复现。
