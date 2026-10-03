# Stage3 → action-space attribution: frozen-state execution report

基线：`e9ad0d20feca2f81acdf93dff231ada89b64363e`。执行日期：2026-09-21。

## 1. 授权与停止点

用户在 Phase0 审查后明确授权补齐最多 16,261 个 pickup routing keys；全部 16,386 keys 解决后自动运行 D0–D4 × 10 states。该授权替代 protocol v1 的旧 Phase0 人工停止点，其科学定义不变。

本轮已完成，**STOP**。没有修改 Stage3、V3、41 个全天场景，没有 service-route rerouting、fallback generation、M3 inference、profile 修改、车辆推进或 FleetPy 全天仿真。没有启动 Stage5。只提供归因证据，不自动选择研究分支。

## 2. 最重要的结论

Stage3 放松能增加可匹配容量，也能在部分状态打开低代价车型置换；但并非所有新增容量都变成同订单 H↔A 自由度。数据不支持“Stage3 可以排除为原因”，也不支持立即修改 reverse 或 evidence 规则。

| 条件 | 最终 E_AV | 最终 U_AV | 最终 M_AV | 最终 O_HA | A1 ≤5% 状态数 | A2 ≤5% 状态数 |
|---|---:|---:|---:|---:|---:|---:|
| D0 CURRENT | 48 | 12 | 12 | 3 | 1/10 | 1/10 |
| D1 NO-REVERSE | 115 | 41 | 39 | 9 | 4/10 | 4/10 |
| D2 NO-CAPABILITY-HARD | 52 | 14 | 13 | 3 | 1/10 | 1/10 |
| D3 NO-EVIDENCE | 139 | 45 | 44 | 7 | 4/10 | 4/10 |
| D4 STAGE3-NEUTRAL | 509 | 119 | 113 | 15 | 6/10 | 7/10 |

E/U/M/O 是**十个状态分别计算后求和**，不是全天独立订单总量，也不是把十个状态拼成一个图求最大匹配。最终层包含 pickup/patience 和 session。M_AV 是 AV 子图容量，不等于 mixed-fleet service，也不是实际派单总数。

### 预注册分类

- `STAGE3_SUBSTITUTION_BOTTLENECK`：**未达到冻结 screening 标准**。D1/D3 各新增 3 个 ≤5% A1 状态，但只有 2 个同时增加最终 O_HA，少于要求的 3 个。
- `STAGE3_CAPACITY_ONLY`：适用于 D2，M_AV 12→13，A1 未新增。这只说明本样本的 M-profile U-turn hard-rule 单因素信号，不涉及 C/A profile。
- `STAGE3_INTERACTION`：联合放松的 screening signal。D4 新增 5 个 A1 状态，而单因素未达完整判据。**不是统计交互效应估计或已证明某两个机制相乘**。
- `DOWNSTREAM_SPATIAL_PATIENCE_BOTTLENECK`：下游损失明确存在，但没有冻结“主导瓶颈”的数值界限，保留 `REVIEW_REQUIRED`，不强制贴排他标签。
- `STOP_STAGE3_INVESTMENT_ON_THIS_SAMPLE`：**不满足**。D4 新增 A1=5、A2=6，超过停止规则的 0–1。

这组分类不等于授权任何上游修改。尤其不能由 D1 推断现实中逆行合法，也不能由 D3 推断缺失预测已被补齐。

## 3. 相对 pickup budget 的关键限定

每个条件独立计算 lexicographic critical/total/carryover 最优值，再求最小 aggregate pickup P* 与参考 x*。A1 固定该条件的 served-order set；A2 固定该条件的三个 count 最优值。所有 gap 是该条件参照下的全局最小 alternative pickup gap（求解器最优性证书与 1e-7 秒数值核验），不是枚举启发式。

| 条件 | A1 0% | A1 0.5% | A1 1% | A1 2% | A1 5% | A2 0% | A2 0.5% | A2 1% | A2 2% | A2 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| D1 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 1 | 4 |
| D2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| D3 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 4 |
| D4 | 0 | 0 | 1 | 3 | 6 | 0 | 0 | 1 | 4 | 7 |

D1/D3 的 A1 新增状态同为 07:30、18:00、18:30；均无 1→0 状态。只有后两者同时增加 O_HA。D4 新增 07:30、08:30、13:00、18:00、18:30；A2 额外新增 12:00，不能把该 A2 结果自动叫同订单车型置换。

07:30 的 A1 最小**绝对** gap 在 D0–D4 都是约 59.749033 秒，但相对 gap 为 6.8345%、4.7739%、6.8345%、3.4050%、3.0400%。这里的 ≤5% 跨线来自条件最优 P* 分母扩大，不能称为车型置换绝对成本降低。这也解释了为何必须同时要求 O_HA 增加，而不能只看 1/10→4/10。

完整逐状态 exact minimum absolute/relative gap（包括 >5% 与不可行）在 `results_v1/endpoints.csv`、JSON。`INFEASIBLE` 的 null gap 表示不存在满足约束的 alternative，不表示 0 或缺失 ETA。有限相对 gap 在 P*=0 时定义为 null。当前结果没有未决 MILP。

## 4. 两层图与下游损失

所有条件共享空间候选全集。每个条件按冻结 eligibility 和 passenger CRN 重选 shared HV+AV Top-K20；没有为了性能截断候选全集。

| 条件 | pre-TopK M_AV | Top-K M_AV | patience M_AV | session M_AV | patience O_HA | session O_HA |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 124 | 124 | 12 | 12 | 5 | 3 |
| D1 | 275 | 274 | 39 | 39 | 15 | 9 |
| D2 | 131 | 131 | 13 | 13 | 5 | 3 |
| D3 | 238 | 238 | 44 | 44 | 12 | 7 |
| D4 | 493 | 492 | 113 | 113 | 28 | 15 |

Passenger 后共同 M_AV=493、O_HA=493。D4 已放松 Stage3，仍从 Top-K M_AV=492 降至最终 113；session 还把 O_HA 28→15。这是本样本下游 pickup/patience/availability 仍很强的直接证据，但不能把空间分布、等待预算、availability 各自的因果贡献分离出来。

与 D0 相比，被新增 AV 挤掉的 HV Top-K state-arcs：D1=1,591、D2=55、D3=1,173、D4=3,925。跨条件生产图不嵌套，D4 不是 assignment 上界。各条件基线 served-set 的增加/移除数量在 `baselines.csv`。

`graph_layers` 给出每个 state/condition 在 SPATIAL、PASSENGER、PHYSICAL_DIRECTION_HARD、CAPABILITY_HARD、STRUCTURAL_ROUTE、PRE_TOPK_EVIDENCE、TOPK、VALID_ETA、PATIENCE、SESSION 的 E_AV/U_AV/M_AV/O_HA。前六层只是条件性机会，不是已具备及时服务能力。

PHYSICAL_DIRECTION_HARD 只是排除所选 route provenance 中除 UTURN_PROFILE_INCOMPATIBLE 外的 hard reasons；**该名称不意味着本轮认证了真实道路法律限制**。没有建立的路线/unknown 在 STRUCTURAL_ROUTE 层排除。D1/D2 使用 Phase0 已冻结的 route-aware original/K1 结果，不能把这些层的差额视作纯独立单规则归因。

## 5. 三类 provenance 漏斗

下表数字为具有 AV option 的 state-order 数 U_AV；分类可以重叠，不可相加归因。

| 原始/最终 provenance 分组与诊断条件 | 空间 | passenger | pre-TopK evidence/eligibility | Top-K | patience | session |
|---|---:|---:|---:|---:|---:|---:|
| 原路线 reverse，D1 | 386 | 265 | 151 | 150 | 29 | 29 |
| 原路线 M U-turn hard，D2 | 45 | 33 | 7 | 7 | 2 | 2 |
| final fallback hardF / evidence-incomplete，D3 | 161 | 114 | 114 | 114 | 33 | 33 |

最后一组 final M_AV=32；可见原先已有结构 fallback 的机会确实被 evidence gate 隔离，但放行 evidence 后仍有 114→33 的时效损失。没有生成新 fallback，没有给这些路线移植 original 的动态 rho/P50。保留的 original predicted service time 仅用于原有 EMPIRICAL_SESSION 车辆窗口约束；AV full-horizon 不需要用伪造服务时间来通过本诊断。

reverse 组最终 M_AV=27；本轮不能区分其中真实 one-way/access 限制与 directed-identity representation 问题。M U-turn 组最终 M_AV=2，而整个 D2 相对 D0 的净 M_AV 只增加 1；分组总量不等于净干预效应。

## 6. Routing QA、工程 QA 与资源

- 所需 16,386 unique keys：125 archive reuse + 16,261 新 SINGLE_SOURCE_MATRIX 调用。
- VALID_ETA=16,386，CERTIFIED_ROUTING_FAILURE=0，unresolved=0。没有 scalar route fallback，没有 ETA 插值，没有第二轮重路由。
- 固定 auto/WGS84/local-minute/beta，保留原坐标数值，复用 frozen rounding key。检查 union、routing config 和 beta hashes。详细 receipt/cache 本地保存，不公开上传逐车坐标。
- 所有 17,621 state-arcs union 与 Phase0 完全对账，50 个条件 Top-K 数量分别对账。
- 50 个基线、100 个 A1/A2 endpoint 完成；所有 MILP 返回 certified optimal/infeasible，无 timeout/unresolved。
- 每个 witness 检查一车一单、count 层次、A1 同 served set 与真实 H↔A flip。A2 单列 served-set 是否变化和同订单 swap 数。
- 独立调用现有 production candidate construction（拦截 solver，纯缓存 ETA）对照十个 D0 图：10/10 arcs/ETA/critical/carry/type 完全相同。额外 routing calls=0。
- 最小合成检查：空图、已知 0.1 秒 swap gap 与预算边界、双订单 count 最优值。
- Routing 93.034 秒，内部逐 key 采样峰值 RSS 248.922 MiB；主实验 81.759 秒，内部 guard 采样峰值 RSS 283.172 MiB。两阶段合计 174.793 秒，不含独立 QA/代码准备。
- CPU 单进程，OMP/BLAS threads=1，无 GPU，无 order×vehicle 稠密矩阵；图和 MILP 使用稀疏格式。单 MILP 10 秒、各执行阶段 600 秒、内部 RSS 2 GiB guard。
- **资源监护限定**：另做了运行中外部进程内存快照，但没有覆盖全程的连续外部采样器；上述峰值明确是内部采样值，不冒充外部认证峰值。未因该记录缺口重跑科学实验。

## 7. 交付与可复现性

代码入口：`complete_attribution_pickup.py` → `run_stage3_attribution.py` → `verify_stage3_attribution.py` → `summarize_stage3_attribution.py`，均在 `stage4/tools/`。主实验已有输出时拒绝覆盖；pickup journal 可恢复，已尝试 keys 不重复调用。可在新的输出目录重现，不要删除现有结果来运行。

可提交材料在本目录 `results_v1/`：comparison、transitions、endpoints、baselines、graph_layers、provenance_funnels 的聚合 CSV/JSON，以及 routing_qa、experiment_summary、independent_qa、classification JSON。含 order/vehicle identity 的 private witnesses、routing receipts/cache 留在本地 ignored output 目录。冻结输入来源仍由 Phase0/protocol 提供。

**下一步保持人工研究决策停止点。** 本报告没有更改科学主线，没有据结果调阈值，没有启动 reverse 全量审计、counterfactual prediction 开发或新的 control 实验。
