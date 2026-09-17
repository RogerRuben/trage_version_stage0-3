# A0 — Stage3 → action-space repository audit

日期：2026-09-18；基线 `cb399b50bbba03795180f6de9a654160b2ec99db`。本轮只读代码、配置、既有汇总和 Parquet schema，未运行 ablation、MILP、路由或状态重建。

## 1. 已核实的入口与场景差异

逐一检查 `stage4/output/final_experiments/MAIN_*/scenario_config.json`：共 27 份，runtime Gamma 均为 null、cost 均为 false。连续 rho 超过 1 并不会在这些场景作为上限删除弧；但非有限 rho 会影响 exposure 是否存在，并因此影响 AV eligibility。

旧十状态来自 `ODD_Q50_M_P70_REFERENCE`，不是 MAIN_Q50_M_P70。其 Gamma static=2.145067625590382、dynamic=0.14934256810554178、speed=0，cost=false。旧 0/10 是这个联合约束下的结果。

若新实验关闭 Gamma，必须重新定义/求出 D0 基线；不能将任何改善全部称为 Stage3 的作用。可以保留同一物理状态，但它仍来自 REFERENCE 历史，不是 Gamma-off 政策演化后的状态。

## 2. 实际 eligibility 比简写多一层

代码来源：`stage4/fleetpy_adapter/test31_demand_adapter.py`、`stage4/dispatch/rolling_or_control.py`、`stage4/dispatch/gate_diagnostics.py`。

- `SpikeRequest.av_smoke_eligible` 是 hard_state=FEASIBLE 且 evidence_complete。
- production AV eligibility 还与 passenger acceptance、`exposure_excess(...) is not None` 取交集；后者要求三个 rho 有限，即使 Gamma 关闭也如此。
- 之后 shared Top-K、pickup/patience，以及按 availability_policy 进行的 session gate 继续作用。
- gate-diagnostic helper 还显式判断 selected route 是否存在、预测时长是否有限且正；不能未经核对就把其诊断谓词当成 production 的完全相同实现。

因此 D3 只把 evidence_complete 布尔值设为 true，仍可能被 NaN rho 拦住。这不是“evidence ablation 无效”的科学结果，而是没有真正移除目标证据门。提案必须定义移除哪些证据检查，并保留其缺失来源。

## 3. 原路线 reverse 与 final interface 不在同一层

来源：`stage3/odd_tod/original_route_suitability.py`、`operational_suitability.py`、`finalization.py`；既有汇总 `stage3/output/odd_tod/final/stage3_final_summary.json`。

原路线 reason `KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE` 由 HISTORICAL_REVERSE_OVERLAY token 触发。这个实现事实不等于逐条证明真实 one-way/access 违法；本轮没有进行 lineage 或法律属性认证。

既有 M profile 汇总：

| 字段 | 已有值 |
| --- | ---: |
| 原路线 FEASIBLE / INFEASIBLE / UNKNOWN | 9258 / 15822 / 4920 |
| final FEASIBLE / INFEASIBLE / UNKNOWN | 15659 / 0 / 14341 |
| final ORIGINAL / FALLBACK / NONE | 14178 / 6401 / 9421 |
| original reverse reason triggered | 14837 |
| reverse family fallback recovered | 6144 |
| maneuver family triggered / recovered | 2050 / 553 |

reason families 可重叠，不把 14837+2050 当成互斥总数。

关键分支：原 INFEASIBLE 触发 K=1 fallback；找到 hard-feasible fallback 后 final hard=FEASIBLE，但 evidence=false、selected P50=null、rho_dynamic=null；未建立 fallback 后 final hard=UNKNOWN、route=NONE、final hard reasons=[]，原原因保留在 `original_route_hard_reason_codes`。

所以直接从 final hard reasons 删除 reverse 可能没有任何改变，却不能据此排除 reverse 经由 fallback/证据机制产生的影响。D1 必须先说明操作的是原路线、最终选中路线还是既有候选池。

## 4. 可用路由/evidence 产品

`test31_stage3_to_stage4_interface.parquet`：90,000 行，包含 final/original hard、选路、fallback 搜索状态、selected P50、rho、reason 等字段。

`test31_fallback_candidates.parquet`：17,668 行，包含 candidate_count/status、distance、Valhalla route time、route reference、failure_reason；它本身不含逐 profile 完整 hard classification 或合法 M3 P50。

`test31_fallback_route_edges.parquet` 已存在，大小约 2.37 MB；可作为现成几何来源，但存在几何不等于 prediction/evidence contract 完备。

Stage4 replay 的 `predicted_service_time_s` 来自 original route descriptor；`selected_service_time_p50_s` 是另一个合并进来的字段。完整 demand loader 使用前者构造 SpikeRequest。不得将 original P50 偷换成 fallback P50，也不得使用 Valhalla route time 冒充 M3 prediction。本轮只记录接口差异，未判定新的生产 bug，未修改接口。

## 5. Pickup arc 归档不足以恢复松门后的图

现有 `stage4/output/paper_enhancement/control_freedom_structure/arcs_*.json` 保存 10 状态、130 条 solver-input arcs（48 AV）：它们已通过 eligibility、Top-K 和 pickup/session 筛选。

这批文件是完整 D0 solver 输入，不是完整 pre-gate candidate universe。此前计算的 13,046 次路由只有通过筛选的弧被归档；adapter cache 为内存对象，并逐状态清空。log 没有完整逐弧 ETA 输出。

`mechanism_validity/neutral_av_identity.csv` 有 neutral 图的候选身份/选中身份/总 pickup，但无所有弧逐条 ETA；不能由总和还原权重。

`routing_determinism/routing_arc_sample.csv` 确有 WGS84、时刻、raw/corrected ETA 字段，但只是抽样，未认证覆盖新十状态×五条件的 required arc union。其他已检查 routing sample 同理。

既有 service route geometry 或 Valhalla service-route duration 不是车辆→pickup 的 ETA。不得用二者替代、用近邻速度填充，或把没有缓存的弧当作不可行。

结论：**在已检查产品中，尚无可确认完整覆盖 D1–D4 的 pickup ETA archive。严格“不重路由”下，目前不能承诺完整执行 10×5。** 这不是实验的负结果；只是一项可识别性/输入缺口。

## 6. 初步 reason taxonomy（待冻结，不改代码）

- certified/legal hard：`CERTIFIED_MOVEMENT_PROHIBITION`，保留认证条件；不能把未知当合法。
- network direction identity：`KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE`，单列；D1 单独研究，不能先宣称全是真违法或全是 artifact。
- capability-specific hard：`CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE`、`UTURN_PROFILE_INCOMPATIBLE`、`CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE`。
- evidence/structural unknown：根据原 reason provenance 保留，不与 hard reason 混为一谈；有限 K=1 未找到路线不证明任何 AV 路线不存在。

M 状态不能评估 C-only roundabout/left-stop-yield 规则的总体影响；D2 对这些规则无变化不等于 C profile 假设不重要。

## 审计状态

A0=COMPLETE；新实验=NOT_RUN。待解决：Gamma 口径、route 选择口径、reason/unknown 映射、Top-K 口径、完整 pickup ETA 覆盖。无需现在重写 Stage3；更不能把缺输入造成的 D4 无改善用于洗清或归责 Stage3。
