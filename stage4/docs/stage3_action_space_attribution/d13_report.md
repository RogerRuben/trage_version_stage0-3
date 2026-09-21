# D13 closure：联合条件与 D4 新增状态阻断核查

日期：2026-09-21。基线结果：c04e623。范围依据同目录 `d13_protocol.md`，先冻结定义再执行。**本轮完成，STOP；不继续组合实验，不修改 Stage3 或41场景，不启动 Stage5。**

## 1. 结果

| 条件 | 最终 E_AV | U_AV | M_AV | O_HA | A1 ≤5% | A2 ≤5% |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 48 | 12 | 12 | 3 | 1/10 | 1/10 |
| D1 | 115 | 41 | 39 | 9 | 4/10 | 4/10 |
| D3 | 139 | 45 | 44 | 7 | 4/10 | 4/10 |
| **D13** | **123** | **43** | **41** | **9** | **4/10** | **4/10** |
| D4 | 509 | 119 | 113 | 15 | 6/10 | 7/10 |

所有 E/U/M/O 都是十个物理状态分别计算后求和，不是全天独立订单或合并图容量。旧条件读取已落盘结果，没有重跑。新条件仅 D13。

D13 两个 endpoint 在 0/0.5/1/2/5% 下均为 **0/0/0/1/4 个状态**。没有新打开的 ≤5% A1 状态：通过状态与 D1 一致，为 07:30、18:00、18:30、21:00。相对 D0 新增三个，仍只有 18:00、18:30 同时增加最终 O_HA，因此不改变上一轮的单因素 screening 结论。

## 2. Exact minimum gap

| 状态 | D13 A1 最小绝对 gap（秒，表内四舍五入） | 最小相对 gap | ≤5% |
|---|---:|---:|---|
| 07:30 | 59.749033 | 4.773869% | 是 |
| 08:30 | 101.996673 | 20.000000% | 否 |
| 12:00 | 223.534380 | 24.019608% | 否 |
| 13:00 | 110.827826 | 9.888060% | 否 |
| 17:00 | 不存在 | N/A | 否 |
| 17:30 | 不存在 | N/A | 否 |
| 18:00 | 11.081515 | 1.158301% | 是 |
| 18:30 | 26.100177 | 3.539823% | 是 |
| 21:00 | 29.611276 | 2.564103% | 是 |
| 23:00 | 不存在 | N/A | 否 |

完整浮点结果和 A2 在 `results_d13/endpoints.json`、CSV。不存在表示经过求解确认没有满足该条件约束的 alternative，并非 missing ETA。各条件自己的 count 最优值、P*、参考订单集合差异在 `state_comparison`。5% 是诊断容差，不是新平台政策。

07:30 仍是相同约59.75秒绝对成本跨过相对预算线的现象。13:00、21:00 比 D1 的相对 gap 下降，也不能脱离绝对 gap 和 P* 分母解释为置换路径变便宜。

## 3. D4 新增五状态：恢复了什么，剩下什么

| 状态 | D13恢复≤5% A1 | D4最优见证中的直接车型变化 | 该变化所需两条arc都在D13中 | D13 route provenance |
|---|---|---|---|---|
| 07:30 | 是 | AV→HV | 是 | ORIGINAL / FEASIBLE |
| 08:30 | 否 | HV→AV | 否 | ORIGINAL / UNKNOWN / MOVEMENT_LOOKUP_UNRESOLVED |
| 13:00 | 否 | HV→AV | 否 | ORIGINAL / UNKNOWN / MOVEMENT_LOOKUP_UNRESOLVED |
| 18:00 | 是 | AV→HV | 是 | ORIGINAL / FEASIBLE |
| 18:30 | 是 | AV→HV | 是 | ORIGINAL / FEASIBLE |

因此，D13在状态级恢复了D4新增五状态中的三个，未恢复08:30、13:00。后两者的已保存D4最优见证，其直接HV→AV置换依赖原路线movement查询未解决的订单。D4诊断性绕过它，D13保留UNKNOWN约束，故该AV arc没有进入D13。

检查整份D4 baseline/A1 witness，还发现其他订单带有 MOVEMENT_LOOKUP_UNRESOLVED、UNRESOLVED_ROUTE_IDENTITY 和少量 UTURN_PROFILE_INCOMPATIBLE。各状态的完整数量见 `blocking_summary` 和 `blocking_reasons`。这十份见证中缺失arc均首先因Stage3未建立/非FEASIBLE而被排除，没有被分到shared Top-K、ETA、patience或session缺失类别。**这是所检查见证的结论，不是整个图中不存在Top-K挤出或下游损失。**

即使某个直接swap的两条arc均存在，也不代表D4的整个served set在D13中可行。每份D4完整见证都还有其他被排除的arc。不能将上表理解为固定D4全套订单后的可迁移性证书。

这些是已保存最优见证的阻断来源，不是唯一最优见证、不构成全图唯一因果解释。本轮没有为证明唯一性追加求解或消融。

## 4. 最重要的科学限定：D13不是纯二元gate并集

D13严格按冻结定义执行：先采用D1的route-aware original/K1选择，再仅对已经建立且FEASIBLE的路线绕过evidence。**不是D1 eligible OR D3 eligible。**

在Phase0的720个唯一订单中，有26个满足D3但不满足D13：

- 23个：D1选择 ORIGINAL / UNKNOWN，原因 MOVEMENT_LOOKUP_UNRESOLVED；
- 3个：D1选择 ORIGINAL / UNKNOWN，原因 UNRESOLVED_ROUTE_IDENTITY。

原因在冻结的D1路线分支：删除reverse后，如果original变为UNKNOWN而不是INFEASIBLE，保留original UNKNOWN，不继续选择fallback。D3则沿用原生产final selected route，其中可有结构FEASIBLE的fallback。两者路线选择不同。

这不是本轮新改的规则，也没有据结果修补它。它解释了为何D13 M_AV=41可以低于D3的44，而O_HA又可以更高。加上各条件重新选择shared Top-K，生产图不保证嵌套。

因此本轮结果**不能**写成“同时移除reverse和evidence，收益严格等于或小于单因素，故两者没有交互”。正确结论是：在冻结D1路线选择语义下，进一步绕过evidence没有新增≤5%的A1状态；D4剩余新增见证涉及结构UNKNOWN，且route-selection branching本身影响可用集合。

也不能声称已经证明movement unresolved是数据错误或真实不可通过。是否由reverse token打断parser、映射缺失或真实复杂movement导致，需要后续来源审计，不在本轮修复。

## 5. QA、资源与范围

- 10个状态的全部D13候选在任何新求解之前完成缓存检查：12,352 unique keys，missing/unresolved=0，新增routing=0。
- 复用上一轮有效receipt与冻结key语义，核对journal SHA。没有service-route搜索、fallback生成、M3调用或profile改动。
- Gamma/cost OFF，condition-local A1/A2定义与上一轮一致。所有新求解optimal/infeasible，无超时未决。
- 独立无求解QA验证十状态一车一单、count最优值一致、AV结构eligibility、pickup/gap/budget重算、A1固定served set与真实车型变化。另有5个D13定义边界断言（FEASIBLE original/fallback通过，UNKNOWN/INFEASIBLE/NONE不通过）。
- 主执行19.880秒；内部采样峰值RSS266.320 MiB，最终MILP最大33 arcs。无GPU、无稠密order×vehicle矩阵。最多10秒/solve、600秒/执行、内部2GiB guard。内部峰值不是连续外部采样认证值。
- compileall与diff check通过。旧结果只读，新产品独立目录，含订单/车辆标识的见证仅本地保存。

公开产物：`results_d13/` 中 graph_layers、endpoints、state_comparison、blocking_summary、blocking_reasons CSV/JSON，及summary、qa、swap_blocking_detail、route_selection_divergence JSON。

## 6. 研究停止点与建议

到此停止action-space组合实验。没有证据支持通过放宽M capability或直接取消evidence来解决问题，也不能把D4的诊断自由度当作可部署策略。

下一项值得提交人工决定的是**来源审计**：reverse身份→方向约束证据，以及这两个未恢复状态中的movement UNKNOWN与original/fallback分支。先查清“KNOWN_REVERSE”和“MOVEMENT_LOOKUP_UNRESOLVED”各自在说什么，再决定是否需要科学修正。不要直接启动全量重算、任意路线预测开发或Stage5。本轮未执行该后续审计。
