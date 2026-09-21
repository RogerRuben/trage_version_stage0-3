# Stage3 端点身份与接口：最小修正设计及核验结果

2026-09-21；基线8fdc726。本轮只制定并核验方案，不修改Stage3产品或重算派单。

## 一、核验结果：不能采用片段拼接捷径

上轮确定了391个unknown encounter涉及88条端点不完整边。本轮仅读取这些边相关的16个局部z12 MVT，采用冻结GraphId、auto图导出属性与generalize=0；没有route/matrix调用。

| 检查 | 结果 |
|---|---:|
| 审计edge/endpoints | 88 / 88 |
| 局部MVT tile请求 | 16 |
| 存储端点位于web-tile边界 | 0/88 |
| 同GraphId仅一份去重片段 | 85/88 |
| 同GraphId两份去重片段 | 3/88 |
| 同层级、片段终点唯一网格节点候选 | 0/88 |
| 可认证修复 | 0/88 |

检查采用整数瓦片网格、方向一致片段与同hierarchy level的node feature，不以最近距离选节点。该诊断既没有证明所有可能MVT处理都失败，也没有证明底层图节点缺失；它排除了本次明确检查的“局部片段连接+唯一网格节点即可恢复”的方案。S2A代码中的clipping注释不能作为这88条的既定根因。

运行约5.38秒，内部采样峰值408.92MiB；边产品SHA前后一致。证据在 `endpoint_repair_design/summary.json`；工具 `stage4/tools/check_endpoint_recovery_design.py`。

## 二、保留已确定的根因，不扩大结论

已确定：movement只使用endpoint-complete edge，boundary使用全量edge；两者输入域不同，parser能输出movement未纳入的组合。

尚未确定：这些边为何没有精确Valhalla端点身份。坐标量化、导出node覆盖、hierarchy连接及其他底层语义必须从原生图记录查清，不能选一个猜测写成根因。

因此修正不能只做以下任一操作：

- 删除boundary里的端点不完整边，让encounter从订单中消失；
- 根据内部可达给391个组合补movement行；
- 用附近OSM/node坐标补ID；
- UNKNOWN直接改FEASIBLE；
- UNKNOWN原路线自动切到已有fallback后默认为证据完整。

## 三、方案A：精确端点取证，先产shadow表

输入限定为88个已登记directed edge GraphId与冻结tiles。实现可使用与冻结tile版本匹配的原生Valhalla GraphReader/导出器；本轮未安装或构建新运行时，也未猜测二进制offset。

精确恢复契约：

1. 对GraphId定位原生directed-edge记录；目标节点从该记录的endnode身份读取，而非几何推断。
2. 起点依据node的outgoing edge索引范围确认edge归属；如用opposing edge交叉核验，必须验证两边身份关联，不默认所有边都有可用对向记录。
3. 核验tile/hierarchy版本、way identity、direction/access，以及已有非缺失端点的一致性。
4. 保留原生节点ID和取证来源。经纬度仅做一致性检查，不决定身份。
5. 发现多个归属、ID不在对应tile、已有端点冲突或版本不兼容时STOP该项，不能退回nearest-node。

shadow表至少包含：stage3_edge_uid、directed_edge_graph_id、endpoint_side、old_node_id、candidate_node_id、source_tile、source_record、evidence_status、conflict_reason。只写独立新目录，不覆盖冻结文件。

这一步的通过条件是精确身份核验，不是“必须恢复88/88”。部分无法恢复应继续UNKNOWN；若不足以支持后续修正，提交证据并停止，不扩为全网重导项目。

## 四、方案B：统一拓扑口径，同时保留缺失证据

应定义单一的 `endpoint_identity_complete` 判据供movement与boundary的可解析视图使用。但必须另存**被排除edge/token的原始证据通道**，否则统一过滤会人为减少UNKNOWN。

建议双通道契约：

| 通道 | 职责 |
|---|---|
| topology-ready boundary/movement | 只消费两端精确身份已确认的边 |
| route evidence diagnostics | 保留每个端点未解决token、断点与原始route位置，传播UNKNOWN |

缺失端点应有明确reason，例如 `ENDPOINT_IDENTITY_UNRESOLVED`，而不是只依赖碰巧生成的 `MOVEMENT_LOOKUP_UNRESOLVED`。reason改名不自动改hard_state：没有其他hard violation时仍UNKNOWN，有hard violation时保留既有hard优先级及未知证据。

验收必须按原始token/encounter对账：旧391个问题不能因为新parser不再输出它们，就计为391个“恢复”。分别标记精确恢复、仍未知、断链、分类变化，分母不能变化。

## 五、movement重建的最小范围

只有方案A拿到可核验端点后，才在shadow graph对受影响边/complex重建必要的movement。保持10m clustering、complex成员、C/M/A profile、reverse规则、动态CDF/M3全不变。

如果精确端点改变complex成员归属或暴露新拓扑、必须重新cluster才能表达，则超出本次最小方案，应STOP并报告，不能顺手重做S2B。

完整movement成立仍需：incoming/outgoing角色、方向连通、实际路线边链连续、已有turn-restriction证据及geometry字段一致。391条中2条记录链不连续，必须单列；不能只因内部BFS可达就认为原路线真实连续。

连通只证明拓扑存在，不认证法律许可、AV能力或预测evidence。不得新增未知turn restriction的默认合法规则。

## 六、branching保持独立

D13揭示的26单original UNKNOWN/fallback分支问题与端点恢复分开处理。当前不改变 `route_variant` 或production fallback政策。

如果之后考虑UNKNOWN时查询既有fallback，必须作为单独的政策/接口变更，保持fallback动态evidence不足时不进入正式dispatch。不能将此政策变更混入端点身份修复，再把结果都归功于bug fix。

## 七、必要核验与停止线

建议修复实施分两次授权：

**R0：只读原生端点取证＋shadow表。** 不改变route状态，不运行优化器。确认证据可得后再进入R1。

**R1：最小代码/接口修正＋固定样本核验。** 验收：

- 原88端点、117 movement keys、391 encounters逐项对账；
- 非受影响网络记录不变，reverse及profile不变；
- 未解决项继续UNKNOWN，不能消失或误通过；
- 两条不连续链单列；
- 检查旧十状态的route readiness差异与原因；是否重算固定状态assignment须在R1授权中明确，不能自动扩大；
- 不重跑41场景、FleetPy全天、M3训练/inference、service-route搜索或Stage5。

## 本轮交付状态

`ENDPOINT_REPAIR_DESIGN = COMPLETE`

`LOCAL_MVT_SHORTCUT = NOT_SUPPORTED_BY_CURRENT_CHECK`

`CERTIFIED_ENDPOINT_REPAIRS = 0`

`PRODUCTION_CHANGED = NO`

下一项具体工作是R0原生有向图端点取证，而不是立即编写“自动修复”。该实施需要确认可用的版本匹配读取方式；没有读取证据之前，不承诺可恢复比例或运营收益。
