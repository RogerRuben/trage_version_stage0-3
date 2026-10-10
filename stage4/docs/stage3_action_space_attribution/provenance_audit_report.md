# Stage3 semantic provenance audit

2026-09-21；承接 7adde03。范围：reverse 全量程序化分类，以及十状态720个唯一订单的 movement/identity UNKNOWN 来源审计。不运行performance ablation，不修产品，不重新计算service rate。

## 结论与下一步决定

**reverse：本次没有证据支持删除或放宽规则。** Test31所有14,837个reverse订单涉及的3,322个唯一reverse identity、85,538个token，都有冻结PBF中显式 `oneway=yes` 及相反OSM节点顺序支持。不是仅凭overlay标签得出的结论。

**movement UNKNOWN：存在确定的产品输入口径不一致。** 720单中391个unresolved encounter（276单、117个唯一movement key）全部涉及Stage3端点身份不完整的边。S2B movement构建剔除这类边，S2B2 boundary构建却使用全量边；S3 parser因而能输出movement表从未覆盖的组合。这是technical/interface unknown，不能解释成AV物理能力不足。

但**不能直接补上lookup行或把UNKNOWN置为FEASIBLE**。88个缺失edge-endpoint在现有表中都没有Valhalla node ID；对应51个OSM node虽都在冻结PBF中，却没有精确的directed graph-node对接。2个encounter的已记录边链还不连续。

建议下一步走**有限的Stage3端点身份/产品口径修正设计与核验**，而不是reverse放宽、capability调参或Stage5。需单独授权修复；本轮到此停止。修复设计必须保留UNKNOWN，直到端点与movement证据均核实。不要通过忽略未知encounter来“修好”路线。

## 分类总表

| Issue | Count / 单位 | Source | Physical restriction? | Representation issue? | Recoverable from frozen data? | Scientific consequence |
|---|---|---|---|---|---|---|
| Reverse，有显式方向支持 | Test31 14,837单 / 85,538 tokens / 3,322 identities | 冻结PBF `oneway=yes`，way节点序，observed begin/end | 冻结网络方向证据支持；不是现实法律认证 | 本轮未发现同方向已导出auto-edge却被误标的实例 | 方向证据已可直接核验 | 保留reverse语义，不将D1 oracle视作可部署政策 |
| 全overlay中方向顺序尚未解明 | 5 identities，Test31出现0次 | 同way有oneway=yes，但端点顺序无法唯一判定 | 未确定 | 未确定 | 需进一步端点核验 | 不影响本轮14,837单结论，不扩展为全6502均已确认 |
| Movement/boundary endpoint scope不一致 | 391 encounters / 276单 / 117 keys | S2B完整端点过滤、S2B2全量boundary、S3 parser | 不构成禁止证据 | 是，产品输入域不一致 | 内部路径可查，但整体movement尚不能直接恢复 | 限定修复目标为身份与接口链，不放宽profile |
| 缺失directed端点 | 88 unique edge-endpoints / 51 OSM nodes | full-network edge/node与冻结PBF | 未确定 | 端点导出/对接证据缺失 | PBF节点存在；精确Valhalla node对接0/88 | 不做nearest-node猜测，不宣称可无条件补齐 |
| 原始route identity未映射 | 19 tokens / 19单 / 2 unique canonical identities | mapping_status=UNMAPPED_FULL_NETWORK_EDGE | 未确定 | 映射缺口明确，根因未完全确定 | 当前mapping无候选；需查底层export身份 | 不称为真实禁止，也不默认为技术可修 |
| 原始source与canonical字段为空 | 4 tokens / 4单 | frozen typed route | 未确定 | 输入身份缺失 | 仅凭当前token不可恢复 | 保留UNKNOWN；不删除断点掩盖缺失 |
| D1/D13 original UNKNOWN阻止沿用D3路线 | 26 unique orders（23 movement、3 identity） | frozen route_variant选择分支 | 分支不是物理禁止证据 | 是，route-selection policy组成部分 | 已有D3结构路线，但仍不代表动态evidence齐全 | D13不是D1/D3候选集并集，不能作简单二元交互估计 |

各行不是互斥集合，不能相加成总问题数。order、token、identity、edge-endpoint单位严格分开。

## A. Reverse identity evidence

### 实际判定链

`s2a_scientific_closure.py` 把 `UNMAPPED_FULL_NETWORK_EDGE` 且canonical traversal为R的身份转为历史overlay，并统一设置AV_ROUTABILITY_VIOLATION。`capability_envelope.py`保留此typed identity；`original_route_suitability.py`看到reverse token即设置KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE。该链本身并不逐条核验oneway。

本次独立读取冻结PBF（SHA与已有binding一致），取得way tags和完整节点序，再使用overlay内observed begin/end方向核验：

- 全6502个overlay：6497个显式oneway=yes且观察方向相反，5个端点顺序未唯一解明。
- 这6497个不依赖roundabout/motorway默认单行推断，也没有相关conditional tag。
- Test31全部85,538个reverse token均落入上述有证据类，覆盖14,837单。
- 按冻结导出auto graph的same-way、同begin/end精确查找，没有发现已有同方向auto edge但被错误归为reverse的情况。

Stage0有 `ignore_oneways` 历史匹配及reverse-oneway canonical映射分支。历史观测允许记录相反方向，与AV按冻结道路规则可否规划不是同一个问题；不能用前者直接否定后者。

这些结论没有认证2016年的真实道路法律状态或OSM事实准确性，也没有证明完整路线上不存在合法替代路线。OSM与Valhalla可能同源，不视作两份独立物理证据。没有实时查询或路线重规划。

## B. Movement UNKNOWN 根因链

### 覆盖范围

十状态720个唯一订单共有13,378个已记录complex encounters；391个查找未解决，涉及276单、117个唯一complex/incoming/outgoing三元组。不是只看08:30、13:00见证。

### 精确代码口径

1. `intersection_complex.prepare_topology`：只有from/to Stage3 node均在node表中的edge进入movement构建。
2. `intersection_complex.build_products`：在上述完整拓扑上枚举incoming/internal/outgoing并检查有向连通。
3. `s2b2_final_freeze.build_boundary_index`：使用全量edge；当一端可映射complex、另一端缺失时，仍可生成INCOMING/OUTGOING角色。
4. `capability_envelope.parse_route_complex_encounters`：消费上述boundary角色。当识别出的三元组不在movement表中，输出UNRESOLVED_MOVEMENT_LOOKUP。
5. S3最终映射为MOVEMENT_LOOKUP_UNRESOLVED，保留UNKNOWN。

391个问题encounter均命中此端点不完整情形。没有发现本范围内“端点完整、内部可达却无理由漏写movement”的实例。

### 为什么最初的path检查不足

391个incoming进入complex后的节点到outgoing离开complex前的节点，在冻结内部图上都可达；但这只核验**内部段**。incoming外端或outgoing外端缺失，不能据此证明整个movement完整。初步内部路径结果因此被endpoint-aware分类取代，正式输出见 `provenance_audit/movement_classification.json`。禁止把初步的internal-path字段解释成整体可恢复性。

88个缺失edge-endpoints均没有Valhalla node ID；51个对应OSM node都在PBF中。现有final node表没有可用的精确Valhalla ID或OSM ID候选；不是简单漏join一个已存在的ID。S2A节点对接代码使用坐标量化匹配，代码注明web-tile clipping可造成端点不吻合，但本轮没有逐一认证88个都是clipping造成。需在后续修复前追查tile/导出链，不能把注释当作全量根因证书。

已记录chain中389个连续、2个不连续。即使之后恢复端点，也必须独立处理这2个连续性问题，不能一批放行。

### Epistemic vs implementation unknown

这里可以确认unknown的生成机制属于技术产品覆盖不一致；但当前输入的缺失身份又构成真实的证据不足。两者并不互斥：**技术链导致epistemic gap**，并不意味着可以跳过证据要求。

与D13的26单分支问题相连：冻结D1删除reverse后，original可变成UNKNOWN，逻辑不会继续选择原来可用的fallback。这是路线选择政策，值得在修复方案中明确，但本轮没有改成UNKNOWN自动fallback，更没有补动态evidence。

## 代表性案例与核验状态

已准备55个可检查记录：方向支持类25个、方向未解明类全部5个、movement类25个。选择规则为现有稳定顺序前25个，目的是追踪证据，不是代表性随机抽样或统计推断。

公开的 `representative_cases.json` 仅含静态地图/身份信息和case编号，不含order ID。完整关联明细留本地ignored目录。样本统一标为**程序化证据；人工审查未完成**，没有冒称完成合作方要求的人工核验。

## QA、资源、交付

- PBF SHA：`4d918b7ed2201a8f7a75fa7fb2974679343c89a10bf83638e7bcf21ac07c1526`，与冻结binding相同。
- 审计前后来源产品SHA对账，无Stage3产品变更。
- 主分类约29.09秒，内部采样峰值662.72 MiB；后续closure单独执行，未将主运行峰值冒充全部进程连续监护值。
- CPU、无GPU，无routing/MILP/inference/training。无service-rate重算。
- 两个工具：`audit_stage3_semantics.py`先生成初步证据，`close_stage3_provenance_audit.py`执行必要的端点closure，后者输出为最终movement分类。完整复现必须执行两者。
- JSON/CSV、来源SHA、QA、端点可恢复性、分支汇总和代表性案例均在同目录 `provenance_audit/`。

**研究决策：不选reverse放宽分支；转入有边界的端点身份/接口修正方案审查。** 这不等于修复已获授权，不预测修复后的service改善，更不重启Stage5。本轮结束。
