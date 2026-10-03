# 下一步方案：SNE 是否提供额外的道路形态信息？

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-25
- Verification Status: UNVERIFIED（方案，尚无 SNE 实测结果）
- Version Label: sne_information_plan_v1
- 基线：`873da71`；本文件是待确认建议，不是作者已批准的预注册，也不授权门控实验。

## 1. 唯一研究问题与边界

在冻结西安路网上，街道方向分布熵能否描述 A/M/D/L 未直接表达、且不主要由道路切分、方向复制、网格相位或低支持造成的形态差异？

允许结果包括：有互补描述价值、主要重复现有信息、主要不稳定、证据不足。没有必须得到正结果的目标。

本阶段不回答 SNE 是否预测 AV 安全、是否扩大可服务订单、是否提高派单收益。没有真实 maneuver capability 标签，不能把统计不相关称为已验证的能力信息增益。

## 2. 已有依据

- [max/mean/tail 诊断](README.md)：M profile 的 D 超出常局部化，但四维联合超出并非普遍只有一次；不能直接用 mean 替代 max。
- [武汉公开材料审查](../wuhan_robotaxi_review/README.md)：公开 SNE 是方向熵，不是 signal/control、合法 movement 或安全概率。
- 作者固定版本 `d8de2c56e6c127542c98c9d224021cd131cd4f77` 的 [compute_grid_sne_wuhan.py](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/5-cci_and_sne/compute_grid_sne_wuhan.py) 已在本地逐行检查：WGS84 几何中点归格，投影后首末点方位角，72 个 5° bin → roll(1) → 36 个 10° bin，edge count 权重，H 与 H/ln(36)，每格至少两条有效边。

原作者道路 GPKG 的生成血缘未由该单个脚本完全说明。因此在西安实现同一计算公式，应叫“author-formula adaptation”，不能宣称已复现武汉论文数值或完全相同的道路总体。

## 3. 建议分两步，先不碰 route exposure

**第一步：network/complex-level 测量有效性。** 用完整冻结路网生成网格描述，再映射到 unique complexes，每个 complex 一行。先区分表示/尺度伪影与真实形态差异。不能直接用 538,355 次经过作独立样本，重复访问会让常用路口过度加权。

**第二步：仅在第一步可解释时，考虑 route-level 描述。** 这一步需要另外明确经过网格的长度权重、partial edge clipping 与 missing coverage，不能把起点网格值或 edge midpoint 值冒充整条路线暴露。本方案不默认执行第二步。

## 4. 第一阶段建议设计（待确认）

### 4.1 输入与总体

复用 `stage3/output/odd_tod/` 下：

- `s2a/stage3_full_network_edges.parquet`：edge identity、geometry、road class、access 与 node identity；只读必要列。
- `s2b/final/stage3_intersection_complexes.parquet` 及 node membership：43,685 个冻结 complexes 不重建。
- `s2b/final/stage3_edge_complex_boundary_index.parquet`：重用 S3.1 boundary D 定义。
- `s3/train_static_complex_reference.parquet`：2,425 个 unique Train-exposed complexes 作为主要比较总体。

完整 network 是空间上下文，不代表全部道路都可由某一 profile 使用。主比较不按历史订单频率加权；全网结果作附加覆盖说明。若空间邻域靠近 tiles 边界，记录截断标记，不把缺道路误当低熵。不得按 Test31 的服务结果筛路网或选参数。

### 4.2 两种描述量严格分开命名

| 名称 | 计算对象 | 用途与限制 |
|---|---|---|
| `SNE_author_formula_xian` | 有向 edge 首末点 bearing，中点归格、edge count，36 个方向 bin | 保留作者公式的表示敏感性；西安网格/路网不同，不能叫武汉原样复现 |
| `orientation_entropy_length_axial` | 同一底图上的几何线段，按格内长度计权，方向模 180°，18 个 10° bin | 我们自己的无向形态描述，不是作者 SNE，也不是 capability score |

第二种先消除可确证的完全重合反向几何重复，再按实际折线切格，以局部线段方位/长度累计。保留真正不同的平行车行道；不能仅因相同 OSM way_id 就去重。近似但不重合的反向几何只记录歧义，不做任意 snapping 合并。若现有 geometry/identity 无法支持可解释去重，停止这一分支并报告缺口。

分别以 ln(36)、ln(18) 归一化；两种原始熵不能直接相减宣称“提升”。保留每格道路总长度、原始 edge 数、有效几何数、道路等级构成和未知率。空格/无效几何为缺失，不能填零；一条直线的零熵是低支持描述，不是证明简单安全。

### 4.3 固定小规模空间敏感性，而不是调参

建议米制网格以冻结网络的适用投影为准，先核实 CRS；主尺度 1,000 m，仅做 500 m / 2,000 m 两个对照。每尺度只比较两个相位：固定投影原点和双轴半格平移。六种组合全部报告，不按 AV/HV 差异、相关性或服务率挑赢家。

这六组合是我们的尺度设计建议，不是论文所用武汉经纬度网格的严格复刻。无需同时扫描几十种尺度。

complex 的位置使用冻结 member nodes 在米制坐标下的确定性中心，并报告跨格成员；不得修改 complex 成员或以 SNE 重新 cluster。相位/中心选择对映射的影响属于待报告的不确定性。

### 4.4 先检查测量是否可信

最小合成检查只需：同一直线切分、反向复制、直交道路、空格、曲线与跨格长度守恒。长度-轴向版本应在同一直线切分和完全重合反向复制下保持不变；作者 edge-count 版本允许不满足，但必须量化，而不是当作实现 bug 修掉。

固定角度旋转合成道路，报告 bin 边界效应；不声称熵对任意旋转严格不变。以与真实数据相同的实现跑这些检查，不执行下载的第三方 main 程序。

### 4.5 再检查描述互补性

在同一组有效的 Train-exposed complexes 上输出：

1. 两种指标及六种空间设定的支持率、分布、秩相关；共同支持与各自缺失同时报告。
2. SNE 与 A/M/D/L 分维 Spearman 相关，不合成新风险分数、不以 p 值判价值。
3. 道路长度/密度、road-class 构成、boundary 截断和有效支持数量的分层结果，避免把网络密度或低样本效应叫互补性。
4. A/M/D 相同、L 相近的 complex 对，检查其方向熵是否仍明显不同。配对规则先固定，低支持组不强配；只将它作为条件描述，不是因果匹配。
5. 最多 12 对地图核查样例：由固定规则选取形态分歧和一致对照，记录选择规则，不能只展示支持 SNE 的漂亮案例。

不拟合 AV 能力标签、不运行 prediction superiority 或 dispatch regression。若以后想用“可预测的额外信息”而非“形态描述互补”，需另定义有效目标和空间分块验证，不能用 A/M/D/L-derived pass/fail 当外部真值制造循环验证。

## 5. 停止与解释规则

- 几何/CRS/身份不足：`INSUFFICIENT_INPUT`，不补猜测。
- 排序主要随切分、方向复制、相位或低支持变动：`MEASUREMENT_SENSITIVE`，停止向能力指标升级。
- 指标稳定，但与既有描述几乎重复、核查缺少可解释差异：`LIMITED_INCREMENTAL_DESCRIPTION`。
- 稳定差异可在地图上解释：最多称 `COMPLEMENTARY_MORPHOLOGY_DESCRIPTOR`，仍不能称决策收益或安全证据。

这些是解释标签，不是已设定数值门槛的机械 PASS/FAIL。当前不伪造有依据的相关系数阈值；所有分布与敏感性先完整呈现，由作者判断是否值得保留。即使结果互补，也不会自动开放第二阶段或派单实验。

## 6. 资源、交付与授权边界

建议单进程 CPU，按 edge batch 与空间索引累计每格少量 bin；不构建 edge×grid、order×vehicle 稠密矩阵，不用 GPU，不重新下载路网。每个空间设定串行，及时释放几何对象。建议软内存预算 2 GiB，预计越界即暂停检查分块，不静默提高预算。

若批准执行第一步，交付一个简短报告、聚合 summary 和小型 QA 图；网格/complex 明细仅本地保存。先记录代码、输入版本、尺度和公式即可，不新建多层 evidence bundle。命令待实现脚本后提供，当前不存在可运行入口，不能把计划当成已运行。

本轮只交付此方案。未计算西安 SNE，未训练，未路由，未修改 Stage3、Gamma、eligibility、M3 或 41 个冻结场景。按研究规划规范，**执行前需要用户选择是否开展上述第一步**；也可以选择到此停止，把 SNE 留在讨论而不进入方法。
