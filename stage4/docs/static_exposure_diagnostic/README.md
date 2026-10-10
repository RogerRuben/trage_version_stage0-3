# A/M/D/L 路线静态复杂度：max / mean / tail 只读诊断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（既有产品描述性分析，不运行派单实验）
- Origin Date: 2026-09-25
- Verification Status: ANALYZED
- Version Label: static_exposure_diagnostic_v1
- 基线代码：`b549b9c092309b818de7c36aff094be7642e85d5`

## 1. 结论先行

**max 与平均/尾部暴露确实表达不同的信息，但目前不能据此把 max 换成 mean，更不能声称这样可以恢复匹配容量。**

最明显的是 M profile 的 D（boundary road-class diversity）：11,703 条路线至少有一次 D>3，但仅 41 条路线 mean(D)>3；在前一组中，65.5% 只有一次超出，超出 encounter 占比的中位数为 6.25%。这支持“D 的路线最大值常由局部少数经过触发”的描述。

但“四维联合超出”不是同一件事。M profile 有 27,839 条路线至少一个 encounter 的任一静态维度超出 cap；这些路线的联合超出占比中位数为 29.4%，只有 11.4% 恰好一次联合超出。因此不能从 D 单维的局部性推广为“现有静态结果主要都是单一极端路口”。

本轮的实际建议是：保留 max 的“最强局部要求”含义，同时把 mean 和 tail 作为独立描述性轴；暂不新增 SNE gate、不替换能力判据、不修改冻结场景。这里的 tail 是经过次数占比，**不是持续时间、连续距离或安全风险**。

## 2. 数据范围与可识别边界

输入为 Test31 的 **原始历史路线**，不是 Stage3 fallback 最终选中路线，也不是 Stage4 已匹配订单子集。读取已有 Parquet 的必要列，不加载 M3 预测、轨迹几何或大型路由矩阵。

| 项目 | 数量 |
|---|---:|
| 原始路线 | 30,000 |
| 至少一个可解析 complex encounter 的路线 | 29,926 |
| 无 encounter 的路线 | 74 |
| encounter 行 | 538,355 |
| 被经过的不同 complex | 2,107 |
| 路线内重复经过同一 complex 的额外行 | 1,821 |
| 每条有 encounter 路线的次数 P50 / P90 / 最大值 | 16 / 34 / 97 |
| 无 unresolved token、无 reverse overlay 且 movement lookup 全部解析的有 encounter 路线 | 9,848 |

全体中有 unresolved token 的路线为 981 条，有 reverse overlay 的为 14,837 条，有 unresolved movement lookup 的为 10,199 条；三类允许重叠，不能相加解释。解析器在身份断点处不能证明沿程全部交叉口都已被覆盖。因此全体统计是 **observed encounters 的条件描述**，并非每条完整物理路线的无遗漏统计。

74 条无 encounter 路线的 max/mean/tail 保持缺失，不当作零复杂度或自动通过。另输出 9,848 条身份及 movement 均解析的子集，减少明显的解析缺口影响；该筛选也不构成真实道路合法性、安全性或动态证据完备的认证。

## 3. 方法：不重新拟合 cap

A = external physical connection count；M = topological movement count；D = INCOMING/OUTGOING boundary edges 的不同道路等级数；L = complex 内部长度（米）。D 完全沿用 S3.1 的 boundary 定义，不读取旧 internal diversity 作为 D。

复用冻结的 43,685 个 complexes、boundary index 和 full-network edge road class。没有重做 clustering、movement 或 profile calibration。

对路线 o 的 n 个已观测 encounter，分别计算每维：

- `max_j = max_i x_ij`：局部最强要求。
- `mean_j = sum_i x_ij / n`：按经过次数等权的平均值。
- `p90_j`：经验 90% 分位数，采用 `higher`；短路线时可能接近或等于 max。
- `tail_jk = count(x_ij > B_kj) / n`：严格超出冻结 profile cap 的经过占比。
- `tail_any_k = count(any_j x_ij > B_kj) / n`：任一维超出的 encounter 占比，同一次经过只计一次。

重复经过按真实 encounter 行保留，不把路线当作 unique-complex 集合。不跨 A/M/D/L 求平均，不混合计数与米；不引入 weighted risk index。数据未提供可直接对应每次 encounter 的可信耗时/长度权重，本轮不以整个 complex 的 L 替代车辆实际行驶距离，也不以 realized travel time 补权重。

cap 是 **unique Train-exposed complexes** 上冻结的分位数，不是路线 mean 分布的分位数。所以 mean 与相同 cap 的比较只是看描述差异，**不是经过校准的 mean-based 备选政策**。本轮不在 Test31 重新选择分位数或阈值。

## 4. 全部已观测 encounter 路线：逐维比较

以下分母统一为 29,926。表中“单次占比”和“tail P50”的条件分母为该行 max>cap 的路线，不能与全体占比混淆。M profile 与 M dimension 名称不同。

| Profile | 维度 | cap | max>cap 路线（全体占比） | mean>cap 路线 | p90>cap 路线 | 超出组中恰好一次 | 超出组 tail P50 |
|---|---|---:|---:|---:|---:|---:|---:|
| C | A | 4 | 28,619 (95.6%) | 22,385 | 27,974 | 7.8% | 41.7% |
| C | M | 9 | 26,180 (87.5%) | 14,372 | 23,449 | 15.2% | 26.7% |
| C | D | 2 | 25,169 (84.1%) | 16,103 | 22,254 | 18.1% | 25.0% |
| C | L | 10 | 28,979 (96.8%) | 23,477 | 28,443 | 7.0% | 45.1% |
| M | A | 5 | 27,181 (90.8%) | 10,942 | 25,076 | 12.9% | 26.9% |
| M | M | 16 | 19,923 (66.6%) | 1,755 | 13,002 | 33.3% | 12.5% |
| M | D | 3 | 11,703 (39.1%) | 41 | 2,612 | 65.5% | 6.25% |
| M | L | 34 | 25,903 (86.6%) | 6,737 | 22,963 | 16.6% | 21.4% |
| A | A | 9 | 16,565 (55.4%) | 91 | 8,669 | 42.7% | 10.0% |
| A | M | 25 | 15,381 (51.4%) | 81 | 7,826 | 41.8% | 10.0% |
| A | D | 3 | 11,703 (39.1%) | 41 | 2,612 | 65.5% | 6.25% |
| A | L | 73 | 18,257 (61.0%) | 336 | 9,430 | 45.2% | 10.0% |

例如 M profile 的 D 有 11,662 条 max>3、mean≤3。这样的差异是“是否遇到过”与“平均遇到多少”的区别，不证明那一次高 D 路口可以被忽略。路线中低复杂度路口增加，也会稀释 mean/tail；这不是车辆获得了通过原高复杂度路口的新能力。

## 5. 四维联合与解析完整子集

| 范围 | Profile | 任一维至少一次超出 / 路线数 | 超出组恰好一次联合超出 | 超出组联合 tail P50 |
|---|---|---:|---:|---:|
| 全部已观测 | C | 29,102 / 29,926 | 5.1% | 50.0% |
| 全部已观测 | M | 27,839 / 29,926 | 11.4% | 29.4% |
| 全部已观测 | A | 21,311 / 29,926 | 35.2% | 12.5% |
| 身份/movement 解析子集 | C | 9,322 / 9,848 | 7.2% | 46.2% |
| 身份/movement 解析子集 | M | 8,768 / 9,848 | 16.2% | 28.6% |
| 身份/movement 解析子集 | A | 6,414 / 9,848 | 43.9% | 13.3% |

子集中 M profile 的 D：3,376 条 max>3，其中 2,370 条仅一次超出（70.2%），mean>3 为 20 条，超出组 tail P50 为 7.14%。局部少数经过这一现象并非只出现在明显有身份/解析缺口的路线中。但子集与全体是不同条件样本，差异不是筛选的因果效果。

联合 tail 表明 C、M、A 的局部极值与广泛经过暴露结构不同。它仍未回答“高复杂度经过是否时间连续”“哪一维最影响实际 AV 能力”或“哪些订单因此失配”。

## 6. 与当前派单机制的关系

当前仓库已有 [A0 action-space audit](../stage3_action_space_attribution/A0_repository_audit.md) 明确区分：27 个 MAIN 场景关闭 Gamma，有限 rho>1 本身不删除弧；非有限 rho 仍可能触发 evidence gate。旧 REFERENCE 的 Gamma-on 口径不同，不能混用。

本轮只读原路线 A/M/D/L，既未检查/替换最终 fallback route，也未重建 vehicle→pickup arcs。因此：

1. 表中的超出数量不是 AV hard infeasible 数量，更不是实际拒绝订单数。
2. max 改成 mean 不能自动修复 unresolved movement、direction identity、缺失 M3 evidence 或 patience。
3. 不能把 11,662 等描述差异直接写成“可恢复订单”“新增 action space”或“服务率提升”。
4. 此结果支持保留多种暴露描述，不足以单独证明需要 SNE，或 SNE 有独立增量价值。

## 7. 质量检查、资源与复现

脚本：[diagnose_static_route_exposure.py](../../tools/diagnose_static_route_exposure.py)。完整聚合结果：[summary.json](summary.json)。仅提交汇总，不导出原始 order_id、位置或逐订单表。

复现（仓库 worktree 根目录）：

```powershell
& D:/anaconda/envs/stage0-valhalla/python.exe stage4/tools/diagnose_static_route_exposure.py
```

本轮计算约 5.28 秒；Windows process peak working set 约 467.9 MiB。单进程 CPU，无 GPU，无大矩阵。耗时和资源值随环境变化，不作为确定性结果指纹。

已执行检查：

- Test31 date、30,000 唯一路线、encounter identity 唯一、静态 join 无缺失、43,685 complexes。
- 逐路线 encounter 数与既有产品一致。
- 四维 max 对全部 30,000 条原始 descriptor 核对，缺失值同位处理，绝对容差 1e-10；不一致数均为 0。
- mean≤max、tail∈[0,1]、tail>0 当且仅当 max>cap。
- 最小合成样本检查区分单次极端与多次极端。
- 六个输入文件运行前后 SHA256 一致；profile 文件 SHA256：`bea54c12a0c013995c3644a0bab84b35413d5df8a6785dfdd6a03fd49f32978e`。

所有输入路径和 SHA256 保存在 summary；没有重新执行上游科学实验，也没有把这些有限检查表述为整个系统的认证。

## 8. 解释风险检查与停止边界

11/11 类解释风险已考虑：

| 风险 | 本轮处理 |
|---|---|
| Simpson 反转 | 全体与明确的解析子集均保留；未声称排除全部分层混杂 |
| 生态谬误 | 路线汇总不推出单一路口安全性或个体 AV 能力 |
| Berkson 选择 | 高质量历史订单及可解析 encounter 均是条件样本 |
| Collider 偏差 | 解析子集不作因果比较或性能收益估计 |
| 基础率忽略 | 明示全体、超出组、无 encounter 的不同分母 |
| 回归均值 | 无干预前后改善估计，不适用 |
| 幸存者偏差 | 不覆盖取消/未服务订单，不外推全城市需求 |
| 多重搜寻 | 无显著性检验，三 profile 四维和两范围全部报告 |
| 分析自由度 | 探索性描述；固定 cap，p90/tail 非优化后的政策阈值 |
| 相关即因果 | 未运行派单反事实，不声称恢复容量或服务收益 |
| 反向因果/时序 | 无 realized time 权重或未来信息输入决策 |

状态：`READ_ONLY_STATIC_EXPOSURE_DIAGNOSTIC_COMPLETE`。

停止于描述层。冻结 profile、Gamma、eligibility、M3、路线和 41 个正式场景均不变。若下一步研究信息增量，应先明确 SNE 是否补充 A/M/D/L 没有表达的街道形态维度，以及分析单位/空间尺度；不能因 mean 数值较小就启动一次放松门控实验。
