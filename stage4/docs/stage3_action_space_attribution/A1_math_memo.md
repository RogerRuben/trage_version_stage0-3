# A1 — Attribution estimand and mathematical boundaries

日期：2026-09-18。仅定义实验，不求新解。

## 1. 固定物理状态，不等于固定可选弧

令 B_t 为同一时点可用车辆、等待订单、空间半径形成的共同潜在弧集合；w_e 为冻结 pickup ETA。D_d 只改变需求侧 Stage3 eligibility，其他条件保持不变。

如果只持有已通过 D0 的 A_t^0，则对它换标签只能删边，无法增加已被过滤的边。目标 A_t^d 必须在共同 B_t 上重新筛选；缺失 w_e 不是 w_e=∞，也不是弧不可行。

所有结果条件于同一历史物理状态，不代表新 eligibility policy 的演化路径或全天服务效果。

## 2. 指标与单位

对每个层级图 G：E_AV 为 AV 弧数；U_AV 为至少一个 AV 邻居的订单数；M_AV 为 AV 子图最大匹配数；O_HA 为同时有 H、A 邻居的订单数。

O_HA 是局部重叠，既不等于联合 assignment 可换车型，也不保证现有所有已服务订单能同时保留。车辆竞争、计数、Gamma、pickup 最优层还会收缩可选解。

分层早期的 O_HA 仅代表通过该层的重叠，不可称为“此刻真的能服务”；只有通过所有 pickup/session 条件的终层图才有该解释。必须补出 routing-evidence availability 和 session 层，不把它们混入 Stage3 loss。

跨十状态相加时，U/M/O 是 state-order 或 state-matching 总数，不是 unique daily orders。Stage3 两步下降 (493−124)/493≈74.85% 是旧固定顺序下、pickup 前的条件差，不是最终 cross-type freedom 的因果份额。

## 3. D4 的上界需要条件，不能自动成立

若共享同一个 B_t、w_e、HV 弧，且松门仅增加 AV 弧，则 E/U/M_AV/O_HA 单调不减。因此 D4 可以成为这些图指标的 eligibility-relaxed 上界。

生产 shared Top-K20 在资格过滤后按 H+AV 一起截断。松开 AV 资格可能把原 HV 挤出 K20，mixed graph 不再包含原图；O_HA、mixed capacity 和当前已服务订单保留不保证单调。不能给不同条件各跑一次 shared Top-K 后仍宣称是 nested edge-addition upper bound。

即便图严格嵌套，每条件重新优化计数和 pickup 后，可选最优解集合也不嵌套。增加边可提高最优服务数或产生唯一更低 pickup 的解，使 strict alternative 指标下降。因此 **D4 的 assignment freedom 不是 D0–D3 的数学上界**；它至多是一个明确口径的 fully relaxed diagnostic comparator。

Gamma 开启且 rho 缺失时，无法既“移除全部 Stage3 evidence”又忠实保留原约束：将 rho 填 0 会偷偷放宽 Gamma，填极大值会偷偷保留证据门。建议五条件全部 Gamma-off，并把其 D0 与旧 Gamma-on 基线分开；待用户冻结。

## 4. A1/A2 应定义两种不同参照问题

推荐 primary 为 **condition-local**：每个 D_d 求自己的词典序计数 l_d*、pickup P_d* 和稳定参考 x_d*。A1 固定 y(x_d*)，A2 固定 l_d*。输出最小 alternative gap g_d 和 g_d/P_d*，P_d*=0 时相对值 N/A。

这回答“该条件自身最优派单附近有多少选择”，不是“相同乘客跨条件替换了多少”。不同条件 y/l/P 变化必须单列，不能直接将 alternative-state 数相减当成纯车辆切换效应。

若研究固定 D0 订单集合能否跨条件换车型，则是另一个 **baseline-anchored** estimand。它需额外固定 y(x_0*)、明确参照 pickup，不宜与 primary 混报。为了不扩大本轮，建议只设计 condition-local，并明确局限。

A2 的 AV service indicator 改变可能只是 AV-served↔unserved。必须报告共同服务订单中的实际 H↔A 改变和 served-set 变化，不能全部称 pure cross-type swap。

## 5. nested chain 是顺序分解，不是独立贡献

spatial→passenger→direction/certified hard→capability hard→evidence→Top-K→pickup→session 可以生成条件顺序损失，但 Stage3 多个原因与缺失证据重叠，换顺序损失分配可能变。

D1/D2/D3 应分别相对于 D0 单因素定义，不是 D1→D2→D3 的累积放松。D4 才是联合移除 comparator。单因素无改善可能因另一个门仍然阻断，不能据此判定该因素没有作用。

## 6. 筛查判断不是因果结论

“D4 只增加 0–1 state”可以预注册为停止投入阈值，但不能证明 Stage3 不是主要原因：strict 层会重优化、Gamma/Top-K口径可能变化、缺失 ETA 会遮蔽、单因素会被共现门压住。

合法判定顺序：输入完整且语义冻结→结果可解释→按预注册标准决定是否继续审计。只有最后一步可以输出 NO_FURTHER_INVESTMENT_ON_THIS_SAMPLE；不得输出全系统因果免责。

若缺少新弧 ETA，完整实验=NOT_IDENTIFIABLE_FROM_CURRENT_ARCHIVE，而不是 D4=0。需要单独提交补充数据方案或接受只做上游资格覆盖审计，不能在本轮悄悄重路由。
