# Control freedom and joint-fleet structure: bounded findings

历史阶段说明：本文记录授权重建前的输入盘点。用户随后授权的有界重建与真实 A1/A2 求解已完成，输入缺口已关闭；最新结果见 [行动空间诊断报告](action_space_diagnostic_report.md)，本文 Toy 2 结论不变。

日期：2026-09-14；起始 commit：d4c32fff9fad4f6733be8efc9cddad5c4e2ccb23。

本轮仅做冻结输入盘点与第二个有限合成例子。生产 solver、路由、预测、FleetPy、V3 和 41 场景不变。

## A. 真实行动空间：当前不能直接计算

实际检查的文件位于 `stage4/output/paper_enhancement/`：

| 来源 | 已保存内容 | 对本诊断的不足 |
| --- | --- | --- |
| frozen_state_prediction_ablation/frozen_epoch_registry.csv | 10 个时点、等待/车辆数量、累计 exposure、state hash | 不是完整逐弧输入 |
| frozen_state_prediction_ablation/selected_assignments_diagnostic.csv | 选中弧的订单、车辆、类型、ETA、exposure | 缺少未选候选弧，不能求第二优解 |
| frozen_state_prediction_ablation/decision_variant_metrics.csv | 计数、订单集合、目标等汇总 | 汇总不能恢复所有可行 assignment |
| mechanism_validity/neutral_av_identity.csv | neutral 图身份、选中身份、pickup 总值 | 不是原资格下的完整带权 mixed graph；无逐弧 ETA |
| mechanism_validity/snapshot_gate_matching.csv | AV gate E/U/M 汇总 | 不是 mixed-fleet 优化输入；Gamma 还在后续联合优化中 |

`frozen_state_prediction_ablation.py` 在内存中构造完整 arcs，但落盘的是选中 assignments 与汇总。`mechanism_validity.load_states()` 会重建物理状态；生产候选捕获路径仍使用 routing adapter。不能将它描述成直接读取现成完整候选表的纯离线求解。

另在 `stage4/output` 中按 arc/candidate/snapshot/state 文件名查找，见到 routing sample、epoch stats、exposure 表，未定位到上述十状态的完整原始 mixed solver-input arc archive。这里只报告已检查范围，不宣称硬盘其他位置绝无归档。

因此 A 的状态是 **INPUT_INCOMPLETE / NOT_EVALUATED**，不是“唯一解”、不是“不存在行动空间”。没有执行 A1/A2 的 MILP，也没有构造相对 gap 表。原始场景为 ODD_Q50_M_P70_REFERENCE，配置含 Gamma，不能用 neutral/no-Gamma 图替代。

下一步必须提供现成完整弧归档，或另行授权有界确定性路由重建。两者都不得静默退化成重跑全天模拟。详细待执行规则见 [诊断任务书](../control_freedom_structure_agent_taskbook.md)。

## B. Joint-fleet 例子：当前目标相同，最优动作随 companion 改变

沿用 Toy 1 的时钟：t=0 当前订单 L→R；本区 pickup=0；任意跨区旅行 600 秒；当前服务 600 秒；未来订单 t=900 同时释放、deadline=960、服务 600 秒；两期 horizon=1800。无主动空驶、Gamma、成本、预测误差或 session 差异。

当前单 pickup deadline 明确定为 t=60，故 C 的 A2 与尚在执行任务的 H2 均不能接当前单。

有两辆 H（H1/H2）与两辆 A（A1/A2），三个区域 L/R/C。H1、A1 当前同在 L，均能服务当前单，所有即时目标并列。A2 在 C，无法按当前本区 pickup deadline 接到当前单。H2 正执行既有任务，t=600 才在 L 或 R 释放，因而不能接当前单；这项既有服务是固定背景，不计入两个动作的新增服务收益。所有车未来均在线。

Context L：H2 在 L 释放；Context R：H2 在 R 释放。H1、A1 的当前属性和当前订单完全相同，H2 的已承诺释放位置是决策时已知信息。不是窥看未来订单。

未来固定三单：oL: L→R、oR: R→L 都是 H-only；oC: C→L 允许 H/A。OD、时刻、资格在两个 context 完全不变。跨区 pickup 不可行。当前 x_A 派 A1，未来 H1=L/A1=R；x_H 派 H1，未来 H1=R/A1=L。

| Companion context | 当前动作 | 未来全部可行弧 | 最大未来服务数 |
| --- | --- | --- | ---: |
| H2→L | x_A | H1→oL，H2→oL，A2→oC | 2 |
| H2→L | x_H | H1→oR，H2→oL，A2→oC | 3 |
| H2→R | x_A | H1→oL，H2→oR，A2→oC | 3 |
| H2→R | x_H | H1→oR，H2→oR，A2→oC | 2 |

两个 context 的唯一最优动作相反；当前服务始终是同一张单，所以是真正的 A1 pure reassignment。A1 未来的零贡献是逐弧检查的结果，不是将其漏计。当前服务在总值中只加 1。

### 边际贡献与可加表示

固定 x_A 时 H1 一直在 L，自己的未来可行弧一直是 H1→oL。去掉 H1 后，两个 context 的 rank 都为 2；加入后分别为 2、3。因此同一 H1 的边际贡献为 0、1，取决于 companion fleet。

设 U(a,c) 为表中精确值。如果存在不看 companion 的固定单车状态价值之和，则在本例中可归并为 F(a)+G(c)，动作差不应随 c 变化。但这里差分别为 −1、+1，交叉差为 −2，不为 0。因此这种精确可加表示无法同时表示四格值。

这是有限反例，不是新的 matching theorem，也不是计算复杂度下界。它否定的是只依赖 focal 车辆自身属性的固定评分及固定 tie-break。若评分或 tie-break 读取其他车的释放位置，已经超出被否定类别。

固定选择 x_A 的概率 q，两个 context 的期望分别为 3−q、2+q，regret 为 q、1−q，故任一固定随机混合至少在一个 context 损失 1/2。允许 q 随 companion 变化则可精确最优。

### 必须保留的简单 baseline

规则“让 H1 下一期覆盖 H2 没覆盖的 restricted 区域”已精确解决全部四格。这是一个简单的 fleet-context rule。故本例证明背景相关与固定局部分数不足，仍然不证明复杂 lookahead、LP/MPC 或新学习模型必要，也不排除具有全车队上下文的标量评分。

## C. 本轮分类

| 项目 | 状态 |
| --- | --- |
| 完整真实候选输入 | INPUT_INCOMPLETE_IN_INSPECTED_ARCHIVES |
| A1/A2 真实行动空间 | NOT_EVALUATED |
| companion-dependent marginal value | ESTABLISHED_IN_FINITE_EXAMPLE |
| context-blind fixed-score limitation | COUNTEREXAMPLE_ESTABLISHED |
| 同信息简单 context rule | EXACT_IN_THIS_EXAMPLE |
| 复杂方法必要性 / 创新 | UNESTABLISHED |
| 真实动态收益 | NOT_TESTED |
| 下一阶段 | STOP_PENDING_COMPLETE_INPUT_OR_RECONSTRUCTION_AUTHORIZATION |

不能用 B 成立替代 A，也不能把 A 缺输入记作 A 失败。V3 不吸收新 toy；不改既有论文结论。

## D. 实际核验

运行 [独立脚本](../../tools/joint_fleet_structure_check.py)：Python 标准库、单进程 CPU、最多四车三单，不读文件、不导入生产模块。预先限定八项：四个全车队格子、两个去掉 H1 的格子、两个固定随机混合 q=0/0.5。结果 PASS、退出码 0，实测调用约 0.082 秒。公式对任意 q 的结论由正文解析证明，而非有限 q 核验推广得出。

未执行真实 MILP、路由、FleetPy 或训练；没有测量峰值 RSS，不填造内存指标。只核查本轮文档链接和暂存 diff，不新建测试框架。这是公式/枚举一致性核验，不是 simulation validation。
