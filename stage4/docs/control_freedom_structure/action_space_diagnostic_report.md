# Frozen-state A1/A2 action-space diagnostic

日期：2026-09-14。执行前 Git 基线：`98246579063ad9f6cf4f955958eb0130d94dd8f2`。本轮在用户明确授权后补齐十个冻结状态的完整带权 mixed candidate arcs，并完成离线求解；此前的 INPUT_INCOMPLETE 已关闭。

## 1. 结论先行

**普通换车空间存在，但严格保持最小 pickup 时，样本中没有 HV/AV 服务分配改变的空间。** 同信息复杂控制器的必要性仍未成立。

- A1 固定订单集合：严格 pickup 下 4/10 状态可改变 assignment，但 0/10 可改变订单由 HV 还是 AV 服务。
- A2 仅固定 critical/total/carry-over 数：严格 pickup 下 5/10 状态可改变 assignment，同样 0/10 可改变 AV 服务指示。
- pickup 预算增加到 2% 仍没有车型分配改变；5% 时 A1/A2 都仅 1/10 有此空间。
- 这不是“真实系统 95% 都唯一”的证据，也不是所有动态控制无用。它只描述原登记十个条件状态、原 gates/Gamma、当前稀疏候选图。

## 2. 重建与一致性

来源：ODD_Q50_M_P70_REFERENCE 的 07:30、08:30、12:00、13:00、17:00、17:30、18:00、18:30、21:00、23:00。固定 q_A=.50、M、acceptance=.70、零 lead/300 秒 patience、Top-K20、原 Gamma；成本层 disabled。

调用原冻结物理状态构造函数，十个 state SHA 全部与登记一致。使用生产 candidate builder，并在 solver 调用前截取其输入；不推进车辆、不模拟下一 epoch。没有调用旧分析脚本的完整 run，也没有重建历史预测 reference 或运行 M3。

pickup 使用既有 SINGLE_SOURCE_MATRIX、auto、WGS84、上海时区分钟粒度和冻结 beta，不更换 route fallback 或参数。每状态清空缓存。累计 13,046 次新路由 arc evaluations，失败 0；最终十状态 solver-input arcs 共 130 条，其中 AV 48 条，与既有 funnel 的 AV 总数一致。

完整图是本轮确定性重建，不冒称旧原始 arc archive 被找回。验证包括：原 P selected identities 全部存在、逐选中弧 ETA 一致、原计数和最小 pickup 一致、历史解满足完整 Gamma。由于旧文件没有完整带权图 hash，不能由此宣称已逐条比对所有旧未选弧。

本轮保留 10 份完整 arc JSON、10 份诊断 JSON，以及 summary/provenance/verification，随本次提交纳入版本控制。原大输出和路由日志不提交。这样后续查看替代解无需再次路由。

## 3. 预算下存在替代解的状态数

数值判定容差独立固定为 1e-7 秒；0% 行表示机器容差内的严格 pickup，不宣称对实数的形式化唯一性证明。

| aggregate pickup 增量预算 | A1 任意换配 | A1 AV 服务指示改变 | A2 任意换配 | A2 AV 服务指示改变 |
| --- | ---: | ---: | ---: | ---: |
| 0% | 4/10 | 0/10 | 5/10 | 0/10 |
| 0.5% | 5/10 | 0/10 | 5/10 | 0/10 |
| 1% | 6/10 | 0/10 | 6/10 | 0/10 |
| 2% | 6/10 | 0/10 | 6/10 | 0/10 |
| 5% | 9/10 | 1/10 | 9/10 | 1/10 |

预算值只作预先固定的诊断剖面，不选为新政策参数。AV 服务指示定义为每单是否由 AV 服务；A2 下可能涉及 served/unserved 切换，所以需要 witness 才能称为同订单 HV↔AV 换配。

## 4. 最小 gap：绝对秒数

“无解”表示在对应订单/计数约束和 Gamma 下，即使不限制 pickup 增量也没有该类替代解；不是时限失败。所有 40 个 alternative 子问题都有明确 solver disposition，无 unresolved。

| 时点 | mixed arcs | 当前服务数 | P* 秒 | A1 任意 gap | A1 AV gap | A2 任意 gap | A2 AV gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 07:30 | 26 | 5 | 874.222699 | 22.012802 | 59.749033 | 22.012802 | 59.749033 |
| 08:30 | 6 | 3 | 609.295913 | 24.157107 | 无解 | 24.157107 | 无解 |
| 12:00 | 18 | 6 | 839.394407 | 0 | 无解 | 0 | 无解 |
| 13:00 | 14 | 4 | 564.594583 | 4.182182 | 无解 | 4.182182 | 无解 |
| 17:00 | 3 | 1 | 72.783519 | 0 | 无解 | 0 | 无解 |
| 17:30 | 22 | 4 | 963.380800 | 0 | 无解 | 0 | 无解 |
| 18:00 | 18 | 3 | 384.159179 | 0 | 无解 | 0 | 无解 |
| 18:30 | 5 | 3 | 450.228059 | 13.050089 | 无解 | 13.050089 | 无解 |
| 21:00 | 17 | 3 | 623.951886 | 2.115091 | 29.611276 | 0 | 29.611276 |
| 23:00 | 1 | 0 | 0 | 无解 | 无解 | 无解 | 无解 |

23:00 固定最优服务数为零，只有空 assignment，不能把它当作非空匹配“严格唯一”的典型。其相对 gap 为 N/A，比例预算仍为零。其余九个状态有非空参考解。十状态前三层的 critical 与 carry-over 最优计数均为零，本轮没有检验高 critical/carry-over 状态的代表性。

### 21:00 的三种区别

1. A1 普通换车：相同订单由另一辆 HV 服务，pickup 从 154.401653 增至 156.516744 秒，总增加 2.115091 秒（0.338983%），没有车型变化。
2. A2 严格并列：同一辆 HV 改接另一订单，两条弧 ETA 同为 249.580754 秒。它改变当前乘客服务对象，不是 pure vehicle reassignment。
3. A1/A2 车型换配：同一订单从 HV 的 219.969478 秒换成 AV 的 249.580754 秒，总增加 29.611276 秒（4.745763%）；已核查 witness 是真正的同订单 H→A。

07:30 的车型换配最小代价为 59.749033 秒，超过本轮 5% 预算，但该有限最小 gap 仍如实报告。其余八个时点在固定当前计数和 Gamma 下无该类解。不能为了让车型变化变多而放宽 Gamma、扩大 K 或改变 patience。

## 5. 验证与资源

生产重建/诊断 Python 耗时 38.130 秒；外部监护调用总计 39.264 秒。每 250ms 外部采样，内外监测的峰值 RSS 均约 260.30 MiB；这是采样峰值，不保证捕捉所有瞬时峰值。无 GPU。最大单状态仅 26 条 solver arcs，未分配 orders×vehicles 稠密矩阵。

外部监护只管理本次启动的 Python PID；达到 2 GiB 或 600 秒即终止该诊断。内部额外检查逐状态路由候选数、总路由数和运行时；MILP 每次上限 10 秒。没有触发上限。

22 个可行 alternative witness 和 18 个 solver infeasibility disposition。单独 verifier 对全部可行 witness 重算一车一单、计数、pickup、Gamma、A1 订单集合及 AV 指示改变；未另做不可行性的独立形式化证明。三项小型 formulation fixtures 检查：已知 1 秒 gap、A1/A2 订单切换区别、Gamma 禁止 AV。全部 PASS。另运行 touched-file py_compile 和 diff check，不跑全仓测试或新实验。

源配置与 beta SHA 前后相同，执行脚本 SHA 与 provenance 一致。tiles 记录现有路径、23 文件目录 inventory hash，历史 Stage0 全 tiles SHA 仅作为来源值，未重算为本轮内容证明；不把目录 inventory 当成全文件内容 hash。

## 6. 科研解释与停止

Toy 2 已证明固定局部车辆分数可能遗漏 companion context，但简单上下文规则仍可最优。真实 A 诊断现在进一步显示：这些状态中，严格即时目标下的可选解主要不是 HV/AV 分配改变；这弱化了立即开展跨车型 reserve/lookahead 方法开发的实际依据。

同车型换车仍可能改变位置、session 与未来价值，本轮没有对此预测收益，也没有声称无用。没有使用实际未来服务来给 witness 排名；释放状态差异未被量化，不据此推断全天收益。

最终分类：A_INPUT_CLOSURE=COMPLETE；A1/A2_DIAGNOSTIC=COMPLETE；STRICT_CROSS_TYPE_FREEDOM=NONE_IN_10_REGISTERED_STATES；METHOD_NECESSITY=UNESTABLISHED；REAL_DYNAMIC_BENEFIT=NOT_TESTED。

建议 **STOP_OR_SIMPLIFY / REVIEW_ONLY**。不由 5% 那一个状态自动选择政策容差，不扩样本、不调 Gamma、不训练 predictor、不接入控制器、不推进 Stage5，V3 与冻结场景保持不变。

## 文件入口

- [汇总结果](../../output/paper_enhancement/control_freedom_structure/summary.json)
- [来源记录](../../output/paper_enhancement/control_freedom_structure/provenance.json)
- [验证记录](../../output/paper_enhancement/control_freedom_structure/verification.json)
- [21:00 替代解 witness](../../output/paper_enhancement/control_freedom_structure/diagnosis_2100.json)
- [离线诊断工具](../../tools/control_freedom_diagnostic.py)
- [有界运行入口](../../tools/run_control_freedom_guarded.ps1)
- [不调用路由的核验工具](../../tools/verify_control_freedom.py)
