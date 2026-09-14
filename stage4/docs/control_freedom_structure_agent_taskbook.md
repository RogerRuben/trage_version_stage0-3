# Agent taskbook: Control freedom and joint-fleet structure diagnostic

日期：2026-09-14。只检查动态价值是否有可选动作以及固定单车规则的结构边界。不是 Stage5，不建立 predictor 或 controller。

执行补记：用户随后明确授权有界确定性路由重建；十个原登记状态已完成，见 [最终诊断报告](control_freedom_structure/action_space_diagnostic_report.md)。以下保留原执行设计及输入门槛，不构成再跑授权。

## 0. 当前执行状态与授权边界

已完成：输入盘点、有限 joint-fleet 反例，见 [结果报告](control_freedom_structure/structure_and_input_report.md)。未完成：真实候选图的 A1/A2 求解，因为已检查归档缺少完整带权 mixed arcs。

本任务书明确区分已完成工作与待执行方案，不自授权限：读取完整现成冻结输入、进行有界离线诊断属于本轮方向；重新调用 Valhalla 重建输入需要另行确认。不得为了补齐输入启动全天 FleetPy、预测推理、重新选择车辆或路由不一致的替代估计。

工作区：`D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla`；分支 `codex/stage2-v5-micro-transfer`。保护生产代码、配置、上游冻结产品、41 场景及 V3。只在新的独立工具和文档目录交付。

## 1. 输入要求：必须先满足

使用原登记的 10 个 ODD_Q50_M_P70_REFERENCE 时点，不按替代解丰富程度选择样本。完整输入必须同时包含 HV 与 AV，不是 neutral identity 图，也不是 AV-only funnel：

- 每弧 vehicle/request identity、vehicle type、pickup ETA（原始精度）、critical/carry-over 标志。
- 启用 Gamma 的逐弧 static/dynamic/speed exposure、累计 AV assignments/exposure 与配置阈值。
- 原时点、完整候选资格口径、来源配置；订单和车辆 identity 可映射。
- 参考 selected set、前三项最优计数与 pickup 目标；必要时由完整图重新求出该时点参考解。
- 若评价未来释放差异，还需每车当前承诺、session、固定服务终点与决策时预测时长。缺这些字段时只报告 assignment 差异，不用实际事后时长补齐。

核对完整候选弧不存在重复 identity，ETA 非负有限，订单标志跨弧一致，参数来源一致。不能只检查 selected set 能复现就称完整候选图已证明相同。只有 counts/hash 而无 ETA 时停止，输出 INPUT_INCOMPLETE。

如果未来授权重建：另列固定 routing mode、tiles/config/source、缓存及时间语义，先设计有界成本预算；不在此任务中自动启动。

## 2. 优化定义

在原 gates 与 Gamma 下重新求 critical→total→carry-over→最小 pickup，得到计数向量 l*、P*、稳定定义的参考 assignment x0。成本层保持该来源场景的 disabled 状态。

A1：固定每张订单的 y_o=y_o(x0)，只改车辆；A2：仅固定三项计数 l*。两个集合分别报告。参考 x0 是完整源图上的 pickup 最优解；若与历史选中解不同，先辨别等价并列还是不一致，不能覆盖原记录。

严格层在数学上使用 P=P*；机器求解另列容差，不将浮点近似当作精确唯一性证明。near-optimal 层固定诊断比例为 {0,0.005,0.01,0.02,0.05}，预算 P≤P*+alpha·P*。绝对秒 gap 必报；P*=0 时相对 gap 为 N/A，所有上述比例对应零增量，不擅加绝对放宽。

这些比例是描述性剖面，不选出“最优政策 alpha”。本轮不以未来服务收益调参。

## 3. 用最小 gap，不枚举所有 assignment

每状态、每 A 版本求两个优化子问题，各自最小化 P：

1. 排除 x0，查找任意不同 arc assignment。
2. 强制至少一张订单的 HV/AV 服务指示改变，查找车型分配不同的 assignment。

总服务数固定时，可用 `sum(x_e for e in selected(x0)) ≤ m−1` 排除原解。m=0 单列无非空替代，不构造错误约束。

第二类对每单定义 b_o=是否由 AV 服务；用相对 b_o(x0) 的二元 Hamming 差至少为 1。A1 下这是实际车型换配；A2 下也可能是 AV-served 与 unserved 之间变化，必须再记录订单集合变化，不能都称 HV↔AV swap。

得到最小 gap 后即可推导五档预算的存在性，不需逐档重复求解。若要宣称同一订单从 H 换 A 或反向，应另有可核查 witness，不能由 AV 总数或 Hamming 差自动推出。

若最优状态已证实，gap=alternative P−P*；负值超数值容差说明参考不一致，停止该状态。时限下有可行 witness 可以证明存在，不能证明最小 gap；无 witness 且未证 infeasible 标记 UNRESOLVED。提供 incumbent/bound 以界定哪些预算存在性已确定。不要把求解器限时当 unique。

## 4. 规模与内存

最多原登记 10 状态，单进程 CPU，无 GPU。每次仅一个稀疏 mixed graph，矩阵按 arc 建列，不构建 orders×vehicles 稠密矩阵；不枚举全部解。

建议硬停止上限：单状态 5000 arcs、单次 MILP 10 秒、整个离线诊断 10 分钟、进程 RSS 2 GiB。是诊断资源界限，不是科学阈值。超限标记 RESOURCE_LIMIT，不截断候选后继续冒充完整图。若环境不能可靠实施内存保护，先报告，不扩大规模。

每个状态先基准最多四层，再 A1/A2 各两类 gap 子问题；不运行新 solver tuning、随机重复或额外场景。使用已有环境，不安装依赖。

## 5. 输出与解释

每状态输出：时点、图大小、Gamma 状态、l*、P*、A版本、alternative 类型、求解状态、最小 gap 或上下界、绝对秒/相对比例、各预算存在性、served-order-set 是否变、vehicle identity 是否变、AV allocation 是否变、少量 witness、runtime/RSS。

汇总必须区分：全部登记状态、输入完整状态、求解已确定状态、未确定状态；十个指定时点不是全天代表性样本。报告几分之几而非断言“95% epochs 无空间”。不给“足够行动空间”后验编造统一门槛。

joint-fleet 检查只否定 context-blind 固定规则；同信息简单上下文规则必须保留。既有例子已表明它仍可最优，不通过增大 toy 来人为制造方法优势。

## 6. 完成与停止

当前真实 A 检查被输入缺口阻止，不得用 B 通过宣告整个方向 GO。后续如果 A-strict 空间少，只说明该版本可能无用；near-optimal 空间多也不自动授权 pickup 政策放宽。

不论结果如何，不训练 predictor、不实现 LP/MPC、不接入生产 solver、不跑真实动态政策、不改 V3。仅提交本轮小文档/独立诊断结果到已授权 origin，禁止 force push；最终报告实际完成/未完成及原因。
