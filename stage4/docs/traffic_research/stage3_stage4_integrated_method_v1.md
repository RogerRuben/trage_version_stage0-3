# Stage3–Stage4 统一方法与数学表述 v1

日期：2026-09-28。依据代码与已有结果基线：`34e543d`。

**状态：方法整理稿，待作者确认；本文件不授权启动实验。** 本轮仅整理方法，不修改冻结模型、profile、运行配置或已有结果，也不执行训练、推理、仿真。原 41 场景仍属于原冻结策略；不能因本文提出统一表达，就将它们改称新策略的实验结果。

## 1. 方法定位与正文结构

**核心问题是异质车队的名义供给如何经由订单—车辆兼容性转化为可匹配能力及最终服务，而不是构建 AV 安全认证模型。** Stage2 提供决策时可用的多目标路线预测；Stage3 将预测与静态网络转化为有解释的运营适配描述；Stage4 在乘客接受、到达耐心、车辆可用性和平台暴露预算下进行稀疏滚动派单。

本文选取一条主方法：**静态与速度域约束 + 交通环境适配 + 保留波动性暴露的滚动分配**。交通状态不是取代 Stage2，也不是叠加到原 12 维动态门控之上；它替换 crawl/stop 的六个 E/Q/C 约束入口，CV/acceleration 的六个 E/Q/C 及其他原有模块保留。

正文可压缩为以下六个小节，其余实现口径放附录：

1. 决策时信息与多目标路线预测。
2. 静态结构、速度域与交通状态表示。
3. C/M/A 情景的交通适配假设。
4. 互补的波动性暴露与累计预算。
5. 稀疏滚动派单模型。
6. 瞬时匹配能力、服务结果及解释边界。

概念关系为：

\[
\text{名义供给}
\longrightarrow G_t(\text{结构、交通、接受、耐心})
\longrightarrow C_t^{\mathrm{eff}}=\nu(G_t)
\longrightarrow Y_{1:T}(\text{滚动策略、历史预算、车辆演化}).
\]

这是解释关系，不是四个变量之间一一对应的确定性等式。

### 1.1 原有研究如何进入这一版

| 原有信息 | 本版作用 | 是否直接形成决策条件 |
|---|---|---|
| Stage0/1 有向路线、traversal、直接观测标签 | 路线身份、训练目标与历史参考 | 上游基础，不重新生产 |
| Stage2 pace / travel time | 相对减速、路线预测时间、到达/完成可行性、运营成本 | 是 |
| Stage2 crawl / stop | 低速占比与交通状态 | 是；替换原 crawl/stop 六项 EQC 入口 |
| Stage2 speed CV / acceleration RMS | 行驶波动性的 E/Q/C 暴露 | 是；保留六项 |
| Stage3 A/M/D/L、movement、方向身份 | 静态利用率与结构性可行性 | 是 |
| Stage3 冻结速度域 | 高速度域的独立适配边界 | 是；不再承担主要拥堵区分角色 |
| 支持度、UNKNOWN、最长拥堵段、完整旧 EQC | 解释、分层统计、可追溯比较 | 不新增门控或成本项 |
| Stage2 其他输入特征/辅助目标 | 参与既有预测表征或作为诊断 | 不要求每个特征再各加一条派单约束 |

“充分利用预测”指把不同输出用于有物理语义的决策环节，不意味着把所有输出重复转成惩罚项。保留维度之间仍可能相关；本方法避免明确的同源重复门控，不声称统计独立或已完全消除冗余。

## 2. 符号、时间与信息边界

用 \(o\) 表示订单，\(v\) 表示车辆，\(k\in\{C,M,A\}\) 表示 AV 研究情景，\(t\) 表示派单时刻。路线 \(r_o=(e_{o1},\ldots,e_{on_o})\) 是有序 traversal/token 序列。同一 canonical edge 的重复访问保留为不同 token，不能按 edge 去重后计算暴露。

设请求释放时刻为 \(a_o\)，到达耐心为 \(P_o\)，则：

\[
d_o=a_o+P_o,\qquad s_o(t)=d_o-t.
\tag{1}
\]

\(d_o\) 是接驾到达截止时间，不是载客行程完成截止时间。现有 canonical replay 使用 observed departure/boarding proxy 作为释放时刻，即 lead=0，\(P_o=300\) s，滚动间隔为 30 s。这是回放假设，不等于真实乘客请求时间已被观测。

令 \(\mathcal I_o\) 为预测决策时已经可用的信息。冻结 M3 对每个 token 输出：

\[
(\widehat p_{oi},\widehat w_{oi},\widehat c_{oi},
\widehat s_{oi},\widehat z^{\mathrm{cv}}_{oi},
\widehat z^{\mathrm{acc}}_{oi})
=f_{\theta^{\mathrm{M3}}}(r_o,\mathcal I_o)_i.
\tag{2}
\]

其中 \(\widehat p\) 为 P50 pace（s/m），\(\widehat w\) 为 token 的预测 P50 时间（s），\(\widehat c,\widehat s\) 分别为 crawl/stop time share；后两项是既有 bounded CV/acceleration 输出。本文的上标 \(z\) 只是原始预测的记号，后文 CDF 变换另用 \(u\)。实现直接复用冻结输出，不重新用全 edge 长度重算 partial traversal 的时间。

\[
\widehat T_o=\sum_i\widehat w_{oi},\qquad
\omega_{oi}=\widehat w_{oi}/\widehat T_o.
\tag{3}
\]

\(\widehat T_o\) 是 token P50 的和，不声称等于联合路线旅行时间分布的严格中位数。所有权重需有限且为正。

**当前实现是 request/route-conditioned 的冻结预测，不是每个 rolling epoch 都重新观测并刷新整张路网。** 时间变化通过预测上下文及 predicted progression 表达，不能把它写成实时交通监测系统。历史路线属于 revealed-route proxy：使用原始路线几何这一信息假设应公开；禁止使用完成后的 realized travel time 或未来拥堵来修正决策。若以后提前请求释放，不能未经检查直接复用原释放时刻的预测。

## 3. Stage3：结构与速度域

### 3.1 结构性状态独立于连续利用率

保留 \(h_{ok}\in\{\mathrm{FEASIBLE},\mathrm{UNKNOWN},\mathrm{INFEASIBLE}\}\)。已知不可通行方向、禁止 movement 或 profile 明确禁用的 maneuver 属于结构性不可行；身份或关键结构证据未解析属于 UNKNOWN。历史反向遍历保留独立 overlay，不能投影为 forward edge。

单纯超过某个经验 envelope 不自动改写为“车辆无法安全行驶”。当前 Stage4 基线仅允许结构与关键预测证据完整的 AV 订单；后文放宽交通 UNKNOWN 的研究口径不改变这一基线。

### 3.2 静态描述与路线聚合

对交叉口 complex \(c\) 定义：

\[
A_c=\text{外部物理连接数},\quad
M_c=\text{拓扑 movement 数},\quad
L_c=\text{内部道路总长度（m）},
\]

\[
D_c=\left|\{\operatorname{roadclass}(e):
e\in\delta^{\mathrm{in}}(c)\cup\delta^{\mathrm{out}}(c)\}\right|.
\tag{4}
\]

\(D_c\) 使用 incoming/outgoing boundary edges，不使用 internal edges。这样 single-node complex 不会仅因内部边为空而被赋予零道路类别多样性。intersection clustering、movement 和 S3.1 修正均不重新生成。

设 \(\mathcal C_o\) 为路线解析出的 complex encounters，对 \(j\in\{A,M,D,L\}\)：

\[
X^{\mathrm{stat}}_{oj}=\max_{c\in\mathcal C_o}j_c,
\qquad
\rho^{\mathrm{stat}}_{ok}
=\max_j X^{\mathrm{stat}}_{oj}/B^{\mathrm{stat}}_{kj}.
\tag{5}
\]

确实没有 encounter 时该项暴露为零；存在 encounter 但缺失描述不能当作零。cap 使用 unique Train-exposed complexes 的 higher quantile，不按订单经过次数重复加权。

### 3.3 速度域不是拥堵状态

设 \(V_e^{\mathrm{domain}}\) 为冻结 Stage3 速度域属性：

\[
\rho^{\mathrm{speed}}_{ok}
=\frac{\max_{e\in r_o}V_e^{\mathrm{domain}}}{V_k^{\max}},
\quad(V_C^{\max},V_M^{\max},V_A^{\max})=(60,80,120)\ \mathrm{km/h}.
\tag{6}
\]

未知速度属性保留 unknown，不填零。速度域是道路环境边界，不能用预测拥堵速度降低它：高速道路在堵车时不会因此变成低速度域道路。本文保留该维度，但不把它作为 C/M/A 交通情景的主要区分依据。

## 4. Stage3：可解释交通状态

### 4.1 Train 参考与预测状态分开

对于 edge \(e\)，先在每个 Train 的 order–edge–day 内平均有效观测 pace，再对这些记录取 20% 分位数：

\[
p_e^{\mathrm{ref}}=Q_{0.20}
\bigl(\{\overline p^{\mathrm{obs}}_{e,o,d}:
d\in\mathrm{Train}\}\bigr).
\tag{7}
\]

现有参考期为 20161009–24。它是历史较低 pace 参考，**不是实测 free-flow speed，也不是法定限速**。参考不随 Test31 结果重新拟合。

令：

\[
R_{oi}=\widehat p_{oi}/p_{e_{oi}}^{\mathrm{ref}},
\qquad L_{oi}^{\mathrm{low}}=\widehat c_{oi}+\widehat s_{oi}.
\tag{8}
\]

\(R\) 表示相对于该道路参考的减速；\(L^{\mathrm{low}}\) 表示预测低速/停车时间份额。二者联合避免把“相对变慢”和“绝对低速占比高”混为同一现象。

### 4.2 互斥状态定义

对预测与参考可估计的 token，按下式构建六类状态（MIXED 拆成两类）：

\[
S_{oi}=\begin{cases}
\mathrm{SC},&R_{oi}\ge2.0\ \land\ L_{oi}^{\mathrm{low}}\ge0.8,\\
\mathrm{CG},&R_{oi}\ge1.5\ \land\ L_{oi}^{\mathrm{low}}\ge0.5,
\quad\text{且不属于 SC},\\
\mathrm{NC},&R_{oi}<1.5\ \land\ L_{oi}^{\mathrm{low}}<0.5,\\
\mathrm{MR},&R_{oi}\ge1.5\ \land\ L_{oi}^{\mathrm{low}}<0.5,\\
\mathrm{ML},&R_{oi}<1.5\ \land\ L_{oi}^{\mathrm{low}}\ge0.5,\\
\mathrm{U},&\text{参考或必要预测不可估计}.
\end{cases}
\tag{9}
\]

NC、CG、SC 分别为非拥堵、拥堵、严重拥堵 **proxy**；MR 是相对减速型 MIXED，ML 是高低速占比型 MIXED。阈值是既有研究规则，不是拥堵真值标定或 AV 能力测定。

ML 可能与道路固有低速、停车或局部控制有关，但不能仅由这些信号认定其原因。MR 也不意味着绝对速度很低。两类 MIXED 保持独立命名，不能一律重命名为拥堵。

### 4.3 可估计性与支持度是两个轴

必要预测必须具有正且有限的 pace，crawl/stop 在 \([0,1]\) 内，其和不超过 1（实现允许 \(10^{-6}\) 数值误差）；参考 pace 必须正且有限。研究状态允许已有有效参考但不足严格支持度的 edge 参与分类，同时标记 LOW_REFERENCE_SUPPORT。

既有严格支持线为 100 order–edge–day 记录且至少 5 个日期；研究分类使用至少 1 条、1 日的有效参考。它不虚构当前窗口样本数，也不宣称低支持预测与高支持预测同样可靠。历史观测 cell 的 30 分钟计数/新鲜度规则属于另一类描述产品，不是本预测接口的实时观测门槛。

路线状态占比、未知占比及低支持占比分别为：

\[
\phi_{os}=\sum_i\omega_{oi}\mathbf1(S_{oi}=s),\quad
u_o=\phi_{o\mathrm U},\quad
\ell_o=\sum_i\omega_{oi}\mathbf1(\text{low support}_{oi}).
\tag{10}
\]

所有 \(\phi\) 使用**完整路线预测时间**为分母，并满足 \(\sum_s\phi_{os}=1\)。低支持是另外一个标记轴，不能和状态占比相加凑成分区。UNKNOWN 不从分母剔除、不当作 NC；缺失整个 token 的预测时间导致路线无法完整计量，不能靠删除 token 伪造完整路线。

## 5. C/M/A 环境适配与订单级预算

**C/M/A 是嵌套研究情景，不是实际车型性能鉴定。** 对已知状态采用以下适配假设：

| 已知交通状态 | C | M | A |
|---|---|---|---|
| NC 非拥堵 proxy | 环境内 | 环境内 | 环境内 |
| ML 高低速占比、无显著相对减速 | 环境内 | 环境内 | 环境内 |
| MR 相对减速、低速占比未高 | 环境外 | 环境内 | 环境内 |
| CG 拥堵 proxy | 环境外 | 环境内 | 环境内 |
| SC 严重拥堵 proxy | 环境外 | 环境外 | 环境内 |
| U 不可估计 | 不新增否决；单列 | 不新增否决；单列 | 不新增否决；单列 |

选择 ML 为环境内并不表示停车负担消失，而是避免在缺少相对减速证据时再增加拥堵否决。CV/acceleration、静态结构和其他既有条件仍可能起作用。

记环境外集合为：

\[
\mathcal O_C=\{\mathrm{MR,CG,SC}\},\quad
\mathcal O_M=\{\mathrm{SC}\},\quad
\mathcal O_A=\varnothing.
\]

\[
b_{ok}=\sum_{s\in\mathcal O_k}\phi_{os},\qquad
g^{\mathrm{traffic}}_{ok}=\mathbf1(b_{ok}\le\beta),\quad\beta=0.05.
\tag{11}
\]

实现比较使用 \(\beta+10^{-6}\)，仅用于浮点容差，不是改变 5% 预算。A 的 \(b_{oA}=0\) 是定义结果，不是其没有未知风险的证据。

5% 表示**单条订单路线中已估计的环境外预测时间份额**。不代表 5% 事故风险、不代表全车队可以有 5% 不适配订单，也不转成 \(b/\beta\) 再叠加到 dynamic rho 或成本。

由于不对交通 UNKNOWN 新增门控，此规则只能称“已估计环境外暴露未超预算”，不能称“至少 95% 的路线已被证明适配”。未知占比高会削弱解释强度，因此结果同时报告 \(u_o\) 与 \(\ell_o\)，不偷偷改阈值补偿。

## 6. 保留 CV/acceleration 的 E/Q/C

### 6.1 冻结 CDF 与路线描述

保留 \(j\in\mathcal J=\{\mathrm{cv},\mathrm{acc}\}\)。用 Train token 预测及预测时间权重构建冻结 weighted mid-CDF：

\[
F_j^{\mathrm{mid}}(z)=
\frac{\sum_{m\in\mathrm{Train}}\widehat w_m
[\mathbf1(\widehat z_{mj}<z)+\tfrac12\mathbf1(\widehat z_{mj}=z)]}
{\sum_{m\in\mathrm{Train}}\widehat w_m},
\quad u_{oij}=F_j^{\mathrm{mid}}(\widehat z_{oij}).
\tag{12}
\]

这是相对 Train 分布的位置，不是风险概率。没有 exact tie 的新值使用其左侧累计权重，不做线性插值。

\[
E_{oj}=\sum_i\omega_{oi}u_{oij},\qquad
Q_{oj}=\sum_i\omega_{oi}\mathbf1(u_{oij}>0.90),
\]

\[
C_{oj}=\max_{I\in\mathcal R_{oj}}
\sum_{i\in I}\widehat w_{oi},
\tag{13}
\]

其中 \(\mathcal R_{oj}\) 是完整、顺序有效路线内连续超过 0.90 的 token 段。E 是平均分位暴露，Q 是尾部时间占比，C 是最长连续尾部时间，单位为秒，**不是份额**。完整路线支持检查在聚合前完成，不能跨缺失序列拼接长段。

保留六个冻结 marginal caps：

\[
B^{\mathrm{var}}_{k,j,m}
=Q_{\pi_k}^{\mathrm{higher}}(\{m_{oj}:o\in\mathrm{Train}\}),
\quad m\in\{E,Q,C\},
\quad(\pi_C,\pi_M,\pi_A)=(.75,.90,.975).
\tag{14}
\]

这里每条 Train 路线一个样本，不能把 CDF 的 token 时间加权再次用作路线 quantile 权重。\(\pi_k\) 是逐维 marginal anchor，不是全条件联合通过率目标。

### 6.2 唯一的动态决策入口

\[
\rho^{\mathrm{var}}_{ok}=
\max_{j\in\mathcal J,m\in\{E,Q,C\}}
\frac{m_{oj}}{B^{\mathrm{var}}_{k,j,m}},
\quad
\rho^{\mathrm{overall}}_{ok}=
\max(\rho^{\mathrm{stat}}_{ok},\rho^{\mathrm{var}}_{ok},\rho^{\mathrm{speed}}_{ok}).
\tag{15}
\]

分母沿用冻结有效正 cap，缺失值不能填零。实现研究 sidecar 中为 `rho_variability`；进入 solver 时占用原 `dynamic` family 槽位。原始 `request.rho_dynamic` 仍可保留以便复现旧基线，不能误把它当新策略输入。

原 crawl/stop 六项不再参与动态 max、累计 Gamma 或额外准入条件；可以保留在输出中作历史对照。\(\rho^{\mathrm{overall}}\) 仅为非补偿性概览，本版**不额外要求它小于等于 1**，也不用加权平均把一种超额抵消成另一种不足。

## 7. Stage4：累计暴露预算

对 \(f\in\{\mathrm{stat,var,speed}\}\) 定义非负超额：

\[
e^f_{ok}=[\rho^f_{ok}-1]_+.
\tag{16}
\]

令 \(N_t^-\) 为当前 epoch 前累计 AV 分配订单数，\(B_{f,t}^-\) 为这些订单对应超额之和。它们按已分配订单累计，不按完成订单、路程或 vehicle-hours 计权。

设 \(x_{vo}\in\{0,1\}\)，\(\mathcal A_t^A\) 为 AV 候选 arcs。对启用的预算 \(\Gamma_f\)：

\[
\frac{B_{f,t}^-+\sum_{(v,o)\in\mathcal A_t^A}e^f_{o,k(v)}x_{vo}}
{N_t^-+\sum_{(v,o)\in\mathcal A_t^A}x_{vo}}
\le\Gamma_f.
\tag{17}
\]

若历史与新增分配均为零，均值约定为零。求解器使用等价的线性形式：

\[
\sum_{(v,o)\in\mathcal A_t^A}
(e^f_{o,k(v)}-\Gamma_f)x_{vo}
\le\Gamma_f N_t^- -B_{f,t}^-.
\tag{18}
\]

因此一条路线可以超过 envelope，但仍受车队历史平均预算限制；这不是“每条路线均必须 \(\rho\le1\)”。`Gamma=None` 表示关闭该 family，`Gamma=0` 不是关闭，而是在有效历史下禁止正超额。HV arcs 不计入分子或分母。

**本版保留三类 Gamma 的机制，不把诊断实验中的 Gamma 全关闭提升为最终方法。** Static/speed 沿用对应既定情景；dynamic 改为 retained-six variability excess。既有数值可作为明确标注的研究预算继承，但不宣称它们已针对新六维定义重新校准，也不为匹配通过率重新选数值。具体实验仍须列出所用三项 Gamma，不能把一个 M 场景数值冒充统一 C/M/A 设置。

从 checkpoint 延续时必须用同一批历史 AV 分配、按新六维定义重新计量 \(B_{\mathrm{var},t}^-\)，保持 \(N_t^-\) 不变；不能把旧 12 维 dynamic 总量直接接入。已有共同 M 物理起点的 Gamma-off 诊断可保留原账本，因为它不参与约束，但该账本不能直接拿去启动 Gamma-on 的 C/A 实验。

订单级 \(\beta\) 与累计 \(\Gamma\) 的区别在于：前者限制交通环境外时间份额，后者限制已分配 AV 订单的平均 envelope 超额。二者分母、对象和含义均不同，不相加、不合成为新风险分数。

## 8. Stage4：稀疏滚动分配

### 8.1 候选构造与时间链

每个 epoch 仅处理未过期等待订单与当前可用车辆。在动态搜索半径内，先施加 AV 订单级接受/结构/证据/交通条件，再执行共享 Top-K，随后查询接驾路线、检查到达耐心及 session 完成条件。原设置保留 2 km 起始半径、失败后按配置扩展、Top-K=20；不构造所有订单与所有车辆的稠密矩阵。

乘客 AV 接受变量 \(a_o^{\mathrm{AV}}\in\{0,1\}\) 沿用固定 seed 的外生情景输入，不训练 choice model。AV 的订单侧条件可写作：

\[
g_{ok}^{A}=\mathbf1(h_{ok}=\mathrm{FEASIBLE})
\,\mathbf1(\text{必要证据完整})\,a_o^{\mathrm{AV}}
\,g^{\mathrm{traffic}}_{ok}.
\tag{19}
\]

令 \(\widehat p_{vo,t}^{\mathrm{pickup}}\) 为经既有校准的接驾 ETA。候选还要求：

\[
\widehat p_{vo,t}^{\mathrm{pickup}}\le s_o(t).
\tag{20}
\]

计划到达、上车与再次可用时间明确分开：

\[
\widehat t^{\mathrm{arrive}}_{vo}=t+\widehat p_{vo,t}^{\mathrm{pickup}},\quad
\widehat t^{\mathrm{board}}_{vo}=\widehat t^{\mathrm{arrive}}_{vo}+\delta_o,
\quad
\widehat t^{\mathrm{free}}_{vo}=\widehat t^{\mathrm{board}}_{vo}+\widehat T_o.
\tag{21}
\]

\(\delta_o\) 仅指配置明确启用且未包含在其他时间中的 pickup overhead；canonical 无新增 dwell 时为零。本版不引入新的停留时间。若车辆 availability policy 为 EMPIRICAL_SESSION，还要求 \(\widehat t^{\mathrm{free}}_{vo}\) 不超过 session 终点。约束按 availability policy 而非仅按车辆标签判定，不能把 identity test 变成 availability-policy 改动。

所有式子是决策用预测计划时间，不能用 realized outcome 逆向决定 arc。当前交通 sidecar 绑定精确 ORIGINAL 路线；fallback 不能继承 ORIGINAL 的交通暴露，本版不开发新的 rerouting 或 fallback。

### 8.2 约束与词典序目标

构造稀疏图 \(G_t=(\mathcal V_t,\mathcal O_t,\mathcal A_t)\)，仅对已有 arcs 建变量：

\[
\sum_{o:(v,o)\in\mathcal A_t}x_{vo}\le1\quad\forall v,
\qquad
\sum_{v:(v,o)\in\mathcal A_t}x_{vo}\le1\quad\forall o.
\tag{22}
\]

再加入式（18）。令 critical 指示 \(c_o(t)=\mathbf1(0<s_o(t)\le30)\)，carry-over 指示为 \(h_o(t)\)。沿用精确逐级求解：

\[
\operatorname{lexmax}\left(
\sum c_o(t)x_{vo},\quad
\sum x_{vo},\quad
\sum h_o(t)x_{vo},\quad
-\sum\widehat p_{vo,t}^{\mathrm{pickup}}x_{vo}
\right).
\tag{23}
\]

注意 critical 优先于总数，与现有 solver 一致，不擅自写成总服务最大化优先。每一级固定前级最优值，不使用大权重拼合。

若既有情景启用 operating-cost 层，先取得 ETA 最优值 \(P_t^*\)，再在保持前三项最优且 \(\sum\widehat p^{\mathrm{pickup}}x\le(1+\epsilon)P_t^*+\varepsilon_{\mathrm{num}}\) 下最小化：

\[
\sum_{(v,o)\in\mathcal A_t}
\alpha_{\mathrm{type}(v)}
(\widehat p_{vo,t}^{\mathrm{pickup}}+\delta_o+\widehat T_o)x_{vo}.
\tag{24}
\]

\(\alpha_H=1\)，\(\alpha_A\) 为既定相对运营成本系数。\(\epsilon>0\) 时是容许 ETA 小幅退让的末级成本选择，不能称为严格 ETA 最优下的纯 tie-break。没有启用成本层的情景不自动增加它。

## 9. 容量机制与结论口径

\[
C_t^{\mathrm{eff}}=\nu(G_t)
\tag{25}
\]

称为 **instantaneous effective matching capacity（瞬时有效匹配能力）**，即当前候选图的 maximum matching cardinality，不是日容量或单位时间服务率。

启用历史 Gamma 后，应另外区分预算可行的最大匹配数：

\[
C_t^{\Gamma}=\max_x\sum x_{vo}
\quad\text{s.t. 式（18）、（22）及二元约束}.
\tag{26}
\]

对于可行状态，\(C_t^{\Gamma}\le\nu(G_t)\)。实际式（23）的优先级选择不应无条件等同于式（26）；全天服务 \(Y_{1:T}\) 又受车辆后续位置、忙闲状态、patience、request timing 和历史预算影响。AV 子图 \(\nu(G_t^A)\) 是失去可替代性的机制证据，不是 mixed-fleet 全天服务的可加分解。

若在固定订单、车辆与未经重新压缩的候选集合上仅删去不兼容边，则最大匹配数不增加；但本文的共享 Top-K 会因 AV 条件变化而重新补入 HV 候选，加上后续车辆轨迹演化，不能由删边单调性直接推出动态总服务必然下降。

名义供给变量继续写为：

\[
q_A=H^A/H^{\mathrm{base}},
\tag{27}
\]

称为 baseline-normalized AV active-hour level/ratio，不默认为 \(H^A/(H^A+H^H)\)。不将预测的决策相关性升级为“已证明决策优越性”。

## 10. 现有证据支持什么、不支持什么

| 方法主张 | 现有依据 | 可写结论与边界 |
|---|---|---|
| 新交通规则与旧 crawl/stop 来自同源预测，应避免直接叠加 | 状态构造、旧 EQC 定义、已完成 F/V/VT 对照 | 采用替换而非额外叠加；不声称所有维度独立 |
| C 对相对减速更敏感 | 共同物理起点 OFF/ON 两窗口及 MIXED 归因 | C 新规则确实改变已有窗口的分配，不证明这种偏好是真实车型能力 |
| M 影响较温和、A 已知类别无新增限制 | 同一组既有短窗口 | 仅限已检查情景；A outside=0 是定义 |
| Stage2 仍有多目标决策用途 | 式（2）、（8）、（12）、（20）、（24）对应源代码 | 可以说 prediction is decision-relevant；没有新增优越性证据 |
| 统一 Gamma-on 策略优于冻结主线 | 尚无完整公平比较支持 | 不作此主张；旧 41 场景及 Gamma-off 诊断不能顶替 |

已有共同状态诊断的作用是说明新规则在哪里生效，不是要求再扩展一轮大审计。本文不新增阈值搜索、不重新选 profile、不新增实验清单；先确认方法，再决定是否需要一个与最终主张对应的最小验证。

## 11. 可直接进入论文的方法概述（英文草稿）

**Opening / scope.** We study how heterogeneous vehicle availability is converted into instantaneous effective matching capacity and realized service. The model separates structural eligibility from continuous operational exposure and passenger acceptance. The AV profiles represent nested research scenarios rather than certified vehicle capabilities.

**Prediction / representation.** A frozen route-conditioned model provides decision-time predictions of traversal duration, pace, crawl and stop shares, and speed and acceleration variability. Predicted traversal durations weight route-level exposure. A Train-only low-pace reference distinguishes relative slowdown from a high predicted low-speed share. Their joint classification yields non-congested, congested, severely congested, and two mixed-evidence traffic states, with estimability and reference support reported separately.

**Policy / non-duplication.** Profile-dependent traffic compatibility limits estimated outside-environment exposure to five percent of each route's total predicted duration. Unknown exposure remains in the denominator and is reported rather than newly vetoed. This traffic rule replaces the crawl- and stop-based E/Q/C decision channels. Static intersection descriptors, the frozen speed-domain envelope, and the six E/Q/C descriptors of speed and acceleration variability are retained. Thus, traffic-state interpretation does not discard the multi-target predictions or impose a second constraint on the same crawl/stop evidence.

**Assignment / interpretation.** Sparse rolling assignment combines vehicle availability, pickup patience, passenger acceptance, structural eligibility, traffic compatibility, and cumulative AV exposure budgets. The solver follows the existing lexicographic priorities and, where enabled, a final operating-cost objective subject to a pickup-time allowance. Maximum matching cardinality characterizes instantaneous graph capacity; it is distinguished from budget-constrained assignment and full-day realized service. The traffic assumptions and exposure budgets define the study scenarios, not safety guarantees.

## 附录 A：实现映射与迁移边界

本文为统一的科学表达，不更名或覆盖旧文件。以下路径相对 repository 根目录：

| 数学模块 | 现有来源 |
|---|---|
| 静态 D、mid-CDF、E/Q/C、caps | `stage3/odd_tod/capability_envelope.py` |
| 静态/速度利用率与结构性语义 | `stage3/odd_tod/operational_suitability.py` |
| Train pace 参考、初始规则 | `stage3/scripts/traffic_state_batch1.py`、`stage3/config/traffic_state_batch1.json` |
| 预测状态、路线时间聚合 | `stage3/scripts/traffic_state_batch2.py` |
| 支持度分轴、两类 MIXED | `stage3/scripts/traffic_state_support.py` |
| profile 环境外集合、5%门控、retained-six exposure | `stage4/dispatch/traffic_research_policy.py` |
| 累计账本、正部超额 | `stage4/dispatch/exposure.py` |
| availability/patience/成本与候选执行顺序 | `stage4/dispatch/rolling_or_control.py` |
| 稀疏约束与逐级求解 | `stage4/dispatch/solver.py` |

默认 `FROZEN` 继续复现旧方法。`TRAFFIC_REPLACEMENT_V1` 是本文组合的研究策略入口；`VARIABILITY_ONLY_V1` 是去除交通门控的诊断，不成为第二条论文主线。底层 frozen dynamic profile 仍保存 12 维以保持复现；新策略仅选取 CV/acceleration 六项，不修改旧 caps 或 CDF。

Stage3→4 解释接口应同时携带精确路线 identity、hard_state、三类连续利用率（dynamic 实际采用 retained-six）、traffic outside/unknown/low-support shares 及解释代码。它们不合成一个“AV safe / unsafe”标签。当前 ORIGINAL sidecar 的范围不能泛化到任意候选路线或接驾路径；交通暴露目前针对订单服务路线，而非完整车辆任务的接驾加载客全过程。

历史证据入口：[共同状态报告](common_state_report.md)、[MIXED 归因](mixed_attribution_report.md)、[F/V/VT 报告](mechanism_report.md)、[当前实现状态](status.md)。

## 附录 B：方法写作自检（不是新实验 Gate）

- **贡献**：围绕兼容性、瞬时匹配能力和服务转化组织；不因增加交通解释而宣称新的 AV 安全模型或预测优越性。
- **清晰性**：区分 \(\beta\) 的路线时间份额、\(\Gamma\) 的订单平均超额、\(\nu(G)\) 的图匹配数和全天服务；区分静态 M 与 profile M。
- **实证强度**：现有短窗口支持规则有作用的位置，不足以宣称统一策略全天优于旧方法；不补造结果。
- **评价完整性**：旧基线、F/V/VT 和共同状态诊断各自保持原来的适用范围，不将 Gamma-off 结果冒充最终 Gamma-on 结果。
- **设计合理性**：排除 crawl/stop 明确重复门控，保留其他互补维度；承认相关性、UNKNOWN、低支持、route proxy 与 timing 假设。最终 Gamma 数值需随所选情景明确列示，而不是从本文公式自动得出。

**交付边界：方法与公式已整理；本轮没有运行新实验。**
