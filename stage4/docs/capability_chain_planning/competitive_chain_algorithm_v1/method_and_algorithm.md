# 竞争感知的能力服务链分解：统一模型与算法设计 v1

日期：2026-10-14，Asia/Singapore。代码对照基线：60a4f459ac956b86646e115c7f6cac591cfeec8a。

状态：METHOD_DESIGN_COMPLETE；本文不表示新增算法已经实现或运行。学术研究技能用于问题、方法和证据边界整理；推导与检查在同一上下文完成，不称独立同行评审。

2026-10-14执行更新：独立有限原型与固定12例已完成，见[实际结果报告](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/competitive_chain_algorithm_v1/results/report.md)。以下保留88d146c中的执行前设计，文中“拟补强／未运行”描述其原始阶段；新增实现没有接入城市控制，12例没有出现额外恢复收益。

[有限计算验证计划](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/competitive_chain_algorithm_v1/bounded_validation_plan.md)

## 1. 主问题不退成上线位置案例

研究对象仍是能力受限AV与既有HV共享需求时的部署、滚动分配和空闲移动。C是当前主要运营情景；数学记号允许其他能力类型，但不因此新增M/A实验。

核心问题：在共同供给、信息和执行条件下，怎样利用任务资格、完成后位置与就绪时间，以及不同车辆争用同一订单的关系，提高资源转换，并解释服务与空驶之间的取舍？

本次选定一条方法主线：

1. 以有限服务链表达任务资格与时空接续；
2. 用共享订单容量产生竞争价格，协调两类资源；
3. 用稀疏主问题与动态规划定价生成有用服务链；
4. 为声明的有限模型建立有效界，恢复并保留可行整数方案；
5. 用同一状态价值接口连接部署和快速滚动动作选择。

不是把列生成、动态规划、网络流分别列成三个“新算法”。候选贡献是它们针对本问题的分解、质量控制与执行连接方式；新颖性仍需与最接近的方法逐项比较。

本文不改冻结M3、能力定义、300/600等待配置、供给、种子、旧结果或论文主稿，不执行训练、路由、推理、仿真或新计算比较。提出的新增实现必须放在独立入口，不覆盖现有控制器。

## 2. 已有证据与已有理论直接复用

| 已有材料 | 已支持的结论 | 不能据此宣称 |
|---|---|---|
| [六组部署报告](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/full_report.md) | 四个预定对照点的C利用率增量为0.390–1.107个百分点；全队服务和空驶不一致占优 | 新算法的收敛、近似比、预测模块独立贡献或全天最优 |
| [调位组件对照](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/operations_v5_ablation/report.md) | 通用调位下改变当轮空驶处理，总空驶下降16.37%、全队少8单；类型模块另有取舍 | 经典最优面原理是本研究首创，或当轮保护保证全天服务不降 |
| [300/600布局准备](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_300_600_v1/preparation_report.md) | 该次45个有效主LP批次均未定价闭合；整数恢复后未评估服务分数 | 这些受限LP分数是全域上界，或恢复的部署继承了LP服务值 |
| [已有结构辨识](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/model_and_structure_v1.md) | 已有整数松弛差、非次模反例、严格网络流特殊情形与对偶增量界 | 一般部署有收益递减、贪心有固定近似比、影子价是真实整数增量 |

旧结构文档保留其当时的运行契约；本文使用当前300秒内接单、独立600秒接驾上限，不复制旧的“300秒内到达”公式。

本轮真正要补的是有效求解界、考虑服务目标的整数恢复和统一执行接口，不重新开展旧反例审计。

## 3. 状态、信息与两套空间表示

真实资源单元是带接单窗口的服务班次，不是采购车辆。物理状态写作

\[
\sigma_v(t)=(k_v,z_v(t),\tau_v(t),e_v,\ell_v(t),c_v(t)).
\]

其中k为能力类型，z为实际位置，tau为完成既有承诺后可用时刻，e为停止接新单时刻，ell为有向道路/动作上下文，c为承诺与进度。

粗层状态写作

\[
i=(k,z,\tau,e),
\]

其中z是既有候选节点投影，tau和e采用现有五分钟格。它用于未来价值估计，不替换真实物理位置和道路上下文。

必须区分：

- 当前执行域：实际道路、方向、动作、接受、窗口和时间链通过现有研究检查的动作；
- 未来规划域：历史任务、冻结M3、空间连接时间代理和稀疏后继构成的有限模型；
- 真实运营结果：执行后观察到的完成、空驶、载客小时与再次可用状态。

粗层连接不是未来有向道路可达证书，其质量界不能当作真实全城容量界。

### 3.1 当前时间契约

对订单o，释放时刻仍采用上车/轨迹出发近似，记为t_rel。合法接单与时间递推为

\[
t_o^{rel}\le t_o^a<t_o^{rel}+300,\qquad
t_o^a<e_v,\qquad
0\le\widehat P_{vo}(t_o^a)\le600,
\]

\[
t_o^p=t_o^a+\widehat P_{vo}(t_o^a),\qquad
\widehat t_o^d=t_o^p+\widehat T_o^{M3}.
\]

未接单在300秒期限过后退出；已承诺工作继续完成，不因接单期限经过而取消。完成可超过e_v，但不能据此在e_v后接新单。已有物理活动的独立停留或移动字段按原执行规则处理，不凭本文新增时长。

粗层的固定时刻是接单格，不是上车格。接驾时间依赖前驱位置；接续就绪时刻包含接驾与M3服务时间，并向上对齐粗格。末次已接服务可完成在规划/接单窗口之后，之后不得继续接单。

### 3.2 信息与场景

现有三个历史情景来自20161010、20161017、20161024，使用已有冻结M3预测，权重为1/3。它们是历史参考样本，不声称已识别真实需求概率分布。

在线只读取已释放请求、当前车辆状态、既有历史参考和决策时预测，不读取Test31尚未发生的真实请求或完成后标签。部署在情景间共享，不能提前知道哪一个历史情景会“实现”后才选择地点。

## 4. 有限服务链与共享需求主问题

情景s中的一次性历史任务集合为O_s。对状态i，P_si表示声明域内的全部合法有限服务链，含空链；每条链最多3项服务。同一订单在不同时间状态、不同车辆或车型下，仍只有一个容量行。

当前声明域采用30分钟范围、300秒粗格、七个就绪格（末格为终止）、2km初始邻域、C最多3个后继、HV保留自身3个与C后继的并集（最多6个）。这些是现有规划限制，不是完整路网的无损压缩。

服务收益保持现有数量目标：

\[
R_p=|O(p)|.
\]

不新增AV优先奖励、加权安全评分或按结果选择收益系数。

### 4.1 固定状态供给的服务链模型

给定状态容量b_si，线性放松为

\[
\begin{aligned}
F_s^{LP}(b)=\max_x\;&\sum_i\sum_{p\in P_{si}}R_px_{sip}\\
\mathrm{s.t.}\;&\sum_p x_{sip}\le b_{si} &&\forall i,\\
&\sum_i\sum_{p:o\in O(p)}x_{sip}\le1 &&\forall o\in O_s,\\
&x_{sip}\ge0.
\end{aligned}\tag{1}
\]

整数版本要求x为非负整数。状态容量可以大于1，表示同一粗状态下的多个班次；整数链可分配给该状态下的资源。分数链不能作为物理服务直接执行。

不能先独立优化C、HV后相加，因为式(1)的订单容量是共同的。

### 4.2 共同部署、情景服务可不同

新上线C资源按共同就绪/接单窗口分组g，数量为n_g；Z_gz是放到候选点z的数量。背景资源供给b_si固定。模型为

\[
\max_{Z,x}\sum_s w_s\sum_{i,p}R_px_{sip},
\]

\[
\sum_z Z_{gz}=n_g,\quad Z_{gz}\ge0,
\quad \sum_p x_{s,g,z,p}\le Z_{gz}\quad\forall s,g,z,
\tag{2}
\]

并保留每情景背景资源容量与订单容量约束。整数版本要求Z和x均为整数。

Z在所有情景中相同，x可随情景变化。这是两阶段有限历史规划模型，不是已经求解的全天随机最优政策。准备地点改变不等于运营中瞬移，也不增加总车时。

### 4.3 与滚动决策的统一：接口统一，不伪称一个全天大MIP

有限联合参考模型用y_va选择共同当前动作，并用各情景的后继链x_vsap连接它：

\[
\sum_{a\in A_v}y_{va}=1,\qquad
\sum_{p\in P_{sva}}x_{vsap}\le y_{va}\quad\forall v,s,a.
\tag{3}
\]

当前客户容量跨车辆共享，未来客户容量在每情景跨车型共享；不重复计算当前已选择服务。空后继隐含允许。忙车只保持既有承诺，其预测完成状态可进入未来供给。

式(3)用于小实例参考和解释。城市实现通过粗层竞争价值、细层当前动作流分解执行，不在每轮显式求完整未来链整数模型，也不声称它与该完整模型等价。

## 5. 竞争价格与可检查的质量界

### 5.1 状态价值

订单容量的非负对偶价格为pi_so。给定价格，状态定价为

\[
\Phi_{\pi_s}(i)=
\max_{p\in P_{si}}\left(R_p-\sum_{o\in O(p)}\pi_{so}\right).
\tag{4}
\]

空链保证Phi不小于0；收益为数量、链长最多3、pi非负时，Phi不超过3。价格不是票价或额外运营费用，是避免共享机会被重复估值的协调量。

现有动态规划在七个就绪格、服务深度和稀疏后继上定价；支持等待与终止。只在声明的图与链长内精确，不对未保留道路/后继作保证。

特别注意：Phi不是F(b+一辆车)-F(b)的真实边际值。一个订单、一个已有资源时，最优对偶可取订单价0、资源价1；Phi=1，但再增加同类资源也只能服务该一单，新增服务为0。对偶退化还可能给出另一组同样最优的价格。因此它是固定价格下的竞争调整价值，不是因果贡献或整数新增收益。

### 5.2 性质P1：固定供给的有效上界

对任何非负订单价格，只要式(4)完整覆盖声明域，就有

\[
U_s(\pi)=\sum_o\pi_{so}+\sum_i b_{si}\Phi_{\pi_s}(i)
\ge F_s^{LP}(b)\ge F_s^{IP}(b).
\tag{5}
\]

证明：每条链均满足R_p<=sum(pi_o)+Phi(i)。乘以任意可行x再求和，订单项由容量1约束控制，状态项由b约束控制，即得上界。这是弱对偶的特定构造，不称新对偶定理。

列生成未闭合时，已求受限主LP值L_R是完整LP的下侧近似，不是完整整数问题上界，也不一定是整数最优值的下界。利用式(4)重新定价全部根状态，可给出式(5)的完整声明域上界，而不要求重新枚举全部路径。

如使用已有资源乘子mu，可将其修复为max(mu,Phi)，也可直接取Phi；只有所有根状态都已覆盖，才具备该证书意义。

### 5.3 性质P2：共同部署的非预知上界

令I_s^fixed为背景状态，Z_set为式(2)的共同部署可行集合。对非负价格，

\[
\begin{aligned}
U^{off}(\pi)=&
\sum_s w_s\left(\sum_o\pi_{so}
 \sum_{i\in I_s^{fixed}}b_{si}\Phi_{\pi_s}(i)\right)\\
&+\max_{Z\in Z_{set}}\sum_{g,z}Z_{gz}
          \sum_s w_s\Phi_{\pi_s}(C,z,g)
\end{aligned}\tag{6}
\]

是共同部署模型的LP/IP上界。证明与P1相同，再对同一个Z取最大值。

现有部署只有分组数量约束、没有驻点开通/容量预算时，最后一项简化为

\[
\sum_g n_g\max_z\sum_s w_s\Phi_{\pi_s}(C,z,g).
\tag{7}
\]

max在情景平均之外：不能用每情景各选一个最优地点的值冒充共同部署最优值。后者可以作为更松的乐观上界，但不是可执行的一阶段方案。

### 5.4 性质P3：整数方案、LP值和证书不能混用

若恢复的共同整数部署Z_int与每情景整数服务链x_int都可行，其评价为Q_int，则

\[
Q_{int}\le F^{IP}_{off}\le F^{LP}_{off}\le U^{off}.
\tag{8}
\]

Q_int必须由真实选定链、共同订单容量与资源容量重建，不能用恢复前LP分数代替。恢复地点后再求的分数服务LP，也不是整数可行成绩。

所有界只对相同场景、网格、候选、链长、收益与连接域有效。它们不是完整路网、未来真实需求、非预知在线最优值或全天服务的界。

数值实现须采用上向舍入/区间上界或可核对有理数计算，并覆盖遗漏路径。普通求解器返回成功、显示gap为0或任意固定容差，不自动成为式(5)–(8)的数值证书。

## 6. 主算法设计：受限定价、界与服务感知整数恢复

以下是拟补强的算法规格，不表示代码已有全部步骤。

    Algorithm 1: Shared-demand competitive chain decomposition
    Input: declared history DAGs, profile masks, fixed supply,
           common fresh groups/sites, existing time/memory budget
    1. Construct nested sparse domains; include idle/empty paths.
    2. Seed the restricted master with feasible path columns.
    3. Solve the sparse restricted LP; extract nonnegative order prices.
    4. Price every fixed and fresh root through the same DP oracle.
    5. Compute a conservative complete-domain upper bound.
       Save restricted-LP value separately from integer feasible value.
    6. Add improving paths while the existing round/time budget permits.
       Stop on certified closure or the original budget, not on hoped-for gain.
    7. Recover one common integer placement; evaluate feasible integer
       scenario chains under shared resource/customer capacities.
    8. Try a bounded set of one-unit placement exchanges, reusing columns.
       Accept only a strictly better evaluated feasible solution.
    9. Return placement, integer chain witness, bound/gap and termination reason.

第7步初始部署复用现有最大余数规则，作为可行种子，而不是最优恢复。第8步只在同组候选点间交换一单位，车辆数量、窗口和候选池不变；固定稳定遍历顺序，不依赖运营结果挑点。当前主目标仍为期望计划服务数，现有兼容热点L1平局规则只在主目标相同时适用。

可行链评价器应保留并返回链见证。候选评价超时或没有更好的可行见证时保留种子，不用分数LP改善替代整数改善。预算包含恢复和评价，不额外延长30秒离线单批上限。

### 6.1 性质P4：恢复过程的有限模型不劣化

保留初始可行对(Z0,x0)，只接受主评价严格提高的可行整数对，并始终保存最好见证，则返回的Q>=Q0。候选整数部署有限，严格改进不会循环。

这只保证实际评价到的可行见证不变差，不保证找到了给定部署的最佳服务链、全局整数部署最优或真实回放收益。若用有限启发式评价，不可把“未找到改善”解释成局部最优。

## 7. 在线执行：当轮服务保护与后继价值

现有物理动作包括SERVE、WAIT、RELOCATE；忙车/未上线资源不进入可自由派单的当前流。每个资源一条WAIT，当前订单容量1，可选移动共用原50辆上限。

令bar_Phi(i)=sum_s(w_s Phi_s(i))。现有接口先量化各状态，而不是先量化差值：

\[
q(\phi)=\lfloor3\phi+1/2\rfloor,\qquad
u_{va}=q(\bar\Phi(i_{va}^{after}))-q(\bar\Phi(i_v^{wait})).
\tag{9}
\]

u在[-9,9]，WAIT的u为0。当前临界、服务、carry-over数量分别记C_t、N_t、G_t，近似后继分数为

\[
A_t=3N_t+\sum_{v,a}u_{va}y_{va}.
\]

现有服务保留顺序为lexmax(C_t,N_t,G_t,A_t)，之后最小化当前空驶。允许暂缓的已有对照顺序为lexmax(C_t,A_t,N_t,G_t)，不是将新权重混入原控制。

部署、派单、调位共享状态价值定义，但不声称独立求和的后继价值再次严格满足所有未来客户容量。共同容量用于生成价格，在线值是分解近似；每300秒刷新不能消除所有当轮组合变化。

### 7.1 性质P5：主最优面内的空驶保证

对已准入、固定的当前图，主目标已认证最优时，令F_star为达到同一四项词典序主目标的动作集合。在该面求最小空驶，可以保持四项主目标不变。

现有实现用互补松弛：非零约化费用弧固定在相应边界，只让零约化费用弧变化；整数容量与节点供给保留网络流整数解。不能把完整未来链模型也称为单商品流。

距离以毫米整数费用求解，物理距离不变。若完整次级求解已证明最优，N个选定动作的真实米数相对该主最优面内最佳真实米数的损失至多N/1000米：每项舍入误差至多0.0005米，比较两个方案即得此界。

若主求解未证明最优、次级超时或原生调用晚返回，分别记录；不能默认有上述保证。当前原生求解器没有可中断API，时间限制不是无条件按时返回的证明。

此性质不保证不同配对后的全天服务相同，也不保证当前主最优面等于长期最优动作集。已有全队失单仍如实保留。

### 7.2 性质P6：量化的一个有限界

在相同主服务面、相同冻结价格、相同准入动作域中，比较连续bar_Phi与式(9)的量化目标。量化步长h=1/3，每个后继状态误差至多h/2。

每个资源恰选一个动作，所有WAIT基准在不同完整选择之间抵消。因此，完成精确量化目标求解的方案，在这个连续状态价值目标上的差距至多N*h=N/3。

证明：每个完整选择的后继状态和有N项，误差至多N*h/2；比较最优连续选择与量化选择，误差最多两倍。该界通常较松，只量化了本地固定价格目标，不是M3预测误差、真实未来服务误差或全天损失界。

现有量化保持不变，不能为了得到更小理论界自动改分辨率或整数范围。

## 8. 能力与就绪单调性：用于稀疏化，不替代算法贡献

### 性质P7

若两个状态的位置、就绪、窗口、收益、价格和连接时间相同，并且P_weak是P_strong的子集，则Phi_weak<=Phi_strong。若允许等待且其他条件固定，较早就绪和较晚停止接单同样不会降低Phi。

证明是可行链集合包含。单任务资格嵌套并不充分：连接、窗口、时间代理、历史姿态或候选截断不同都可能破坏链包含。

因此，较强类型保留较弱类型的稀疏后继，可防止独立Top-K人为反转能力关系。当前C Top3与HV Top3并集合成HV Top6是此性质的一种实现。它只保留声明图中的嵌套，不恢复被双方共同删掉的其他链。

当前MOVE的零时间到达估值是就绪单调性上界：如果它也不优于WAIT，实际耗时到达不会改善该固定价格价值。它是筛选依据，不是实际移动或道路证书。

这些性质可支持方法解释与正确实现，但本身是经典包含关系，不单独宣称原创单调性定理。

## 9. 可选增量：延迟路线验证

此项未实现，本轮不列为主算法第一批必做。它针对路由/资格判断成本，而不是再次优化已很快的流求解器。

    Algorithm 2: Optimistic current graph with lazy validation
    1. Use the same declared Top-K universe and known common exclusions.
    2. Form an optimistic graph that contains every truly admissible arc
       in that universe; compute a current-cardinality upper bound.
    3. Keep an independently checked feasible incumbent, including WAIT.
    4. Validate arcs capable of changing the selected solution.
    5. Remove only proven inadmissible arcs; update validated times/state.
    6. Re-solve within the unchanged decision budget.
    7. Execute only an entirely validated incumbent; return gap/status.

### 性质P8：仅限当前数量的上下界

若已验证图G_valid包含于真实准入图G_true，而G_true包含于乐观图G_plus，则

\[
|M_{valid}|\le \nu(G_{true})\le \nu(G_{plus}).
\tag{10}
\]

左侧是任何已验证可行匹配；右侧是乐观图最大匹配数。两者相等时认证声明候选域的当前最大数量；若不等，记录差距，不宣称最优。

前提非常重要：不能用没有保守性的ETA排除候选；未知不能当作已经证明不可行；更新实际ETA后必须重新检查600秒、承诺时间与后继状态。未验证边不得执行。

式(10)不自动给未来价值、空驶或完整词典序目标的上界。若候选基于乐观ETA产生了更高后继分数，需要另证其上界；没有该证明时，只报告当前数量界。维持可行备选也不自动证明不可中断的原生调用能按时返回。

延迟约束/验证是已有一般思想；本问题中的分离方式、有效界和查询节省仍待实现与最近方法比较。

## 10. Stage1–3如何实质参与

| 来源 | 进入决策的位置 | 不作的延伸 |
|---|---|---|
| Stage1方向路线、traversal和movement | 当前准入检查、历史任务与接口身份、客户路线固定 | 不重做地图匹配，不把规划链当观测轨迹 |
| Stage2冻结M3 P50/pace | 完成后就绪时刻、历史接驾时间代理、后继链可行性和价值 | 不读取真实完成标签，不宣称预测独立运营增益 |
| Stage3道路/路口动作及交通研究适配 | 任务能力掩码和当前连接检查，决定不同类型的链域 | 不重复加同源风险指数，不称AV安全认证 |
| Stage4当前状态与供给 | 共同订单竞争、上线位置、承诺保护、当前动作与空闲移动 | 不把班次窗口数量当采购车辆数 |

已完成六组只改变初始C布局；共同在线控制相同。因此其结果支持部署效果，不单独辨识本方法全部在线组件。

## 11. 代码对照与新增范围

| 组件 | 基线实现 | 本次规格状态 |
|---|---|---|
| 当前300/600规划时间链、嵌套后继、七格DP | [scheme_a_match_deadline_value.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/scheme_a_match_deadline_value.py) | 已有；只对照，不改 |
| 共享订单LP、最多六轮定价、零供给状态同源定价 | [scheme_a_two_scale_value.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/scheme_a_two_scale_value.py) | 已有；完整域数值上界待补 |
| 跨情景共同部署、受限主LP、最大余数恢复 | [coarse_layout.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/coarse_layout.py) | 已有；整数服务见证及交换评价待补 |
| 量化后继差值与真实动作接口 | [scheme_a_two_scale_native.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/scheme_a_two_scale_native.py) | 已有；不改策略顺序 |
| 当前稀疏整数流与主最优面空驶 | [scheme_a_current_flow.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/scheme_a_current_flow.py)、[current_flow_secondary.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/current_flow_secondary.py) | 已有；经典执行支撑 |
| 小实例共同动作与链联合参考 | [scheme_a_city_master.py](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/dispatch/scheme_a_city_master.py) | 可复用建模/解质量思想；旧数据时钟需区别 |
| 延迟路线验证 | 尚无相应正式入口 | 可选提案，未实现、未运行 |

## 12. 计算与内存

粗图保存任务、稀疏后继、候选点邻域与实际访问状态缓存。七格、最多三单DP的工作量受任务数、稀疏弧数及类型/结束窗口缓存数共同控制，不构建订单×车辆×时间稠密矩阵。

根邻域不能一边要求包含全部2km内可行首项，一边无条件声称线性内存。应统计sum_z(|heads_z|)、DP数组、列数和非零元；极密邻域可能很大，达到既有上限时报告停止，不偷偷剪掉首项后仍引用单调性证书。

当前动作层仍是整数稀疏流，不包含未来链变量。第一批算法补强优先复用图、列和DP缓存，不重新请求道路、不加载PyTorch/GPU、不重新训练。保留现有5秒粗价值、30秒离线单批、10秒当前决策与稀疏规模限制。

## 13. 文献边界与贡献候选

[Al-Kanj、Nascimento与Powell（EJOR 2020）](https://doi.org/10.1016/j.ejor.2020.01.033)已有价值函数辅助派单、调位、充电与时间/电量单调性。不能把“使用未来价值”或“部署与运营联动”作为排他性新意。

[Zhang等（TRE 105293）](https://doi.org/10.1016/j.tre.2026.105293)联合公司车辆配置、众包司机对分区加价的响应与需求分布鲁棒处理；正文主模型为单期区域供需，具体可解性命题证明在另附E-Companion，本次未独立检查证明。我们不复制其定价/司机行为，也不能仅因车型标签不同就排除其最近邻地位。

[Liao等（TRE 105284）](https://doi.org/10.1016/j.tre.2026.105284)提供规模、补贴与利用问题背景，采用宏观容量表达，不替代任务级接续。我们没有其需求选择、利润和就业识别，不把载客转换当利润。

[Yan等（TRE 105309）](https://doi.org/10.1016/j.tre.2026.105309)强调架构分工、执行截止和公平比较；这些是评价依据，不是本研究的新架构定理。

本轮贡献候选是“能力/时态链域 + 共享机会价格 + 有界稀疏求解 + 整数恢复 + 服务保护执行”的问题特定算法。P1–P8均是有条件推导，其中多项属于经典原理；算法组合的新颖性与效益仍待计算比较和最近邻方法全文对照。没有声称新复杂度定理、固定近似比、在线后悔界或投稿保证。

## 14. 下一步与停止点

接下来只实现Algorithm 1的有效界和整数服务评价/恢复，按[固定有限计划](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/competitive_chain_algorithm_v1/bounded_validation_plan.md)做一次小规模算法比较。延迟验证只保留规格，先不与主补强一起扩展。

本次停止在方法与计算计划写好；未启动任何比较、全天或新的优化运行。旧理论、实例、运行配置与结果全部保留。本文由AI辅助组织与推导，最终算法取舍与科研表述归作者。

## 15. 本轮只做的文档与构造数学校对

新文件的14个本地链接已逐项确认目标存在；数学块开闭与改动检查在提交前核对。没有运行求解器、算法基准、路由、推理或FleetPy。

同一上下文进行了三项小型构造计算，作为推导的 sanity check，不替代证明或独立验证：

- 复用旧两资源四任务反例，穷举整数值为3、分数见证为4；检查625组非负价格，式(5)的上界均覆盖该分数见证。
- 两情景两地点的构造中，共同地点的计划值/上界为1，逐情景先知选点为2，核对式(7)中max与情景平均不能互换。
- 两资源、状态值0到3的四分之一网格上，核对15,272对量化排序；最大连续损失0.5，小于P6的2/3界。

这些是固定常数与有限路径的直接算术，不是新算法运行成绩；12例有界计算计划仍未执行。
