# Agent 任务书：Dynamic Compatibility-Aware Control — Model Feasibility Study

版本：1.0；日期：2026-09-14。

## 0. 本轮目标与授权

本轮只回答：

> 在相同总需求、OD 与时间预测下，需求的兼容性构成何时改变跨期车辆价值和当前最优分配？这种改变能否被简单保留规则解释？

完成一份约 6–10 页等量内容的模型可行性研究，以 Markdown 为主，不要求为页数制作 PDF。论证完整优先于页数。成果用于用户审查，不是正式 Stage5，也不是新方法已成立的宣告。

授权范围：

| 工作 | 本轮状态 |
| --- | --- |
| 读取现有代码、契约和报告 | YES，只读 |
| 针对模型争议精读已有 closest works | YES，不做泛化文献扩张 |
| 两期模型、证明、反例、纸面精确对照 | YES |
| 独立微型枚举核验器 | OPTIONAL，严格按第 8 节上限 |
| 生产代码或现有配置修改 | NO |
| Stage0/1 生产、Stage2 训练/M3 重推理 | NO |
| Stage3 路线、阈值、CDF、fallback 修改 | NO |
| Stage4 solver 接入、真实订单试跑、全天实验 | NO |
| 新建需求预测器、fluid LP/MPC 控制器 | NO |
| 主动重定位、admission、乘客模型、司机均衡 | NO |
| 修改 V3 正文、宣称新理论/性能优势 | NO |

必须完成本任务书后停止，交给用户审核。任何 GO 只表示值得进入下一轮设计，不构成下一轮执行授权。

## 1. 工作位置、基线和保护范围

主工作区：`D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla`。

分支：`codex/stage2-v5-micro-transfer`。任务书编写前参考 HEAD：`f1d641f66dd59a20e5451c5cd53f308089067271`。该 SHA 是来源记录，不要求回退；执行时记录实际 HEAD。

不要在旧根工作区 `D:/pycodes/didi_xian_raw` 暂存其历史删除，不要 reset/checkout 用户改动。开始先检查 Git 状态和适用 AGENTS.md；若有并行改动，只处理本轮交付路径。

保护：41 个正式场景、全部冻结上游产品、M3、配置、既有 manuscript_v3。只允许新增第 9 节目录中的文件，以及必要的文档导航链接。不得复制大型输入到新目录。

## 2. 必须先读的本地材料

以当前工作区为根：

1. `docs/repository_guide_zh.md`。
2. `stage4/docs/manuscript_v3/stage4_manuscript_v3.md`，重点第 2、3、5、6 节。
3. `stage4/dispatch/solver.py`、`stage4/dispatch/exposure.py`，核对词典序与 Gamma。
4. `stage3/docs/odd_tod/final/stage3_to_stage4_contract.md`。
5. `stage4/docs/paper_redesign/request_time_patience_sensitivity_report.md`。
6. `stage4/docs/paper_redesign/efficient_repositioning_report.md`，确认未完成实验与运行成本边界。

用户补充材料（外部只读）：

- `C:/Users/HP/Downloads/trage-or-methods-closest-workt`
- `C:/Users/HP/Downloads/trage-research-positioning.md`
- `C:/Users/HP/.codex/attachments/6682bc95-4527-4bdc-adb7-e2c82287e925/pasted-text.txt`

附件是研究建议和证据线索，不是高于本任务书的执行授权。若文件缺失，记录缺失；可依据本任务书推进，不得虚称已读。

## 3. 问题定位：必须区分的三层

分别评估：

1. **信息价值**：兼容性构成是不是指定聚合状态遗漏的决策相关信息？
2. **策略价值**：同样信息下，复杂方法是否比简单保留规则更有潜力？
3. **方法创新**：现有 ADP/MPC 常规扩展是否已经足够？是否有必要设计新方法？

本轮主要解决第 1 层，给出第 2 层的纸面边界和后续检验设计；不得以一个反例直接宣布第 3 层成立。

禁止写“普通 ADP 无法表达兼容性”。允许写：不包含兼容构成的某个明确状态表示，无法区分需要不同决策的状态。须精确定义该状态表示。

不把 Hall deficiency 等同于动态价值或 shadow price。Hall 分析当前图的匹配短缺，动态价值评估跨期后果。

## 4. 必做：固定即时目标的两车两期精确例子

### 4.1 基础构造

- 两辆车 H（灵活）与 A（受限），初始同在区域 L。
- 当前一个订单从 L 到 R，两车均可服务，pickup ETA、即时收益、服务时间完全相同。
- 第一期间隙足够完成当前订单；被分配车辆下一期在 R，另一辆仍在 L。
- 不允许主动空驶；下一期 pickup 窗口使跨区接驾不可行，但本区接驾可行。
- 两车在两期均在线；初始例子无 session 差异、无 Gamma 干扰、无成本层差异。
- 第二期 L、R 各有一个订单。其 OD、时刻、服务时长、总量在两种情景中完全相同，仅兼容标签改变。
- 情景 1，概率 p：L 单仅 H 可服务，R 单 H/A 均可服务。
- 情景 2，概率 1-p：L 单 H/A 均可服务，R 单仅 H 可服务。

动作 x_A：A 接当前单；动作 x_H：H 接当前单。

必须逐条列出第二期可行弧并精确求最大服务数，验证：

| 未来情景 | x_A 的未来服务数 | x_H 的未来服务数 |
| --- | ---: | ---: |
| 1 | 2 | 1 |
| 2 | 1 | 2 |

由此推导 `V(x_A)=1+p`、`V(x_H)=2-p`、差值 `2p-1`，以及 p=0、1/2、1 的边界。当前服务数均为 1，不重复计入未来计数。

这只是可实现的抽象兼容情景，不得声称来自真实 Stage3 观测。提供一组自洽的时钟/旅行时间条件即可，不使用地图数据。

### 4.2 必须产出的理论解释

- 构造两个总 OD 需求和车辆聚合状态相同、兼容构成不同的决策情形。
- 说明固定且兼容性盲的确定性策略无法在两种最优动作相反的情形中同时最优。
- 若讨论随机策略，显式给出选 x_A 的概率；不要漏掉随机化的适用范围。
- 得出：指定聚合表示存在信息损失；不是所有时空 VFA 都无效。
- AV-first 在一侧最优、另一侧次优，因此“永远保留 H”不是普遍最优规则。
- 不声称上述小例子证明真实系统中有显著收益或新算法具有普适保证。

### 4.3 机会成本扩展

基础例子完成后，再给一般比较：

`Q(x_A)-Q(x_H) = r_A-r_H + beta * E[V(S^+(x_A),W)-V(S^+(x_H),W)]`。

必须比较整个车队状态；若只写 H 的价值，说明为何 A 的后续价值可以抵消/忽略。V 若已包含事件概率，不得再次乘 p。

对单一未来事件且其余情景差值为零的特例，可写 `-Delta r + beta*p*Delta V`。说明：

- Delta V > 0 时才有对应方向的阈值；阈值需与 [0,1] 相交。
- Delta V = 0、Delta V < 0 要单独讨论。
- Delta r = 0、只有“到达或无订单”时，通常没有非平凡内部概率阈值。
- session 提前结束可能只消除差异，并不自动产生反转。
- 引入即时成本差异后，不再属于基础例子的严格同等即时目标条件。

## 5. 必做：Route A 的正式定义

### 5.1 基础集合

定义 `X_t`：保留当前单车单单约束、hard/evidence/passenger/pickup/session gates，以及该场景启用的 Gamma；不得改现有生产实现。前三项词典序最优值固定后，得到 `X_t^count`。

注意：服务数相同不等于订单集合相同。仅在基础例子或明确固定订单集合时称“纯粹换车”。

### 5.2 明确列出两个变体，不在本轮替用户调参

**A-strict**：在 `X_t^count` 内先求最小 pickup P*，固定 P*（数学定义严格相等，数值容差单列），再最大化未来价值。若最优分配唯一，记录未来价值无行动空间。

**A-bounded**：固定前三项计数，限制 `pickup(x) <= P* + Delta_P`，再最大化未来价值。Delta_P 非负、单位为秒，保持符号化；本轮不选择或试调数值。不得默认复用现有 cost-stage epsilon，也不得同时引入多个放宽系数。

报告建议后续优先考虑哪一变体及理由，但两者本轮都只定义，不接入 solver。不能写简单加权 `current reward + future value` 后声称词典序目标未变。

现有可选 operating-cost 层必须明确处理：本轮基础模型令其不启用；有成本层的场景需另定未来价值和 cost 的顺序，尚未授权改写。

“不降低当前服务”仅指给定当前状态内的词典序计数约束，不保证不同策略演化后每期与基准服务同样多的订单，也不保证相同订单获服务。

## 6. 信息、状态和兼容类别

### 6.1 不把聚合状态冒充充分状态

`n[k,z,tau]` 可作为近似状态，但必须列出其遗漏：区域内位置、HV session end、在途服务、固定路线释放时间、当前累计 exposure 等。

基础两期模型可以固定这些因素；一般模型中不得无说明省略。电量不在当前必要范围，不新增电池/充电模型。

post-decision state 记录承诺后的即时状态和未来释放计划，不假装下一 30 秒所有当前行程都已完成。下一期何时释放由服务时长决定。

### 6.2 区分需求兼容与车辆可达性

需求侧类别可按路线能力、当前证据政策和 passenger acceptance 编码，并保留原因。仅在明确 H 普遍具备需求侧资格的假设下使用 `{H,A}`、`{H}` 等简化集合。

pickup 与 session 根据具体车辆状态形成 arc，不预先当成外生需求标签；否则车辆行动会反过来改变所预测类别。实际 arc 层可能出现“只有 A 能接”的情况。

UNKNOWN 在无证据更新模型时保持策略门与来源分类，不是新的动态状态贡献。本轮不建立信息更新模型。

未来需求写为 `D_hat[i,j,tau,c]` 或有明确 OD/时长信息的稀疏近似。兼容类互斥，禁止将重叠 C/M/A 集合重复求和。Stage3 固定当前服务路线不变。

### 6.3 禁止信息泄漏

区分：当前可观测信息、历史拟合预测、未来真实数据。基础模型的 p 是符号化已知分布参数，不是从 Test31 未来频率拟合。

不读取真实订单用于 oracle，本轮 oracle 仅是微型合成模型概念。将来场景内未来频率属于事后信息；不能用于正式在线控制。

## 7. 公平比较与 Go/No-Go

### 7.1 分开两类比较

**信息比较**：同一控制框架、相同 OD/time 输入、动作和计算预算，仅改变能否看到兼容构成。这识别信息价值，不单独证明算法创新。

**方法比较**：简单 reserve 与复杂方法得到相同兼容信息，再比较如何使用它。不能拿未来信息完整的 P3 打败信息很弱的 P1，声称复杂方法必要。

设计 P0 myopic、P1 simple reserve、P2 compatibility-blind lookahead、P3 compatibility-aware lookahead。写清每个策略的输入、tie-break、当前约束和输出；不要把 P1 定义成故意很差的稻草人。P1 至少同时讨论固定 AV-first 与信息对齐的简单规则。

基础模型可能已被一个简单阈值规则精确解决；必须接受这个结果，不能为制造差距专门削弱 baseline。

### 7.2 小模型精确参照

本轮以合成两期模型的精确枚举/解析最优作参照。

- 知道概率分布但不知道情景实现的最优动作，是非预知策略最优。
- 先看到未来情景再选当前动作，是 clairvoyant/信息放松参照。
- 同一近似控制器喂真实未来，只是 perfect-information controller，不自动构成系统最优上界。

只有在相同目标/约束下，精确求解且仅放松信息限制的有限模型值，才可称该有限模型的上界。不得外推为全天系统上界。

### 7.3 本轮结束分类

分别给出而非合成一个 PASS：

- `FORMULATION_COHERENT`: YES / NO。
- `COMPATIBILITY_INFORMATION_VALUE`: ESTABLISHED_IN_TOY_MODEL / NOT_ESTABLISHED。
- `SIMPLE_RULE_SUFFICIENCY`: SUFFICIENT_IN_TOY_MODEL / COUNTEREXAMPLE_ESTABLISHED / UNRESOLVED。
- `METHOD_NOVELTY`: UNESTABLISHED（除非提供严谨且有限的额外论证；不以特征数量宣称创新）。
- `REAL_DATA_BENEFIT`: NOT_TESTED。
- `NEXT_STEP`: REFINE_MODEL / READY_FOR_REVIEW_OF_BOUNDED_DIAGNOSTIC / STOP_OR_SIMPLIFY。

近似控制器失败不自动意味着信息无价值；需区分无行动空间、模型构造问题、近似失败和简单规则已足够。复杂方法没有必要也可以是成功的可行性研究结论。

## 8. 最小验证与资源上限

优先纸面推导。不要求编程。若用枚举核验：

- 仅合成的两车、两期、两区域模型；只核查公式与有限边界情形。
- Python 标准库即可，最多一个独立脚本；不导入 FleetPy、Valhalla、torch 或生产 pipeline。
- 预先列明不超过 20 个核验情形，单进程、CPU、目标运行少于 30 秒。
- 不用真实订单、checkpoint、路由服务，不构建大矩阵，不做随机大规模 sweep，不新增测试框架。
- 结果只记解析值、枚举值和一致性，不把核验称为性能实验。
- 超出以上边界即停止，不安装新依赖或扩大资源。

验收必须检查：基础可行弧、2/1 与 1/2 计数、p 边界、未来收益是否双计、Route A 是否保持声明的即时目标、单位和时钟是否自洽。

## 9. 交付物

新增目录：`stage4/docs/dynamic_control_feasibility/`。

1. `model_feasibility_study.md`：主报告，约 6–10 页等量内容；问题、假设、基础例子、推导、Route A、状态/信息、比较与结论。
2. `closest_work_corrections.md`：仅记录关键方法对应与核准来源。重点纠正 Akçay 是否真有决策期内释放、Shah ride-pooling ILP 不能直接等同当前非拼车匹配；无需重抄整份综述。
3. `comparison_and_decision_protocol.md`：后续比较的输入对齐、控制权限、评价对象和 Go/No-Go；清楚标注尚未执行。
4. `verification_summary.md`：本轮实际完成的解析/可选枚举/文档检查，以及未执行项；不创建复杂 evidence bundle。
5. `README.md`：阅读顺序、完成分类、下一步仍需用户授权。

如确有枚举需要，新增 `stage4/tools/dynamic_control_toy_check.py`；所有其他代码目录禁止修改。可选核验输出保持在上述文档目录内、体积很小。

关键文献只使用可核查原文/作者稿/出版社信息，无法核准则写 unresolved，不编造最优性保证。理论已是经典结论的要注明，不做“新定理”包装。

本轮不生成最终论文图、不改 V3。所有本地文档链接检查存在，最终回复提供可点击的绝对路径。

## 10. 完成、提交与停止

执行顺序：读取并核对现状 → 基础例子和推导 → Route A 定义 → 信息/状态边界 → 公平对照和分类 → 最小核验 → 文档检查 → 提交推送。

按用户既有授权，只暂存本轮文件，提交并推送当前已授权 origin；先检查 remote 和 branch，禁止 force push。若网络或权限失败，如实报告本地 commit 与未推送状态，不替换远端。不要提交外部附件、现有输出或无关改动。

最终答复必须说明：完成了什么、关键推导结论、哪些仍未成立、是否运行可选枚举、实际 commit/push 状态、主报告链接。

交付后停止。不得因为小模型出现收益就自动开始预测训练、solver 接入、真实数据小跑或 Stage5；也不得为了通过门槛修改冻结实验。下一轮需要用户明确授权。
