# A2 — Ten-state Stage3/action-space experiment proposal

日期：2026-09-18；状态：**DRAFT_FOR_USER_FREEZE / EXECUTION_NOT_AUTHORIZED**。只提交提案，不能据本文启动计算。上限仍为 10 states × 5 conditions；不做全天、不改生产，不新生成 service route 或 prediction。

## 1. 冻结前需要决定的五项

| 项目 | 建议 | 当前限制 |
| --- | --- | --- |
| Gamma | 五条件全部 off、cost off；命名 D0_STAGE3_CURRENT_GAMMA_OFF | 与旧 Gamma-on 0/10 不同，不能直接复用该数字 |
| Route 口径 | D0 复现 final；D1/D2 若要测原原因传播，应允许在已有 ORIGINAL/K1 FALLBACK 池内重派生选路 | 这包含既有路线选择机制，不是只改最终一个布尔值；需冻结选择规则 |
| Pickup ETA | 仅使用同位置、时刻、模式、beta 的既有 arc ETA | 当前未确认完整 required union；不允许新路由或插值 |
| Top-K | 推荐生产 shared Top-K，报告条件间 HV displacement；不称 nested upper bound | 真正单调 common-universe 口径是另一个实验设计，不能混用 |
| Assignment 参照 | condition-local A1/A2，明确各条件 counts/order-set 的变化 | 不增加 baseline-anchored 第二套优化 |

目前 **pickup coverage 阻止完整执行**，route-selection 语义也未冻结。若无法补齐既有归档，本提案应保持待定，不执行“看起来完整”的残缺实验。

## 2. 条件定义草案

所有条件固定物理车辆/历史状态、订单/截止时间、acceptance CRN、半径、session、原 pickup 时钟。只讨论 M；不扩 C/M/A 矩阵。

| 条件 | 目标变更 | 必须保留/标注 |
| --- | --- | --- |
| D0 CURRENT | 原 final Stage3 eligibility | 新定义 Gamma-off；完整 production evidence 谓词，不只布尔标记 |
| D1 NO-REVERSE | 原/候选路线判定中仅移除 reverse-overlay prohibition | 保留 certified restriction、capability 与其他 unknown；不得删除包含其他原因的整条 reason list |
| D2 NO-CAPABILITY-HARD | 仅移除 capability-specific hard codes | 保留 direction identity 规则与 certified restrictions；“PHYSICAL-HARD”名称不足以表达尚未法律认证的 reverse，建议改名 |
| D3 NO-EVIDENCE-GATE | 保留选中路线与 structural hard，诊断性绕过 AV evidence flag 与有限 rho 资格检查 | 不伪造 rho/P50；Gamma-off。fallback 没有 P50仍记为缺失；仅当前 pickup/assignment诊断，不推进服务 |
| D4 STAGE3-NEUTRAL | 移除 AV 的 Stage3 type-specific eligibility | hypothetical-service oracle，不声称已有合法路线或动态证据；保留 passenger、spatial、pickup、availability/session；不是生产 policy |

D3 不能将结构未知身份、未证可行 movement、route=NONE 自动全当 evidence 缺失删除；结构 unknown 和动态 evidence unknown 要先建立显式映射。

对于 D1/D2，可选口径需用户选定后才实现：

- 固定 final selected route：最纯粹，但 NONE 不能恢复原路线，可能遮蔽原 reason→fallback→evidence 的传播，不能回答全部 reverse 问题。
- 既有路线池重派生：仅复用已存在 original/fallback，不新路由；在新 hard 规则下先重新判原路线，再按冻结 limited fallback 规则选择，所有 evidence 必须绑定对应 route。若无原路线反事实所需静态/unknown信息，则该订单 unresolved。不能借旧最终聚合字段猜新 hard 状态。

建议第二口径用于目标归因，但必须先审计所需 descriptor/reason 是否足够；现有 fallback candidates schema 本身不够。禁止自动生成缺失 movement evidence 或跑 M3 补洞。

## 3. 无路由输入覆盖检查（冻结后第一步）

不运行模拟，按旧登记重建可用物理状态并核 state SHA。构造五条件所需 pre-pickup sparse union，对候选弧查找既有缓存的精确 key：source/pickup WGS84、timestamp/minute、routing mode、beta及配置来源。

分别报 required_arc_count、exact_eta_available_count、missing_eta_count、coverage；这些是输入审计，不是 routing failure rate。未知 ETA 不能按不通过 patience处理，也不能按通过处理。

完整主实验需要所有所需弧有有效 ETA 或已归档的真实 routing failure。否则停止该轮；不仅删掉缺失最多的订单/状态后报告完整十状态结果。可汇报不依赖 ETA 的 upstream eligibility 层，但终层 M/O/A1/A2 保持 N/A。

这一检查只在协议冻结后执行；本轮未重建 state 或构造 union，A0 不声称已经测出覆盖率。

## 4. 结果表

每状态×条件：route source/provenance 状态、AV E/U/M、最终 O_HA、HV弧数、Top-K 排挤 HV 数、routing-evidence覆盖、原约束计数、P*、参考订单集合变化、A1/A2 最小 gap、绝对秒与相对值、严格/0.5%/1%/2%/5% alternative 状态、实际同订单 H↔A witness、求解状态。

每 nested 层另外报告 E_AV/U_AV/M_AV/O_HA，保留如下细分：

spatial → passenger → direction/certified hard → capability hard → structural unknown/route established → evidence → shared Top-K → archived ETA availability → pickup/patience → session。

该 chain 在单条件内是筛选顺序；跨条件重算 Top-K 后不宣称 mixed graph 单调。结构未知单列避免把 UNKNOWN 擅自映射为 legal hard 或纯动态缺证。

缺预测时长导致 HV session 无法判定的弧仍按原规则，不用 realized time补齐。AV FULL_HORIZON 在 hypothetical D3/D4 下只做即时可接性，不据缺失 duration 计算释放或未来收益。

## 5. 拟议 screening 门槛（供审查，不算已冻结）

用户提出的“明显”“很大”需量化。建议以五档预算中预先选定的 5% 为最高诊断预算，同时完整报 strict；不据结果修改预算。

- **INVESTIGATE 某单因素**：相对新 D0，至少 3 个状态同时满足终层 M_AV 增加至少 1、O_HA 增加至少 1；且至少 3 个状态由没有变为有 A1 同订单 H↔A 替代（≤5%）。这是投入筛查阈值，不是显著性检验。
- **STOP_INVESTMENT_ON_THIS_SAMPLE**：D4 相对 D0，A1 和 A2 各自新增可变状态均≤1，且没有任何单因素达到上述 investigate 规则。
- 其他情形（例如容量改善但 A1未打开，D4与单因素分歧）：**MIXED / REVIEW_REQUIRED**，不强行三选一，不自行追加实验。

这里以“新增状态”逐状态算 0→1，另报 1→0，不只用净差；assignment freedom 不保证单调。若 5% 以上才有替代，报真实 gap，不能将其作为本轮通过证据，也不能称政策已允许5%。A2-only 的改善单独报告，不能替代 A1 的车辆灵活性证据。

任一输入缺口、reason taxonomy未覆盖、route语义不一致：**INCONCLUSIVE_INPUT_OR_SEMANTICS**，不触发 Stage3 免责或整改。

## 6. 执行资源与禁止项

待用户冻结后：CPU、单 Python 进程、每次一个状态、稀疏矩阵、单状态条件≤5000 arcs、单 MILP≤10秒、全程≤10分钟、外部采样RSS上限2GiB；超限即partial/unresolved，不截断图冒充完整。

不写实验执行器、不执行上述 50 条件，是本轮交付边界。以后也不根据 screening 自动进入分支：reverse provenance、evidence pipeline、capability sensitivity 均须单独审查授权。不得修改 Stage3 frozen profiles、Gamma、41 场景、V3、Stage2模型；不得通过重路由规避本轮“不重路由”要求。
