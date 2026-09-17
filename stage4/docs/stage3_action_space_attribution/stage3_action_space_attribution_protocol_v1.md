# Stage3 action-space attribution protocol v1 — frozen

冻结日期：2026-09-18；依据用户最新审查指令。本文只记录已冻结规则，不授权执行后续所有步骤。当前执行授权 **Q0 + Phase0 only**；pickup 补全与最终 50 条件须等用户审查两步输出后再次授权。

## Scientific scope

10 个原登记 M-profile REFERENCE 物理状态；D0–D4 全部 Gamma OFF、cost OFF。它们是参考历史状态下的局部反事实，不是全天关闭 Gamma 后的系统演化。原 Gamma-on 的 0/10 仅供 Q0 对账，不是新 D0。

Primary endpoint 是 A1 同一批订单中的真实 H↔A alternative；A2 为 secondary。E_AV、U_AV、M_AV、O_HA 单列，不要求 M_AV 增加才能判定 substitution freedom 有效应。

## Conditions

| 条件 | 冻结定义 |
| --- | --- |
| D0 CURRENT | 当前 production Stage3 eligibility；Gamma/cost OFF |
| D1 NO-REVERSE | 在 original-route provenance 中只移除 historical reverse-overlay prohibition，保留其他 hard/unknown/evidence；若原路线 FEASIBLE 用原路线，仍 INFEASIBLE 才查既有 K1 pool |
| D2 NO-CAPABILITY-HARD | 同样 route-aware 逻辑；M 仅移除 UTURN_PROFILE_INCOMPATIBLE，保留方向及认证禁止；不声称评估 C-only 规则 |
| D3 NO-EVIDENCE | 已建立结构路线且 final hard FEASIBLE；绕过 evidence_complete 与 finite-rho eligibility；不放行 UNKNOWN/NONE，不填 rho/P50 |
| D4 STAGE3-NEUTRAL | 移除 AV 的 Stage3 type-specific serviceability；保留 acceptance/spatial/pickup/availability/session；fully relaxed comparator，不是 assignment 上界 |

没有新 service-route search、fallback rerouting、M3 inference 或 profile 改动。D1/D2 重派生使用既有 original 与 frozen K1 fallback edges，必要时仅调用冻结静态 evaluator。没有合法 fallback 动态证据就仍然缺失，禁止移植 original P50。

原 hard failure 曾掩盖的未知信息仍须保留，例如 reverse 与 unresolved identity 同时出现时，删除 reverse 不得把 unresolved identity 也抹掉。

## Execution gates

1. Q0：原 Gamma/reference 语义，重新核对旧十状态 counts/P*/A1/A2 gap 与状态；任何不一致停止。
2. Phase0：不求新 MILP、不路由；核 physical state SHA，恢复 route counterfactual，构造各条件 shared Top-K20 union，输出 required/archived/missing pickup arcs。无法恢复 route hard state 标 COUNTERFACTUAL_ROUTE_UNRESOLVED。
3. Phase0 仅三种最终状态：READY、READY_AFTER_BOUNDED_PICKUP_ETA_COMPLETION、NOT_IDENTIFIABLE。前两种只是输入门槛，不越过本轮的人工审查停止点。
4. 用户另行授权后才补 union 缺失的 vehicle→pickup ETA；相同位置、时间、Valhalla 配置/mode、WGS84、beta。不补 service route 或 prediction。
5. 再授权后才执行 10×5；保持每状态单独处理、无车辆推进。

## Two graph levels

共同 pre-Top-K 空间 universe：仅改变 Stage3 eligibility，可报告 E/U/M_AV/O_HA 的加边关系。

Production downstream：每条件各自 shared Top-K20→pickup→patience→session；单列新 AV 挤掉的 HV，不要求跨条件嵌套。早期 O_HA 不是已经通过时间约束的真实可服务重叠。缺失 ETA 不等于不可行。

## Assignment endpoints

每条件自己求 l_d*、P_d*、x_d*。A1 固定每个 y_o(x_d*)，A2 只固定 critical/total/carry-over 计数。报告各条件参考订单集合改变，A2 service indicator 变化不自动叫同订单车型替换。

预算为 aggregate pickup 的 0/0.5%/1%/2%/5%；数值容差与预算分开（沿用 1e-7 秒核验容差）。最小绝对/相对 gap 必报，即使超过5%。P*=0 相对 gap 为 N/A。5% 不是新平台政策。

## Screening

- INVESTIGATE_STAGE3_FACTOR：某 D1/D2/D3 相对 D0 至少3个状态新增≤5%的 A1 H↔A alternative，且这些状态 O_HA 增加；不要求 M_AV 增加。
- CAPACITY_ONLY_EFFECT：M_AV 改善但 A1未打开，单列容量效应，不推断控制自由度已改善。
- INTERACTION_REQUIRED：单因素弱而 D4打开 A1，记联合放松现象；reason co-occurrence 是待审后续分析，不新增实验。
- STOP_FURTHER_STAGE3_INVESTMENT_ON_THIS_SAMPLE：D4相对D0新增 A1≤1且新增A2≤1，无单因素达到 investigate。
- 上述后两类中的“明显”未额外指定数值时，报告实际增减状态和 M/O 数值，不擅自选阈值强制归类；可以保留 REVIEW_REQUIRED。多类信号可同时报告，不混成因果免责。

按状态分别记 0→1 与1→0；不得只报净差。结论限于这十个M状态，不能写 Stage3 不是原因或已经证明真实道路限制。

## Fallback provenance and resources

在现有条件表内附带各状态 fallback/hardF/evidence-incomplete 订单数、AV空间机会、D3 Top-K以及最终patience通过数。Phase0只能提供前几项，最终patience保持 NOT_RUN。

CPU单进程，稀疏图，2GiB外部采样监护、每MILP10秒、每执行阶段最多10分钟；超限记录未完成，不改变图或样本。原state hashes、配置、profile、route products只读。完成Q0+Phase0后立即报告并停止，自动提交推送本轮小型材料。
