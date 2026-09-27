# UNKNOWN / MIXED 与参考支持度分离

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized descriptive analysis
- Origin Date: 2026-09-27
- Verification Status: UNVERIFIED before execution
- Version Label: traffic_state_support_v1

## 执行前口径

1. Validation 仍限 25–27 日原 29,851 条动态完整历史路线，复用冻结 M3，分类阈值不改。
2. UNKNOWN 原因互斥优先级：预测非法 → 无 edge reference → reference 非法 →
   order-days<100 且 days<5 → 仅 order-days 不足 → 仅 days 不足。另报支持度与状态交叉表。
3. 研究性状态与支持度分开：只要预测合法且有有限正 edge q20，即计算状态；
   未达到原 100 order-days / 5 dates 记 LOW_REFERENCE_SUPPORT，不伪称可靠。
   没参考不补道路等级均值，不填畅通；原严格状态保留。
4. MIXED 拆成 RELATIVE_SLOWDOWN_WITHOUT_HIGH_LOW_SPEED_SHARE 与
   HIGH_LOW_SPEED_SHARE_WITHOUT_RELATIVE_SLOWDOWN。它们描述阈值组合，不标为红灯/接客真值。
5. Train 稳定性：只用 09–20 的有效 order-edge-day 平均 pace；以 SHA256(date,order_id,固定种子)
   将订单日分为两半，同订单所有 edge 同组。各半独立求 edge q20，报告对称相对差
   abs(qA-qB)/((qA+qB)/2)，按完整 09–20 的 order-day 数 1–9/10–29/30–99/100+ 分组。
   同时报日期数<5与>=5，不把组标签直接称为置信等级。
6. 用未参与上述参考拟合的 Train 21–24 完成事件 cell，比较两个 q20 对同一 pace/low-share
   的状态分配差异。分母仅是 joint 有效且两半都有参考的 cell，同时公开无法比较数。
   这是内部可重复性诊断，不是独立交通分类准确率；样本量组可能混有道路类型/日期结构差异。
7. 不选择新的最低支持阈值，不因 Validation UNKNOWN 下降而调阈值。
   研究性可计算性不等于状态正确性，更不等于 AV 通行或新的 hard gate。

## 输出与资源

`stage3/output/traffic_state_support/`：Train 分半参考、Validation token/route 解释产品。
`stage3/docs/traffic_state_support/`：原因表、交叉表、稳定性报告、配置/输入 SHA、资源摘要。
命令 `python -m stage3.scripts.traffic_state_support`；单进程 CPU，逐日窄列读取，
只保留紧凑 Train 数值池，不建 edge×time 稠密矩阵。1800s 硬超时、0.2s RSS 采样。
保留冻结 profile/M3/原参考，绝不重跑 Stage0/1、训练、派单或读取 Test31。
完成报告后停止，不自动选择预算或生产策略。
