# 对称路线能力接口与M全天配对比较

## Material Passport

- academic-research-suite / experiment-agent；2026-10-04；run + validate。
- 执行状态：两组已完成。旧3小时超时与旧6小时进程消失的部分产品保留；2026-10-05按新授权在独立加速目录完成第二组（执行代码6b46f54），复用MYOPIC。为内部ANALYZED，不是独立复跑VERIFIED。
- 输入与执行前代码提交：`d50b02b`、`8559cf7`。协议在新增native全天结果之前提交。
- 仅两组M全天条件，不新增C/A动态条件，不覆盖冻结基线，不按结果调参。

## 1. 本轮先修正了什么

旧接口把historical reverse overlay整单判为AV方向不可路由，HV却绕过这道门槛。因此旧A覆盖14,613/30,000不能解释成A驾驶能力只有48.71%；C/M也受到同一道不对称门槛及反向区段路口解析中断的影响。

新opt-in研究接口先定义共同的数据支持与道路约束集合，再判断能力：

\[
\mathcal O_C\subseteq\mathcal O_M\subseteq\mathcal O_A=\mathcal O_{HV}.
\]

A/HV相等的是研究路线能力，不是实际服务量。乘客接受、车辆位置、可用时段和接驾耐心仍属于派单条件。
研究层为有冻结物理几何参考的历史反向身份建立独立virtual directed identity，反转INCOMING/OUTGOING和切向，重新解析maneuver；不把forward edge冒充实际反向遍历，也不复制forward movement或restriction。
历史观测方向可用于回放是明确研究假设，不证明所有OSM方向冲突都合法或都是地图错误。certified prohibition同样作用于HV与AV。

| 路线能力 | 旧标签数量 | 新标签数量 | 共同集合内兼容率 |
|---|---:|---:|---:|
| C | 5,299 | 9,275 | 32.70% |
| M | 13,847 | 24,201 | 85.31% |
| A | 14,613 | 28,367 | 100% |
| HV | 无共同标签 | 28,367 | 100% |

共同集合28,367单；其余1,633单对HV/AV共同排除，不计为AV能力失败或patience-expired。排除原因：1,497单仅路线身份支持不足，118单还缺必要输入，18单仅缺必要输入。
6,388个反向身份获得研究表示，complex encounter从538,355增至579,778（+41,423）。这不是跳过反向maneuver后直接放行C/M。
C保留信控左转、禁止掉头/环岛的限制；M允许非信控左转和环岛、禁止掉头。C环境外状态为MR/CG/SC，M为SC，A/HV为空；环境外暴露预算5%。U不自动否决，ML不重复叠加拥堵约束。

## 2. 同一新口径下的唯一全天比较

Test31，M，baseline-normalized AV active-hour level `qA=0.5`，passenger acceptance `0.7`；零提前请求释放、300秒到达耐心、30秒rolling、current TopK20、确定性single-source matrix。
两组从0秒独立干净native状态开始，使用相同车队抽样、稳定接受随机数和28,367单共同模型集合。原30,000单全量对账。
HV继续采用经验session，AV覆盖全天测量窗并允许已接受任务drain；能力集合相等不意味着可用时段、位置和需求接受完全相同。
Gamma、平台cost及repositioning关闭；旧连续静态/动态/speed描述保留，但旧反向区段静态聚合可能不完整，本轮不把它当准入约束或完整复杂性真值。

Train需求模板来自20161010/17/24全天，共28,335条，分别9,479/9,392/9,464；复用冻结M3预测，不用Test31未来需求拟合。
busy剩余时长模型复用上一轮3,400个Train配对，不重新拟合。新策略先保持同状态critical/current service/carry-over最优面，再最大化预期下一服务数、最后最小化接驾ETA；保护的是当轮计数，不是每个订单，也不是全天优势保证。

固定172,800秒行政drain上界，不从Test31未来实际行程时长推导优化器里的availability终点；日vehicle-hours仍按24小时归一化。
本轮输入派生曾因未先筛选冻结complete-dynamic订单而显式停止，修复的是Train订单集合入口，不逐token丢弃、不用实际时长补值；已完成Test31产品复用，未重算。

## 3. 全天结果

第二组原尝试运行10,804.062秒后触发行政超时，中止产物保存在`full_day/aborted/SERVICE_PRESERVING_LOOKAHEAD_10800s/`。原尝试不计为完整全天结果。旧6小时重跑最后落盘为模拟14:30、9,255单、3,928.703秒；文件保存于2026-10-05 00:44，12:51检查时已无对应进程，原exec会话不可读取，退出原因无法证实。原始RUNNING summary及部分产物保留，不把它改成伪造的完成结果。以上为旧尝试历史；下表的第二组来自新授权的独立加速版完整执行，不拼接旧部分数据。

| 指标 | MYOPIC | SERVICE_PRESERVING_LOOKAHEAD |
|---|---:|---:|
| 服务订单 | 18,702 | 19,398 |
| AV服务 / HV服务 | 4,338 / 14,364 | 4,647 / 14,751 |
| 共同集合服务率 | 65.9287% | 68.3823% |
| 原30,000单服务率 | 62.3400% | 64.6600% |
| 共同集合中耐心超时 | 9,665 | 8,969 |
| 共同输入排除 | 1,633 | 1,633 |
| 已服务订单平均等待，秒 | 170.774 | 181.427 |
| 等待P50 / P90 / P95，秒 | 184.178 / 281.298 / 290.846 | 199.914 / 284.502 / 292.302 |

配对新增获服务2,573、丢失服务1,877、净增696；共同获服务16,825单平均等待增加10.0895秒。是吞吐—等待权衡，不是Pareto改善。等待指标条件于已服务，不把未服务订单当零等待。完整数据与解释见[加速执行报告](../acceleration_v1/report.md)。

## 4. 资源与执行正确性

MYOPIC：运行5,119.375秒（85.323分钟），native Python峰值549.230 MiB，2,893个dispatch epoch，最大TopK候选2,294、有效OR arcs 104，routing失败0，arc evaluations 2,331,641；solver总计17.478秒。
其two-stage扩展变量/非零元为0，是未构建未来扩展模型，不代表原当轮稀疏OR没有求解。
原30,000单 = 18,702已服务 + 9,665耐心超时 + 1,633共同输入排除。
native任务全部完成，接驾耐心、任务不重叠、车辆/订单/位置及物理时间链对账通过，受保护执行输入SHA未变。
MYOPIC实际AV分配包含1,947单历史反向路线，HV包含7,018单；AV的M能力违规为0。这证明新接口并非只生成离线标签，旧反向否决没有继续潜藏在native分配中。

第二组加速版完成8,158.813秒（135.980分钟），父进程峰值663.555 MiB、进程组step采样峰值1,024.496 MiB；最大7,194变量、28,072非零元，回退/当轮最优面违规/路由失败均0，全部接受任务完成drain。实际AV历史反向路线2,127单，M能力违规0。六个共同冻结输入SHA未变。
每轮20,000变量/150,000非零元/10秒求解上限不变；旧3小时和6小时部分记录保留。详见[行政重跑记录](administrative_retry_record.json)，该文件是旧尝试历史，不是新版完成状态。
仅一个native情景主进程运行，加速版另有两个独立路由worker；数值线程环境1、CUDA禁用，无稠密订单×车辆矩阵。输入派生峰值1,790.020 MiB，不跨日期累计token表。

## 5. 解释边界

本轮是内部全天配对验证，Test31已用于开发，不是未见Test。不能把新旧接口的服务变化归因于派单策略，也不声称单模块因果消融。
只有一日M动态比较，不外推到C/A、其他日期、真实安全性或全天必然优势。A是human-equivalent研究上界，不是实际AV车型认证。
未来ETA是Train OD-chord pace近似；每车每情景最多一次下一服务，不是完整多次服务随机VRP。剩余时长模型的原训练支持只有两个时间带；route time proxy是冻结traversal预测聚合，不称校准的joint route P50。
封存699处信号包含693处地图/高德证据及6处旧推断；2026控制补齐用于研究情景，不能写成2016控制真值。
11类解释风险与AV/HV分工、服务选择偏差已在[完整报告](../acceleration_v1/report.md)记录。不进行基于订单独立性的显著性检验，也不把一日M结果升级成普遍决策优越性。

## 6. 文件与复现

- [定义闭环](definition_closure.md)、[执行前协议](protocol.md)、[输入数字](input_summary.json)。
- 已完成：[MYOPIC summary](myopic_summary.json)、[第二组summary](../acceleration_v1/service_preserving_lookahead_summary.json)、[配对analysis](../acceleration_v1/analysis.json)。
- 本地产物：MYOPIC位于`stage4/output/symmetric_flexibility_v1/full_day/MYOPIC/`；第二组位于`stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD/`，含assignments、cohort_outcomes、solver_trace、epochs及fleet_accounting。
- 执行入口：`stage4.analysis.symmetric_research_prepare`及`stage4.analysis.symmetric_flexibility_full_day`，详见协议。
- 旧18窗口、41场景、旧负结果、冻结profile/CDF/M3/路网/信号产品均保留。
