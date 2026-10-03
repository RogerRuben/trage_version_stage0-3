# 独立交通研究策略：小窗口动态对照

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent / run
- Date: 2026-09-28
- Verification: 执行完成；基线截断前精确复现；比较结果为描述性分析
- Protocol: `e13325d`; execution code: `c3692e9`; attempt: `retry1`
- Status: COMPLETE，进程退出码0；未新增实验条件、未调参

## 设计与运行

按已批准的 `stage4/config/traffic_research_window.json`，从同一10:30原生checkpoint顺序运行
FROZEN和TRAFFIC_REPLACEMENT_V1两个条件。q50是基线归一化AV active-hour level，不是混合车队占比；
profile M，乘客接受参数0.7，30秒滚动，300秒到达耐心，cohort为[10:30,10:45)释放的379单。
10:45后停止新请求，10:50结束派单，继续排空任务至11:19（40740秒）。没有新增dwell、lead或repositioning。

运行命令见 [status.md](status.md)。本地产品位于
`stage4/output/traffic_research/window/retry1/{FROZEN,TRAFFIC_REPLACEMENT_V1}/`，
每个条件保存assignments、cohort_outcomes、epochs、exposure四个parquet及summary.json。
摘要另存 [window_summary.json](window_summary.json)，不依赖被Git忽略的输出目录即可审查核心数字。

## 结果

| 指标 | FROZEN | 5%研究策略 |
|---|---:|---:|
| cohort订单 | 379 | 379 |
| 匹配 / 到期未匹配 | 254 / 125 | 254 / 125 |
| cohort AV / HV匹配 | 22 / 232 | 22 / 232 |
| 已匹配订单平均等待(s) | 187.9833 | 187.9833 |
| checkpoint后分配（含早前等待单） | 266 | 266 |
| 其中AV分配（含早前等待单） | 24 | 24 |
| routing arc evaluations / failures | 37,857 / 0 | 37,857 / 0 |
| 运行时间(s) | 79.047 | 81.484 |
| 进程峰值RSS(MiB) | 532.594 | 542.270 |

两个cohort outcome表逐字段精确相同；全部266条新增分配的请求、车辆、派单时刻、pickup ETA、
pickup time及service end time精确相同。匹配数差为0，不是只看汇总相近。
两个条件顺序运行，合计约160.53秒；第二个峰值是同一进程的累计高水位。无GPU、无稠密大矩阵。

## 为什么本窗口未改变派单

研究分支41个epoch的 `traffic_policy_pruned_av_opportunities` 总计为0：
本次M环境外定义（严重拥堵预测时间占比）与5%预算没有额外删除进入该门控的AV机会。
24条新增AV分配的环境外占比平均0.2577%，最大1.9981%，均低于5%。
这不代表全部请求都没有严重拥堵，也不代表该门控在C或其他窗口不会起作用。

去掉crawl/stop六项确实改变了dynamic exposure，不是开关未生效：

| 动态量 | FROZEN | 研究策略 |
|---|---:|---:|
| checkpoint历史累计量 | 95.645053 | 61.650633 |
| 结束累计量（695次历史及新增AV分配） | 99.616281 | 64.288317 |
| 结束累计均值 | 0.143333 | 0.092501 |
| 固定gamma_dynamic | 0.149343 | 0.149343 |

static累计均值两者均为2.118923，gamma_static为2.145068；speed均为0。
动态预算变宽没有在此条件下转化为更多分配。仅凭这些最终数值不能认定究竟哪个其他约束是唯一瓶颈，
也不能推导本策略全天无效。

历史累计量按新CV/acceleration定义重算，但历史分配、计数、static/speed及物理状态不改。
因此这是同一冻结物理状态下的条件性延续，不是用新策略从零运行全天。

## UNKNOWN与验证范围

cohort中原AV-eligible且有研究描述的订单，其路线UNKNOWN占比的订单平均为8.4872%；
24条新增AV分配的UNKNOWN均值为9.6777%，最大25.7069%。这些是订单均值，不是合并时间加权占比。
UNKNOWN没有新增拒绝或惩罚，但也没有被证明属于适配环境。5%只约束已知环境外暴露。

- 2,026个范围内原AV-eligible请求的日期、ORIGINAL路线身份、冻结rho及预测时长预检通过。
- FROZEN在需求截断前与既有deterministic continuation逐分配精确一致。
- 请求/车辆/位置/AV availability对账通过，已接任务全部完成，物理任务无重叠。
- pickup时间误差最大约9.87e-10秒，service时间误差0；均满足1e-6秒校验阈值。
- 研究AV分配均满足5%门控；配置、Stage3 profile、M3 checkpoint和物理checkpoint保持原hash。
- 先前32项测试及compileall已通过；本轮未改执行代码，不重复仿真或扩展测试。
- 本checkpoint没有启用完整shadow gate列。读盘时尝试调用完整gate守恒检查因缺少
  `gate_av_n0_spatial`而停止；随后按实际schema完成分配及cohort对账。
  不声称此次输出验证了完整funnel守恒；相关新增字段逻辑已有专项测试覆盖。

## 解释与停止边界

这次支持的结论是：独立开关和不重复叠加的替代定义已能接入原生动态派单，冻结基线保留；
在一个预定M窗口中，暴露账本改变但物理分配没有变化。
不能据此声称提高服务、策略统计等效、C/M/A整体有效、AV安全认证或全天泛化。
Test31仍是legacy benchmark，不是新增未见测试集；既有pickup校准的回放局限未由本次试验消除。
已匹配等待均值属于选择后的样本，不单独解释为乘客总体效应。
两个固定条件完成后停止，没有依据零差异改预算或寻找另一个有差异窗口。

首次预检失败的 `stage4/output/traffic_research/window/summary.json` 原样保留；
它在仿真前停止，与本次已授权retry1的完成状态分开记录。
