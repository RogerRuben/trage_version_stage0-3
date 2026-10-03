# Relative-MIXED 离线解释与归因协议

2026-09-28；用户授权沿既定三步推进；基线 a8be540。运行前固定。

仅复用 Validation25–27、legacy Test31 已有预测、Train参考、道路类别和已完成
C/M两窗口F/VT结果。无训练、推断、路由或派单；不修改任何策略与5%预算。

1. 按日统计状态的预测时间加权速度(3.6/pace)、参考速度(3.6/q20 pace)、
   pace比值、crawl/stop/low-speed分布。速度是pace倒数的描述量，不是实测瞬时速度。
   relative-MIXED另按canonical_highway和既有reference_support分层；保留缺失类别。
   加权分位数使用累计权重左连续逆CDF，不做跨日token池。
2. C路线分成 CONGESTION_ONLY、RELATIVE_ONLY、BOTH_INDIVIDUALLY、COMBINED_ONLY、
   PASS、MISSING。congestion=congested+severe，relative=relative-MIXED。
   超标完全沿用 >0.05+1e-6。UNKNOWN不填0，缺少路线描述标MISSING。
   Test31原合格路线分别汇总；Validation不虚构Stage4资格。
   进一步把outside分为满足原参考支持/低支持贡献，报告去掉低支持贡献后是否仍超标；
   这是账面分解，不是授权删掉低支持token或改变完整路线分母。
3. 现有F→VT全部cohort订单配对，连接精确ORIGINAL路线描述，报告所有
   before/after服务类型及 gained/lost/retained/unserved与归因类别交叉计数。
   M实际门控仅severe，C分类在M中只作旁列诊断；不把HV损失说成直接AV门控损失。

资源：逐日必要列、单进程CPU、无GPU/稠密矩阵；2GiB告警，1800秒硬超时。
输出独立目录 stage4/output/traffic_mixed_attribution_v1；摘要/报告放本docs目录。
检查：四种拒绝分类互斥完备、边界容差、缺失保留、加权分位数、订单配对唯一性、
与既有policy占比及窗口gain/loss对账。原冻结输入前后不变。
不检验显著性、不用结果选阈值，不推断真实AV能力或独立交通准确率。
