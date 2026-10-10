# 完整布局已完成；第一次全天组未正常收尾

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run。
- Origin Date：2026-10-10；Verification Status：UNVERIFIED（没有完整全天组）；Version Label：attempt_005_interruption。

## 已完成并可复用

执行代码为 `aaf73e20f9c4a46ad5ae74cc3ef307b01bf2cf86`。真实离线30秒已落实：第39批整数主问题用时12.915秒，保持严格最优性要求。恢复阶段1,034.976秒，补齐剩余9批；总计47/47批、851个AV槽位，所有47批期望服务非零。该数据是预测规划支持，不是实际服务收益。

2026-10-10核对现有完整布局与当前固定协议的全部输入SHA完全一致。该布局可以直接复用，不重新计算、不改预测或策略。

## 第一组中断事实

模拟进程26632已不存在，原工具会话句柄失效，旧摘要仍留有 `RUNNING`。最后日志写入为2026-10-10 00:42:59（+08:00），无Python异常栈。相关时段应用日志没有筛出可用的Python／Valhalla崩溃事件，**具体终止原因及退出码未知**，不能断言是算法、内存、系统崩溃或某个具体应用动作导致。

最后持久化进度是08:00（日秒28,800）：分配1,750单、完成1,561单、过耐心438单；进度时运行2,465.998秒、RSS960.56 MiB。最后日志晚于这个进度，不能把08:00当作精确停止轮次。

没有写出完整或部分 `assignments`、`cohort_outcomes`、`solver_trace` 产品，也没有可精确恢复的原生模拟状态。因此不能计算最终服务率、不能将该前缀当作一组结果、也不能从08:00直接续接。

原第一组目录完整移至本地：

`stage4/output/capability_chain_planning/scheme_a_city_v1/interrupted_attempts/hotspot_baseline_005/`

原日志、完整布局及路由缓存保留，没有删除或覆盖原始数据。

## 用户授权后的处理

用户明确允许：仅将第一组从午夜按原条件重跑，随后完成未启动的第二、第三组；复用完整布局，不重做布局。数据、能力、目标顺序、链限制、种子、实时10秒和每组10800秒时限均不变，不新增实验条件。

新执行仍是 `python -m stage4.analysis.scheme_a_city_experiment --resume`。采用Windows无窗口、脱离工具作业的后台启动；单一模拟进程12292，一枚仅等待并记录真实退出码的轻量进程15152，不并行运行模型，不自动重试。2026-10-10 15:00:28（+08:00）启动。

本地运行记录为 `execution_attempt_006_process.json`，最终真实退出记录为 `execution_attempt_006_exit.json`；后者仅在模拟实际退出后生成。没有创建OS定时任务或循环重启机制。

上述只是执行管理调整，不把此前中断归因为已证明的工具缺陷，也不宣称三组实验已经完成。
