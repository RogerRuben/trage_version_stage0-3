## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-10
- Verification Status: UNVERIFIED (全天实验未完成；以下为真实退出记录与源码诊断)
- Version Label: scheme_a_attempt_007_stopped_v1

# 第七次执行：持久化 HiGHS 的 run 警告被提前当作接口错误

## 执行结果

- 命令：`D:/anaconda/envs/stage0-valhalla/python.exe -m stage4.analysis.scheme_a_city_experiment --resume`。
- 工作目录：`D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla`。
- 执行代码：`4e8a208c2643f52246b035aae871e6a56c8c287d`。
- 配置 SHA256：`8022d88c8c1377fd87f925bf7f0e35584b9bb3f500bc5cb8fd9d16ea9a0ef47e`，字节未改变。
- 独立包装器回执：2026-10-10 18:51:51.981 启动，19:57:19.838 退出（UTC+08）；仿真退出码 1，分析未执行。
- 第一组 `HOTSPOT_BASELINE` 的 group runtime 为 3924.123 秒，即 65.40 分钟；在日秒 30600（08:30）失败。
- 第二组 `CHAIN_LAYOUT_BASELINE` 和第三组 `CHAIN_LAYOUT_JOINT` 均未启动，完整全天结果为 **0/3**。
- 47 批、851 个 AV 槽位的完整布局直接复用，没有重做布局。
- 监控采样的进程 RSS 最高约 972 MiB，未观察到内存超限；此值不是 Windows 精确峰值。
- 没有新增情景、修改阈值、使用 GPU、构造稠密全城矩阵、接受未认证 incumbent 或自动重试。

第一组从午夜执行。旧第六次第一组现场仍保存在 `stage4/output/capability_chain_planning/scheme_a_city_v1/stopped_attempts/hotspot_baseline_006`。本次现场未删除或覆盖。

## 已确认的故障与证据边界

真实异常为：

```text
RuntimeError('bundled persistent HiGHS rejected an operation: HighsStatus.kWarning')
```

堆栈经过城市耦合主问题的科学目标循环、`scheme_a_joint_solver._run_milp`，最终位于 `persistent_city_milp.py` 的 `self._checked(h.run())`。不是 `passModel`、`addRows` 或 `setSolution` 返回的警告。

本地 SciPy 的 `_highspy/_highs_wrapper.py` 对 `highs.run()` 仅在返回 `kError` 时立即失败，否则继续读取 `getModelStatus()`。新增持久化接口将所有非 `kOk` 返回值直接抛错，因此还未读取模型状态就退出。实际的模型状态、目标层名称、原始/对偶界和 gap 都没有保存。

**不能把这次包装器异常解释为“已经避免了底层求解超时”。** HiGHS 的调用返回状态与模型最优性状态是两个对象；`kWarning` 不是最优性证明，也不等于不可恢复的 API 错误。固定版本源码中，MIP 求解的返回状态来自模型状态；之后还可能检查最优解数值质量。这里只确认了错误的提前终止／诊断丢失路径，不能凭现有输出恢复具体模型状态。[HiGHS 1.12.0 源码](https://github.com/ERGO-Code/HiGHS/blob/v1.12.0/highs/lp_data/Highs.cpp)

该调用又发生在旧第六次同样的 08:30 时刻。底层仍遇到时间上限是一个合理怀疑，但 **尚未被记录证明**；也不能确认具体是哪一个科学目标层。没有保存本次失败稀疏模型，不能宣称已经精确定位其最优性难点。

### 关于累计时钟：撤回未经路径核对的诊断

监控说明中曾依据通用 HiGHS 时钟文档，推测复用对象后直接设置剩余秒数会重复扣除先前求解时间。进一步核对固定 **1.12.0 的 MIP 路径**发现，`callSolveMip()` 每次构造 `HighsMipSolver`，该对象有自己的计时器，运行时启动并据此判断 MIP 时间上限。因此不能把累计 `getRunTime()` 文档直接套用为本次二进制 MILP 的确定根因，也不应据此给 MIP time_limit 加累计偏移。

这一诊断已撤回；目前没有为它修改代码或预算。[HiGHS 1.12.0 MIP 计时路径](https://github.com/ERGO-Code/HiGHS/blob/v1.12.0/highs/mip/HighsMipSolver.cpp)，[通用时钟文档（不替代固定版本 MIP 路径核对）](https://ergo-code.github.io/HiGHS/dev/interfaces/c_api/#Highs_zeroAllClocks)

## 能说清楚的效率改善：相同前缀，不是完整全天

从两次日志的 08:00 原始 JSON 行取得：

| 项目 | 第六次旧代码 | 第七次优化代码 |
|---|---:|---:|
| group runtime（秒） | 6757.988 | 3031.080 |
| group runtime（分钟） | 112.63 | 50.52 |
| matched | 1750 | 1750 |
| completed | 1561 | 1561 |
| expired | 438 | 438 |
| 当次进度 RSS（MiB） | 688.29 | 950.54 |

观察到前缀耗时缩短 55.15%，旧／新耗时比约 2.23。当天对应的三项服务计数一致，但没有逐订单比较本次所有中间动作。两次不是冷缓存控制基准，旧运行可能填充了共用路由缓存，故不能把全部改善独立归因于某项代码优化，更不能将 2.23 倍作为完整全天或后两组速度。

最后完成的 08:15 进度中，连接检查累计 2219.419 秒、graph/OR 652.929 秒、整数主问题 281.069 秒、链构建复用 2436 次。计时项有嵌套，不能相加；连接处理仍是主要成本。第一组最后距离等难层能否在原 10 秒共同预算内证明最优，仍未闭环。

## 保存的产物

基目录：`stage4/output/capability_chain_planning/scheme_a_city_v1`。

| 文件 | 规模／性质 |
|---|---|
| `execution_attempt_007_process.json` | 真实命令、PID、启动时间及来源提交 |
| `execution_attempt_007_exit.json` | 仿真退出码 1；分析退出码为空；没有自动重试 |
| `execution_attempt_007.log` | 进度行、完整异常堆栈和路由日志 |
| `HOTSPOT_BASELINE/summary.json` | STOPPED、失败时刻 30600、group runtime |
| `HOTSPOT_BASELINE/progress.json` | 最后完成的 08:15 快照；1712 单已完成，不能当全天结果 |
| `HOTSPOT_BASELINE/partial_assignments.parquet` | 2245 行，349533 字节；包含失败前已提交工作，不等于 2245 单已完成 |
| `HOTSPOT_BASELINE/partial_solver_trace.parquet` | 1020 行，184662 字节；未包含失败调用的完整稀疏模型 |
| `layout/placements.json` | 原 47 批完整布局，未改动 |

模拟与包装器进程均已退出。没有运行 `scheme_a_city_results`，没有生成三组比较，也没有为 partial 输出服务率或策略优劣结论。此次提交仅包含公开的停止记录与状态；私人逐订单产物仍不提交到 Git。

## 下一步最小范围（尚未执行）

1. 仅修持久化 `run()` 返回值处理与失败诊断：保留 `kError` 的失败；对 `kOk/kWarning` 读取并记录真正模型状态，仍只接受通过既有认证的 `kOptimal`。时间上限仍必须失败，不使用 incumbent fallback。
2. 用一个针对性接口检查覆盖警告／非最优状态；确认时间上限能到达既有 `DecisionTimeout` 诊断。必要时核对固定版本 MIP 计时行为，不进行参数搜索。
3. 修复后再由作者决定是否原样重跑第一组及顺序完成后两组。没有可精确恢复的完整仿真快照，不能声称可以从 08:30 无损续跑；完成布局可以继续复用。

本轮不自动修改仿真源码或开始第八次执行。研究问题、三组比较、能力、种子、数据、原 10 秒共同决策上限及行政时限保持原样；不增加审计阶段或实验条件。
