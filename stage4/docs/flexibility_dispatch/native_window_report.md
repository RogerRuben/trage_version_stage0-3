# 研究兼容性与前瞻分工：真实小窗口结果

## 结论

2026-10-04；academic-research-suite / experiment-agent；**原定 18/18 组 COMPLETE，配对分析 COMPLETE**。
本轮按顺序完成数据连接、两个 native 窗口、结果分析与最小正确性验证，没有因负结果增加实验或调整参数。
**新 Stage3 研究兼容性接口已接通；当前 LOOKAHEAD 近似不具备替代 MYOPIC 的实证支持。**
AV_FIRST 仅产生很小的净服务增益，更多作用是改变 HV/AV 分工；不能据此宣称稳定、显著或全天优势。
不修改论文，不覆盖冻结 profile / M3 checkpoint / 41 个 canonical 场景。

运行前代码与配置提交：`dcdb39e`；完整设定见 [native_window_protocol.md](native_window_protocol.md)。
输出为 `stage4/output/flexibility_dispatch_v1/`；逐条件产品保留在本地，不上传订单明细。

## 1. 数据连接实际完成了什么

信号输入使用封存 **699 处完整正例并集 / 693 处地图高德支持 / 6 处旧来源**，HTTP 调用为 0。
56 处按 node identity、571 处按唯一近邻关联，72 处未强行关联；627 个已关联物理位置对应 596 个 frozen complex。
与既有 1,629 个 OSM-positive complex 取并集后，overlay 共 2,020 个 positive complex；没有重新 clustering。
物理位置与 complex 不同单位，不能把 699、596、2,020 当作相同对象比较。

未列为 positive 的 control 使用显式研究假设，不写成“实测无灯”；2026 补采也不写成 2016 现场真值。
缺失 movement 复用 boundary 切向 bearing 分类，Test31 共有 13,896 次 fallback、0 次 unresolved maneuver，保留来源记录。
已知 reverse / unresolved route identity 检查仍然保留；**消除 movement lookup 缺项，不等于所有路线身份问题都消失。**

30,000 单中 29,864 单具备 movement/traffic 分类输入，C/M/A 兼容集合分别为 5,299/13,847/14,613。
该 data-ready 计数不表示路线全部通过 direction check；兼容性嵌套 \(K_C\subseteq K_M\subseteq K_A\) 已检查。
Train-only 需求库来自 20161010/17/24 三个周一，模板数 1,062/1,113/1,225，共 3,400；未拟合 Test31 后续需求，未重新推理或训练 M3。
详细输入来源：[input_connection_summary.json](input_connection_summary.json)。

## 2. 真实 native 配对结果

两个共同 M 物理起点：10:30、17:30；每次纳入随后 15min 全部请求，30s rolling，300s 接驾耐心，并 drain 已接受任务。
上午 379 单、晚高峰 419 单，共 **798 个不同订单**；三种 profile 是同一 cohort 的研究情景，不是三份独立样本。
三策略共享 availability policy、乘客接受、原始服务路线与 M3 P50；Gamma、cost、repositioning 本轮关闭。
每个 profile 内的三策略才是主要配对比较；继承的旧任务继续完成，但不计入新 cohort 的服务数。

| 窗口 | Profile | Cohort | MYOPIC | AV_FIRST | LOOKAHEAD | AV_FIRST 净变化 | LOOKAHEAD 净变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| 10:30 | C | 379 | 254 | 255 | 251 | +1 | −3 |
| 10:30 | M | 379 | 267 | 267 | 264 | 0 | −3 |
| 10:30 | A | 379 | 268 | 268 | 267 | 0 | −1 |
| 17:30 | C | 419 | 200 | 201 | 191 | +1 | −9 |
| 17:30 | M | 419 | 225 | 226 | 212 | +1 | −13 |
| 17:30 | A | 419 | 228 | 228 | 217 | 0 | −11 |

| Profile：两个窗口合计 | MYOPIC 服务数/率 | AV_FIRST 服务数/率 | LOOKAHEAD 服务数/率 | LOOKAHEAD 率变化 |
|---|---:|---:|---:|---:|
| C | 454 / 56.89% | 456 / 57.14% | 442 / 55.39% | −1.50pp |
| M | 492 / 61.65% | 493 / 61.78% | 476 / 59.65% | −2.01pp |
| A | 496 / 62.16% | 496 / 62.16% | 484 / 60.65% | −1.50pp |

AV_FIRST 的 AV 服务量，C 为 23→28、M 为 86→103、A 为 90→106；对应总服务净变化仅 +2/+1/0。
因此“更多使用 AV”和“增加总服务”不能互相替代。
LOOKAHEAD 各 profile 合并 gained/lost 分别为 51/63、39/55、45/57；它改变了实际服务身份与动态轨迹，而不只是留下相同分配结果。
完整逐组 AV/HV、等待及配对 gained/lost 见 [native_window_analysis.json](native_window_analysis.json)。

## 3. 如何解释负结果

LOOKAHEAD 在六个 window×profile 配对中均低于 MYOPIC。本批结果不能支持“加前瞻就提高服务”，也不能被小实例的有利案例覆盖。
所有组 routing failure、资源回退均为 0；模型没有因为资源上限或未证最优而悄悄退回基线。
这排除了**本轮记录到的这些工程失败**作为损失解释，但不等于已经识别了具体因果来源，也不等于证明所有模型假设都合理。

前瞻的以下近似均应保留在解释中，而不是事后调整：

- 三个历史 Train 周一 + Poisson resampling，不是已校准的 Test31 到达预测；乘数 3 来自既定样本配额。
- future pickup ETA 使用预测 OD chord pace，不是当前 Valhalla 的精确 future routing；future compatibility 来自历史模板。
- 每车每情景最多一次下一服务，不是完整多服务 stochastic VRP。
- busy 状态沿用 booked M3 P50；若预测完成已过但仍 observed busy，只延至下一步，未建立条件剩余服务时间分布。

这些是潜在误差来源，**本轮没有进一步分解证明哪一个主导了负收益**。
只出现 4 个 LOOKAHEAD epoch“已有 current arc 却选择 0 单”，其余策略为 0；这不足以把全部损失归因于完全不派单，也不是同状态 maximum-matching 分解。
已服务订单的均值等待受服务对象组成影响，例如晚高峰 M 的 LOOKAHEAD 等待略低却少服务 13 单，不能称作等待改善而忽略流失。
只检查两个窗口，不报显著性、不将多 profile 当作独立重复，也不推广为全日、跨日期或真实车型的安全结论。

## 4. Stage3 与 Stage4 的当前定位

Stage3 主表达可以保留为**显式 maneuver/control + 预测交通环境的研究兼容集**，而不是第四种能力 UNKNOWN 或一串不透明 veto。
同一 cohort 的兼容量，上午 C/M/A 为 63/170/177，晚高峰为 15/201/213；C 场景在所选时段的路线覆盖差异很大。
但道路控制与交通是联合规则，不能仅凭这些计数把差异全部归因于拥堵，更不能写成实测安全边界。
未观测控制属性仍是有来源标记的情景假设，接驾路径没有逐段 C/M/A 认证。

Stage2 的 M3 没有被抛弃：它进入当前服务完工/session 条件、busy 后续状态、交通状态和历史前瞻 ETA。
**使用了预测，并不自动建立 prediction 的决策优越性。** 当前保持 decision-relevant，不升级为 decision-superior。
A/M/D/L、speed、CV/acceleration 描述与旧预算接口保留；本轮关闭连续预算/成本，是为了隔离分工模型，而不是永久删除这些特征。

Stage4 以 MYOPIC 保留参照；AV_FIRST 可作为简洁分工对照，LOOKAHEAD 保持独立可切换研究原型，暂不晋升主策略。
如后续继续研究，应先明确前瞻输入/状态近似的改进假设，不把下一步变成参数搜索或更多 gate。
**本任务在当前预定批次与分析结束；未擅自追加实验、重训、重规划或修改论文。**

## 5. 最小正确性与资源

18/18 condition 写出产品；全部 cohort 订单有 matched/expired 归属，配对身份一致。
接驾耐心、实际时间链、车辆任务不重叠、请求/车辆/位置状态、AV availability 与新 AV 的研究路线兼容性核对通过；所有 accepted task 已完成 draining。
配置、frozen profile 和 M3 checkpoint 的运行前后 SHA 一致。
这些是实施正确性检查，不作为统计效果或科学有效性的 PASS 代用品。

运行耗时 **1,972.36s（32.87min）**；native 进程峰值 **668.19 MiB**；单进程 CPU，无 GPU、无 request×vehicle 稠密矩阵。
最大当前 Top-K pairs **1,986**，最大有效当前 arc **72**；最大成功模型 **6,413 变量 / 29,410 非零元**，低于预定 20,000/150,000。
实际 routing arc evaluations **821,025**，failure **0**；资源/超时 fallback **0**。
epoch 耗时汇总：候选生成 115.91s、当前路由 1,340.35s、控制/研究 adapter 425.73s，其余约 90.37s。
控制耗时包含 forecast、模型构造与求解，不能全部叫 HiGHS 时间；当前路由约占总 wall time 68%，不需要 GPU 或大矩阵优化。
峰值为该串行进程的 lifetime peak，不是 18 组分别隔离测量的峰值；数据准备是另一进程，不包含在该 wall/RSS 统计中。

最终相关回归：**34 passed，2 deselected**；compileall 与 diff check 成功。未重跑两项全车队/session 数据测试，未扩展 full-day benchmark。

## 6. 文件与复现

- [运行协议](native_window_protocol.md)、[输入连接](input_connection_summary.json)。
- [完整运行汇总](native_window_summary.json)、[配对计数](native_window_comparison.json)、[结果分析](native_window_analysis.json)。
- 数学定义与历史小实例：[method_v1.md](method_v1.md)、[test_report.md](test_report.md)。
- 本地 `stage4/output/flexibility_dispatch_v1/input/`：信号关联、control overlay、30,000 单 sidecar、Train templates。
- 本地 `.../windows/<profile>_<cut>_<policy>/`：assignments、cohort_outcomes、epochs、solver_trace、summary。

```powershell
# 只重新汇总既有结果，不触发仿真。
conda run -n stage0-valhalla python -m stage4.analysis.flexibility_analysis
```

科学分类：`BOUNDED_EXPLORATORY_NATIVE_COMPARISON`；当前 LOOKAHEAD promotion：`NOT_SUPPORTED_BY_THIS_BATCH`。
