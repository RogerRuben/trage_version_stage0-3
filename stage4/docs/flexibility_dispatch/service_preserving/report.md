# 保护当前服务的前瞻：M 两窗口完成报告

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / run + validate。
- 日期：2026-10-04；版本：service_preserving_v1。
- 执行状态：`2/2 COMPLETE`；解释状态：`ANALYZED`，不是独立重跑复现认证。
- 运行前提交：`2efae48`；旧18条件基线：`57636ec`。
- 仅本次用户授权的三步：诊断、独立策略及Train剩余时长、M的两个原定窗口。

## 1. 结论

**当前服务保护 + Train条件剩余时长的联合策略，在两个M开发窗口都增加了服务数。**
798个不同cohort订单中，MYOPIC为492单，新策略为501单，净增9单，服务率61.65%→62.78%，增加1.13个百分点。
82轮当前决策最优面保护全部通过，未发生路由失败或资源回退。
等待时间有所增加，不能称为所有指标全面改善；仅两个窗口，也不证明全天、跨日期、C/A或真实部署优势。
按协议到此停止，不追加实验、调参、重训、reroute或论文修改，不替换41个canonical场景。

## 2. 三步工作完成情况

1. **单次诊断**：[diagnostics.md](diagnostics.md)。区分共同direction/identity、movement/control、traffic规则的作用，并核对Train与replay时长口径。
2. **独立策略**：`SERVICE_PRESERVING_LOOKAHEAD`。先保留critical、总服务数、carry-over的最优值，再以预期下一服务数选车辆分配，最后最小接驾时间。旧LOOKAHEAD与MYOPIC实现保留。
3. **唯一两个新增native条件**：M、10:30和17:30，原checkpoint及15min cohort，30s rolling、300s patience。六份旧reference直接复用。

Train剩余时长使用20161010/17/24已有3,400个预测—观察配对。条件经验分布不读取Test31实际结束时间，不重新推理或训练M3。
current admission与新任务M3时长不修改；仅估计已接受、仍busy任务的未来可用状态。无经验尾部支持时只省略prospective state，不取消实际任务。
运行前完整定义：[protocol.md](protocol.md)；模型：[train_remaining_time_model.json](train_remaining_time_model.json)。

## 3. 配对结果

| 窗口 | Cohort | MYOPIC | AV_FIRST（复用） | 旧LOOKAHEAD（复用） | 新策略 | 新策略vs MYOPIC |
|---|---:|---:|---:|---:|---:|---:|
| 10:30 | 379 | 267 | 267 | 264 | 273 | +6 |
| 17:30 | 419 | 225 | 226 | 212 | 228 | +3 |
| 合计 | 798 | 492 | 493 | 476 | 501 | +9 |

相对MYOPIC，上午gained/lost为15/9，晚高峰为11/8；合计26/17。
所以“保护当前数量”不是保护每一个订单身份，也不是没有任何个体订单损失。
相对旧LOOKAHEAD，服务数由476→501，增加25；这是目标函数与剩余时长适配的联合变化，不能分别声称某一模块带来25单。

| 分工 | MYOPIC | 新策略 | 净变化 |
|---|---:|---:|---:|
| AV | 86 | 87 | +1 |
| HV | 406 | 414 | +8 |
| Total | 492 | 501 | +9 |

这不是通过“大量提高AV使用率”获得的结果。更合理的解释是当前服务优先面内的时空分配有所改善；尚未识别所有动态机制的独立贡献。

## 4. 必须保留的等待权衡

| 窗口 | MYOPIC已服务订单平均等待 | 新策略已服务订单平均等待 | 两策略共同服务订单的平均差 |
|---|---:|---:|---:|
| 10:30 | 178.24s | 183.08s | +5.71s（258单） |
| 17:30 | 182.00s | 187.77s | +2.53s（217单） |

前两列服务对象不同，不能直接解释为同订单处理效应；共同服务子集也属于事后选择，只作描述。
本策略以服务数量和既定优先级为前层目标、接驾时间为后层目标，因而可能接受这种权衡。不为降低等待临时加权或重新选择结果。

## 5. 实施核对与资源

- 当前lexicographic最优面保护：82/82。实际总选择数逐轮等于同状态MYOPIC最优服务数；critical/carry层也在执行时断言相等。
- cohort归属：379=273+106；419=228+191。全部matched/expired对账，订单身份与旧reference一致。
- native实际时间链、车辆任务不重叠、请求/位置/车辆状态、AV availability、新分配AV路线兼容性及accepted task draining通过。
- 原配置、profile、M3 checkpoint、Train模板、route sidecar、共同物理checkpoint与旧reference产品运行前后SHA不变。
- 两个窗口运行分别91.27s和125.69s；总217.25s（约3.62min）。单进程native lifetime峰值618.36MiB，无GPU。
- 最大当前Top-K pairs=1,799，最大有效current arcs=72；最大成功模型6,003变量/27,508非零元，未构造稠密订单×车辆矩阵。
- routing failure=0；resource/timeout fallback=0。不是因为工程回退才得到与基线接近的结果。
- 上午32个busy vehicle×epoch状态因无经验尾部支持而省略，晚高峰为0；不是32个订单被删除或32辆车提前结束任务。
- 条件可用时刻相对旧近似的均值顺延约315.86/316.25秒；重复状态计数不能当作唯一车辆数，顺延也不是预测准确性的证明。
- 单次诊断进程峰值961.21MiB，与native是不同进程，不能将二者相加作为并发内存。

最小回归：28 passed，2 deselected（原全车队/session数据测试不在本次改动范围）；compileall、diff check通过。
这些核对证明产品与实现按定义运行，不替代策略有效性或统计显著性论证。

## 6. 研究定位与停止条件

分类：`BOUNDED_WINDOW_IMPROVEMENT`。
本次支持继续把“保护当前服务的灵活性分配”作为候选研究方法，不支持恢复旧的、不保护当前服务的LOOKAHEAD主张。
原负结果完整保留；Test31已用于方法开发和比较，不包装为未见测试。
全日、C/A、其他日期、连续Gamma与cost共同启用均未验证，不能从这两个M窗口外推。
经验时长模型拟合的是Train GPS观察窗代理，不是独立认证的车门/周转时长。逐traversal中位数求和仍不能称作joint route median。
新增策略和Train时长同时变动，本次不增加factorial ablation；到此停止，后续工作另行授权。

11/11统计解释风险已检查。主要限定是窗口选择、共用cohort、成功服务子集、事后方法开发和联合改动归因；无显著性检验或“全天胜出”结论。

## 7. 可打开的交付文件

- [单次诊断](diagnostics.md)、[数字](diagnostics.json)、[执行协议](protocol.md)。
- [配对分析](analysis.json)、[运行汇总与输入来源](summary.json)、[Train剩余时长模型](train_remaining_time_model.json)。
- 本地明细：`stage4/output/flexibility_service_preserving_v1/`，每窗口含assignments、cohort_outcomes、epochs、solver_trace及summary；不上传订单明细。
- 核心实现：`stage4/dispatch/flexibility_model.py`、`flexibility_native.py`、`remaining_time.py`。
- 诊断入口：`python -m stage4.analysis.flexibility_diagnostics`，执行开始后禁止重新拟合该批次模型。
- 执行入口：`python -m stage4.analysis.flexibility_service_preserving --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy`；已有完成结果不重复启动。

预注册与结果分开提交，冻结基线保留，本轮三步全部完成。
