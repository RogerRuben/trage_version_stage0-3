# 研究兼容性与前瞻分工：真实小窗口运行协议

2026-10-04；academic-research-suite / experiment-agent。本协议在 native 对照结果产生之前保存。
承接 [method_v1.md](method_v1.md) 的稀疏两阶段下一服务模型，完成普通数据接口连接后直接运行，不新建审计阶段。
不改论文、不重新训练、不重新聚类、不重跑 41 个 canonical 场景。

## 1. 固定范围与比较对象

配置为 `stage4/config/flexibility_windows_v1.json`。
Test31 的 10:30 和 17:30 两个既有物理 checkpoint，均继承 M 场景的同一车辆位置、已接受任务及 availability policy。
每个起点顺序运行 C/M/A × MYOPIC/AV_FIRST/LOOKAHEAD，共 18 次 continuation；同一 profile 内才是三策略的主要配对比较。
这不是三个 profile 各自跑到该时刻后形成的天然车队状态，也不是 frozen canonical 的重现。

每个窗口纳入随后 900 秒释放的全部订单，允许 300 秒接驾耐心，30 秒滚动，并继续物理 draining 到所有已接受任务完成。
窗口末尾停止纳入新需求，但不把未服务订单从分母删除。配对使用相同订单 cohort，分别记录 gained/lost/net change。
保留 source checkpoint 的乘客接受率与 seed、HV session / AV full-horizon availability；不重新选择车队。
本轮 Gamma、operating cost、repositioning 关闭，避免同时改变多个决策层。

## 2. 信控与 movement 输入

只读 20261004 封存版：完整正例并集 **699**，其中 **693** 有地图/高德支持、6 处仅旧来源；没有 HTTP 请求。
位置先尝试既有成员 node ID，再用 35m 内 surface complex 的唯一近邻关联，次近距离差至少 5m；不跨层强制吸收。
实际 56 处按 node identity、571 处按唯一近邻关联，72 处未强制关联；得到 596 个新集合关联的 complex。
新正例与既有 1,629 个 OSM-positive complex 取并集，研究 overlay 的 positive complex 共 2,020 个。
**699 个物理位置与 2,020 个冻结 complex 是不同计数单位。** 输出保留逐位置关联依据，不把 72 处未关联写成无灯。

未有 positive evidence 的 complex 使用显式 `RESEARCH_ASSUMED_NON_SIGNAL_NOT_OBSERVED_ABSENCE` 情景假设；不是实测无灯，也不是 2016 年控制真值。
C 的主规则为信控左转可行、非信控左转/环岛/U-turn 不可行；M 增加非信控左转/环岛，A 再增加 U-turn。
非信控直行/右转不另设 C veto。已知方向不可通行与 certified prohibition 保留。
movement 优先复用 frozen lookup；缺项用 boundary edge 切向 bearing 分类，保留 fallback provenance，不把 lookup 缺失直接解释成不可通行。
Test31 共修复 13,896 次 encounter 的缺失分类，剩余 unresolved maneuver 为 0。

交通环境沿用已有 Train reference 与冻结 M3 prediction 生成的状态：C 的环境外集合为 MR/CG/SC，M 为 SC，A 为空；环境外时间预算 .05。
ML 不追加拥堵 veto，U 只描述，不冒充非拥堵。A/M/D/L、speed、CV/acceleration EQC 仍保留描述字段，不再叠加额外 hard gate。
30,000 单中 29,864 单具有研究输入，C/M/A 兼容量为 5,299/13,847/14,613；这些是离线集合，不是实际可派单量。

## 3. Train-only 前瞻输入

预定采用与 Test31 同为周一的 20161010、17、24 日，共 3,400 条所需时段模板。
只读取订单身份、释放时间、OD 坐标、冻结 M3 预测及研究兼容集。Train 原始端点由 GCJ02 转 WGS84，保留既有转换流程。
不使用 Test31 后续实际请求来采样、选参数或计算未来 ETA；不会从未接受订单读取“未来车辆终点”。

每个 Train 日是一种权重 1/3 的需求情景。未来 600 秒内的模板按固定 seed 做 Poisson 抽样，强度乘数 3 来自 Test 30,000 / Train 每日 10,000 的既定样本配额；**不是 Test31 实际到达量校准**。
释放时间仅加预定 ±15 秒 jitter。场景截止于本次 cohort 的需求纳入终点，避免为不再纳入的窗口外订单保留车辆。
该强度与时间带只是低成本预测近似，未宣称经统计校准或跨日期泛化。

当前接驾仍使用既有 deterministic Valhalla（Top-K20、单来源 matrix）。
后续候选使用 Train M3 P50 / OD 直线距离的 30min 中位 pace 近似接驾 ETA，搜索半径 2km，每个请求取 5 个不同车辆及其可行状态。
它不是精确路由，也不是实现未来拥堵；明确标记 `TRAIN_M3_MEDIAN_OD_CHORD_PACE_BY_30MIN`。
所有策略共享同一当前候选条件与预测服务时间；只有 LOOKAHEAD 使用情景 recourse。

## 4. 时间链与基线区别

真实 replay 的物理服务时间仍由既有环境推进；它只用于完成任务与事后评价，不进入前瞻优化。
已接受任务的下一可用时间由历史 decision-time 接驾 ETA + 一次 overhead + M3 预测服务时间计算，位置为已接受任务终点。
若预测完成早于现在但 native 状态仍 busy，则不能立即当作 idle，最早延到下一 rolling step。
当前 assignment / scenario recourse 均检查接驾 deadline 与显式 availability end；未释放实际订单不进入优化。

三个研究策略都使用 original-route 的冻结 M3 P50；旧 replay 的 NONE/FALLBACK 标记不再决定是否提供该决策预测。
有旧预测时 P50 本来相同；这项 common-input 连接不改 physical realized duration，不宣称与旧 canonical assignment 集合完全一致。
本轮 additional pickup overhead 为 0，不额外重复计停留时间。
继承时已接受的任务继续完成，研究兼容检查只约束新分配，不追溯撤销历史任务。

## 5. 资源与预定异常处理

CPU 单进程、顺序运行；OMP/OpenBLAS 一线程；不使用 GPU、不创建全订单×全车辆稠密矩阵。
MILP 上限 20,000 变量 / 150,000 非零元，每 epoch 10 秒；仅资源上限或未证最优/超时允许退回同一当前 MYOPIC。
退回次数与原因逐 epoch 落盘，不把 fallback 称为 LOOKAHEAD 最优解；数据/物理错误立即停止。
每个 continuation 最多 1,800 秒，2,048 MiB RSS 提醒。分批释放 checkpoint 与 DataFrame；日志报告 RSS、稀疏变量/非零元、路由及 solver 耗时。

最小正确性检查：相同 cohort、接驾耐心、时间链、车辆不重叠、已接受任务 drain、AV route compatibility、profile/M3 checkpoint/config 未改变。
预运行必要回归 **34 passed，2 deselected**；compileall 成功。两个需要全量车队数据的既有测试不在本轮重跑。

## 6. 命令与解释边界

```powershell
$env:FLEETPY_ROOT='D:/pycodes/didi_xian_raw/.external/FleetPy'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
conda run -n stage0-valhalla python -u -m stage4.analysis.flexibility_prepare
conda run -n stage0-valhalla python -u -m stage4.analysis.flexibility_windows `
  --fleetpy-root $env:FLEETPY_ROOT
# 中断后的显式续跑添加 --resume，完整 condition 不重复启动。
```

输入连接记录：[input_connection_summary.json](input_connection_summary.json)。运行器写 `native_window_summary.json`、`native_window_comparison.json` 及小窗口报告。
不以两个探索性窗口宣称统计显著、全天优势、安全认证或方法新颖性；保留可能的不利结果。
只有实际配对结果支持时才讨论该近似前瞻策略的服务价值，预测本身不自动升级为 decision-superior。
