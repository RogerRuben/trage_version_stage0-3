# 稀疏两阶段分工模型：小实例执行报告

## Material Passport

2026-10-04；academic-research-suite / experiment-agent；状态 COMPLETE（小实例范围）。
这轮检验 OR 实现与反例，不检验真实全日服务提升，没有 p-value、显著性或车型安全结论。

## 本轮实际完成

Stage3 明确 C/M/A 兼容集；Stage4 实现三个独立、可切换的策略和一次后续服务模型。
旧冻结 solver、模型 checkpoint、profile 和已有仿真结果均未修改。
新增数据记录使用 **699 处完整正例并集 / 693 处地图高德支持**的 20261004 封存版；manifest SHA256：
`f7d7e4aabfdb5304997497421868466e29e402d67c581b5ea679a4e7c467d454`。

**没有运行 native FleetPy 真实动态窗口。** 699 处位置到冻结 complex/movement 的正式关联与 Train-only 后续需求输入仍待接入；不能把本轮小实例称为真实数据效果。

## 固定小实例结果

各实例均有 1 辆 HV、1 辆 C-profile AV；当前服务一次，随后最多每车服务一次。
评分使用共同的 ex-post 下一服务最优延续，当前动作不能改写。

| 构造情形 | MYOPIC 服务数 | AV_FIRST 服务数 | LOOKAHEAD 服务数 |
|---|---:|---:|---:|
| 后续订单依赖 HV，当前订单两车均兼容 | 1 | 2 | 2 |
| 后续订单需要附近的 AV，HV 接驾太远 | 2 | 1 | 2 |
| 预测后续依赖 HV，但实际后续需要附近 AV | 2 | 1 | 1 |

第一行证明保留更灵活车辆具有可能的服务价值；第二行说明路线兼容性不能取代时间/空间条件。
第三行是主动保留的不利例子：前瞻没有无条件支配即时策略。
这些是**存在性示例**，不是从真实 replay 中抽样估计的收益大小。

另外，两个后续情形概率为 .75/.25 时，前瞻选择当前 AV，期望总服务 1.75；交换概率后选择当前 HV。
全部场景共用一个当前动作，未按未来情形分别选择一个事后动作。

## 数学与回归验证

首轮 focused tests：`16 passed`，1.99 秒。pytest 默认缓存因目录权限产生一次非测试失败 warning；测试和 compileall 均成功。
最终相关回归：**32 passed，2 deselected，1.02 秒**，关闭 pytest 缓存后无 warning。
包括新模型、原交通策略、原 ODD budget/cost kernel 和 sparse rolling baseline；两个会读取真实车队/session 的测试明确不运行，避免把本轮扩大为数据实验。
新增 Python 文件 `compileall` 成功；提交前 staged `diff --check` 成功。没有运行全量数据仿真。

覆盖的是独立穷举最优值、嵌套兼容集、预测信息时间、busy/available、一次 overhead、接驾 deadline、session completion、已知请求不重复、资源上限及 post-service origin 一致性。

## 执行资源与可复现命令

固定小实例脚本：0.070 秒，进程峰值 **77.85 MiB**；未使用 GPU，未创建订单×车辆稠密矩阵。
具体变量数、非零元数与解保存在 [small_instances_result.json](small_instances_result.json)。每个实例只有个位数变量，独立于已有大数据产品。

```powershell
conda run -n stage0-valhalla python -m pytest -p no:cacheprovider `
  stage4/tests/test_flexibility_model.py `
  stage4/tests/test_traffic_research_policy.py -q

# 扩展既有 FleetPy 相关模块回归时，先指定已安装的本地库。
$env:FLEETPY_ROOT='D:/pycodes/didi_xian_raw/.external/FleetPy'
conda run -n stage0-valhalla python -m pytest -p no:cacheprovider `
  stage4/tests/test_flexibility_model.py `
  stage4/tests/test_traffic_research_policy.py `
  stage4/tests/test_odd_aware_decision_kernel.py `
  stage4/tests/test_rolling_or_baseline.py -q `
  -k 'not active_vehicle_hour_accounting and not av_full_horizon_and_hv_session'

conda run -n stage0-valhalla python -m stage4.analysis.flexibility_small_instances `
  --signal-release D:/pycodes/didi_xian_raw/map_data/signal_gapfill_20261004
```

输入与例子固定；耗时和进程内存随环境变化，离散选择与得分可重现。
脚本记录正例表的行数、唯一身份与摘要 SHA，不复制完整采集 evidence，不再启动 HTTP。

## 对研究主线的含义

下一层数学应围绕**兼容性稀缺与车辆时间/位置机会成本**，不能只比较进一步删边的比例。
Stage2 预测服务时间已经进入完成/再次可用时间；需求预测的不准确性则必须和策略价值一起解释。
本轮尚不能将 prediction 定位升级为 decision-superior，也不以这些手工例子证明实际 Stage4 效果提高。
无需新增一轮 audit/gate：后续只做数据接口连接和原定小窗口对照。
