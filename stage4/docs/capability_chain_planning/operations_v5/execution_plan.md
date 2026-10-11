# v5 C运营：一次固定全天对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；run。
- Origin Date: 2026-10-11。
- Verification Status: PREREGISTERED_BEFORE_EXECUTION；此页不含新实验结果。
- 代码准备基线：`e76c337ad213779c2c66e2146b7b78e67610cef8`。实际生产commit由运行入口写入`input_binding.json`。
- 作者最新指令“完成下一步”承接上一批已完成的三项代码修改；不扩大为参数网格、三组重跑或M/A敏感性。

## 比较对象

只执行v5的`CHAIN_LAYOUT_BASELINE`：已有联合初始布局 + 保持当轮服务的四层主目标。复用v4同名组的完整结果，不重跑基线。

| 内容 | 固定值 / 来源 |
|---|---|
| 日期与订单 | 20161031；相同28,367单公共高质量队列 |
| 能力与乘客假设 | C；接受率0.7；冻结Stage3路线及接口规则不变 |
| 供给与布局 | 原逐小时上线窗口及47批完整布局；AV计划车时为基线车时约0.10 |
| 等待语义 | 300秒内接单；预测接驾不超过600秒；接单后不再因等待取消 |
| 决策规模 | 原30秒滚动、稀疏Top-K、原候选半径；不构建订单×车辆稠密矩阵 |
| 资源预算 | 单个原生子进程；实时派单10秒；可选调位10秒；整组10,800秒；原RSS/启动内存限制不变；无GPU |
| 实际变化 | AV按类型兼容历史需求调位；当前主最优流面内最小化空驶；新增最终合法候选/接口阶段诊断 |
| 历史信息 | HV原20161009–24的16天需求参考；AV原20161010/17/24三个周一模板；不读取Test31未来订单 |

执行命令（工作目录为当前worktree）：

```powershell
D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.scheme_a_two_scale_experiment --root . --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --config stage4/config/scheme_a_two_scale_city_v5.json --group CHAIN_LAYOUT_BASELINE
```

仅新增行政单组选择入口；默认旧三组顺序不改变。相同输出目录不可换组扩展，失败产物不覆盖、不自动重试。异常时保留现场并报告。

## 预定输出与停止点

新原生产品仅写入`stage4/output/capability_chain_planning/two_scale_city_v5/CHAIN_LAYOUT_BASELINE/`。报告写入本目录，保留v4文档和落盘产品。

比较服务数/率、接单和接驾等待、接驾与调位空驶、C有效班次内载客/接驾/移动/空闲车时，以及已存在最终合法C候选后由HV获单的原因。保留全部订单分母，区别班次内与尾部执行车时。

同时报告实际墙钟时间、RSS、候选/路由规模和次级求解完成情况。没有基线旧诊断的字段明确为未观测，不用“附近有C”代替“存在合法C候选”。

这是一次单日期、单固定种子、两项运营变化的组合比较；不能从收益或损失独立归因到其中一项，不能声称全天最优或推广性显著改善。次级求解的数学保证只限当轮已准入候选图。

完成一次后分析、提交、推送并停止；不因结果不理想调参数，不自动增加实验。
