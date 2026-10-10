# 交通状态解释层 v1

本层回答“路线的预测交通环境是什么、哪些部分证据不足”，不回答“是否允许派 AV”。
它是 profile-neutral 的附加产品，不能被当作一组新的独立风险约束。

## 同源信息只约束一次

| 信息 | 本轮用途 | 不允许的重复使用 |
|---|---|---|
| M3 crawl/stop → 现有 E/Q/C | 保留冻结的动态 envelope 与现有决策逻辑 | 在已有约束外再叠加拥堵门控、惩罚或成本 |
| M3 pace/crawl/stop → 五类交通状态 | 解释拥堵/严重拥堵的预测时间占比和连续时长 | 把与 E/Q/C 的一致性称为独立验证或双重证据 |
| speed_cv / acceleration_rms | 保留原维度，不能由拥堵类别代替 | 宣称拥堵状态已涵盖所有动态行为 |
| speed-domain / 静态结构 / 方向 | 保持原语义与决策逻辑 | 把拥堵等同于结构不可通行或法定限速违规 |

因此本轮不生成 U_C/U_M 门槛、不增加 rho、不更改 hard_state、不追加决策 reason_codes，
也不选择“拥堵暴露预算”。C/M/A 在同一路线上看到相同交通解释；profile 差异仍由现有 envelope 表达。
之前“C 非拥堵、M 加拥堵、A 加严重拥堵”仍只是未部署的情景假设，而非已证实的 AV 能力。
若未来测试替代定义，应明确是替代消融，不能默认与同源 crawl/stop 约束同时叠加。

## 产品与接口

本次仅导出已有 Validation 20161025–27 的完整动态历史路线：
`stage3/output/traffic_explanation_v1/date=YYYYMMDD.parquet`。
没有读取 Test31，没有重新推理、训练、CDF 拟合或派单；冻结的 Stage3→Stage4 主表未写入。

键：`date + order_id + selected_route_reference`；当前仅支持 `ORIGINAL:<order_id>`。
`attach_explanations` 是显式调用的附加函数，目前未接入生产 evaluator/solver。
调用者必须使用同一冻结历史路线来源，不能拿重算后的同名 ORIGINAL 路线冒充旧路线；
汇总文件绑定本次来源文件 SHA。FALLBACK、日期不匹配或无记录时状态为
`UNAVAILABLE_FOR_SELECTED_ROUTE`，暴露值为空，绝不能填 0。

主要附加字段（统一 `traffic_` 前缀）：

- `non_congested_proxy_share / congested_proxy_share / severe_congestion_proxy_share`
- `mixed_evidence_share / unknown_share`：和上面三项构成总和为 1 的完整分区。
- `total_time_s`：冻结 M3 预测 P50 路线时间，所有占比均使用该分母。
- `longest_congested_run_s`：拥堵或严重拥堵连续预测时长，未知与路线序列断点会中断连续段。
- `support_basis`：Train directed-edge reference 至少 100 order-days / 5 dates。
- `current_window_support_assessed=false`：这里不是实时观测窗口支持度认证。
- `explanation_status / explanation_codes / source_family / use_policy / explanation_version`。

独立 `traffic_explanation_codes` 只做描述，不与 hard/unknown/soft 决策原因合并。
保留原输入的全部列、行顺序及索引；拒绝覆盖任何已有字段。

## 必须随产品保留的解释边界

状态是 decision-time M3 预测的 proxy，不是真实拥堵标签；参考为 Train 低 pace 分位，
不是真实自由流速度或标准 TTI。参考支持度不是实时交通可观测性、预测置信度或安全概率。
未知占比保留在总分母内，不能删掉后重新归一化；mixed 也不能强行并入三类之一。
本表不包含已完成行程的 `obs_*`、direct-observation coverage 或 realized target，
避免将回顾性局部测量当成提前派单可用信息。

现有 batch2 的相关性仍只是同源描述性结果。新表没有带来新的预测信息，不能据此宣称
预测性能提高、AV 安全性提高或服务率改善。加入解释列前后决策字段相同不等于重跑了仿真；
本轮没有重跑仿真。

## 复现与最小 QA

在 worktree 根目录执行：

```powershell
D:/anaconda/envs/stage0-valhalla/python.exe -m stage3.scripts.build_traffic_explanations
D:/anaconda/envs/stage0-valhalla/python.exe -m pytest -q -p no:cacheprovider stage3/tests_odd_tod/test_traffic_explanation.py
```

每日窄列读取并写出，无跨日期 token 累积、无全路网×时间稠密矩阵、无 GPU。
构建时检查暴露分区、冻结 profile 不变、来源文件不变，逐行验证原 C/M/A 动态 rho 未变。
单元测试检查原 hard/rho/EQC/reason 保留、fallback/缺失不继承、重复键及非法暴露拒绝。
运行结果、输入与输出 SHA、抽样进程 RSS 见同目录 `summary.json`。
