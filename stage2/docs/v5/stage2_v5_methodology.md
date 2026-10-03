# Stage 2 v5 方法与冻结边界

## 定位

Stage 2 v5 是决策时点可用的随机服务状态预测模块。它复用冻结的 Stage 0 地图匹配、Stage 1 标签和 Stage 2 v4 因果路线产品，不重新生产 20161028–30，也不修改 Stage 0、Stage 1 或 v4。

输出表示 traversal/route 的物理 travel time、pace、条件分布、路线内相关场景，以及 RTS/LCS raw 辅助状态；不表示 AV 安全、失效、接管或 ODD 概率。

## 时间协议

开发与选择：

- Train：20161009–20161021
- Validation-model：20161022–20161023
- Calibration：20161024
- Development temporal evaluation：20161025–20161027

结构、特征、loss 和超参数冻结后运行三个 rolling-origin folds：

1. Train 09–18；Validation 19–20；Calibration 21；Evaluation 22–23。
2. Train 09–20；Validation 21–22；Calibration 23；Evaluation 24–25。
3. Train 09–22；Validation 23–24；Calibration 25；Evaluation 26–27。

20161031 是 `legacy_frozen_benchmark`，只用于与冻结 v4 的版本可比性，不称为未见最终 Test。其拟合协议是 Train 09–24、Validation 25–26、Calibration 27、Benchmark 31。

每个协议独立拟合 Train-only normalization、vocabulary、baseline、availability/IPW 和深度模型；校准参数只读对应 Calibration 日期。Stage 2 日期角色与 Stage 1 物理分区严格解耦：09–24 读取 Stage 1 train，25–27 读取 validation，31 读取 test。

## 标签与泄漏边界

主分布目标是冻结 Stage 1 质量门槛确认的 direct `observed_sec_per_m`。完整 traversal 时间由预测 pace × `allocated_distance_m` 派生；插值、interval-supported 和 unresolved 时间不作为 link 主监督，缺失标签保留 NaN 和显式 mask。

Stage 1 RTS/LCS percentile/CDF 使用 09–24 拟合，因此 development 和 rolling folds 将两类 percentile tail mask 置零。模型结构选择和 rolling 科学结论仅使用 pace、物理时间、RTS raw、LCS raw 与路线服务时间。Percentile 只用于 legacy 或描述性报告。

## 模型与工程约束

v5 使用正值 log-normal pace head，输出 mean/P50/P90/P95 和训练得到的 scale。动态头按结构保证 `crawl + stop <= 1`。overlap chunk 使用 `1 / overlap_supervision_count`，使每个物理 traversal 每 epoch 总监督权重为 1。

forecast horizon、history age 与 support 显式控制 recent/profile gate；ordinary concatenation、without recent、without profile 消融只在开发划分运行，不在 rolling folds 重复。

路线场景比较 independent、shared route latent 和 residual block。选择只使用 Validation，scale/dispersion/offset 只使用 Calibration，Evaluation 只做报告。正式输出包含路线 mean/std/P50/P90/P95/CVaR、外部阈值 timeout probability、scenario seed、generator ID 与输入 hash。

生产路径采用按日流式读取、稳定排序、vectorized groupby/NumPy 聚合和批量场景生成。禁止逐订单/逐 edge 全表扫描、`groupby.apply`、axis=1 apply 和循环内 concat。全量 Stage 2 重训前必须通过静态复杂度审计、micro-benchmark、dry run、profile、reference 等价测试和联合回归。
