# Stage 2 v5 方法与冻结边界

## 定位

Stage 2 v5 是决策时点可用的随机服务状态预测模块。它从冻结的 `stage2-v4-final` 派生，不修改 Stage 0、Stage 1 或 Stage 2 v4。输出表示 traversal/route 的运行时间、pace、RTS/LCS 辅助状态及其不确定性；它不表示 AV 安全、失效、接管或 ODD 概率。

## 科学评估协议

模型拟合前已提交 `stage2_v5_split_freeze.json`：

- Train：20161009–20161024
- 模型选择：20161025–20161026
- 校准：20161027
- 新 final test：20161028–20161030，只允许运行一次
- 20161031：仅为已公开结果的 frozen legacy benchmark

10 月 28–30 日原始归档存在，但尚无冻结的 Stage 0/1 产品。生成这些产品时必须复用冻结的上游逻辑与质量门槛，不能根据 final test 修改特征、参数、baseline 或校准。

## 物理服务时间目标

全量审计确认，`travel_time_s` 与 `observed_travel_time_s` 是同一 direct GPS interval 时间的副本，不是每次 traversal 的完整 edge 实现时间。冻结数据也没有启用 engine 时间分配。因此：

- 主分布目标为 direct `observed_sec_per_m`；
- 完整 traversal 时间由预测 pace × `allocated_distance_m` 派生；
- direct time 仅用于高直接距离覆盖率的辅助/敏感性分析；
- interpolated/interval-supported/unresolved token 不作为 link 时间监督；
- 缺失标签保持 NaN 并配显式 mask，绝不填 0；
- `time_observation_valid` 不能单独使用，必须联合 `measurement_source == direct_observed`、正时间和正距离。

冻结样本中 15,649,455 个 route token 有 5,309,097 个通过物理窗口质量门槛的 direct pace（33.93%）；在 5,775,530 个有 direct 时间的 traversal 内覆盖 91.92%。另有 453,737 个 pace 虽可由正时间/正距离机械相除，但因时间不足、距离不足、不连续或速度不可能而不进入监督。该冻结上游质量门槛消除了最高 250,000 s/m 的退化短距离比值；正式监督 pace 的最大值为 17.33 s/m，P99 为 0.297 s/m。

## 模型结构

v5 使用正值 log-normal pace head，明确训练 `pace_log_scale`，并直接导出 mean/P50/P90/P95。旧版未进入 loss 的 LCS/RTS scale 不再输出。

动态状态约束为：

```text
stop = sigmoid(presence) * sigmoid(positive)
crawl = (1 - stop) * sigmoid(crawl_logit)
```

由结构保证 `crawl + stop <= 1`。Stop 评估拆分为 occurrence 与 positive share。

forecast horizon、history age 和 support 共同控制 recent/profile gate。horizon 和 history age 对 recent 权重具有结构性非增约束，support 对 recent 权重具有非减约束；ordinary concatenation、移除 recent、移除 profile 都是正式消融。

## 重叠监督与选择性观测

每个物理 traversal token 计算 `overlap_supervision_count`，所有 loss 使用 `1 / overlap_supervision_count`。同一 token 在一个 epoch 内总权重严格为 1。

标签可用性作为独立 head。正式比较 complete-case、component-mask 与 stabilized IPW：

```text
w = M * mean(q) / max(q, 0.05)
maximum weight = 10
```

必须报告 availability AUC/Brier、权重分位数、clipping 数和 ESS。IPW 不稳定或无改善时保留 component-mask 主结果，并将 IPW 作为敏感性结果。

## 路线状态聚合

路线均值、平均 tail 概率、条件 tail 严重度、加权 tail persistence、最长连续 tail 权重占比和加权 coverage 分开计算。无 tail 时 conditional severity 为 NaN，`tail_event_present=false`。LCS 使用预测 traversal 时间加权；RTS 使用路线距离加权。

## 联合场景

正式候选包括 independent、shared route latent shock 与 residual block。所有模式保留相同 traversal 边缘 log-normal 分布；路线时间严格等于 traversal scenario 求和。输出 mean/std/P50/P90/P95/CVaR90/CVaR95，以及针对外部服务时间阈值的 timeout probability。seed、generator ID 与输入 hash 都进入 provenance。

## 性能纪律

生产路径采用分区流式读取、单次稳定排序、factorize/index lookup、NumPy `bincount`/`add.at` 和批量场景生成。逐订单/逐 edge 全表扫描、`groupby.apply`、`axis=1 apply`、循环 concat 均为阻断缺陷。reference 实现只用于小样本数值等价测试。

全量 fit/transform/evaluate 只有在静态审计、micro-benchmark、分层 dry run、profile、reference 等价和内存检查共同 PASS 后才允许启动。
