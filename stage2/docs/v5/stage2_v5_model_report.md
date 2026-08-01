# Stage 2 v5 模型报告

## 开发集结论

科学状态由代码根据同集指标和 paired bootstrap 计算为：`PREDICTIVE_BASELINE_VALIDATED`。

RC-MSTNet v5 的 20161025–26 加权 pace MAE 为 **0.028951 s/m**，强 HistGradientBoosting baseline 为 **0.030140 s/m**，相对改善 **3.94%**。两天的 order-cluster bootstrap 中，`v5 − tree` 绝对误差差值及 95% CI 分别为：

- 20161025：-0.001330，CI [-0.001406, -0.001258]
- 20161026：-0.001048，CI [-0.001159, -0.000874]

10 月 28–30 日 final test 尚未读取。

## 点预测

| Date | Model | N | MAE | RMSE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| 20161025 | RC-MSTNet v5 | 237,553 | 0.02903 | 0.05369 | 0.6442 | 0.7332 |
| 20161025 | Tree | 237,553 | 0.03036 | 0.05783 | 0.5643 | 0.7111 |
| 20161026 | RC-MSTNet v5 | 238,711 | 0.02887 | 0.07118 | 0.4121 | 0.7285 |
| 20161026 | Tree | 238,711 | 0.02992 | 0.06273 | 0.5320 | 0.7090 |
| 20161027 calibration | RC-MSTNet v5 | 237,196 | 0.02830 | 0.05139 | 0.6189 | 0.7361 |

v5 的 MAE 和 rank correlation 稳定改善；20161026 RMSE 比 tree 差，说明少量大误差仍存在，必须作为非阻断风险保留，不能只汇报 MAE。

## Pace 分位数校准

| Date | P50 coverage | P90 coverage | P95 coverage |
|---|---:|---:|---:|
| 20161025 | 0.5050 | 0.9002 | 0.9474 |
| 20161026 | 0.5075 | 0.8994 | 0.9473 |
| 20161027 | 0.5012 | 0.9008 | 0.9488 |

log-normal scale 直接进入 NLL 训练；旧版未训练的 LCS/RTS scale 已完全移除。P50≤P90≤P95 与正 pace 在 merge verify 中逐行检查。

## Availability 与 IPW 敏感性

service-time availability prevalence 约 0.339，AUC 为 0.962–0.963，Brier 为 0.072 左右。稳定化 IPW 无 clipping，P99 权重约 4.9–5.1，但 ESS 只有约 89,600–90,500（complete-case 的约 38%）。

IPW 加权后，v5 MAE 为 0.03202/0.03273，tree 为 0.03399/0.03342；v5 仍占优，但估计方差明显增大。因此正式主结果保留 component-mask，IPW 作为选择性观测敏感性结果，不把缺失标签填 0。

## Stop 两部评估

Stop occurrence prevalence 只有约 0.68%–0.70%。v5 occurrence AP 为 0.343–0.346、ROC AUC 为 0.941–0.948、Brier 约 0.0054；positive stop-share MAE 为 0.131–0.144。expected stop-share 的 RMSE 优于 always-zero，但 MAE 略差于 always-zero，说明稀有事件下不能用总体 MAE 单独证明有效。

## v4/v5 同集状态目标

在 20161025–26 的完全相同物理 traversal 与 mask 上，v5 在 crawl、stop、speed-CV、acceleration-RMS 和 RTS raw 的 MAE 上优于 v4；LCS raw 略差。加权 MAE 如右：crawl 0.17070 vs 0.17312，stop 0.00545 vs 0.00563，speed-CV 0.06314 vs 0.06331，acceleration 0.11943 vs 0.11951，RTS 0.09972 vs 0.10073，LCS 0.06254 vs 0.06243（均为 v5 vs v4）。

逐日 order-cluster bootstrap 显示 crawl、stop、speed-CV 和 RTS 的 v5 改善在两天均显著；acceleration 差异不显著；LCS 在 20161025 不显著、20161026 显著偏向 v4。LCS 仍是明确的非阻断弱项。LCS/RTS tail 的 AP、ROC AUC、Brier、ECE、Lift@5% 和 Lift@10% 均见 `state_tail_metrics.csv`，未将其中任何指标解释为 AV 安全或失效概率。

## Horizon gate

gate 始终位于 [0,1]。在 20161025–27，≤5 分钟 horizon 的 recent 平均权重约 0.692，而 ≥30 分钟约 0.595–0.631；结构上 horizon/history age 增大只会降低 recent 权重，support 增加只会提高 recent 权重。

四种结构均使用相同 Train、验证日期、随机种子、优化器和 early-stopping 规则完整重训。20161025–26 同样本加权 pace MAE 为：

| History fusion | Validation MAE | 相对 horizon gate |
|---|---:|---:|
| horizon gate | **0.0289513** | 0 |
| without profile | 0.0290202 | +0.0000689 |
| ordinary concatenation | 0.0290210 | +0.0000698 |
| without recent | 0.0290412 | +0.0000900 |

因此预注册主指标选择 `horizon_gate`。20161025 上三种消融相对 gate 的 order-cluster bootstrap 95% CI 全部大于 0；20161026 方向仍有利于 gate，但 CI 跨 0；20161027 calibration 日三种差值再次全部显著大于 0。结论是显式门控带来小幅且跨日方向稳定的 MAE 收益，而不是数量级提升。详细结果见 `horizon_gate_ablation.csv` 与 `horizon_gate_ablation_bootstrap.csv`。

## 训练工程

最终模型在 RTX 4060 Laptop GPU 上使用 AMP，13 个 epoch，best epoch=9，总 runtime 1,819 秒。性能审计发现并修复了 NPZ 成员按 mini-batch 重复解压，以及百万级 profile count 在 fp16 下溢出的两个问题。任何非有限 Train/validation loss 现在都会即时硬失败。
