# Stage 2 v5 同集 service-pace baseline 报告

## 结论

当前最强开发集 service-pace baseline 是冻结 v4 参数规格的 `HistGradientBoostingRegressor`。它在 20161025–26 的同一 direct-pace 样本上，按样本数加权 MAE 为 **0.030140 s/m**，稳定优于 strict historical profile 与 v4 static entry-time pace。RC-MSTNet v5 在相同行上达到 **0.028951 s/m**，两天配对 bootstrap CI 均低于 0，因此代码计算的开发期科学状态为 `PREDICTIVE_BASELINE_VALIDATED`。

10 月 28–30 日新 final test 未被读取。20161031 结果只作为已公开的 legacy benchmark。

## 同集结果

| Date | Model | N | MAE (s/m) | RMSE (s/m) | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| 20161025 | HistGradientBoosting | 237,553 | 0.03036 | 0.05783 | 0.5643 | 0.7111 |
| 20161025 | strict history | 237,553 | 0.03143 | 0.05668 | 0.5908 | 0.6695 |
| 20161025 | v4 static entry pace | 237,553 | 0.03147 | 0.05950 | 0.5313 | 0.6695 |
| 20161026 | HistGradientBoosting | 238,711 | 0.02992 | 0.06273 | 0.5320 | 0.7090 |
| 20161026 | v4 static entry pace | 238,711 | 0.03095 | 0.06388 | 0.5082 | 0.6737 |
| 20161026 | strict history | 238,711 | 0.03095 | 0.06106 | 0.5723 | 0.6737 |
| 20161027 calibration | HistGradientBoosting | 237,196 | 0.02952 | 0.05242 | 0.5980 | 0.7157 |
| 20161031 legacy | HistGradientBoosting | 717,805 | 0.02849 | 0.04839 | 0.6168 | 0.7126 |

完整 CSV 同时包含 global mean、highway × time-bin mean 与 edge rolling mean。

## Paired order-cluster bootstrap

负值表示 tree 的绝对误差更低：

| Date | 对照 | Tree − control MAE | 95% CI |
|---|---|---:|---:|
| 20161025 | strict history | -0.001075 | [-0.001184, -0.000962] |
| 20161026 | v4 static entry pace | -0.001025 | [-0.001111, -0.000938] |
| 20161027 | strict history | -0.000905 | [-0.000984, -0.000837] |
| 20161031 legacy | v4 static entry pace | -0.001521 | [-0.001561, -0.001480] |

每个日期使用 500 次 order-cluster bootstrap，所有模型在完全相同的有效 pace 行上比较。baseline 没有使用评估集重新拟合或调参；tree 最多使用 500,000 个稳定哈希选择的 Train 标签。

## 当前科学状态

`PREDICTIVE_BASELINE_VALIDATED`：强 baseline、v4、v5 已在同集上完成比较；该状态只适用于开发协议。20161028–30 的一次性 final test 尚未执行，因此不能据此创建最终 release/tag。
