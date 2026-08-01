# Stage 2 v5 service-pace baseline 报告

所有 baseline 都在各协议的 Train 日期上重新拟合，并与 v5 使用完全相同的 direct-observed pace 行。每个 rolling fold 分别重拟合 Train-only normalization、vocabulary、tree 与历史统计，没有使用未来 fold 数据。

## 主要比较

| 协议 | v5 MAE (s/m) | Tree MAE (s/m) | 相对变化 |
|---|---:|---:|---:|
| Development evaluation 25—27 | 0.028746 | 0.029871 | -3.77% |
| Rolling Fold 1 | 0.028697 | 0.029810 | -3.73% |
| Rolling Fold 2 | 0.029767 | 0.029922 | -0.52% |
| Rolling Fold 3 | 0.044483 | 0.029735 | +49.60% |
| Rolling pooled | 0.034286 | 0.029822 | +14.97% |
| Legacy 20161031 | 0.028135 | 0.028493 | -1.26% |

Rolling pooled 是主要科学稳定性结论，因此 baseline 未被总体战胜。Fold 3 的异常来自冻结概率均值的极端尾部；正式比较保留原始结果。

## Legacy 同排产品

20161031 上，strict historical profile MAE 为 0.030015，v4 static entry-time proxy 为 0.030014，tree 为 0.028493，v5 mean 为 0.028135，v5 P50 为 0.027335。实际冻结 RC-MSTNet v4/v5 的辅助状态目标另存于 `protocols/legacy/legacy_v4_v5_state_metrics.csv`，避免把 v4 static proxy 误称为 v4 深度模型。

20161031 是 legacy frozen benchmark，仅用于版本可比性；20161028—30 未生产、未读取。
