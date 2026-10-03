# Stage 2 v5 性能门禁报告

当前性能门禁：`PASS`。

## 静态复杂度审计

扫描 17 个 v5 production Python 文件，无阻断发现。3 处 parquet 循环读取均为有界的一分区一读取，用于 target audit 或 dry run；不跨日累计 DataFrame。

## Micro-benchmark

| Hotspot | 100k rows | 500k rows | 5×倍率 | 500k rows/s |
|---|---:|---:|---:|---:|
| CDF mapping | 0.360 s | 1.786 s | 4.96× | 279,943 |
| Route aggregation | 0.435 s | 2.196 s | 5.04× | 227,713 |
| Prediction shard merge | 0.003 s | 0.010 s | 3.40× | 47,854,217 |
| History lookup | 0.313 s | 1.526 s | 4.88× | 327,687 |
| Scenario aggregation | 0.011 s | 0.055 s | 5.20× | 9,067,268 |

所有 10k→50k 和 100k→500k 检查均低于阻断阈值 8×。100k CDF 的 100/1,000/10,000 cohort 用时分别为 0.293/0.330/0.638 秒，未表现为 N×K 扫描。最大基准内存增量约 284 MB。

## 分层 dry run

| 阶段 | Rows | Runtime | Peak RSS |
|---|---:|---:|---:|
| 1 train bucket | 87,789 | 0.060 s | 104 MB |
| 1 train day | 714,536 | 0.220 s | 119 MB |
| 1 validation day | 701,601 | 0.215 s | 121 MB |
| 20161031 legacy read-only rehearsal | 2,116,712 | 0.112 s | 459 MB |

上述 runtime 是列裁剪、target mask 与只读推理预演的吞吐，不代表深度模型完整训练时间。全量训练时间必须在 v5 tensor shard 和训练 worker 完成后另行实测，不能用这里的约 4 秒线性扫描外推冒充模型 runtime。

## 证据

- `stage2_v5_static_complexity_audit.json`
- `stage2_v5_performance_benchmarks.json`
- `performance_benchmarks.csv`
- `performance_profile_hotspots.txt`
- `runtime_by_stage.csv`
- `memory_by_stage.csv`
- `stage2_v5_preflight.json`

