# Stage 2 v5 路线情景报告

开发协议在 20161022—23 比较 independent、shared route latent 和 residual block，选择 `shared_route_latent`。随后只使用 20161024 calibration 拟合路线时间 scale、dispersion 与 offset：

```text
route time scale = 1.3496731736
route dispersion multiplier = 2.9
route offset = 16.8771 s
scenario seed = 20261009
scenario count = 1000
```

冻结校准在 development temporal evaluation 20161025—27 的覆盖为：

| Date | P50 | P90 | P95 |
|---|---:|---:|---:|
| 20161025 | 0.4903 | 0.8959 | 0.9384 |
| 20161026 | 0.5093 | 0.8982 | 0.9337 |
| 20161027 | 0.4957 | 0.9047 | 0.9413 |

三个 rolling folds 的合并覆盖为 P50 0.4851、P90 0.8802、P95 0.9234；覆盖门通过。20161031 legacy 最终拟合的冻结校准覆盖为 P50 0.6099、P90 0.9461、P95 0.9685，路线 mean MAE 为 166.68 秒。

路线情景严格等于 traversal 情景逐路线求和，输出 mean/std/P50/P90/P95/CVaR90/CVaR95。任何 timeout threshold 必须来自 Stage 3 或外部服务约束，不能使用订单结束后才可得的真实时长作为决策输入。

由于 rolling pace 聚合未战胜 tree，情景产品当前只允许用于 `READY_FOR_ROUTE_SCENARIO_PROTOTYPE`，不能作为已通过 Stage 3 科学准入的正式输入。
