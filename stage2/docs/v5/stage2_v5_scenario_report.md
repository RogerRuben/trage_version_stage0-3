# Stage 2 v5 路线场景报告

## 选择结果

在 20161025–26 比较 independent、shared route latent 与 residual block 后，选择 `shared_route_latent`。所有方案使用相同 traversal log-normal 边缘分布、1,000 个固定 seed 场景，路线场景严格等于 traversal 场景求和。

未校准场景暴露出明确的位置偏差：路线总时长平均低估约 25%，shared latent 的 P90/P95 coverage 只有约 0.34/0.41。原因是 direct pace 监督只覆盖可靠 GPS interval，并不能自动恢复订单级停车、未直接计时区段和其他路线总时长开销。

## 校准

只使用 20161027 calibration 拟合：

```text
route time scale = 1.3401549203
route dispersion multiplier = 2.8
route offset = 16.4230 s
```

校准日拟合内覆盖为 P50=0.5000、P90=0.9028、P95=0.9402；mean MAE 为 196.1 s，RMSE 为 312.0 s。该覆盖是 calibration-fit 诊断，不是无偏泛化结果，最终是否可接受只能由一次性 20161028–30 final test 判断。

## 正式场景 provenance

- generator：`stage2_v5_route_scenarios.1`
- model：`shared_route_latent`
- seed：20261009
- scenario count：1,000
- shared route rho：0.35
- 输出：mean/std/P50/P90/P95/CVaR90/CVaR95，以及针对外部阈值的 timeout probability

任何 timeout threshold 必须由 Stage 3 或外部服务约束提供，不能使用真实订单时长作为决策输入。

