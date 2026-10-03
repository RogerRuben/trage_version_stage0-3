# Stage 2 v5 模型报告

## 结论

Stage 2 v5 已按冻结的日期协议完成开发、三组 rolling-origin 评估和 20161031 legacy benchmark。全过程复用 Stage 1 与 Stage 2 v4 冻结产品，没有生产或读取 20161028—30，也没有重跑 Stage 0/1。

开发划分选择 `ordinary_concatenation`：其 20161022—23 validation-model pace MAE 为 0.028581，优于 horizon gate 的 0.028694。结构冻结后，20161025—27 development temporal evaluation 的 v5 MAE 为 0.028746，tree 为 0.029871，相对改善 3.77%，逐日 paired bootstrap 的 95% CI 均低于 0。

## Rolling-origin 稳定性

| Fold | Evaluation | v5 MAE | Tree MAE | 结果 |
|---|---|---:|---:|---|
| 1 | 20161022—23 | 0.028697 | 0.029810 | WIN |
| 2 | 20161024—25 | 0.029767 | 0.029922 | WIN |
| 3 | 20161026—27 | 0.044483 | 0.029735 | LOSS_OR_MIXED |

六个评估日中 v5 胜出 5 日，三个 fold 中胜出 2 个；但按全部有效行聚合，v5 MAE 为 0.034286，tree 为 0.029822，v5 高 14.97%。因此 rolling 科学门失败，不能据开发集结果宣布模型稳定优于 baseline。

Fold 3 的只读诊断表明，20161026 的冻结 log-normal 均值中有一行达到 7584.05 s/m，该行独自贡献当天 MAE 的 52.47%。这解释了聚合不稳定性，但正式指标不做截尾、不改写、不事后调参。

## 20161031 legacy benchmark

20161031 仅用于与冻结 v4 的版本可比性，不是未见最终 Test，也不参与模型或超参数选择。最终拟合使用 Train 20161009—24、Validation 20161025—26、Calibration 20161027。

在 717,805 条 direct-pace traversal 上：v5 mean MAE 为 0.028135，tree 为 0.028493（v5 改善 1.26%），strict historical profile 为 0.030015。实际 RC-MSTNet v4/v5 的六个辅助 raw 目标与两个 percentile tail 也在完全相同 traversal 行上比较；tail 结果仅作描述。

## 数值稳定性修复

Legacy 推理发现一个订单的 23 条 traversal 在 AMP 下出现非有限输出。根因是跨期 `feature_age_s` 经 Train-only 标准化后达到 105,492，转换到 FP16 时溢出。预测器现在只对出现非有限值的 batch 使用 FP32 重算，并在 manifest 记录回退次数；31 日仅一个 batch 回退，修复后非有限 availability 数为 0。若 FP32 仍非有限则硬失败。

## 当前准入

- 工程正确性：PASS
- 时序与防泄漏：PASS
- 性能工程：PASS
- Legacy benchmark：PASS
- Rolling-origin 科学门：FAIL
- Stage 3 准入：`READY_FOR_ROUTE_SCENARIO_PROTOTYPE`

由于 rolling 聚合未战胜 tree，本轮不创建最终 release tag，也不宣称 `READY_FOR_STAGE3`。
