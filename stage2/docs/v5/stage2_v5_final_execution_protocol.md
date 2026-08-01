# Stage 2 v5 一次性 final-test 执行协议

## 冻结边界

20161028–30 是 split freeze 中预注册的新 final test。模型、强 baseline、history gate、scenario 类型、校准参数和 Stage 3 准入阈值必须先提交，再生成 `stage2_v5_development_freeze.json`。冻结后不得依据 final-test 结果修改任何上述对象。

20161031 仅为已公开的 legacy benchmark，不再称为 untouched test。

## 隔离上游

Stage 0 使用 `stage2/config/stage2_v5_final_stage0.yaml`，只改变日期、每日 10,000 单配额以及隔离输出路径；匹配、质量、单行道、预处理、核心订单和动态标签阈值与 `stage0/config/stage0_v6_final.yaml` 完全一致。配置差异由 `stage2_v5_final_upstream_plan.json` 自动审计。

Stage 1 不重新拟合。`stage2.v5.final_upstream` 对新日期应用 `stage1/models/stage1_v3_final`，并复用冻结 Stage 1 的区间分类、守恒、主外键、CDF/reference、support 与聚合函数。旧版写死的日期 allowlist 由 v5 split freeze 接管，其他 bucket 验证不放宽。

Stage 2 v4 不重新拟合。新日期特征使用冻结的 `stage2/output_v4/causal_history_store`、entry-time 算法和 revealed-route 特征构造器，输出到 `stage2/output_v5/final_upstream/`。

## 唯一执行顺序

```powershell
conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage2/config/stage2_v5_final_stage0.yaml `
  build-stage1-input --resume

conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage2/config/stage2_v5_final_stage0.yaml `
  verify-stage1-input `
  --input stage2/output_v5/final_upstream/stage1/input_v1

conda run -n stage0-valhalla python -m stage2.v5.final_upstream transform-stage1 --resume
conda run -n stage0-valhalla python -m stage2.v5.final_upstream build-route-features --resume
conda run -n stage0-valhalla python -m stage2.v5.final_upstream build-shards --resume

D:/anaconda/envs/pytorch/python.exe -m stage2.v5.final_inference

conda run -n stage0-valhalla python -m stage2.v5.prediction `
  --prediction-root stage2/output_v5/final_upstream/stage2/deep_predictions `
  --output-root stage2/output_v5/final_upstream/stage2/predictions

conda run -n stage0-valhalla python -m stage2.v5.final_evaluation
conda run -n stage0-valhalla python -m stage2.v5.verification
```

`final_evaluation` 只读取一次冻结预测，输出同样本 deep/tree 指标、order-cluster bootstrap、pace 分位数覆盖与冻结校准后的相关路线场景覆盖。无论结果好坏，`post_test_tuning_count` 必须保持 0。
