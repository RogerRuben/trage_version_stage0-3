# Stage3 Rolling, Stage4 Interface, and 300k Scaling Report

## 1. Completed scope

This run extended the compact pipeline through 20161023 and built a strict Stage3 rolling validation chain:

- upstream 20161020–20161023 Stage0/Stage1 products;
- route-conditioned estimated-time datasets for 20161020–20161023;
- RC-MSTNet held-out predictions for folds 4–7;
- IIS movement dataset and held-out movement predictions for 20161020–20161023;
- daily readiness and leakage audits for 20161017–20161023;
- true Stage3 rolling warehouse with three temporal folds;
- Stage3 Core DeepSets and Core+IIS+dropout evaluation;
- Stage3-to-Stage4 condition-vector export;
- Stage4 capability mapping and ODD-gated dry-run scaffolding;
- 300k feasibility audit.

## 2. Methodology boundary

The Stage3 output is a trajectory-informed operational stress / ODD-relevant condition vector. It is not AV crash risk, disengagement probability, real ADS safety risk, or a verified AV failure probability.

Future AV operational data should calibrate the vehicle capability response layer: remote assistance, fallback, route rejection, AV-specific costs, and ODD thresholds. It should not require rebuilding the Stage0–Stage3 order-environment representation.

## 3. Upstream extension results

20161020–20161023 all passed Stage0 readiness. Median order P90 match distance stayed around 8.76–8.80 m; matcher success ratio stayed around 96.65–97.01%; observed traversal ratio stayed around 81.17–81.24%.

Route-conditioned datasets were rebuilt at 15,000 orders/day:

| day | route-conditioned rows | orders |
| --- | ---: | ---: |
| 20161020 | 455,765 | 15,000 |
| 20161021 | 460,229 | 15,000 |
| 20161022 | 459,671 | 15,000 |
| 20161023 | 446,030 | 15,000 |

## 4. Daily readiness

The daily readiness audit passed for 20161017–20161023. Held-out link predictions exist for 15,000 orders/day. IIS movement prediction order coverage is approximately 98.2–98.5%.

Output:

`stage3/output/daily_readiness_20161017_23/daily_readiness_summary.csv`

## 5. IIS waterfall

For 20161020–20161023:

- mean order prediction coverage: 98.33%;
- mean order Stage3 join coverage: 98.33%;
- unknown loss ratio: 0.0%.

Dominant remaining losses are label-unobservable severity cases and physical/topological non-applicability, not engineering join failure.

Output:

`stage2/output/iis_coverage_audit_stage3_20161020_23/`

## 6. Stage3 rolling folds

Fold definitions:

| fold | train | validation | test |
| ---: | --- | --- | --- |
| 1 | 20161017–20161019 | 20161020 | 20161021 |
| 2 | 20161018–20161020 | 20161021 | 20161022 |
| 3 | 20161019–20161021 | 20161022 | 20161023 |

Warehouse and leakage audit:

- `stage3/output/rolling_stage2_prediction_warehouse/manifest.json`
- `stage3/output/rolling_stage2_prediction_warehouse/audit/leakage_audit.json`

Both passed.

## 7. Stage3 model results

Core DeepSets is the default model. IIS is evaluated as an optional gated branch.

Summary output:

- `stage3/results/rolling/stage3_rolling_metrics_by_fold.csv`
- `stage3/results/rolling/stage3_rolling_metrics_summary.csv`
- `stage3/results/rolling/stage3_core_iis_deltas.csv`

Observed IIS pattern:

- Overall AP improved in all three folds.
- Overall Lift@Top10 improved in two of three folds.
- PMIS performance decreased consistently with IIS branch.

Conclusion: IIS should remain a gated optional branch / auxiliary explanatory modality. It should not become a mandatory Stage3 input.

## 8. Stage3 output interface

The Stage3 condition-vector schema is defined in:

- `stage3/docs/stage3_odd_condition_vector.md`
- `stage3/config/stage3_output_schema.json`

Core Stage3 outputs were exported for all three test folds:

- `stage3/output/stage4_inputs_core/fold=1/stage4_inputs.parquet`
- `stage3/output/stage4_inputs_core/fold=2/stage4_inputs.parquet`
- `stage3/output/stage4_inputs_core/fold=3/stage4_inputs.parquet`

Stage4 export audit passed.

## 9. AV data scarcity boundary

The methodology document is:

`stage3/docs/methodology_without_av_operational_data.md`

It explicitly separates:

1. order-environment representation;
2. vehicle-specific capability mapping;
3. dispatch optimization.

## 10. Stage4 capability mapping and dry run

Scenario profiles:

`stage4/config/vehicle_capability_profiles.json`

Scripts:

- `stage4/scripts/build_vehicle_capability_mapping.py`
- `stage4/scripts/audit_vehicle_capability_mapping.py`
- `stage4/scripts/run_stage4_odd_gated_dry_run.py`

Capability mapping audit passed. The dry run confirms that Stage3 outputs can be consumed by Stage4 without manual feature edits.

Important limitation: current scenario thresholds are intentionally illustrative and too permissive to create strong AV/HV feasibility separation in most folds. This validates the interface but is not yet a final dispatch counterfactual.

## 11. 300k scaling

The 300k feasibility audit is:

`stage2/output/deep_v3_scaling_300k/feasibility/scaling_300k_feasibility_report.md`

Result: `BLOCKED_REQUIRES_UPSTREAM_REBUILD`.

Reason: the current route-conditioned dataset is 15k orders/day. With a seven-day train window, the maximum available training size is about 105k orders/fold. A real 300k train-orders/fold experiment requires rebuilding route-conditioned upstream data to about 45k orders/day. This run therefore does not relabel the 105k setting as 300k.

## 12. Known limitations

- Stage3 uncertainty is currently inherited mainly from Stage2 conformal intervals; folds 4–7 used single-seed conformal uncertainty.
- IIS has high coverage after rebuild, but its predictive gain is heterogeneous and not uniformly positive across targets.
- Stage4 dry run is a scenario/interface validation, not a calibrated operational dispatch result.
- 300k scaling remains pending until 45k/day upstream route-conditioned datasets are rebuilt.

## 13. Next recommended work

1. Rebuild route-conditioned upstream data at about 45k orders/day for the selected train windows.
2. Run one true 300k Lite/Full RC-MSTNet scaling point.
3. Calibrate Stage4 vehicle capability profiles with external AV data or clearly designed scenario priors.
4. Keep Core DeepSets as default Stage3 model; use IIS as optional gated branch unless further subgroup evidence is stable.

## 14. Key output paths

- `stage3/output/rolling_stage2_prediction_warehouse/`
- `stage3/output/rolling_order_targets/`
- `stage3/output/rolling_order_features/`
- `stage3/results/rolling/`
- `stage3/output/stage4_inputs_core/`
- `stage4/output/capability_mapping/`
- `stage4/output/odd_gated_dry_run/`
- `stage2/output/deep_v3_scaling_300k/feasibility/`

## 15. Git commit

The final commit hash is reported in the delivery response. A commit cannot
stably contain its own hash inside a tracked file because editing the file
changes the hash.
