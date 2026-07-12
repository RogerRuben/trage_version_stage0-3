# Stage2 link-level prediction

## Route-conditioned main line

The current Stage2 main line is route-conditioned stress prediction:

```text
given route -> route-conditioned ODD-stress prediction -> calibrated stress vector
```

The map-matched completed route is used as a revealed proxy for the platform's
assigned/planned service route. Route generation, route choice and navigation
routing are outside the main experiment. The OD-based shortest-path and
historical-fastest products remain audit/appendix material for route-choice
mismatch and label-observability analysis.

Contract:

```text
stage2/docs/stage2_route_conditioned_contract.md
```

Build the route-conditioned datasets from the actual-route causal product:

```powershell
python stage2/scripts/build_stage2_route_conditioned_dataset.py `
  --source-root stage2/output/actual_route_oracle_causal_dataset `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/route_conditioned_dataset
```

This writes:

```text
stage2/output/route_conditioned_dataset/
  route_conditioned_estimated_time_train.parquet
  route_conditioned_estimated_time_validation.parquet
  route_conditioned_estimated_time_test.parquet
  route_conditioned_oracle_time_train.parquet
  route_conditioned_oracle_time_validation.parquet
  route_conditioned_oracle_time_test.parquet
  fold=1/
  fold=2/
  fold=3/
```

Only the estimated-time product is eligible for Stage3. The oracle-time product
is an upper-bound diagnostic for the cost of entry-time uncertainty.

Build a true actual-entry oracle state product and compare it with the
estimated-time main setting:

```powershell
python stage2/scripts/attach_stage2_actual_entry_lagged_state.py `
  --route-root stage2/output/routes/actual_route_planned_time_oracle `
  --state-root stage2/output/lagged_state_store `
  --output-root stage2/output/actual_entry_oracle_causal_dataset

python stage2/scripts/run_stage2_threefold_fair_evaluation.py `
  --dataset-root stage2/output/actual_entry_oracle_causal_dataset `
  --output-root stage2/output/route_conditioned_eval/oracle_actual_entry `
  --targets lcs rts pmis

python stage2/scripts/audit_stage2_estimated_oracle_gap.py
```

The gap audit is written to:

```text
stage2/output/route_conditioned_eval/gap_audit/estimated_oracle_gap_report.md
```

## Deep v3: RC-MSTNet main-candidate test

Deep v3 promotes deep learning from the old 30k structural probe to a formal
main-candidate test under the final route-conditioned estimated-time protocol.
The first implemented candidate is:

```text
RC-MSTNet =
  link semantic encoder
  + lagged-state temporal encoder
  + local route convolution
  + route Transformer
  + multi-task LCS/PMIS/RTS heads
  + route-level auxiliary head
```

IIS remains movement-level and is handled by a separate applicability/severity
script. The RTS stress-correlation module is deferred until Model B/C
feasibility is stable.

Build sequence manifests:

```powershell
python stage2/scripts/build_stage2_deep_v3_sequences.py `
  --dataset-root stage2/output/route_conditioned_dataset/estimated_time_daily `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3/data_manifests
```

Train RC-MSTNet:

```powershell
python stage2/scripts/train_stage2_rc_mstnet.py `
  --dataset-root stage2/output/route_conditioned_dataset/estimated_time_daily `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3/feasibility_100k/rc_mstnet `
  --prediction-root stage2/output/deep_v3/rolling_predictions/rc_mstnet `
  --max-train-orders 100000 `
  --epochs 5
```

The optimized trainer uses balanced sampling across every training date,
pre-encoded contiguous order arrays, aligned `4 x 24` lagged-state channels,
length-bucket batches, CUDA AMP, pinned transfers, and metric-only validation
during epochs. On Windows/16 GB RAM, keep `--num-workers 0`; the contiguous
arrays remove the need for multiprocessing and avoid worker-side dataset
copies. `--max-train-orders` is a total fold budget and is distributed evenly
across the fold's training dates.

For 100k+ experiments, first build fold-aware daily mmap tensor shards. The
builder fits normalization statistics and categorical vocabularies on training
dates only, reads one day at a time, stores float16 inputs plus float32 targets,
and fingerprints metadata/order selection for safe resume:

```powershell
python stage2/scripts/build_stage2_deep_v3_tensor_shards.py `
  --dataset-root stage2/output/route_conditioned_dataset_5k/estimated_time_daily `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3_tensor_shards_5k `
  --folds 1,2,3 `
  --max-train-orders 5000 `
  --max-eval-orders 1000 `
  --max-seq-len 96 `
  --feature-dtype float16
```

Train directly from the shards:

```powershell
python stage2/scripts/train_stage2_rc_mstnet.py `
  --tensor-shard-root stage2/output/deep_v3_tensor_shards_5k `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3_5k/p5_mmap_rolling_5k/rc_mstnet `
  --prediction-root stage2/output/deep_v3_5k/p5_mmap_rolling_5k_predictions/rc_mstnet `
  --folds 1,2,3 `
  --max-train-orders 5000 `
  --max-eval-orders 1000 `
  --max-seq-len 96 `
  --batch-size 64 `
  --epochs 2 `
  --num-workers 0
```

The order budgets passed to the trainer are descriptive when tensor shards are
used; the effective sample sizes are fixed by the shard manifests.

Audit every shard before a formal run:

```powershell
python stage2/scripts/audit_stage2_deep_v3_tensor_shards.py `
  --tensor-shard-root stage2/output/deep_v3_tensor_shards_5k
```

Formal 100k rolling run (completed locally):

```powershell
python stage2/scripts/train_stage2_rc_mstnet.py `
  --tensor-shard-root stage2/output/deep_v3_tensor_shards_100k `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3_100k/formal_rolling/rc_mstnet `
  --prediction-root stage2/output/deep_v3_100k/formal_rolling_predictions/rc_mstnet `
  --folds 1,2,3 `
  --max-train-orders 100000 `
  --max-eval-orders 0 `
  --max-seq-len 96 `
  --batch-size 128 `
  --epochs 5 `
  --hidden-dim 128 `
  --layers 3 `
  --heads 4 `
  --num-workers 0
```

The formal test AUC/AP pairs are `0.8667/0.4376` for LCS,
`0.8716/0.4398` for PMIS, and `0.8234/0.3875` for RTS. See
`stage2/docs/stage2_deep_v3_formal_100k_report.md` for the full rolling,
bootstrap, order-level, and resource audit.

The Stage2 finalization and strict Stage3 prototype are summarized in:

```text
stage2/docs/stage2_deep_v3_stage3_prototype_report.md
```

For a resource-safe local run:

```powershell
python stage2/scripts/train_stage2_rc_mstnet.py `
  --dataset-root stage2/output/route_conditioned_dataset_5k/estimated_time_daily `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3_5k/rolling_5k/rc_mstnet `
  --prediction-root stage2/output/deep_v3_5k/rolling_5k_predictions/rc_mstnet `
  --folds 1,2,3 `
  --max-train-orders 5000 `
  --max-eval-orders 1000 `
  --max-seq-len 96 `
  --batch-size 64 `
  --epochs 2 `
  --num-workers 0
```

Train movement-level IIS:

```powershell
python stage2/scripts/train_stage2_rc_mstnet_movement.py `
  --dataset-root stage2/output/iis_movement_causal_dataset `
  --fold-config rolling_threefold_config.json `
  --output-root stage2/output/deep_v3/feasibility_100k/rc_mstnet_movement
```

Evaluate against the LightGBM route-context benchmark:

```powershell
python stage2/scripts/evaluate_stage2_deep_v3.py
python stage2/scripts/summarize_stage2_deep_v3.py
```

Current scale status:

```text
stage2/output/route_conditioned_dataset_15k/estimated_time_daily contains
15,000 orders/day for 20161009-20161019. The formal 100k three-fold experiment
is complete. A 300k experiment would require a larger upstream daily sample;
the mmap tensor-shard loader itself is no longer the RAM bottleneck.
```

Stage2 starts from the compact split pipeline outputs and builds a supervised link-level prediction dataset.

Current split:

```text
Train:      20161009-20161015
Validation: 20161016
Test:       20161017
Matcher:    local_topology_fmm
```

The first Stage2 table is:

```text
stage2/output/link_dataset/
  train.parquet
  validation.parquet
  test.parquet
  manifest.json
```

Each row is one:

```text
order_id x link_id x link_seq
```

Main prediction targets:

```text
target_lcs_pct
target_iis_pct
target_rts_pct
target_pmis_pct
target_high_lcs_90
target_high_iis_90
target_high_rts_90
target_high_pmis_90
```

GNS is retained as a static road-structure/context feature (`gns_pct_link`) rather than the first main prediction target.

Build command:

```powershell
python stage2/scripts/build_stage2_link_dataset.py `
  --stage1-output-root stage1/output/prediction_split `
  --split-config split_config.json `
  --output-root stage2/output/link_dataset
```

Use `--dry-run` or `--max-parts-per-day` for a quick schema/count check before writing the full split datasets.

## Stage2 audit, baselines, and order aggregation

Data contract:

```text
stage2/docs/stage2_data_contract.md
```

Label audit:

```powershell
python stage2/scripts/audit_stage2_labels.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/label_audit
```

Single-target LightGBM baselines:

```powershell
python stage2/scripts/train_stage2_baselines.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/baselines `
  --max-train-rows 1000000
```

Order-level aggregation for Stage3 candidates:

```powershell
python stage2/scripts/aggregate_stage2_order_features.py `
  --dataset-root stage2/output/link_dataset `
  --prediction-root stage2/output/baselines `
  --output-root stage2/output/order_features
```

The first baseline excludes post-trip realized trajectory fields from model inputs. IIS is always trained and evaluated with `iis_valid`; missing IIS is not filled with zero.

## Strict prediction-time and predictability audit

The existing link dataset is now explicitly classified as an oracle-route
upper-bound dataset because its route sequence and link time come from the
completed trip. The deployable contract is documented in:

```text
stage2/docs/stage2_decision_time_contract.md
stage2/docs/stage2_target_reorganization.md
```

Estimate variance/repeatability and compare raw versus percentile stress:

```powershell
python stage2/scripts/estimate_stage2_predictability_ceiling.py
```

Build leakage-safe rolling profiles and completed-bin lagged traffic features:

```powershell
python stage2/scripts/build_stage2_rolling_profiles.py
python stage2/scripts/build_stage2_lagged_traffic_features.py
```

The lagged feature output is still marked `oracle_route_upper_bound` until its
prediction timestamps come from a planned route. Build explicit oracle or
shortest-path route prototypes with:

```powershell
python stage2/scripts/build_stage2_planned_route_dataset.py --route-source actual_matched_route
python stage2/scripts/build_stage2_planned_route_dataset.py --route-source shortest_path
```

Check whether enough consecutive days exist for multi-fold rolling/OOF work:

```powershell
python stage2/scripts/build_stage2_rolling_fold_manifest.py
```

## Three-fold strict pre-dispatch extension

The 20161018-20161019 extension is configured by
`rolling_extension_config.json`; the final fold definitions are frozen in
`rolling_threefold_config.json`.

The strict pipeline is:

```powershell
python stage0/scripts/extract_order_od.py --archive 22-4-8.rar --start-date 20161009 --end-date 20161019
python stage0/scripts/audit_order_od.py
python stage2/scripts/build_stage2_strict_targets.py --skip-existing
python stage2/scripts/build_stage2_od_planned_routes.py --order-od-root stage0/output/order_od_audited --route-source historical_fastest_path --output-root stage2/output/routes/historical_fastest_path
python stage2/scripts/build_stage2_od_planned_routes.py --order-od-root stage0/output/order_od_audited --route-source shortest_path --output-root stage2/output/routes/shortest_path
python stage2/scripts/build_stage2_actual_route_oracle.py
python stage2/scripts/build_stage2_actual_route_planned_time_oracle.py
python stage2/scripts/audit_stage2_planned_routes.py
python stage2/scripts/build_stage2_lagged_state_store.py
python stage2/scripts/attach_stage2_planned_lagged_state.py --planned-route-root stage2/output/routes/historical_fastest_path
python stage2/scripts/build_stage2_iis_movement_dataset.py
python stage2/scripts/run_stage2_threefold_fair_evaluation.py --targets lcs rts pmis
```

Strict targets are `raw expected stress`, frozen raw-tail exceedance,
historical uncertainty, and percentile anomaly as an auxiliary target. IIS is
represented as planned-movement applicability plus conditionally observed
severity; missing severity is never filled with zero.

The OSM line layer is not fully noded at every internal intersection. Planned
routing therefore tries directed endpoint topology, then undirected endpoint
topology, then a recorded same-layer geometry-noded link-graph fallback. This
improves route success without moving OD snaps to a distant giant component,
but its use and planned/actual route overlap must be reported. The actual-route
planned-time product is an explicit post-trip oracle upper bound and is never a
deployable feature source.

Run IIS separately on the native movement dataset:

```powershell
python stage2/scripts/run_stage2_threefold_fair_evaluation.py `
  --dataset-root stage2/output/iis_movement_causal_dataset `
  --output-root stage2/output/rolling_fair_eval_iis `
  --targets iis
```

The link evaluation compares static, rolling-profile, dynamic-state,
topology-propagation and route-context ablations on common prediction rows.
Tail probabilities are isotonic-calibrated on each validation day and
uncertainty is evaluated using validation-normalized conformal intervals.

## Deep upper-bound experiments

The deep upper-bound round asks whether stronger tabular models, route-sequence encoders, or topology-aware GNN-sequence models can materially improve high-stress link identification before Stage3 calibration.

Strong tabular baseline:

```powershell
python stage2/scripts/train_stage2_full_tabular.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/deep_baselines/full_tabular `
  --prediction-root stage2/output/deep_predictions `
  --max-train-rows 3000000 `
  --num-boost-round 500 `
  --tail-weight 4.0
```

## Deep Modeling v2: route-aware, dual-graph, contrastive, and layered evaluation

RouteLocalTransformer:

```powershell
python stage2/scripts/train_stage2_route_local_transformer.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/deep_baselines_v2/route_local_transformer `
  --prediction-root stage2/output/deep_predictions_v2 `
  --profile-path stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet `
  --max-train-orders 30000 `
  --max-eval-orders 15000 `
  --pretrain-epochs 1 `
  --epochs 2
```

DualGraphRouteTransformer:

```powershell
python stage2/scripts/train_stage2_dual_graph_route_transformer.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/deep_baselines_v2/dual_graph_route_transformer `
  --prediction-root stage2/output/deep_predictions_v2 `
  --profile-path stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet `
  --max-train-orders 30000 `
  --max-eval-orders 15000 `
  --epochs 2
```

Layered evaluation:

```powershell
python stage2/scripts/evaluate_stage2_v2_slices.py `
  --dataset-root stage2/output/link_dataset `
  --prediction-root stage2/output/deep_predictions_v2 `
  --baseline-prediction-root stage2/output/deep_predictions `
  --output-root stage2/output/deep_baselines_v2
```

Summary:

```powershell
python stage2/scripts/summarize_stage2_v2_models.py --stage2-output stage2/output
```

The v2 report is written to:

```text
stage2/output/deep_model_v2_report.md
```

Important interpretation note:

```text
The current v2 deep runs are 30k-order structural probes. They are not a final
head-to-head comparison against full_tabular_lgbm_3m_tail.
```

Controlled v2 plan:

```text
stage2/docs/stage2_deep_modeling_v2_controlled_plan.md
```

Generate a fair-budget scaling plan without running it:

```powershell
python stage2/scripts/run_stage2_controlled_scaling.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/controlled_scaling `
  --budgets 30000 100000 300000 1000000
```

Run a strict 30k-order full-tabular comparison point:

```powershell
python stage2/scripts/train_stage2_full_tabular.py `
  --dataset-root stage2/output/link_dataset `
  --output-root stage2/output/deep_baselines/full_tabular_30k_orders `
  --prediction-root stage2/output/deep_predictions_30k `
  --max-train-orders 30000 `
  --max-train-rows all `
  --profile-scope sampled_train_orders
```

Bootstrap slice confidence intervals for candidate gains:

```powershell
python stage2/scripts/bootstrap_stage2_slice_ci.py `
  --prediction-file stage2/output/deep_predictions_v2/dual_graph_route_transformer_test.parquet `
  --dataset-file stage2/output/link_dataset/test.parquet `
  --output-csv stage2/output/deep_baselines_v2/bootstrap_dual_graph_test_ci.csv `
  --model-name dual_graph_route_transformer `
  --split test
```

Hybrid fusion is supported by:

```text
stage2/scripts/train_stage2_hybrid_fusion.py
```

For publication-grade fusion, train-split deep predictions should come from OOF / rolling models rather than in-sample deep predictions.
