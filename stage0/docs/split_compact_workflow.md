# Compact temporal-split workflow

The primary experiment uses the fixed split in `split_config.json`:

- train: 20161001-20161007
- validation: 20161008
- test: 20161009

Stage0 retains link-level products and a small set of point-level case traces.
Stage1 fits travel-time references and cohort histograms on train dates only,
then transforms all nine dates. Validation and test data never participate in
label calibration.

## Output contract

```text
stage0/output/
  order_base/
  hmm_link_traversals/
  hmm_turn_movements/
  stage0_order_link_poi_behavior/
  matcher_comparison/
  case_traces/
  poi/
  reports/
  manifests/
  logs/

stage1/output/prediction_split/
  models/
  primitives/
  link_labels/
  order_labels/
  validity/
  reports/
  manifests/
```

## Preflight and plan validation

```powershell
python .\stage0\scripts\storage_preflight.py `
  --output-root .\stage0\output --days 9 --retention-mode compact

python .\stage0\scripts\run_split_stage01.py `
  --split-config .\split_config.json `
  --archive .\22-4-8.rar `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi .\map_data\西安市POI数据.csv `
  --dry-run
```

Remove `--dry-run` to execute. Use `--phase stage0`, `stage1`, or `audit` to
resume one phase. The controller is resume-safe at Parquet partition level.

Use `--phase stage0 --limit-days 1` for the single-day closure, then rerun with
`--limit-days 3` and finally `--limit-days 9`. Completed compact days are
detected and skipped.

## Compact retention

Before pruning a completed day, `export_case_traces.py` saves representative
HMM/geometric point traces and `export_threshold_sensitivity.py` saves the
point-dependent robustness statistics. `prune_day_outputs.py` refuses deletion
unless all required compact products and the case index exist. It is a dry run
unless `--execute` is explicitly supplied.

The retained Stage2 inputs are HMM link traversals, turn movements, static POI
exposure, order-link POI behavior, and Stage1 link/order labels. Full point-level
matches are working data, not long-term experiment outputs.

After each gate, `check_stage0_readiness.py` writes `reports/stage0_readiness.csv`
and `.md`. A `REVIEW` day is not automatically expanded to the next gate.

## Stage2 gate

Run `check_stage1_label_coverage.py` and `summarize_split.py` after all audits.
The split summary deliberately reports `NOT READY` whenever any date or required
test-day coverage metric is missing.
