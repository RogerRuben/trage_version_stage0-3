# Current completion status: compact Stage0/Stage1 split pipeline

Last updated: 2026-07-07

## Experiment split

The project has been reorganized around a compact temporal prediction split instead of an all-month retention workflow.

- Train: `20161001`–`20161007`
- Validation: `20161008`
- Test: `20161009`
- Stage1 fit scope: train dates only
- Stage1 target scope: train + validation + test

The split is defined in `split_config.json`. The training window overlaps the National Day holiday period, so train-to-validation/test distribution shift must be audited before using downstream Stage2 results.

## Implemented pipeline changes

### Stage0

- Added split-aware controller: `stage0/scripts/run_split_stage01.py`.
- Added compact retention workflow so the pipeline can keep link-level products while pruning heavy point-level intermediates.
- Added storage preflight with compact/full retention estimates: `stage0/scripts/storage_preflight.py`.
- Added case trace export before pruning: `stage0/scripts/export_case_traces.py`.
- Added threshold sensitivity export before pruning: `stage0/scripts/export_threshold_sensitivity.py`.
- Added daily pruning gate: `stage0/scripts/prune_day_outputs.py`.
- Added link-level quality summary: `stage0/scripts/link_level_quality_summary.py`.
- Added Stage0 readiness gate: `stage0/scripts/check_stage0_readiness.py`.
- Added legacy migration helper for existing 20161001 outputs: `stage0/scripts/migrate_legacy_stage0.py`.
- Updated POI processing so POI products are static Stage0 artifacts under `stage0/output/poi/`.
- Lowered HMM worker defaults and changed KDTree candidate query workers to reduce multi-process memory pressure.

### Stage1

- Moved Stage1 scripts into `stage1/scripts/`.
- Added `stage1/README.md`.
- Updated Stage1 label builder to require explicit `--fit-dates` and `--target-dates`.
- Updated Stage1 outputs to `stage1/output/prediction_split/`.
- Added per-dimension cohort level/sample-size fields.
- Excluded inferred or low-quality traversal rows from realized LCS/IIS/RTS/PMIS labels while retaining GNS.
- Added split-aware Stage1 coverage and summary scripts:
  - `stage1/scripts/check_stage1_label_coverage.py`
  - `stage1/scripts/summarize_split.py`
  - `stage1/scripts/summarize_stage1_validity.py`

## Completed validation

### Static checks

- Python compile smoke passed for the updated script set.
- `git diff --check` passed.
- `run_split_stage01.py --dry-run` validated the 7+1+1 split plan.
- Partial `--phase all --limit-days 1` guard correctly rejects incomplete all-phase runs.
- Storage preflight for 9 compact days passed with the available disk headroom.

### Stage1 smoke test

A temporary Stage1 smoke run verified:

- label building with split-style fit/target arguments,
- per-dimension cohort fields,
- missing-day-safe coverage checks,
- split-aware audit output.

Temporary smoke output is not part of the committed repository state.

## Completed data run

The compact Stage0 chain has completed for `20161001`.

Readiness summary from `stage0/output/reports/stage0_readiness.md`:

- Status: PASS
- Orders: 119,018
- HMM fallback ratio: 8.63%
- HMM success ratio: 97.82%
- Median order P90 match distance: 9.71 m
- Observed traversal row ratio: 60.46%
- Observed route length ratio: 76.57%

Link-level quality summary:

- Observed traversals: 1,775,563
- Inferred traversals: 1,161,187
- Observed traversal row ratio: 60.46%
- Inferred traversal row ratio: 39.54%
- Observed route length ratio: 76.57%
- Inferred route length ratio: 23.43%

Interpretation: inferred route links are mostly short routed-but-unobserved links. They are retained for route continuity but excluded from realized LCS/IIS/RTS/PMIS label construction.

## Known caveats

- Existing `20161001` case traces were exported before the case selector was hardened, so high-LCS/high-RTS examples should be treated as diagnostic traces rather than final paper-ready examples.
- Legacy `stage0_output/` still exists locally and may retain hardlinked heavy files. It should not be deleted without explicit user approval.
- `gh` is not installed in the local environment, so GitHub publishing currently uses plain `git push` rather than the GitHub CLI PR workflow.

## Recommended next commands

Run the 3-day Stage0 gate:

```powershell
python -u stage0\scripts\run_split_stage01.py `
  --split-config split_config.json `
  --archive 22-4-8.rar `
  --roads map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi map_data\西安市POI数据.csv `
  --phase stage0 `
  --limit-days 3 `
  --workers 8
```

If the 3-day gate passes, run the full 9-day Stage0 split:

```powershell
python -u stage0\scripts\run_split_stage01.py `
  --split-config split_config.json `
  --archive 22-4-8.rar `
  --roads map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi map_data\西安市POI数据.csv `
  --phase stage0 `
  --limit-days 9 `
  --workers 8
```

Then build and audit Stage1:

```powershell
python -u stage0\scripts\run_split_stage01.py `
  --split-config split_config.json `
  --archive 22-4-8.rar `
  --roads map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi map_data\西安市POI数据.csv `
  --phase stage1 `
  --workers 8

python -u stage0\scripts\run_split_stage01.py `
  --split-config split_config.json `
  --archive 22-4-8.rar `
  --roads map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi map_data\西安市POI数据.csv `
  --phase audit `
  --workers 8
```
