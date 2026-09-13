# Mixed-autonomy ride-hailing research repository

Current working line: Stage0 v6 → Stage1 v3 → frozen Stage2 v5.2 M3 → Stage3 ODD/TOD route interface → Stage4 FleetPy / sparse rolling assignment → manuscript and mechanism analysis.

Start with the [detailed repository guide / 仓库详细说明](docs/repository_guide_zh.md). It explains current versus historical code, input/output contracts, environments, entrypoints, tests, local-only artifacts and maintenance boundaries.

## Current navigation

- [Stage0 → Stage1: route, traversal and direct interval products](stage0/docs/stage0_to_stage1_contract.md)
- [Stage1 → Stage2: labels, masks and decision-time restrictions](stage1/docs/stage1_to_stage2_contract.md)
- [Stage2 v5.2 → Stage3: frozen M3 predictor](stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md)
- [Stage3 → Stage4: fixed routes, hard state and continuous suitability](stage3/docs/odd_tod/final/stage3_to_stage4_contract.md)
- [Stage4: completed formal experiment report](stage4/docs/final_experiments/stage4_final_experiment_execution_summary.md)
- [Manuscript V3](stage4/docs/manuscript_v3/README.md)

## Important boundaries

The repository contains several generations of code. The root `run_pipeline.py` and the [compact split workflow](stage0/docs/split_compact_workflow.md) describe earlier workflows, not the current final-production entrypoint or date protocol. Earlier stage authorization fields are historical snapshots.

Most raw data, tiles, model payloads and generated products are local and ignored by Git; a few compact output/evidence files are tracked explicitly. A clone is not a complete data backup. Older Stage2 v4/v5 modules still support current imports and must not be deleted merely because their version is older.

See the guide before running production, training, cleanup or migration. No single root command currently reproduces every frozen research phase.
