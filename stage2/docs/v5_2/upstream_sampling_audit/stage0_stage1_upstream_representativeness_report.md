# Stage 0/1 upstream sampling representativeness audit

- Audit status: `PASS`
- Raw candidate coverage: `FULL`
- Final classification: `UP-D`
- Conclusion: `MULTI_STAGE_SELECTION_COMPOUNDS_SPARSE_CONTEXT_ATTRITION`
- Sampling-hash representativeness: `PASS`
- Stage 0/1 rerun, model training and inference: `NO / NO / NO`

## Frozen scope

- Raw support fit: 20161009–20161024 only.
- Evaluation coverage: 20161025–20161027 and 20161031.
- Time: Asia/Shanghai, 48 half-hour bins.
- Space: fixed EPSG:32649 1 km × 1 km grid.
- Primary raw sparse context: origin-grid × departure-time-bin.

## Funnel

- Raw candidates: `1,999,515`
- Processed opportunities: `361,965`
- Stage 0 accepted orders: `220,000`
- Stage 0 rejected orders: `141,965`
- Quota-unprocessed candidates: `1,637,550`
- Link traversals: `15,649,455`
- Direct-observed traversals: `5,775,530`

## Mechanism separation

- Raw rare-context share: `2.2006%`
- Accepted rare-context share: `1.9582%`
- Rare/common conditional Stage 0 acceptance: `54.6631%` vs `61.6634%` (difference `-7.0002%`, ratio `0.8865`).
- Stage 0 quality-selection effect material: `True`
- Stage 1 supervision attrition material: `True`

### Stage 1 target-valid availability in rare versus high raw contexts

| target | rare rate | high rate | pp difference | rate ratio | material |
|---|---:|---:|---:|---:|---|
| crawl | 100.0000% | 100.0000% | 0.0000% | 1.0000 | False |
| stop | 100.0000% | 100.0000% | 0.0000% | 1.0000 | False |
| speed_cv | 62.9308% | 67.9423% | -5.0116% | 0.9262 | True |
| acceleration_rms | 40.0509% | 46.4181% | -6.3671% | 0.8628 | True |

## Distribution preservation

| dimension | target | comparison | JSD | TV |
|---|---|---|---:|---:|
| time_bin | orders | raw_vs_accepted | 0.000095 | 0.008340 |
| origin_grid | orders | raw_vs_accepted | 0.004704 | 0.058560 |
| time_bin | crawl | accepted_orders_vs_target_valid_traversals | 0.000419 | 0.015420 |
| origin_grid | crawl | accepted_orders_vs_target_valid_traversals | 0.001996 | 0.041851 |
| time_bin | stop | accepted_orders_vs_target_valid_traversals | 0.000419 | 0.015420 |
| origin_grid | stop | accepted_orders_vs_target_valid_traversals | 0.001996 | 0.041851 |
| time_bin | speed_cv | accepted_orders_vs_target_valid_traversals | 0.000615 | 0.017709 |
| origin_grid | speed_cv | accepted_orders_vs_target_valid_traversals | 0.001908 | 0.040681 |
| time_bin | acceleration_rms | accepted_orders_vs_target_valid_traversals | 0.000876 | 0.021401 |
| origin_grid | acceleration_rms | accepted_orders_vs_target_valid_traversals | 0.001645 | 0.035773 |

## Interpretation

Raw demand concentration, Stage 0 quality selection, and Stage 1 label availability jointly shape sparse-context representation; no single stage explains the observed sparsity.

The audit is descriptive and does not estimate a causal selection effect. Raw demand concentration, hash/quota opportunity, Stage 0 quality selection, and Stage 1 supervision attrition are reported separately.

## Stop state

`SPARSITY_STRESS_TEST_AUTHORIZED=NO`

`TRANSFER_V2_AUTHORIZED=NO`

`PHASE_D_AUTHORIZED=NO`

`STAGE3_AUTHORIZED=NO`
