# IIS 15k Rebuild and Stage3 Follow-up Report

## Scope

This round addressed the first-priority gate for the next phase:

- storage budget audit before large jobs
- IIS coverage waterfall audit
- IIS movement dataset rebuilt on the same 15k/day keys as RC-MSTNet
- fast IIS movement model trained for coverage verification
- Stage3 warehouse rebuilt with 15k IIS predictions
- Stage3 single-chain Core vs Core+IIS dropout probe rerun

Dates covered in this report are the current held-out chain:

- Stage3 train: 20161017
- Stage3 validation: 20161018
- Stage3 test: 20161019

The 20161020-20161023 extension is not yet available because route-conditioned,
strict target, tensor shard, and held-out prediction artifacts for those dates
do not exist yet.

## Storage Budget

`stage2/scripts/audit_stage2_storage_budget.py` was added and run.

Result:

- status: PASS
- estimated free space after planned jobs: 81.22 GB
- retained-space policy: at least 20% of D drive or 50 GB, whichever is stricter

Output:

- `stage2/output/storage_audit/storage_budget.json`
- `stage2/output/storage_audit/storage_budget_report.md`

## IIS Coverage Diagnosis

`stage2/scripts/audit_stage2_iis_coverage_waterfall.py` was added and run on the
old IIS prediction store.

Old result:

- mean Stage3 IIS order prediction coverage: 6.26%
- dominant loss: `movement_dataset_key_missing` and `prediction_not_generated`

Interpretation:

The low Stage3 IIS coverage was primarily an engineering/data-key coverage issue.
The old IIS movement dataset covered about 933 orders/day, while the RC-MSTNet
warehouse covered 15,000 orders/day.

## IIS 15k Rebuild

`stage2/scripts/build_stage2_iis_movement_dataset.py` was updated to read the
15k/day route-conditioned estimated-time dataset directly.

New output:

- `stage2/output/iis_movement_causal_dataset_15k/`

Per-day scale:

- candidate movement rows: about 424k-439k/day
- movement orders: about 14.7k/day
- applicable ratio: about 38%
- realized severity observed ratio: about 35%

The canonical movement key is:

```text
date | order_id | movement_seq | from_link_id | node_id | to_link_id
```

## IIS Fast Coverage Model

`stage2/scripts/train_stage2_rc_mstnet_movement.py` was updated to read only the
needed columns and preserve the canonical movement key.

Fast verification run:

- dataset: `stage2/output/iis_movement_causal_dataset_15k`
- max train rows/fold: 200,000
- epochs: 2
- output: `stage2/output/deep_v3/iis_movement_15k_fast`

Test results:

| Fold | Applicability AUC | Applicability AP | Severity AUC | Severity AP |
| ---: | ----------------: | ---------------: | -----------: | ----------: |
| 1 | 0.9980 | 0.9967 | 0.8205 | 0.4757 |
| 2 | 0.9977 | 0.9963 | 0.8370 | 0.4901 |
| 3 | 0.9960 | 0.9936 | 0.8405 | 0.4936 |

This run is sufficient for Stage3 coverage verification. A longer IIS training
run can be scheduled later if IIS becomes a major Stage3 modality.

## Stage3 Warehouse with IIS 15k

New output:

- `stage3/output/stage2_prediction_warehouse_iis15k`

Leakage audit:

- status: PASS

Movement prediction coverage:

| Split | Date | Movement Rows | Movement Orders |
| --- | --- | ---: | ---: |
| train | 20161017 | 428,688 | 14,733 |
| validation | 20161018 | 436,887 | 14,755 |
| test | 20161019 | 439,195 | 14,734 |

After rebuilding the warehouse, IIS Stage3 joined order coverage recovered to
98.27%. Engineering join loss is no longer the dominant loss.

Remaining IIS missingness is now mostly:

- label unobservable: no realized severity label
- physical/topological non-applicability: invalid movement topology or low-degree
  continuation contexts

## Stage3 Single-chain Probe with IIS 15k

Order targets and features were rebuilt:

- `stage3/output/order_targets_iis15k`
- `stage3/output/order_features_iis15k`

IIS feature coverage:

- train: 98.22%
- validation: 98.37%
- test: 98.23%

LightGBM remains sparse and should not be a required Stage3 modality:

- train: 6.41%
- validation: 1.08%
- test: 0.97%

### Core DeepSets

Output:

- `stage3/output/deepsets_iis15k_core`

Test:

| Target | AUC | AP | Spearman | Lift@Top10 |
| --- | ---: | ---: | ---: | ---: |
| LCS | 0.8067 | 0.1942 | 0.5933 | 4.2814 |
| PMIS | 0.8204 | 0.2834 | 0.5715 | 4.0198 |
| RTS | 0.7914 | 0.3583 | 0.5103 | 3.6909 |
| OVERALL | 0.7808 | 0.4612 | 0.5416 | 3.0702 |

### Core + IIS + Modality Dropout

Output:

- `stage3/output/deepsets_iis15k_iis_dropout`

Test:

| Target | AUC | AP | Spearman | Lift@Top10 |
| --- | ---: | ---: | ---: | ---: |
| LCS | 0.7963 | 0.1842 | 0.5812 | 4.2216 |
| PMIS | 0.8153 | 0.2687 | 0.5595 | 3.8130 |
| RTS | 0.7897 | 0.3562 | 0.5064 | 3.7037 |
| OVERALL | 0.7806 | 0.4652 | 0.5471 | 3.0891 |

Interpretation:

IIS coverage is now fixed, but the simple order-side IIS branch is not yet a
clear universal win. It slightly improves OVERALL AP and Lift@Top10, but weakens
LCS/PMIS and most single-dimension metrics. For now, Core DeepSets remains the
clean Stage3 default, while IIS should be kept as an optional gated branch and
evaluated again under true multi-date rolling folds.

## New Stage3/Stage4 Contracts

Added:

- `stage3/config/stage3_rolling_fold_config.json`
- `stage3/scripts/build_stage3_rolling_warehouse.py`
- `stage3/scripts/audit_stage3_rolling_warehouse.py`
- `stage4/docs/stage4_input_contract.md`

These are ready once 20161020-20161023 Stage2 held-out prediction artifacts are
generated.

## Not Completed Yet

The following remain blocked on upstream date extension or a separate long job:

- 20161020-20161023 Stage0/Stage1/Stage2 extension
- true Stage3 three-fold rolling evaluation
- Stage3 validation-only calibration across three folds
- compact 300k tensor shard builder and 300k Lite/Full scaling run
- final Lite vs Full Stage2 model freeze
- Stage4 release gate

## Current Answer to the Key Questions

1. IIS missing was mostly engineering coverage before rebuild; after rebuild,
   Stage3 order coverage is 98.27% and the remaining loss is mainly label
   unobservability plus physical/topological non-applicability.
2. IIS applicability order coverage on unified 15k/day keys recovered to about
   98.27%.
3. IIS provides a small overall Stage3 signal in this single-chain probe, but not
   enough to make it mandatory.
4. Stage3 DeepSets has not yet been validated on three rolling test days because
   20161020-20161023 upstream artifacts are missing.
5. Modality dropout did not clearly improve all targets; it should remain an
   optional robustness setting.
6. Stage3 overall probability calibration still needs the multi-fold rolling
   protocol.
7. Stage3 uncertainty still needs the multi-fold rolling protocol.
8. 300k scaling has not been run yet.
9. Lite vs Full cannot be frozen until 300k is complete.
10. Stage4 remains HOLD.
