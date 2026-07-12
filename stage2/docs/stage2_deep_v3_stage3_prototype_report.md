# Stage2 Deep finalization and Stage3 prototype report

## Executive decision

```text
Stage2-L: RC-MSTNet retained as main LCS/PMIS/RTS model
Stage2 baseline: LightGBM retained as benchmark, not stacked by default
Stage2-I: separate movement applicability + conditional severity model retained
Stage3: strict one-chain prototype complete; formal multi-fold Stage3 not complete
Stage4: HOLD
```

## Cross-fold and cross-seed evidence

On identical LightGBM/RC-MSTNet prediction rows, link-level AUC, AP, Spearman,
Lift@Top5%, and Lift@Top10% deltas favor RC-MSTNet in every fold and target.
Old LightGBM OOF coverage is limited: common orders are 1,000/164/149 in folds
1/2/3, so paired order-level comparisons have wide intervals.

The 1,000-replicate paired order-cluster bootstrap gives mean-across-fold
intervals:

| target | delta AUC 95% CI | delta AP 95% CI | delta Spearman 95% CI |
|---|---:|---:|---:|
| LCS | [0.0427, 0.0619] | [0.1506, 0.2017] | [0.0818, 0.1030] |
| PMIS | [0.0472, 0.0656] | [0.1288, 0.1820] | [0.1093, 0.1337] |
| RTS | [0.0617, 0.0824] | [0.1035, 0.1488] | [0.1233, 0.1536] |

Order-level delta intervals cross zero for all targets. This does not negate
the full 15k-order RC-MSTNet order discrimination result; it means the paired
comparison against the old 1k LightGBM sample is underpowered.

Across seeds 42, 2026, and 3407:

```text
AUC standard deviation: 0.0004-0.0008
AP standard deviation:  0.0013-0.0024
order q90 Lift@Top10% coefficient of variation: 0.9%-1.9%
```

The incremental result is stable to initialization, dropout, and batch order.

## Structural ablation

All variants use 100k orders/fold, three rolling folds, seed 2026, and three
epochs.

| model | LCS AUC/AP | PMIS AUC/AP | RTS AUC/AP |
|---|---:|---:|---:|
| Neural tabular | 0.7816 / 0.2432 | 0.7665 / 0.2183 | 0.6563 / 0.1740 |
| + dynamic temporal | 0.8072 / 0.2748 | 0.8034 / 0.2691 | 0.7387 / 0.2512 |
| + local route | 0.8673 / 0.4377 | 0.8729 / 0.4412 | 0.8247 / 0.3876 |
| + route Transformer | 0.8682 / 0.4386 | 0.8730 / 0.4425 | 0.8249 / 0.3883 |
| single target | 0.8704 / 0.4471 | 0.8719 / 0.4383 | 0.8311 / 0.3982 |
| + route auxiliary | 0.8618 / 0.4243 | 0.8674 / 0.4297 | 0.8152 / 0.3734 |

The local route encoder is the dominant structural contribution. The global
Transformer adds only a small increment. Single-target heads help LCS and RTS;
PMIS benefits slightly from shared multi-task representation. Route auxiliary
supervision trades some link accuracy for small, inconsistent order-tail gains.
The frozen formal model is not changed retroactively; a future model-contract
revision should reconsider auxiliary weight and target-specific heads.

Average parameter/training cost ranges from 88k parameters and 45 seconds for
the neural-tabular model to 752k parameters and about 70 seconds for the full
three-epoch model. Peak reserved CUDA memory remains below 442 MB.

## Calibration and uncertainty

Validation-selected isotonic calibration is selected for all folds/targets.
Raw RC-MSTNet probabilities were already well calibrated; test ECE changes to:

```text
LCS  0.00425
PMIS 0.00220
RTS  0.00230
```

Three-seed validation-normalized conformal intervals achieve:

| target | coverage | mean width | uncertainty-error Spearman |
|---|---:|---:|---:|
| LCS | 0.9033 | 0.3493 | 0.4377 |
| PMIS | 0.9020 | 0.2663 | 0.4085 |
| RTS | 0.9001 | 0.3446 | 0.6495 |

Uncertainty is therefore informative for conservative decision rules,
especially for RTS.

## IIS

The movement model remains separate. Test applicability AUC is
0.9974-0.9986. Conditional severity/tail AUC is 0.8450-0.8853. Missing severity
is not filled with zero. Current movement predictions cover only the historical
1k-order planned-route sample, so Stage3 IIS coverage is low and explicitly
flagged.

## Stage3 warehouse

Canonical mapping uses only Stage2 test-day predictions:

```text
Fold 1 test / 20161017 -> Stage3 train
Fold 2 test / 20161018 -> Stage3 validation
Fold 3 test / 20161019 -> Stage3 test
```

The warehouse contains 45,000 orders and passes its leakage audit. It excludes
realized targets and post-trip realized features. LightGBM availability is only
6.7%/1.1%/1.0% at order level; IIS availability is 6.2%/1.1%/0.9%.

## Stage3 prototype results

Test metrics:

| model | target | AUC | AP | Spearman | Lift@Top10% |
|---|---|---:|---:|---:|---:|
| Rule q90 | LCS | 0.7459 | 0.1328 | 0.5429 | 3.3698 |
| Order LightGBM | LCS | 0.7933 | 0.1779 | 0.6281 | 3.9390 |
| DeepSets/route attention | LCS | 0.8092 | 0.1946 | 0.5897 | 4.4760 |
| Rule q90 | PMIS | 0.7772 | 0.2411 | 0.5196 | 3.5178 |
| Order LightGBM | PMIS | 0.8256 | 0.3028 | 0.6024 | 4.1026 |
| DeepSets/route attention | PMIS | 0.8249 | 0.3021 | 0.5721 | 4.1007 |
| Rule q90 | RTS | 0.7613 | 0.3219 | 0.4438 | 3.3988 |
| Order LightGBM | RTS | 0.7810 | 0.3442 | 0.5229 | 3.6288 |
| DeepSets/route attention | RTS | 0.7904 | 0.3593 | 0.5129 | 3.6718 |
| Rule q90 | Overall | 0.7472 | 0.3613 | 0.5150 | 2.9308 |
| Order LightGBM | Overall | 0.7778 | 0.3903 | 0.5132 | 3.1584 |
| DeepSets/route attention | Overall | 0.7882 | 0.4093 | 0.5316 | 3.2683 |

DeepSets/route attention is the best prototype for LCS, RTS, and overall tail
screening. Order LightGBM is marginally better for PMIS and for some continuous
ranking/error metrics. The appropriate current conclusion is a DeepSets main
prototype with order-LightGBM sensitivity analysis, not a universal deep win.

Stage3 currently has one strict temporal train/validation/test chain, not three
independent Stage3 folds. It is therefore a prototype and not yet a finalized
Stage3 model claim.

## Stacking and scale decision

Shallow LightGBM + RC-MSTNet stacking does not consistently improve RC-MSTNet.
Most AUC deltas are negative; occasional AP/Lift gains are target/fold-specific
and occur on very sparse common rows. Stacking is not retained.

The 5k-to-100k improvement shows that 100k is not demonstrably saturated, so a
300k point is scientifically worthwhile. It is not run in this round because
15k/day provides at most 105k train orders/fold and the current disk cannot hold
the required 45k/day upstream intermediates plus three-fold shards safely.
Running 300k requires a compact direct upstream builder or additional storage.

## Final answers

1. RC-MSTNet gains are stable across folds and seeds at link level.
2. Dynamic state helps; local route context supplies the largest gain; the
   Transformer increment is small.
3. Full-sample RC-MSTNet order discrimination is strong, but paired order-level
   gain over old LightGBM OOF is statistically underpowered.
4. Validation-only isotonic calibration is effective; raw calibration was
   already good.
5. Ensemble/conformal uncertainty identifies higher-error observations.
6. Shallow stacking is not consistently better and is rejected.
7. 100k is not proven saturated; 300k is deferred for data/storage reasons.
8. DeepSets is the best overall Stage3 prototype; order LightGBM remains a
   PMIS/continuous-metric sensitivity model.
9. Proposed Stage3 vector: LCS, PMIS, RTS raw/tail; IIS applicability and
   conditional severity/tail; per-dimension uncertainty; overall high-stress
   probability.
10. Stage4 is not ready: Stage3 needs additional rolling held-out dates/folds
    and broader IIS/LightGBM coverage or an explicit missing-modality policy.
