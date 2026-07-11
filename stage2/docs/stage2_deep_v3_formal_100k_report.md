# Stage2 Deep v3 formal 100k rolling report

## Protocol

The formal experiment uses the deployable route-conditioned estimated-entry
time contract:

```text
route proxy: completed map-matched service route
feature clock: estimated link entry time
lagged-state rule: feature availability time < estimated link entry time
train budget: 100,000 orders per fold
validation/test: complete 15,000-order days
folds: 09-15/16/17, 10-16/17/18, 11-17/18/19
model: RC-MSTNet, hidden=128, layers=3, epochs=5
```

The upstream product contains 15,000 orders on each day from 20161009 through
20161019. Estimated-entry-time coverage and strict availability pass rate are
both 100%. Link-state coverage is at least 99.81%.

The three fold-aware tensor products contain exactly 100,000 training orders
per fold, 15,000 validation orders, and 15,000 test orders. The tensor-shard
audit passes for all folds.

## Link-level rolling test results

| target | model | AUC | AP | Spearman | MAE | RMSE | Lift@Top5% |
|---|---|---:|---:|---:|---:|---:|---:|
| LCS | LightGBM route-context | 0.8144 | 0.2775 | 0.5538 | 0.1326 | 0.1710 | 4.6962 |
| LCS | RC-MSTNet 100k | 0.8667 | 0.4376 | 0.6521 | 0.0967 | 0.1478 | 6.6194 |
| PMIS | LightGBM route-context | 0.8099 | 0.2714 | 0.4876 | 0.1222 | 0.1529 | 4.4227 |
| PMIS | RC-MSTNet 100k | 0.8716 | 0.4398 | 0.6072 | 0.0769 | 0.1274 | 6.2694 |
| RTS | LightGBM route-context | 0.7545 | 0.2724 | 0.2564 | 0.1906 | 0.2273 | 4.0916 |
| RTS | RC-MSTNet 100k | 0.8234 | 0.3875 | 0.3948 | 0.1054 | 0.1982 | 5.2956 |

RC-MSTNet gains over LightGBM are:

| target | delta AUC | delta AP | delta Spearman | delta MAE | delta Lift@Top5% |
|---|---:|---:|---:|---:|---:|
| LCS | +0.0523 | +0.1601 | +0.0982 | -0.0358 | +1.9231 |
| PMIS | +0.0617 | +0.1684 | +0.1196 | -0.0453 | +1.8467 |
| RTS | +0.0690 | +0.1150 | +0.1384 | -0.0852 | +1.2041 |

Order-block bootstrap intervals were computed separately for every test fold
with 100 rounds. Across folds, the AUC 95% intervals lie within:

```text
LCS:  0.8600-0.8708
PMIS: 0.8673-0.8752
RTS:  0.8192-0.8279
```

These are absolute RC-MSTNet confidence intervals, not paired confidence
intervals for the gain over LightGBM.

## Order-level tail separation

The earlier absolute event `order q90 stress >= 0.90` was nearly empty and is
not a stable order-level metric. Order audit now defines high stress as the
empirical top decile of each order aggregation within the evaluation split and
uses raw-score aggregation for continuous metrics and tail-probability
aggregation for ranking metrics.

Test-day q90 aggregation results:

| target | order AUC | order AP | order Spearman | Lift@Top5% | Lift@Top10% |
|---|---:|---:|---:|---:|---:|
| LCS | 0.7743 | 0.3200 | 0.5736 | 4.3737 | 3.5038 |
| PMIS | 0.7890 | 0.3329 | 0.5466 | 4.3871 | 3.5727 |
| RTS | 0.7699 | 0.3275 | 0.4645 | 4.4805 | 3.5505 |

## Resource use

```text
tensor shards:       3.19 GB
training time/fold:  112-116 seconds for five epochs
throughput:          4,299-4,457 orders/s
padding efficiency: 99.1%-99.2%
CUDA reserved peak: approximately 442 MB
```

End-to-end wall time is longer than pure training time because each epoch runs
full-day validation and the final checkpoint materializes complete validation
and test link predictions.

## Decision

The formal 100k experiment changes the Stage2 model decision. RC-MSTNet is now
the main candidate for LCS, PMIS, and RTS because it improves overall ranking,
tail detection, continuous error, and order-level tail separation across all
three rolling folds. LightGBM remains a required strong baseline and potential
fusion/calibration component.

This result does not cover IIS, which remains a separate movement-level
applicability/severity task. Before Stage3, the rolling validation predictions
must be used for calibration and uncertainty estimation; train in-sample
predictions must not be used. The completed-route path is still an assigned
route proxy, not a deployable route-choice prediction.
