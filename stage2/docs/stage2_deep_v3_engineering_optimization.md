# Stage2 Deep v3 engineering optimization

## Scope

This optimization round targets the local RTX 4060 Laptop GPU (8 GB VRAM) and
16 GB system RAM. It changes data loading and numerical execution, not the
Stage2 target definitions or the estimated-entry-time deployability contract.

## Implemented changes

1. `max_train_orders` is allocated evenly across all training dates. A 5k or
   10k scaling point no longer silently samples only the first training dates.
2. Long routes retain pickup- and dropoff-side context instead of using only
   the first `max_seq_len` links.
3. Pandas order groups and per-item conversions were replaced by contiguous,
   pre-encoded arrays with order offsets.
4. Lagged-state input was reduced from a sparse `4 x 96` tensor to aligned
   `4 x 24` channels. The removed cells were structural zeros.
5. Route-length bucket batches reduce padding while batches remain shuffled.
6. CUDA AMP, TF32, pinned transfers, non-blocking copies, and stable
   BCE-with-logits tail losses are enabled.
7. Epoch validation accumulates metrics without constructing per-link Python
   rows. Full prediction parquet files are materialized only for the selected
   checkpoint.
8. Manifests record date allocation, encoded bytes, preparation time,
   throughput, padding efficiency, and peak CUDA memory.

## Local benchmark

Configuration:

```text
fold = 1
train orders = 5,000
validation orders = 1,000
test orders = 1,000
sequence length = 96
batch size = 64
hidden dimension = 96
Transformer layers = 2
epochs = 2
```

Observed optimized resource metrics:

```text
data preparation:       8.14 s
training:               4.04 s total / 2 epochs
training throughput:    2,476.7 orders/s
link throughput:        70,447 links/s
encoded train storage:  80.8 MB
peak CUDA allocated:    269 MB
peak CUDA reserved:     500 MB
```

The earlier pandas-per-order implementation required about 708 seconds for two
epochs at the same nominal 5k order/model setting. The samples are not exactly
identical: the optimized benchmark correctly spreads 5k orders over all seven
training days, whereas the earlier loader consumed its budget from the first
date. The timing comparison therefore diagnoses the engineering bottleneck but
is not a scientific model-performance comparison.

Optimized fold-1 test metrics after two epochs:

| target | AUC | AP | Spearman | Lift@Top5% |
|---|---:|---:|---:|---:|
| LCS | 0.8148 | 0.3219 | 0.4883 | 4.3510 |
| PMIS | 0.7952 | 0.2907 | 0.4681 | 4.0475 |
| RTS | 0.7400 | 0.2469 | 0.2211 | 3.7058 |

The complete 5k-order, three-fold rolling smoke test also passed. Across folds,
data preparation took 3.9-4.2 seconds and two training epochs took 3.3-4.2
seconds. Length bucketing achieved 91.6%-92.0% padding efficiency and peak CUDA
reserved memory remained about 830 MB.

Three-fold test means:

| target | AUC | AP | Spearman | Lift@Top5% |
|---|---:|---:|---:|---:|
| LCS | 0.7948 | 0.2579 | 0.4861 | 4.3494 |
| PMIS | 0.7984 | 0.2716 | 0.4548 | 4.0716 |
| RTS | 0.7237 | 0.2331 | 0.2066 | 3.3038 |

These are engineering smoke-test results with only two epochs and 5k balanced
training orders per fold. They establish rolling stability of the optimized
loader; they do not replace the formal larger-scale LightGBM comparison.

## Scale boundary addressed by P5

After P0-P4, `read_dates` still built a multi-day pandas frame before encoding
it. The current 5k/day product supports at most 35k training orders per
seven-day fold, and merely raising `max_train_orders` on a larger product would
have moved the bottleneck back to RAM. P5 below resolves that loader boundary;
the remaining prerequisite for 100k-300k is rebuilding the upstream
route-conditioned estimated-time product at the corresponding order scale.

## P5: disk-backed daily tensor shards

P5 removes the remaining multi-day pandas boundary. A new builder fits fold
metadata incrementally from training dates and writes one mmap-compatible shard
per fold/split/date. Each shard contains:

```text
static_numeric.npy   float16
dynamic.npy          float16
categorical.npy      int32
target.npy           float32
tail.npy             uint8
mask.npy             uint8
offsets.npy          int64
lengths.npy          int32
ids.parquet          lazy-loaded for final prediction only
manifest.json
```

Metadata and selected-order SHA-256 fingerprints prevent stale shards from
being reused after configuration changes. Only one daily pandas frame is live
during fitting/encoding. Training opens the arrays with NumPy mmap and locates
orders through per-day offsets.

`audit_stage2_deep_v3_tensor_shards.py` validates array shapes, finite values,
offset/length consistency, ID row counts, and fold/day metadata fingerprints.
All three 5k folds pass this audit.

The 5k-order three-fold shard product occupies 176.2 MB, including only 1.9 MB
of ID parquet files. Per-fold training input is approximately 41 MB, compared
with about 81 MB for the earlier float32 in-memory arrays.

P5 three-fold test means:

| target | AUC | AP | Spearman | Lift@Top5% |
|---|---:|---:|---:|---:|
| LCS | 0.7962 | 0.2588 | 0.4851 | 4.3300 |
| PMIS | 0.7993 | 0.2731 | 0.4562 | 4.1431 |
| RTS | 0.7266 | 0.2368 | 0.2092 | 3.4292 |

The mmap metrics closely track the P0-P4 in-memory smoke test. Per-fold data
preparation at training startup fell from roughly four seconds to 0.035 seconds;
the main benefit at 100k+ scale is bounded RAM rather than the small 5k timing
difference.
