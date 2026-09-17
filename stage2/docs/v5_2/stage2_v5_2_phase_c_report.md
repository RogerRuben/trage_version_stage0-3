# Stage 2 v5.2 Phase C development report

- Direction: `FAIL`
- Formal spatial adopt: `FALSE`
- Phase D authorized: `NO`
- M5 authorized: `NO`

## Frozen protocol

- Protocol hash: `725a71e76c4685f3a8ba0b0b584a9e10b29df426723a11313cf3a89b2bac6790`
- Train: `20161009-20161021`
- Validation: `20161022-20161023`
- Calibration: `20161024`
- Evaluation: `20161025-20161027`
- Frozen tau: `p25 / 3.0`

## M4 frozen tau consumption

- Relationship audit: `PASS`
- Tau freeze file SHA: `5900d6184d151d8093528f1fa04a1afd73a75d07fd0187da7da22abdb44296ec`
- Transfer-tuning support SHA: `12303d329143440041f5271413796dfb69ee12e7dcb1023a235345f908bcfdae`
- Development support SHA: `4bc907af5990c93f23730978ab4f2c049a5ed2ff7fb87ca9c5157bd6b33c2d98`

## Aggregate 20161025-20161027

| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.119805 | 0.199367 | 0.063480 | 0.006030 | NA |
| M1 | 0.119277 | 0.171922 | 0.063041 | 0.005321 | 0.029356 |
| M2 | 0.118909 | 0.169546 | 0.062753 | 0.005457 | 0.029034 |
| M3 | 0.118869 | 0.169445 | 0.062721 | 0.005425 | 0.029015 |
| M4 | 0.118870 | 0.169337 | 0.062726 | 0.005449 | 0.029016 |

## Selected checkpoints

| Model | Path | SHA-256 |
|---|---|---|
| M0 | `stage2/output_v5_2/development/M0/model/m0_micro_tree.joblib` | `1139c9aba37f20444c8d9c7ed6e8592328b7fbc9b3c77d07536d2f23a82542f8` |
| M1 | `stage2/output_v5_1/development/deep_model/best_model.pt` | `d4e05f4197ab163d5537e98b16ca5b46746284db9ca4ee3029c55a3295982bb2` |
| M2 | `stage2/output_v5_2/development/M2/epoch_004.pt` | `bf2ef5f45a10b6a91282fadc99e3638be4d1569088de6c76cc2226b8b65dd0fc` |
| M3 | `stage2/output_v5_2/development/M3/epoch_004.pt` | `965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde` |
| M4 | `stage2/output_v5_2/development/M4/epoch_004.pt` | `dba21e9ae3c0e78e68a8986ef4119228d7707eb5ab7118c972ffa82872fc92a2` |

## Low-support MAE

| Model | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| M0 | 0.129459 | 0.367438 | 0.091756 | 0.103851 |
| M1 | 0.124076 | 0.361517 | 0.095580 | 0.105073 |
| M2 | 0.124879 | 0.346513 | 0.091302 | 0.104912 |
| M3 | 0.124457 | 0.345119 | 0.091629 | 0.108872 |
| M4 | 0.123937 | 0.347341 | 0.091508 | 0.111224 |

## Unseen MAE

| Model | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| M0 | 0.132259 | 0.400642 | 0.086325 | 0.111409 |
| M1 | 0.133921 | 0.422654 | 0.088135 | 0.102369 |
| M2 | 0.134190 | 0.407184 | 0.089202 | 0.113762 |
| M3 | 0.134583 | 0.414275 | 0.087765 | 0.111445 |
| M4 | 0.135644 | 0.400834 | 0.090381 | 0.121428 |

## Evaluation support counts

Counts are target-valid unique traversals and are identical across paired models.

| Group | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| overall | 363917 | 780240 | 529941 | 780240 |
| low | 208 | 654 | 337 | 654 |
| unseen | 56 | 176 | 85 | 176 |

## Per-date overall MAE

### 20161025

| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.119615 | 0.202640 | 0.063598 | 0.006091 | NA |
| M1 | 0.118978 | 0.175066 | 0.063129 | 0.005415 | 0.029717 |
| M2 | 0.118600 | 0.172847 | 0.062859 | 0.005518 | 0.029401 |
| M3 | 0.118529 | 0.172680 | 0.062810 | 0.005474 | 0.029379 |
| M4 | 0.118525 | 0.172582 | 0.062815 | 0.005500 | 0.029378 |

### 20161026

| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.119971 | 0.197127 | 0.063510 | 0.006018 | NA |
| M1 | 0.119441 | 0.169712 | 0.063078 | 0.005371 | 0.029386 |
| M2 | 0.119147 | 0.167300 | 0.062804 | 0.005531 | 0.029051 |
| M3 | 0.119112 | 0.167255 | 0.062777 | 0.005501 | 0.029044 |
| M4 | 0.119108 | 0.167143 | 0.062790 | 0.005527 | 0.029049 |

### 20161027

| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.119829 | 0.198331 | 0.063330 | 0.005982 | NA |
| M1 | 0.119412 | 0.170987 | 0.062916 | 0.005176 | 0.028964 |
| M2 | 0.118980 | 0.168490 | 0.062597 | 0.005323 | 0.028649 |
| M3 | 0.118966 | 0.168398 | 0.062577 | 0.005299 | 0.028621 |
| M4 | 0.118976 | 0.168284 | 0.062574 | 0.005318 | 0.028619 |

## Frozen relative comparisons

Positive percentages favor M4.

| Comparison | Group | Acc RMS | Crawl | Speed CV | Stop | Four-target mean |
|---|---|---:|---:|---:|---:|---:|
| M4 vs M1 | overall | 0.341% | 1.504% | 0.500% | -2.394% | -0.012% |
| M4 vs M1 | low | 0.112% | 3.921% | 4.261% | -5.855% | 0.610% |
| M4 vs M1 | unseen | -1.287% | 5.163% | -2.548% | -18.618% | -4.323% |
| M4 vs M3 | overall | -0.000% | 0.064% | -0.008% | -0.435% | -0.095% |
| M4 vs M3 | low | 0.417% | -0.644% | 0.132% | -2.161% | -0.564% |
| M4 vs M3 | unseen | -0.788% | 3.245% | -2.981% | -8.958% | -2.371% |
| M4 vs M2 | unseen | -1.084% | 1.560% | -1.322% | -6.739% | -1.896% |

## Directional gate

- Low-support wins vs M1: `3/4`
- Low-support mean relative improvement: `0.610%`
- Overall no target degrades over 2%: `FALSE`
- Unseen no worse than M2: `FALSE`
- Pace guard: `PASS`
- Temporal leakage: `0`

The pre-registered continuation gate does not pass. The result is classified as `C-FAIL`. Structured representation retains modest value, but the support-aware spatial transfer path has direct counter-evidence from the overall-stability, unseen-versus-M2, and M3-to-M4 comparisons. Spatial transfer expansion must stop; no retuning or rerun is authorized.

## Runtime and memory

- Instrumented wall-clock total: `3023.9 s`
- Peak RSS: `not available` (the host CIM query was permission denied; no estimate was substituted).

## Verification

- Base combined suite: `128 passed`
- GPU v5.2 suite: `81 passed`
- compileall: `PASS`

Phase D remains unauthorized. No M5/M6, rolling folds, tau reselection, or 20161028-30 data were run.
