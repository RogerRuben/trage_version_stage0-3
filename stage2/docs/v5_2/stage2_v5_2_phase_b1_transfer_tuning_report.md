# Stage 2 v5.2 Phase B1 transfer-tuning report

Status: `PASS — PHASE_B1_TRANSFER_TUNING_COMPLETE`

## Scope

- Train: `20161009-20161018`
- Validation and tau selection: `20161019-20161020`
- Models: M0, M1, M2, M3, and M4 at Train-support p25/p50/p75
- Transfer source rows: `8,594,152`
- Audited temporal tokens: `9,954,428`; leakage count: `0`
- Unique validation traversals: `1,439,737`
- No dates after 20161020 were used for model or tau selection.

## Core validation MAE

| Model | acceleration_rms | crawl | speed_cv | stop | pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.120961 | 0.183083 | 0.063997 | 0.005525 | NA |
| M1 | 0.120349 | 0.152894 | 0.063459 | 0.004881 | 0.027546 |
| M2 | 0.120072 | 0.152865 | 0.063320 | 0.004693 | 0.027325 |
| M3 | 0.120056 | 0.152717 | 0.063284 | 0.004662 | 0.027310 |
| M4 p25 | 0.120050 | 0.152826 | 0.063278 | 0.004651 | 0.027312 |
| M4 p50 | 0.120051 | 0.152812 | 0.063277 | 0.004660 | 0.027307 |
| M4 p75 | 0.120047 | 0.152784 | 0.063280 | 0.004658 | 0.027305 |

## Tau decision

The frozen selection metric was macro normalized MAE over the four core micro
targets; pace and RTS were excluded. Candidate scores were p25 `0.9867471`, p50
`0.9871989`, and p75 `0.9870411`, so the minimum was p25.

- Selected candidate: `p25`
- Selected tau: `3.0`
- Rolling reselection: prohibited
- Freeze file SHA-256: `5900d6184d151d8093528f1fa04a1afd73a75d07fd0187da7da22abdb44296ec`

This result freezes tau for later phases. It is not the three-fold spatial
adoption decision and does not authorize M5, Phase C, or Phase D by itself.

## Execution evidence

- Phase B0 smoke: `stage2_v5_2_phase_b0_smoke.json` — PASS
- Metadata audit: `stage2_v5_2_input_metadata_audit.json` — PASS
- One-bucket correctness: `stage2_v5_2_one_bucket_correctness.json` — PASS
- Real-kernel scaling benchmark: `stage2_v5_2_performance.json` — PASS (14/14)
- Tau metrics, selection, and freeze: `stage2/output_v5_2/transfer_tuning/`
