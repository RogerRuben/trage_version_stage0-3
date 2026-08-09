# Stage 2 v5.2 Phase B1 frozen transfer-tuning report

Status: `PASS — PHASE_B1_FROZEN_COMPLETE`

Phase C authorization: `NO`

## Frozen scope

- Train: `20161009-20161018`
- Validation: `20161019-20161020`
- Models: M0, M1, M2, M3, M4-p25, M4-p50, M4-p75
- Frozen tau: Train-support p25, `tau = 3.0`
- Overall unique validation traversals: `1,439,737`
- No training, inference, tau selection, or hyperparameter change was performed while producing this report.

Tau 3 was selected by the preregistered metric. The overall differences among
tau 3, 39, and 382 are very small and do not establish statistical superiority.

## Overall MAE

| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |
|---|---:|---:|---:|---:|---:|
| M0 | 0.120961 | 0.183083 | 0.063997 | 0.005525 | NA |
| M1 | 0.120349 | 0.152894 | 0.063459 | 0.004881 | 0.027546 |
| M2 | 0.120072 | 0.152865 | 0.063320 | 0.004693 | 0.027325 |
| M3 | 0.120056 | 0.152717 | 0.063284 | 0.004662 | 0.027310 |
| M4-p25 | 0.120050 | 0.152826 | 0.063278 | 0.004651 | 0.027312 |
| M4-p50 | 0.120051 | 0.152812 | 0.063277 | 0.004660 | 0.027307 |
| M4-p75 | 0.120047 | 0.152784 | 0.063280 | 0.004658 | 0.027305 |

The transfer framework has no material overall degradation. M3 and M4 are
extremely close, so the overall table cannot establish that support-aware
gating is effective.

## Support-group sample counts

| Group | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| Overall target-eligible unique traversals | 243,789 | 530,556 | 357,834 | 530,556 |
| Low-support target-eligible unique traversals | 148 | 424 | 231 | 424 |
| Unseen target-eligible unique traversals | 42 | 141 | 72 | 141 |

Counts are unique traversals with an available label for the named target.
Every low/unseen target group is non-empty, but the groups are small. The
evaluation manifests do not store a separate group-wide population independent
of target-label availability, so this report does not reconstruct or invent it.

## Low-support MAE

| Model | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| M1 | 0.134206 | 0.360233 | 0.096989 | 0.100175 |
| M2 | 0.133493 | 0.362481 | 0.095796 | 0.097088 |
| M3 | 0.131420 | 0.358460 | 0.096139 | 0.099593 |
| M4-p25 | 0.131255 | 0.356724 | 0.096615 | 0.102211 |
| M4-p50 | 0.132019 | 0.357717 | 0.095451 | 0.100173 |
| M4-p75 | 0.133370 | 0.359874 | 0.095647 | 0.097759 |

Relative improvement versus M1; positive means lower MAE:

| Model | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| M2 | +0.531% | -0.624% | +1.230% | +3.082% |
| M3 | +2.076% | +0.492% | +0.876% | +0.581% |
| M4-p25 | +2.199% | +0.974% | +0.386% | -2.032% |
| M4-p50 | +1.629% | +0.698% | +1.586% | +0.002% |
| M4-p75 | +0.623% | +0.100% | +1.384% | +2.413% |

M4-p25 improves three low-support targets relative to M1, but the gains are
small and stop worsens. Its mean four-target relative improvement is only about
`0.382%`, below the preregistered 2% continuation threshold.

## Unseen-edge MAE

| Model | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| M1 | 0.134644 | 0.390181 | 0.085688 | 0.103588 |
| M2 | 0.131572 | 0.394700 | 0.087560 | 0.113385 |
| M3 | 0.133540 | 0.385205 | 0.086864 | 0.103077 |
| M4-p25 | 0.130600 | 0.372771 | 0.086894 | 0.116635 |
| M4-p50 | 0.131328 | 0.381155 | 0.086658 | 0.114461 |
| M4-p75 | 0.131461 | 0.391930 | 0.086884 | 0.114357 |

M4-p25 versus M2 structure-only is `+0.739%` Acc RMS, `+5.556%` crawl,
`+0.761%` Speed CV, and `-2.866%` stop. These mixed results do not demonstrate
stable unseen improvement. Any M4/M2 difference may arise from shared-backbone
fine-tuning and cannot be attributed to the edge gate itself.

## M3 to M4-p25 incremental value

Positive means M4-p25 has lower MAE than M3.

| Group | Acc RMS | Crawl | Speed CV | Stop |
|---|---:|---:|---:|---:|
| Overall | +0.005% | -0.071% | +0.009% | +0.247% |
| Low-support | +0.126% | +0.484% | -0.495% | -2.629% |
| Unseen | +2.202% | +3.228% | -0.035% | -13.153% |

Support-aware weighting adds no stable incremental value over ordinary
ID+structure concat in B1.

## Frozen scientific conclusion

Classification: `CASE_C`.

> B1 shows no convincing support-aware transfer evidence; keep the protocol
> frozen and test temporal robustness before any adoption claim.

This does not negate the small overall structured-transfer improvements. It
does require the support-aware claim to remain weak/preliminary and prohibits
retuning the model to obtain a more favorable case.

## NO_HISTORY temporal sentinel

- Existing shard files inspected: `132`
- Total audited temporal tokens: `9,954,428`
- Strict observed-history tokens: `9,954,044`
- NO_HISTORY fallback token copies: `384`
- Unique physical NO_HISTORY rows: `278`
- Fallback share: `0.003858%`
- Minimum strict observed-history age: `1.0 s`
- Minimum NO_HISTORY sentinel audit age: `1.0 s`
- Temporal leakage: `0`

The fallback is an explicit no-history sentinel represented by a positive audit
age; it carries no target-day future observation. It is not a one-second-old
real observation.

## Evidence

- Evidence bundle: `stage2_v5_2_phase_b1_evidence_bundle.json`
- Phase verification: `stage2_v5_2_phase_b1_verification.json`
- Status manifest: `stage2_v5_2_status_manifest.json`
- Tau freeze: `stage2/output_v5_2/transfer_tuning/stage2_v5_2_tau_freeze.json`

Phase C remains unauthorized pending review.
