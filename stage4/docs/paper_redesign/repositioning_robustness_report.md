# R1 repositioning robustness stop report

## Decision

`REPOSITIONING_ROBUSTNESS = STOPPED_AT_NO_REPOSITION_GATE`

The pre-execution reproducibility gate did not pass 4/4. In accordance with Sections 8, 22, and 24 of the frozen taskbook, no repositioning-enabled scenario was run and no scientific storyline category was assigned.

Recommended disposition: `REVISE_REPOSITIONING_ROBUSTNESS`.

## Frozen identity

- Branch: `codex/stage2-v5-micro-transfer`
- Base enhancement commit: `b91fd3e165d7064fb58cddaed90dba51ab13f97f`
- Policy/code pre-registration commit: `94b4eb20580abe455600025568b1e79f3cf9cb63`
- Policy: `TRAIN_TOD_DEMAND_BALANCE`
- Policy version: `stage4_repositioning_r1.1`
- Train reference: Stage1 Train, 20161009--20161024
- Train reference SHA256: `97f9931d9a0d114059699ea92b692df80bb1d54e065c757c075d723d7291ab07`
- Empty deadheading qualification: operational abstraction, not ODD-certified

## No-reposition reproduction

| Anchor | Status | Exact result |
|---|---|---:|
| `MAIN_Q25_M_P70` | completed | yes |
| `MAIN_Q50_M_P70` | completed, gate failed | no |
| `MAIN_Q75_M_P70` | not run after hard stop | not evaluated |
| `BENCH_AV_M` | not run after hard stop | not evaluated |

For Q25, all 19 summary values, the request-outcome fingerprint, and the assignment fingerprint matched canonical output exactly.

For Q50, the rerun produced 18,141 matched requests versus 18,133 canonical, 11,859 patience expirations versus 11,867 canonical, and service rate 0.604700 versus 0.604433. Request and assignment fingerprints both differed. This is a real trajectory-level difference rather than JSON or floating-point presentation noise.

## First divergence diagnostic

The first selected-assignment divergence occurs at 2016-10-31 18:20:30+08:00 for order `ea90853f16ba4c0fcb2c0d8481cb6fd8` and vehicle `HV_S3_02649`.

The vehicle identity, order identity, assignment epoch, path distance (522 m), beta, service time, and pre-assignment vehicle position are identical. The Valhalla raw pickup time differs:

- canonical: 60.000 s;
- rerun: 60.804 s.

This changes corrected pickup ETA from 221.337276 s to 224.303195 s and subsequently changes vehicle timing and later assignments.

The Q50 canonical run recorded zero failed matrix arcs. The no-reposition rerun recorded 980 failed matrix cells, of which 972 succeeded through the already-frozen identical single-route fallback and 8 remained failed. By contrast, Q25 recorded zero matrix failures in both canonical and rerun and reproduced exactly.

A bounded diagnostic repeated the first divergent origin/destination as a one-source Valhalla matrix request 100 times. All 100 calls returned 60.000 s and 0.522 km. This indicates that the observed drift is associated with full-run sparse-matrix batch/cache composition rather than the repositioning policy or a different route geometry. It does not establish a canonical baseline implementation defect.

## Resource and execution discipline

- Runs were sequential and CPU-only.
- No dense order-by-fleet or repositioning matrix was constructed.
- Q50 wall-clock runtime was 4,420.664 s.
- Q50 peak RSS reported by the runner was 1,337.383 MB.
- No GPU was used.
- Q75, all-AV M, and all four enabled R1 scenarios were not started after the stop condition.

## Scientific status

No valid baseline/reposition pairs exist, so system effects, N5/N0 effects, patience effects, reposition burden, and the 17:00--18:59 effect cannot be reported. `SUPPORTS`, `QUALIFIES`, and `CHANGES CURRENT STORY` are therefore all unassigned.

This stop is not labeled `IMPLEMENTATION DEFECT FOUND`: the evidence establishes failure of the required exact reproducibility gate, but it does not show that the frozen canonical baseline itself is scientifically incorrect.

## Required next decision

Before R1 can resume, a new authorized protocol must resolve how Valhalla sparse-matrix batch variability is handled. The frozen taskbook does not permit substituting a contemporaneous paired control or relaxing exact reproduction after observing this failure.

Gamma frontier remains unauthorized.
