# Deterministic dwell window — 5/6 complete

> Superseded by [six-condition final results](dwell_window_final_results.md).
> The authorized q75/120 retry completed; this partial report is preserved as history.

## Material Passport

2026-09-26; academic-research-suite / experiment-agent. Execution commit
168b5ba30054420d94318d679cf8058990e85757. Frozen configuration SHA256:
53a0eae43d6c8abae1017ae54874d8224a6123d5fded574785dbfaeaa9bbd57f.
Status: PARTIAL, not six-condition completion. No manuscript edits.

The process continued after the conversation was interrupted. Five conditions
completed; q75 / 120 s hit the preregistered 1,800-s per-condition wall-time
budget and exited. Last periodic progress was simulation second 41,400
(11:30); the exact final processed epoch was not persisted. No accepted complete
result exists for that condition. Do not infer its outcome from partial progress.

## Completed results

All conditions use the same 1,667-order release cohort [11:00,12:00), native
10:30 initialization and SINGLE_SOURCE_MATRIX. Results are compared with the
new q-specific zero-overhead controls, not the original 41 scenarios.

| q level | Extra pickup s | Matched | Expired | Matched % | Change vs new zero control |
|---|---:|---:|---:|---:|---:|
| q50 | 0 | 1,068 | 599 | 64.07 | — |
| q50 | 60 | 1,034 | 633 | 62.03 | -34 orders / -2.04 pp |
| q50 | 120 | 1,017 | 650 | 61.01 | -51 orders / -3.06 pp |
| q75 | 0 | 700 | 967 | 41.99 | — |
| q75 | 60 | 682 | 985 | 40.91 | -18 orders / -1.08 pp |
| q75 | 120 | NOT COMPLETE | NOT COMPLETE | NOT COMPLETE | NOT ESTIMATED |

In the completed conditions, the matched-order mean request-to-arrival remains
roughly stable (q50: 176.16 / 176.07 / 178.58 s; q75: 187.49 / 187.31 s).
Request-to-service-start adds the configured 0/60/120 s. These are means among
different selected matched cohorts, not paired passenger improvements.

Available online vehicle-hours in the measurement window:

| q level | Extra s | HV available h | AV available h |
|---|---:|---:|---:|
| q50 | 0 | 108.40 | 231.37 |
| q50 | 60 | 103.15 | 228.23 |
| q50 | 120 | 95.27 | 226.12 |
| q75 | 0 | 40.80 | 350.90 |
| q75 | 60 | 37.95 | 348.26 |

The completed runs are consistent with extra stationary time reducing subsequent
service and online idle time in this conditional replay. They do not establish
a full-day effect, empirical dwell calibration, or a general monotonicity theorem.
q75 has a different fleet/session composition, not merely more vehicles added
to an unchanged q50 fleet. Cross-q differences are therefore not a pure causal
effect of adding AVs.

## Integrity and resources

- All five matched + expired totals equal 1,667; accepted trips fully drained.
- Pickup conservation maximum error < 1e-9 s; service conservation error 0 s.
- Predicted HV completion includes overhead and passes session admission.
- Actual HV overruns are 76/68/70 for q50 0/60/120 and 34/30 for q75 0/60;
  these are post-checkpoint assignments, not only the measurement cohort.
  They reflect predicted-vs-realized duration differences, not permission to
  omit overhead from admission.
- Zero routing failures in all five completed runs.
- Completed-condition runtimes total 3,416.64 s, excluding the timed-out run.
- Recorded peak RSS across completed conditions: 431.25 MiB; last periodic
  q75/120 RSS: 395.73 MiB. No observed memory-limit event; no GPU.
- Config and checkpoint hashes unchanged. No result-dependent tuning.

The final condition's slowdown cause is not established by these logs. A wall
time budget firing is not evidence of an algorithmic correctness failure or an
out-of-memory event. There was no automatic retry, following the experiment
execution skill's failure-handling rule.

## Completion boundary

Preserve all original directories and the stopped batch summary. With explicit
authorization, retry only q75/120 into a new attempt directory, keeping scientific
settings and checkpoint unchanged. A proposed 3,600-s wall-time limit would be
an execution-budget amendment, not a change to routing/dispatch parameters;
record it explicitly before running. The five completed conditions need no rerun.

Small machine-readable evidence: `strict_state_results/dwell_window_partial_summary.json`.
Full local products: `stage4/output/paper_enhancement/dwell_deterministic_window/`.
Seven related tests passed before execution. This report is descriptive;
one deterministic run per condition supports neither p-values nor sampling
confidence intervals. No all-six conclusion is claimed.
