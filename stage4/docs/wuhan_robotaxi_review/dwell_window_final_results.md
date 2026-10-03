# Deterministic dwell window — six conditions complete

## Material Passport

2026-09-26. academic-research-suite / experiment-agent. Status: COMPLETE,
6/6 conditions executed and checked. Protocol/execution baseline 168b5ba;
authorized q75/120 retry d1603d4. Frozen scientific configuration SHA256:
53a0eae43d6c8abae1017ae54874d8224a6123d5fded574785dbfaeaa9bbd57f.

The user authorized a new deterministic-routing baseline after a single-source
versus batch-matrix discrepancy was identified. These results are not exact
reproductions or replacements of the original 41 batch-routing scenarios.
No manuscript, model, capability profile, or canonical output was modified.

## Design and estimand

q50/q75 are baseline-normalized AV active-hour levels, not percentages of the
mixed fleet. Each condition starts from its q-specific verified native canonical
10:30 state. Extra pickup overhead is 0/60/120 s for both HV and AV, beginning
with post-checkpoint assignments. Pre-existing tasks remain unchanged. Routing
is SINGLE_SOURCE_MATRIX throughout; M profile, 0.70 passenger acceptance, Gamma,
cost policy, Top-K, zero lead and 300-s arrival patience stay frozen.

Warmup: 10:30–11:00. Measurement release cohort: [11:00,12:00). All six cohorts
contain exactly the same 1,667 order IDs. No new requests after 12:00; waiting
requests reach assignment/deadline and all accepted tasks finish before scoring.
This is conditional window continuation, not a full-day dwell counterfactual.

## Main results

| AV level | Added pickup s | Matched | Expired | Matched % | Delta vs corresponding new zero control |
|---|---:|---:|---:|---:|---:|
| q50 | 0 | 1,068 | 599 | 64.07 | — |
| q50 | 60 | 1,034 | 633 | 62.03 | -34 / -2.04 pp |
| q50 | 120 | 1,017 | 650 | 61.01 | -51 / -3.06 pp |
| q75 | 0 | 700 | 967 | 41.99 | — |
| q75 | 60 | 682 | 985 | 40.91 | -18 / -1.08 pp |
| q75 | 120 | 655 | 1,012 | 39.29 | -45 / -2.70 pp |

In this frozen window, adding 120 s reduces completed-cohort matching by 4.78%
relative to the q50 zero control and 6.43% relative to the q75 zero control.
The q50-over-q75 ordering persists across all three durations. This is a
descriptive result for these conditions, not a theorem or an isolated causal
effect of AV addition: the two q levels also have different HV/session supply.

## Time chain and fleet availability

| Level | Extra s | Mean request→arrival s | Mean request→service-start s | HV idle-online h | AV idle-online h |
|---|---:|---:|---:|---:|---:|
| q50 | 0 | 176.16 | 176.16 | 108.40 | 231.37 |
| q50 | 60 | 176.07 | 236.07 | 103.15 | 228.23 |
| q50 | 120 | 178.58 | 298.58 | 95.27 | 226.12 |
| q75 | 0 | 187.49 | 187.49 | 40.80 | 350.90 |
| q75 | 60 | 187.31 | 247.31 | 37.95 | 348.26 |
| q75 | 120 | 186.65 | 306.65 | 38.58 | 344.15 |

Waiting means are computed among matched orders, whose identities change with
condition; they are not paired passenger effects. Service-start waiting may
exceed 300 s because patience constrains arrival, not the end of stationary
pickup service. Online idle hours integrate fixture windows minus physical
busy intervals within [11:00,12:00), including inherited tasks. They are not
guaranteed matchable hours. HV idle hours alone are not monotone for q75
(37.95→38.58), so do not claim each resource statistic must decrease with dwell.

## Integrity, execution and provenance

- All six matched + expired totals equal 1,667; all accepted tasks drained.
- Same exact cohort IDs verified across the six outcome files.
- Maximum pickup-time conservation error < 1e-9 s; service-time error 0 s.
- No overlapping physical assignments; predicted HV admission includes dwell.
- Native request/position/availability reconciliation passed.
- Routing failures: 0 in every condition; sparse CPU processing, no GPU.
- Original five completed conditions were not rerun.
- q75/120 first attempt hit its 1,800-s budget and remains preserved. User
  authorized a separate retry with a 3,600-s execution budget only; it completed
  in 1,373.19 s (22.89 min), with peak RSS 396.82 MiB.
- Highest recorded RSS across completed runs: 431.25 MiB.
- Successful-condition runtimes total 4,789.83 s, excluding the failed attempt
  and setup/diagnostic work. Different wall runtimes are not scientific outcomes.
- All 26 files under the original result directory retained their hashes;
  scientific config and checkpoint hashes also remained unchanged.
- Seven related tests passed before the original execution; retry module
  compilation passed. This is not a second independent repeat of all conditions.

Actual post-checkpoint HV overruns are 76/68/70 for q50 and 34/30/29 for q75
(0/60/120 s). They use realized durations, whereas admission uses predicted
durations. Lower overrun counts alongside fewer assignments are not evidence
of improved prediction or session compliance quality.

## Interpretation boundary

The new sensitivity evidence shows that added pickup occupancy changes later
matching in this setting; it does not overturn the observed cross-q ordering.
No empirical dwell-time calibration has been established: the extra 60/120 s
were predefined sensitivity levels, not fitted to the external paper's data.
Existing GPS trip durations may already include stops; nothing was subtracted
from them. Results remain conditional on canonical prehistory, one chosen
window, frozen predictions and the day-specific replay environment. No new
early-release inference, passenger choice model or full-day validation occurred.
One deterministic realization per condition supports no sampling p-values or
confidence intervals, and the new zero control must not be mixed with the old
batch-routing control in effect calculations.

The authorized six-condition task is now complete. No additional experiment or
manuscript modification is started automatically.

## Artifacts

- `strict_state_results/dwell_window_complete_summary.json`: all six aggregates.
- `strict_state_results/dwell_window_retry_receipt.json`: retry and preserved hashes.
- Local original products: `stage4/output/paper_enhancement/dwell_deterministic_window/`.
- Local retry products: `stage4/output/paper_enhancement/dwell_deterministic_window_retry1/`.
- Previous partial report remains an audit record, superseded by this report.
