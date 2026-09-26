# Stage4 Result Analysis

This report is a deterministic contrast analysis of the frozen Test31 scenarios. It reports no p-values, inferential confidence intervals, causal effects, or population-level behavioral estimates.

## 1. Main fleet-transition result

Across the 27 main scenarios, marginal mean service rate fell from 0.7258 at q_A=.25 to 0.5984 at .50 and 0.3924 at .75. The .25→.75 change was -0.3334 (-45.9%). This is a modeled effective-capacity result, not a claim about AV technology outside Test31.
Conditional P95 pickup waits remained tightly clustered (289.3–293.0 s) while service rates ranged 0.3544–0.7309; similar waits among served passengers therefore did not imply similar platform performance.

## 2. Acceptance interaction

Raising p_A from .40 to 1.00 changed service at q_A=.25 by C/M/A = 0.0053/0.0087/0.0087, versus 0.0363/0.0477/0.0436 at q_A=.75. The descriptive q_A×p_A contrasts were C/M/A = 0.0311/0.0390/0.0350.
The acceptance parameter becomes operationally more consequential when a larger fraction of active vehicle-hours is assigned to AVs, but it remains a frozen scenario-level probability rather than a calibrated Xi'an behavioral estimate.

## 3. Capability interaction

The C→A service gain at q_A=.25 was 0.0024/0.0085/0.0058 for p_A=.40/.70/1.00; at q_A=.75 it was 0.0254/0.0330/0.0327. Capability improvement mitigated reference-envelope restrictions, but did not restore the all-HV service level.

## 4. Benchmark/extreme interpretation

The all-HV benchmark served 0.7889 of requests. The M-profile mixed scenarios at p_A=.70 served 0.7297, 0.6044, and 0.4013 at q_A=.25/.50/.75; the all-AV M composition extreme served 0.1515. Active vehicle-hour substitution therefore did not preserve effective service capacity in this modeled system.
All-AV cases are composition extremes, not performance upper bounds.

## 5. Temporal mechanism

In the defined morning/evening peak windows, cohort service for BENCH_HV versus MAIN_Q25/50/75_M_P70 and BENCH_AV_M was 0.7705/0.6959/0.5661/0.3743/0.1381; corresponding mean queue pressure was 0.1222/0.1512/0.1921/0.2298/0.2566; mean total available stock was 378.5/370.6/377.6/413.5/467.0. Service fell and queue pressure rose even as more total vehicles remained available, supporting a modeled effective-service-capacity constraint rather than a shortage of logged total stock. The outputs do not record HV/AV available-stock counts separately, so the analysis does not attribute that constraint to a fabricated vehicle-type inventory path.

## 6. Multi-family ODD result

Mean positive assigned-exposure shares for C were static/dynamic/speed = 0.9248/0.7739/0.7073; for M = 0.8698/0.4764/0.0006; for A = 0.6394/0.1822/0.0000. Speed was active for C, nearly inactive for M, and inactive for A: the dominant reference-envelope dimensions changed with the assumed capability profile.

## 7. ODD policy trade-off

STRICT/REFERENCE/UNCONSTRAINED service rates were 0.5532/0.6038/0.6044, with AV assignment shares 0.0113/0.1244/0.1217. Relative to UNCONSTRAINED, REFERENCE service loss was 0.0007 (0.1%) and STRICT loss was 0.0512 (8.5%).
REFERENCE exposure reductions versus UNCONSTRAINED were static/dynamic/speed = 0.0962/0.0563/not-defined (zero unconstrained denominator). STRICT operates at the zero-exposure boundary; this is not evidence of safety.
REFERENCE Gamma was calibrated once from q_A=.25, profile M, p_A=1, UNCONSTRAINED and then held fixed. Positive slack or zero binding epochs does not imply that the policy had no assignment effect.

## 8. Cost robustness

Within eta-matched epsilon=.05 versus 0 comparisons, normalized cost per matched order changed by -1.60% to -0.54%, AV assignment share by -0.0174 to 0.0145, and P95 wait by -0.19 to 0.29 s. These are within-eta comparisons; raw cost levels are not compared as if the objective were invariant across eta.
Each dispatch decision permits a bounded local pickup-objective relaxation; full-day pickup performance emerges endogenously and is not guaranteed to worsen by at most 5%.

## 9. Managerial implications

1. Higher modeled AV penetration reduced marginal service by 0.3334 between q_A=.25 and .75; this applies to the frozen Test31 fleet/session construction.
2. Acceptance mattered more at high penetration: the M-profile p_A gain increased from 0.0087 at q_A=.25 to 0.0477 at .75; p_A remains a scenario parameter.
3. Capability C→A improved high-penetration service by 0.0330 at q_A=.75,p_A=.70, but the resulting service remained below the all-HV benchmark by 0.3775.
4. Reference-envelope relevance shifted across profiles: speed positive activity moved from 0.7073 (C) to 0.0006 (M) and 0.0000 (A).
5. REFERENCE retained 99.9% of UNCONSTRAINED service while reducing static and dynamic final exposure by 9.6% and 5.6%; this is an operational-envelope trade-off, not a safety probability.

## 10. Limitations / wording constraints

- Deterministic frozen-scenario contrasts provide no replication-based uncertainty, inferential significance, or causal identification.
- Waiting quantiles are conditional on served passengers; service rate and expiration remain the capacity outcomes.
- Available vehicle stock is logged only in total, not separately for HV and AV.
- Exposure is reference-envelope utilization, not accident risk, failure probability, or certification.
- Cost is normalized and changes definition with eta; no currency conversion or cross-eta raw-cost ranking is made.
- No new simulation, routing, solver run, parameter tuning, or alternative seed was used.

Recommendation: `GO_PAPER_RESULTS_DRAFT`
