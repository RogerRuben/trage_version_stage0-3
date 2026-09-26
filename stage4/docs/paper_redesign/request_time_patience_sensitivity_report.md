# Request-time / patience sensitivity

**RT_SENSITIVE — QUALIFIES_CURRENT_STORY.** Gate losses remain, but their magnitude and sampled cross-fleet ordering depend on the latent release-time assumption. No full-day service conclusion is inferred.

## Recovered transformation

[Original Train parameters](recovered_rt_parameters_report.md) reproduce the old fingerprint. Gap P25/P50/P75 = 420/748/1733 seconds. OD-dependent lower bound, stable jitter, clipping and UTC boundary are unchanged. No arbitrary offsets were substituted.

The transform covers all 30000 Test31 replay orders. As in the canonical scenario, the loader includes 60 seconds after local midnight: two orders start there. A strict midnight cutoff would drop them.

| Scenario | Orders | Mean lead (s) | P50 (s) | P90 (s) | Clipped share |
| --- | ---: | ---: | ---: | ---: | ---: |
| RT-Low | 30000 | 311.013 | 283.474 | 473.908 | .167 |
| RT-Base | 30000 | 498.148 | 495.457 | 611.825 | 0 |
| RT-High | 30000 | 1308.781 | 1361.268 | 1484.796 | 0 |

## Conditional paired design

Four physical states: q_A=.50/.75, M, acceptance=.70, at 12:00 and 17:30 local. q50 source: ODD_Q50_M_P70_REFERENCE; q75: MAIN_Q75_M_P70. Clocks were chosen before RT results; both are clocks used by the existing ten-state study. Exact canonical fixture selection is restored and positions read from frozen assignment histories; no new fleet policy or simulation.

Within each state, vehicle identity/position/availability, route descriptors, routing clock, acceptance CRN, gate rules, K20 and 300-second patience stay fixed. Only release and derived waiting membership/failed-round proxy/remaining patience change. Original microsecond timestamps remain unquantized.

This is **conditional on canonical prior assignments**, not counterfactual history. Frozen route descriptors are held fixed, not asserted to be available at an earlier RT release. This is a mechanical timing sensitivity, not an earlier-decision M3 forecast validation. Waiting cohorts differ; absolute M increases are not matched-order treatment effects.

## Results

M-before is maximum AV matching after routing, before patience. M-after is solver-eligible capacity, not selected assignments.

| q_A | Clock | Variant | Waiting | M-before | M-after | M retention | Imminent |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| .50 | 12:00 | Zero | 76 | 14 | 4 | 28.57% | 8 |
| .50 | 12:00 | Low | 136 | 36 | 9 | 25.00% | 6 |
| .50 | 12:00 | Base | 136 | 27 | 7 | 25.93% | 17 |
| .50 | 12:00 | High | 153 | 36 | 10 | 27.78% | 15 |
| .50 | 17:30 | Zero | 85 | 10 | 1 | 10.00% | 13 |
| .50 | 17:30 | Low | 143 | 28 | 3 | 10.71% | 14 |
| .50 | 17:30 | Base | 133 | 37 | 4 | 10.81% | 15 |
| .50 | 17:30 | High | 164 | 42 | 8 | 19.05% | 16 |
| .75 | 12:00 | Zero | 102 | 14 | 3 | 21.43% | 9 |
| .75 | 12:00 | Low | 137 | 36 | 9 | 25.00% | 7 |
| .75 | 12:00 | Base | 136 | 27 | 8 | 29.63% | 17 |
| .75 | 12:00 | High | 153 | 36 | 11 | 30.56% | 15 |
| .75 | 17:30 | Zero | 106 | 15 | 0 | 0.00% | 12 |
| .75 | 17:30 | Low | 145 | 30 | 5 | 16.67% | 15 |
| .75 | 17:30 | Base | 133 | 37 | 6 | 16.22% | 15 |
| .75 | 17:30 | High | 164 | 42 | 11 | 26.19% | 16 |

Final U equals M except q50/17:30/RT-High, where U=9 and M=8: nine orders have an option but only eight can be matched simultaneously. Output includes both and E/U/M for every gate. State-summed M-after is 8 / 26 / 25 / 40 (zero/Low/Base/High). Pre-patience M = 53 / 130 / 128 / 156; aggregate retention = 15.09% / 20.00% / 19.53% / 25.64%. These are instantaneous count sums, not daily unique orders.

Passenger/structural/evidence/patience all remove M across RT variants. Top-K does not reduce pre-patience M in these four states. Yet sampled q75-vs-q50 ordering changes: zero-lead M is 3 vs 4 at noon and 0 vs 1 evening; High gives 11 vs 10 and 11 vs 8. This warns against timing-independent claims; it does not show a full-day ordering reversal.

**The effective-capacity conversion interacts with reconstructed request lead time and pickup patience.** The mechanism survives, but quantitative and cross-fleet generalizations from zero lead need qualification. These scenarios do not identify true request times.

## Scope and QA

No solver/vehicle progression run. Selected count and expired-order counts are not estimated; canonical history is not RT counterfactual history. Imminent = currently waiting with 0 < remaining patience <= 30 seconds. Deadline-ever-eligible coverage remains unavailable from stored logs.

16 comparisons, 5979 routed arcs, zero failures, 54.71 seconds. No GPU/dense matrix. Focused tests verify release-only field changes and original-transform reproducibility. Physical hashes agree within each four-variant group; q50 zero-lead E/U/M reproduce the prior noon/evening funnel.

Outputs under stage4/output/paper_enhancement/mechanism_validity: request_time_sensitivity.csv, test31_rt_lead_summary.csv, request_time_sensitivity_summary.json. Run python -m stage4.analysis.rt_patience_sensitivity after recovery. All 41 canonical outputs remain unchanged.
