# Effective-capacity gate observability audit

> Historical audit note: this document records the frozen-log limitation found before prospective logging was authorized. The limitation is now resolved by the four-anchor same-unit rerun documented in `prospective_gate_decomposition_report.md`; the old audit is retained for provenance rather than used as the current conclusion.

## Status

`STOPPED_AT_TASKBOOK_GATE_D4`: a complete candidate-opportunity funnel is not identifiable from the frozen logs. No missing gate is inferred, and no new simulation was started.

## Directly logged evidence

| scenario_id | acceptance_rejected_nearby_av_opportunities | missing_exposure_nearby_av_opportunities | all_fleet_patience_arc_exclusions | selected_av_assignments | selected_total_assignments | final_assigned_av_share |
| --- | --- | --- | --- | --- | --- | --- |
| MAIN_Q25_M_P70 | 621898 | 0 | 2047960 | 1187 | 21891 | 0.0542 |
| MAIN_Q50_M_P70 | 1835602 | 0 | 2653140 | 2207 | 18133 | 0.1217 |
| MAIN_Q75_M_P70 | 4107009 | 0 | 3353235 | 3134 | 12039 | 0.2603 |
| BENCH_AV_M | 0 | 0 | 985131 | 4545 | 4545 | 1.0000 |

The acceptance and missing-exposure columns count nearby **AV candidate-opportunities** removed before routing. The patience column counts **all-fleet candidate arcs** after routing. Final assignments use an **assignment** unit. These quantities therefore cannot be chained into retained/lost percentages.

## Observable gates

- Passenger rejection: directly logged as nearby AV opportunities removed, but without the common nominal denominator.
- Missing exposure: directly logged as nearby AV opportunities removed, but without the common entering denominator.
- Selected AV assignments and final assigned-AV share: directly logged.
- Request-level passenger acceptance is logged separately; it is not a substitute for candidate-opportunity retention.

## Unobservable or non-comparable gates

- Nominal nearby AV opportunities before route/acceptance/evidence gates.
- Route-ready AV opportunities, because non-ready requests suppress AV spatial queries without a counter.
- AV-only pickup-feasible opportunities, because patience exclusions combine HV and AV arcs.
- Candidate-level Gamma attrition. The four required anchors are UNCONSTRAINED, so Gamma is not an active gate there.

## Scientific finding

The frozen logs support selected attrition diagnostics, but not the requested end-to-end effective-capacity funnel. A funnel with retained/lost shares would require new prospective logging and rerunning the anchors; reconstructing it from the current aggregates would invent unavailable states.

## Decision

Classification: `QUALIFIES CURRENT STORY`. The existing effective-capacity result remains descriptive, while exact attribution across all proposed gates is not identified. Under the taskbook stop condition, repositioning, Gamma-frontier, and prediction-ablation runs are not authorized in this execution.
