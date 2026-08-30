# Scientific strengthening execution summary

## Authorized base

- Branch: `codex/stage2-v5-micro-transfer`
- Canonical manuscript commit: `3ed344d7a86ec232db10e0d8913fb08a06ec98d5`
- Frozen canonical experiments: unchanged
- Enhancement experiments: separate registry; no simulation run was started

## Storyline redesign

Status: `PASS`.

The proposed central narrative is a data-to-decision framework that turns observed ride-hailing trajectories into prospective operational knowledge and then into ODD-aware mixed-fleet decisions. The seven candidate paper-level messages and the offline/decision-time/counterfactual information boundary are frozen in the planning documents.

## Technical role map and manuscript architecture

Status: `PASS`.

Seven scientific layers have explicit inputs, outputs, information regimes, paper roles, and evidence boundaries. Existing content has been mapped into `KEEP`, `MOVE`, `EXPAND`, `COMPRESS`, `APPENDIX`, or `DELETE`. No full-manuscript rewrite was performed.

## Figure plan

- Planned main figures: 7, including the main framework specification and three P0 experiment figures.
- Planned appendix figures: 5 candidates, with placement conditional where stated.
- Framework figure specification completed: `YES`.
- Framework figure drawn: `NO`.

## Effective-capacity gate decomposition

Status: `STOPPED_AT_TASKBOOK_GATE_D4`.

Directly observable evidence includes nearby AV opportunities rejected by passenger acceptance, nearby AV opportunities removed for missing exposure, selected AV assignments, and final assigned-AV share. The logs do not preserve a common nominal nearby-AV denominator, route-ready AV opportunity counts, AV-only patience-feasible counts, or candidate-level Gamma attrition. Mixed units cannot be chained into retained/lost shares.

Main finding: selected attrition diagnostics are available, but an end-to-end effective-capacity funnel is not identified from the frozen logs. The scientific story is therefore `QUALIFIES CURRENT STORY`, not rejected.

## Downstream workstreams

Under the taskbook stop condition, the following were not executed:

- AV repositioning robustness;
- Gamma service–exposure frontier;
- prediction-to-decision ablation;
- theory propositions;
- central AV transition curve;
- optional P1/P2 robustness.

No implementation defect, future-information leakage, or cumulative-budget violation was observed; execution stopped solely because completing the requested gate funnel would require invented states or a prospective logging rerun.

## Recommendation

`REVISE_SCIENTIFIC_STRENGTHENING`

Before authorizing later workstreams, choose one of two scientifically clean paths:

1. accept a partial gate-attribution result and explicitly authorize continuation from repositioning; or
2. authorize prospective gate counters plus reruns of the four anchors, preserving the frozen canonical outputs and writing only to the enhancement registry.

Do not rewrite Manuscript v2 yet.
