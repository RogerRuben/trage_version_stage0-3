# Scientific strengthening execution summary

## Authorized base

- Branch: `codex/stage2-v5-micro-transfer`
- Canonical manuscript commit: `3ed344d7a86ec232db10e0d8913fb08a06ec98d5`
- Frozen canonical experiments: unchanged
- Enhancement experiments: separate registry; four authorized shadow-logging reruns completed

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

Status: `PASS — GO_REPOSITIONING_ROBUSTNESS`.

Prospective shadow counters were added for one common unit: `(waiting order, available AV, decision epoch)`. The four authorized anchors were rerun in isolation, and all 19 canonical summary metrics, request-outcome fingerprints, and assignment fingerprints reproduced exactly. Canonical experiment products were not overwritten.

The comparable p=.70 anchors show that N5/N0 dispatch-eligibility conversion declines from 0.0939% at q=.25 to 0.0445% at q=.75, a 52.6% relative reduction. Passenger retention changes only modestly; structural readiness, complete decision evidence, and pickup feasibility under patience deteriorate more clearly. Shared Top-K is separately reported as algorithmic compression, while routing return is 100% and N5→N6 is treated as dispatch competition rather than eligibility attrition.

Main finding: the end-to-end effective-capacity mechanism is now directly observed. It `QUALIFIES CURRENT STORY`: the result is not primarily an acceptance story, but a joint conversion through structural, evidence, sparse-candidate, and patience gates.

## Downstream workstreams

The following later workstreams remain unexecuted in this gate-specific authorization:

- AV repositioning robustness;
- Gamma service–exposure frontier;
- prediction-to-decision ablation;
- theory propositions;
- central AV transition curve;
- optional P1/P2 robustness.

No implementation defect, future-information leakage, or cumulative-budget violation was observed. Gamma remains outside this funnel because all four anchors are `UNCONSTRAINED`.

## Recommendation

`GO_REPOSITIONING_ROBUSTNESS`

The prospective gate decision point is closed. The next scientific robustness task is the pre-specified repositioning analysis; it should remain separate from the frozen canonical experiment products.

Do not rewrite Manuscript v2 yet.
