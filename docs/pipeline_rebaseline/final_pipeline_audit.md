# Final Stage 0-4 canonical pipeline review

## Decision

The **canonical engineering smoke audit is PASS**. This certifies
contracts, field availability, explicit manifests, the 1,000-order-per-day
Stage 0-3 chain, and the 1,000-order Safe/O0 counterfactual functional test.

The **formal experiment release gate remains HOLD**. A functional smoke pass is
not evidence that the formal Stage 2/3 models or full-scale Stage 4 study are ready.

## Independent review

| Review | Status |
| --- | --- |
| Information leakage | PASS |
| Data lineage | PASS |
| Mathematical definitions | PASS |
| Counterfactual inputs | PASS |
| Experiment governance | PASS |

Every PASS above is recomputed from manifests, field registries, audit JSON,
and Stage4 logs; it is not a hand-written acceptance marker.

## Frozen temporal chain

- Stage1/Stage2 upstream fit: 2016-10-19.
- Stage3 train: 2016-10-20.
- Calibration: 2016-10-22 only.
- Test and Stage4 smoke: 2016-10-23.
- Stage2 applies one dispatch-time cutoff to all links in an order.

## Evidence summary

- Raw input: 1,000 complete orders on each of 20161019/20/22/23, extracted by two-pass streaming.
- Stage0: exact interval time/distance conservation and zero unflagged illegal directed transitions.
- Stage1: partition-invariant median; 88,823 cohort CDF models with zero monotonic/tail failures.
- Stage2: 3,000 held-out downstream orders; zero post-decision availability and zero realized-duration permission rows.
- Stage3: validation-only calibration; test core-overall AUC/AP/Brier/ECE = 0.7682/0.3978/0.1428/0.1112.
- Stage4: 1,000 completed, 0 cancelled, 12 AV assignments, zero ODD violations, zero historical-duration reads.

Stage4 numbers are functional-test outputs and must not be used as research findings.

## Formal experiment blockers

- Stage0 clipped-core directed route continuity is only 15.7%-17.5% in the smoke sample.
- Stage2 and Stage3 smoke estimators are lightweight engineering models, not formal RC-MSTNet/DeepSets refits.
- Canonical dispatch-time IIS is unavailable and remains NA.
- Only one engineering smoke split and one Stage4 functional run have been audited.

## Artifact governance

- Canonical manifests: 8.
- Frozen exploratory legacy artifacts: 166.
- Deprecated artifacts: 1.
- Unknown legacy artifacts: 0.
- Config hash: `8b452c91e1ad8811e14bf66b56cd38f4a94643a7cb4a6bc4cd07c7f0ea0059d5`.
- Canonical run id: `stage0_4_rebaseline_v2-20260715T113310Z-s20260715-a7cdb62eea`.

## Release conclusion

The rebaseline engineering skeleton is reproducible and audited. Formal Stage4
experiments must remain disabled until the listed blockers are resolved and a
new canonical version supersedes this engineering smoke.
