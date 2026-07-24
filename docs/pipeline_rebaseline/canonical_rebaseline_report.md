# Stage 0-4 canonical pipeline rebaseline report

## Outcome

The canonical engineering smoke is **PASS**. Formal Stage 4
experiments remain **HOLD**. Legacy Stage 1-4 outputs remain exploratory or
deprecated and were not used by this chain.

## Frozen time chain

- Upstream Stage 1/2 fit: 2016-10-19.
- Stage 3 train: 2016-10-20.
- Calibration: 2016-10-22 only.
- Test and Stage 4 smoke: 2016-10-23.
- Stage 2 uses one order-level dispatch cutoff for every route link.

## Computed results

- Raw/Stage0/Stage1: 1,000 complete orders on each of 19, 20, 22, and 23 October.
- Stage2/Stage3: 1,000 held-out orders on each downstream day.
- Stage4 Safe/O0: 1000 completed, 0 cancelled.
- AV assignments: 12 (1.20%); audited AV ODD violations: 0.
- Historical realized-duration reads: 0.
- Candidate truncation: 9.50%; peak sparse edges: 57.

These Stage4 figures are functional-test outputs, not research findings.

## Mathematical and semantic fixes

- Stage1 v2 uses partition-invariant fixed-bin quantiles and ordered-support CDF interpolation.
- The core composite is LCS/GNS/RTS; PMIS remains a separate interaction output and IIS a conditional modality.
- Stage3 expected values are continuous regression outputs, not q90 aliases.
- Calibration is selected using validation only; the extended probability remains NA because canonical IIS is unavailable.
- Stage4 service execution uses predicted duration plus a pre-generated residual; historical duration is not read.

## Remaining blockers

- Stage0 clipped-core directed route continuity is only 15.7%-17.5% in the smoke sample.
- Stage2 and Stage3 smoke estimators are lightweight engineering models, not formal RC-MSTNet/DeepSets refits.
- Canonical dispatch-time IIS is unavailable and remains NA.
- Only one engineering smoke split and one Stage4 functional run have been audited.
