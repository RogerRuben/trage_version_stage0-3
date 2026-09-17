# Stage4 S4 ODD-Aware Mixed HV/AV Decision Kernel

Recommendation: `GO_ODD_AWARE_DECISION_KERNEL`

## Canonical base

- S3 commit: `84783ca1696c325cc3ed31ab3efd8747f4133ece`
- FleetPy commit: `0379f9725a147ff33c674de4884cdf89fd787fa9`
- Frozen S3 outcome: 1458 requests, 1215 matched/completed, 243 expired.

## Added mechanisms

- Exogenous passenger AV acceptance, independent of trip outcomes.
- Separate static/dynamic/speed reference-envelope excess.
- At most three cumulative Gamma rows in the existing CSR MILP.
- Normalized operating-time cost as the final optional lexicographic level.

## Neutral reproduction

- Requests/matched/completed/expired: 1458/1215/1215/243
- First-window/carry-recovered/critical-matched: 1131/84/1
- Runtime: 42.024s
- Fingerprint: `a90f1285813cfe5fc9fedeeb6514ed5b204ad5de7a5e230316639d8e1ff2c961`
- Exact S3 aggregate reproduction: `True`

## Exposure semantics

Exposure is reference-envelope exceedance. It is not a safety, failure, accident, or legal probability.

## Cat-eye checks

- Acceptance: `PASS`
- Cumulative exposure: `PASS`
- Cost tie-break: `PASS`

## Computational impact

- S3/S4 neutral runtime: 43.404/42.024s.
- Valid sparse arcs: 5406; solver p50/p95/max: 0.003726/0.016192/0.023967s.
- cKDTree + Top-K 20 + CSR only; CPU-only; no order-by-fleet matrix and no per-vehicle tick trace.
- Canonical Gamma rows/cost solves: 0/0.

## Interpretation limits

S4 does not identify best Gamma values, true passenger preferences, monetary cost ratios, AV penetration, economic superiority, or safety improvement.
