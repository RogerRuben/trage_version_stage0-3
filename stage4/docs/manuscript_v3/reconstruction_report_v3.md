# Manuscript V3 reconstruction report

Scope: TR-B / TR-E dual-target story reconstruction. Source revision: `3c643852d01b8396971a29c55c291d45a3af6486`. Existing experimental outputs are reused; no experiment, inference, simulation, solver run or figure rendering is part of this revision.

## Changes delivered

| Requested change | V3 implementation |
| --- | --- |
| Replace the pipeline-centered story | Preferred title A; rewritten abstract; capacity-substitution opening and conclusion |
| Establish a capacity object | Section 2 separates active supply, dispatchable vehicles, compatibility graph, maximum matching and rolling service |
| Exactly three contributions | Conceptual, methodological, empirical/managerial; no independent prediction contribution or RQ3 |
| Model before implementation | Sections 2–3 are generic; Xi'an implementation begins in Section 4 |
| Correct Gamma mathematics | Section 3.4 uses system-wide cumulative mean exposure, matching the solver's AV coefficient `exposure - Gamma` and RHS `Gamma * av_assignments - accumulated_exposure` |
| Replace propositions | Main text contains cumulative guarantee, same-epoch feasible-set monotonicity and separate-family control; Appendix D contains proofs and the pickup-quality epsilon guarantee |
| Reorganize results | Supply/service, E/U/M conversion, recovery levers, timing/information boundaries |
| Remove development narrative | Main text omits project-stage names, cache provenance, fingerprint recovery and old repeated-edge percentages |
| Reframe discussion | Substitutable service capacity and four managerial levers; five short scope limitations |
| Complete references | Ten cited records across four literature groups; source-access limitations documented separately |
| Decide figures after writing | Five-figure plan follows the revised model/results; no final artwork |

## Scientific distinctions retained

- The 41 full-day outcomes remain conditional on zero request lead and 300-second patience.
- The ten-state AV matching sequence is 718/493/238/124/124/124/12/12, not total mixed-fleet capacity or daily unique orders.
- The four-state RT final matching totals are 8/26/25/40; these are not pooled with the ten-state sample. Timing sensitivity conditions on existing physical states, prior assignments and fixed descriptors.
- Ordinary maximum matching does not incorporate coupled Gamma rows. Section 2.3 separately defines policy-constrained capacity and its upper bound by graph capacity.
- Same-epoch Gamma monotonicity does not imply monotonic daily service or every lower-priority lexicographic objective.
- The epsilon cost stage bounds aggregate pickup ETA after retaining critical/total/carry-over optima; it is not a service-count relaxation.
- Prediction remains DECISION-RELEVANT, not decision-superior. RT provenance remains fingerprint-verified reconstruction, not recovery of the original manifest.

## Verification scope

The cumulative state and objective hierarchy were checked against `stage4/dispatch/exposure.py`, `stage4/dispatch/solver.py` and the existing theory notes. Numerical narrative and tables were compared with the retained manuscript evidence and mechanism outputs, including `matching_capacity_summary.csv` and request-time sensitivity products. Main-text terminology and citation-placeholder scans are clean. Markdown structure, local document links and staged whitespace are checked before submission.

Only this new documentation directory is changed. Older V2/V2.2 artifacts remain historical rather than being silently rewritten. No production code, model checkpoint, policy parameter or experimental result is modified. This is a manuscript draft for review, not a submission-ready typeset article or a journal-acceptance claim.
