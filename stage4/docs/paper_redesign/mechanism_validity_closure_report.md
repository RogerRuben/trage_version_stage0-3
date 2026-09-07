# Mechanism validity closure

Base SHA: b2e3f9cabfff9855cc50bbf23be17a0436a84685. Branch: codex/stage2-v5-micro-transfer. Working tree was clean before edits.

**Scientific classification: IMPLEMENTATION_DEFECT**, using the taskbook's narrow definition: AV-neutral relabeling changes the candidate graph before explicit AV restrictions. This is a failed neutral-session invariant, not proof that the published service decline is caused by this branch. Taskbook B5 stopped the experiment sequence at the first failing snapshot.

1. **Does neutral relabeling preserve the problem?** No. In the hash-verified 07:30 state, candidate E changes 109 -> 110 because the same session-end timestamp is enforced for HV but bypassed for AV. U=25, M=24, selected identities, and pickup objective remain identical. One snapshot was run; the other nine remain pending.
2. **Which substantive gates remove capacity?** Not established. The neutral counterexample itself removes/adds a redundant edge, but C was not run. Earlier opportunity-edge percentages must not be presented as demonstrated maximum-matchable-order loss.
3. **Does K20 preserve capacity?** Not established; K10/20/40/80 sensitivity remains pending. No canonical K was changed.
4. **Is the mechanism request-time robust?** Not established. D1 identifies zero-lead observed-departure release and 300-second patience in all 41 canonical configurations, confirmed on 30,000/30,000 replay orders. RT-Low/Base/High describe a different ABM pipeline. D2–D5 were not run.
5. **Does the current story survive?** It remains unvalidated at the requested mechanism level. The recorded full-day scenario results are unchanged, but the stronger conversion-to-matchable-capacity explanation needs B/C/D closure. The discovered branch favors the AV-labeled clone, so it is not evidence that hidden anti-AV filtering explains the full-day decline.
6. **Common evaluator?** NOT_RUN. Retain DECISION-RELEVANT. Differences in the variants' own selected exposure measures do not establish superiority under a shared external evaluator.

## Delivered correction and evidence

- Corrected 9.1% to mean 9.1 additional solver-input AV arcs in the manuscript and number audit.
- Corrected candidate-arc Jaccard to mean selected-AV-arc Jaccard.
- Preserved all other numerical results and frozen experiment products.
- Added a bounded production-path neutral diagnostic and one focused test.
- Retained exact candidate and assignment identities in neutral_av_identity.csv; summary.json records the fail-fast result.
- Other required result CSVs explicitly contain NOT_RUN status with empty metrics, never fabricated zeros.

## Scope and next decision

The actionable issue is whether session-end admissibility should be inherited independently of the HV/AV label in the neutral experiment. A local explicit session-policy field would make that comparison reviewable; changing the canonical operational semantics or rerunning full days was not undertaken. Resume the remaining analyses after that policy is settled. No model training, broad tests, full-day simulation, new fleet protocol, paper-wide rewrite, or literature search occurred.

QA: one focused test PASS; touched-file compileall PASS; result CSV schema check PASS; git diff --check PASS. Candidate construction uses the actual production method, routing is identical for paired coordinates, and the maximum matching is computed from the captured candidate graph using sparse matrices.
