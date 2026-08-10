# Stage 2 v5.2 → Stage 3 final contract

Contract schema: `stage2_v5_2_to_stage3_contract.2`

Stage 2 status: `STAGE2_FINAL_FROZEN`

Stage 3 authorization: `NO`

## Frozen predictor

- Model: `M3` (`structured_representation`).
- Checkpoint: `stage2/output_v5_2/development/M3/epoch_004.pt`.
- Checkpoint SHA-256: `965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde`.
- Training manifest SHA-256: `2f4acbe5244498598021395382f4bd20a39e22b23b77c4453c19e2914709a388`.
- Stage 3 must not retrain Stage 2, reselect a checkpoint, reselect tau, or alter
  the frozen feature construction.

## Deployable outputs

Stage 3 may consume only the decision-time predictions `travel_time_p50`,
`pace_p50`, `crawl`, `stop`, `speed_cv`, and `acceleration_rms`, plus the frozen
identity, support, provenance, distance, and completeness fields required to
interpret them. `RTS` remains diagnostic-only and must not enter ODD–TOD
compatibility, AV feasibility, fallback selection, or Stage 4.

These dynamic outputs are **predicted operational-condition proxies**. They are
not AV accident, safety, takeover, or failure probabilities.

## Route and inference boundary

- The historical Test31 route is the fixed HV route and the first AV candidate.
- Original-route products retain ordered directed-edge/traversal identity.
- A future Stage 3 candidate-route adapter may tokenize a candidate directed-edge
  sequence, construct decision-time entry/horizon features, and run this exact
  frozen M3 checkpoint.
- Candidate predictions must never be copied from the historical route and may
  not be manually imputed. If the frozen input contract cannot be satisfied,
  the candidate dynamic state is `UNKNOWN`.
- Stage 3 owns any later bounded fallback search and returns one fixed AV route
  or no feasible route. Stage 2 performs no path search.

## Downstream decision boundary

Stage 4 receives fixed HV/AV routes, feasibility, service time, distance,
exposure, and unknown state. **Stage 4 has no route decision variable.** It must
not receive candidate-route search or Stage 3 threshold fitting.

## Frozen scientific interpretation

- B1: no convincing support-aware transfer evidence (`CASE_C`).
- Phase C: M4 failed the pre-registered continuation rule.
- Spatiotemporal diagnostic: structured M3 signal is positive while the M4 gate
  has no stable incremental support (`DIAG-B`).
- Upstream audit: demand concentration, Stage 0 selection, and Stage 1 label
  availability compound sparse-context attrition (`UP-D`).

Accordingly M3 is the final engineering predictor; M4 is a rejected ablation;
M5/M6, Phase D, and Transfer-v2 are cancelled.

## Authorization

This contract freezes the interface only. S1–S8, Stage 3 execution, and Stage 4
remain unauthorized until the user explicitly approves the next phase.
