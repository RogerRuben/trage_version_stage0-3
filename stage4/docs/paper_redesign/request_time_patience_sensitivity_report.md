# Request-time / patience sensitivity: parameter recovery pending

**BLOCKED_MISSING_FROZEN_RT_PARAMETERS.** Neither RT_ROBUST nor RT_SENSITIVE is established. No substitute lead-time scenarios were fabricated.

D1 is complete: all 41 canonical configurations use 300-second patience, and replay request_time equals departure_time for all 30,000 orders. This is a counterfactual timing assumption, not a demonstrated code defect.

## Located transform

stage4/scripts/build_decoupled_abm_environment.py, attach_request_times uses:

- response/pickup lower bounds and an OD-distance-dependent minimum;
- Train gap_p25_sec, gap_p50_sec, gap_p75_sec clipped by the existing lead cap;
- fractions 0.25/0.50/0.75, stable order/scenario jitter, and business-boundary clipping;
- simulated_request_time = observed_boarding_time - latent_request_lead_sec.

Retained source defaults are matching_response_sec=30, minimum_pickup_sec=90, max_request_lead_sec=1800 and warmup_minutes=60. These defaults do not supply the missing fitted quantiles. The existing UTC business-boundary convention must be disclosed when transferring the transform.

The old pipeline defaults to Train 20161019–20161022 and target 20161023. It writes request_time_chain_stats into stage4/output/decoupled_environment/manifest.json. Neither that manifest nor the original driver-chain table is present in the active worktree or main checkout. The available summary covers 114,356 old target-day orders; its means cannot identify the three training quantiles.

Recovery checks covered both checkouts, retained Stage4 data/output/docs, artifact manifests and Git history. The tracked legacy artifact manifest contains no chain parameters. A backup path has been requested from the user.

## Remaining execution

Once the parameter source is available, apply the same transform to Test31. Hold q_A=.50/.75 physical states, route descriptors, routing timestamps and candidate rules fixed; vary request release and its derived patience. Use normal and evening states/windows, and report waiting orders, AV-option U, maximum matching M and patience retention. Retaining canonical prior-assignment history must be described as a conditional-state assumption, not a counterfactual daily history.

Patience reduces M from 124 to 12 across the ten existing zero-lead states. This strengthens the need for sensitivity but does not establish robustness or sensitivity by itself. No full-day service rate is inferred from these metrics.
