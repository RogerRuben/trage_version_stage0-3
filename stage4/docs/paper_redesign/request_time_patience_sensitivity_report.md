# Request-time / patience sensitivity status

D1 completed: all 41 canonical scenarios use trajectory departure as request release, zero lead time, and 300-second pickup patience. The 30,000 Moderate-profile replay timestamps exactly match Stage1 departure timestamps. See canonical_request_time_setting.md for the source chain.

D2–D5: **NOT_RUN after the B5 hidden-asymmetry stop**. Neither RT_ROBUST nor RT_SENSITIVE is established. No short-window or full-day service rate is reported.

RT-Low/Base/High belong to the separate 20161023 decoupled ABM method; no such setting is present in the frozen 41-scenario runtime configurations. The legacy mean lead times cannot be assigned to Test31 as if they were frozen order-specific request times. This is a method-attribution discrepancy, not evidence from a completed sensitivity experiment.

Remaining authorized analysis after the identity issue is resolved: retain q_A=.50/.75, M, p=.70 physical states and apply the existing RT transformation with its recorded chain bounds; compare waiting orders, AV-option orders, maximum matching, and patience retention. Do not create an alternative fleet or tune RT values to the outcomes.
