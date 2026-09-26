# Deterministic-routing dwell window v1

## Material Passport

2026-09-26. academic-research-suite / experiment-agent. User accepted the next
step after being offered a new shared deterministic-routing baseline. This is
an authorized conditional simulation sensitivity analysis, not a paper edit or
an attempt to reproduce the original 41 batch-routing scenarios.

## Frozen design

Six conditions: q50/q75 × additional pickup overhead 0/60/120 s. All use
SINGLE_SOURCE_MATRIX, the verified native canonical 10:30 checkpoint, M profile,
0.70 stable passenger acceptance, original Gamma, cost and Top-K. Zero lead,
300-s arrival patience and 30-s dispatch remain unchanged. Pre-checkpoint active
trips retain their original zero additional overhead. No full-day optimization,
new prediction, retraining or future-information feature construction.

Warmup 10:30–11:00; fixed release cohort [11:00,12:00). Stop new releases at
12:00, continue matching through the final deadline and drain accepted trips.
Compare each treatment only to its own q-specific new zero-overhead control.
Do not subtract results from the old batch-routing baseline. The previous
partial q50 single-source zero continuation is an exact regression reference
through 11:31, not a performance target or newly selected window.

Native pickup boarding duration implements the additional stationary time for
both HV and AV. `pickup_time` remains arrival; `service_start_time` is arrival
plus overhead. Route travel duration is unchanged. Planned completion/session
admission and operating time include overhead. Route exposure/Gamma weights
remain unchanged. Additional overhead is a sensitivity parameter, not an
empirically calibrated correction to GPS duration. Stops already inside the
GPS trip duration are not subtracted.

## Checks and outputs

- Native lifecycle tests at all three durations; HV admission with overhead.
- Pickup and service-time conservation; nonoverlapping physical assignments;
  all accepted trips drained; matched + expired equals the fixed cohort.
- Arrival patience checked separately from request-to-service-start time.
- Actual HV overruns are reported separately from predicted admission violations.
- Exact online and idle vehicle-hours in [11:00,12:00), including inherited tasks.
- Local per-condition assignments, cohort outcomes, epochs, exposure and summary.
- Small aggregate report committed; full data/checkpoint pickle stays local.

One CPU process, sparse arcs only, cache released every epoch. 30-minute
per-condition budget, 2-GiB RSS advisory threshold. No automatic failed-run retry.
Config: `stage4/config/dwell_deterministic_window.json`. Checkpoint and config
hashes recorded by the runner; config committed before execution.

Interpretation stays within these two fleet conditions and this fixed window.
One deterministic run per condition yields no sampling-based uncertainty or
population-level significance claim. Waiting-time means among matched orders
are selected-cohort summaries, not changes for identical passengers. The
canonical prehistory is held fixed: this is not a full-day dwell counterfactual.
