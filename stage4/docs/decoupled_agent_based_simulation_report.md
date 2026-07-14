# Decoupled agent-based simulation report

This version upgrades the previous historical-consistency prototype into a
demand-supply decoupled counterfactual ABM.

Current P0-corrected implementation points:

- 114,356 test-day orders enter the demand stream.
- 2,191 unknown-condition orders are HV-only for ODD purposes and receive an
  ETA baseline, not a stress fallback.
- Stage3 full-day scale drift is audited; default capability gating now uses
  raw full-day Stage3 outputs.  The previous full-day rank/quantile remapping is
  disabled by default because it uses the test-day distribution and can
  artificially shape AV feasibility.
- Core ODD and conditional IIS gates are separated; IIS unavailable does not
  close AV feasibility.
- HV supply is generated from prior-day empirical supply with a Saturday
  weekend anchor and weekday stabilizer.
- AV share is defined by vehicle-hours and is approximately 5%.
- Matching is run on sparse candidate edges; no dense all-order-by-all-vehicle
  matrix is constructed.  The current executable simulator is a 30-second
  discrete-time sparse-matching simulator, not a full priority-queue
  event-driven engine.
- Scenario net profit is an evaluation metric.  Window matching uses marginal
  operating contribution plus mechanism-specific constraints.

P0 smoke status after the correction:

- The decoupled environment was rebuilt for 114,356 orders with three CRN
  environment replications.
- A replication-1 Safe GlobalMatch / O0 / RT-Base smoke run completed with
  `max_candidates_per_order=20`.
- The smoke run is not a final dispatch result: candidate truncation remains
  high and the full 3-replication experiment is intentionally not claimed.
- Raw full-day Stage3 scale drift currently makes the moderate AV service gate
  infeasible for the full-day condition-known set; this is reported as a model
  scale/alignment blocker, not hidden by a fallback capability score.
- Preassignment remains disabled by default pending a two-layer state model for
  current service plus reserved next order.
- Three-Stakeholder Balanced currently has only a proxy HV-stress edge filter;
  formal zone-time stress budget and minimum zone service constraints remain
  required before it can be interpreted as the final balanced mechanism.
