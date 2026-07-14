# Decoupled agent-based simulation report

This version upgrades the previous historical-consistency prototype into a
demand-supply decoupled counterfactual ABM.

Key implementation points:

- 114,356 test-day orders enter the demand stream.
- 2,191 unknown-condition orders are HV-only for ODD purposes and receive an
  ETA baseline, not a stress fallback.
- Stage3 full-day scale drift is audited; capability gating uses
  fold-3-reference quantile-calibrated capability scores while preserving raw
  Stage3 outputs.
- Core ODD and conditional IIS gates are separated; IIS unavailable does not
  close AV feasibility.
- HV supply is generated from prior-day empirical supply with a Saturday
  weekend anchor and weekday stabilizer.
- AV share is defined by vehicle-hours and is approximately 5%.
- Matching is run on sparse candidate edges; no dense all-order-by-all-vehicle
  matrix is constructed.
- Scenario net profit is an evaluation metric.  Window matching uses marginal
  operating contribution plus mechanism-specific constraints.

The first completed full run set covers replication 1 for Safe GlobalMatch,
ODD-Gated Price-Aware, Three-Stakeholder Balanced, O0-O3 operations, and
RT-Low/Base/High request-time sensitivity for the main mechanism.  Replications
2 and 3 have fixed CRN environment files and can be resumed with the same
commands.
