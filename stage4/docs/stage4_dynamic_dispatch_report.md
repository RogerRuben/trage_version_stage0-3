# Stage4 Dynamic Dispatch Report

This report summarizes the first rolling dispatch simulator run using the frozen Stage3 condition vector and dimension-specific capability mapping.

## Simulator features

The simulator now includes:

- rolling order stream by `decision_time`;
- pending queue;
- patience cancellation;
- AV/HV fleet initialization;
- vehicle location and availability state;
- pickup candidate generation;
- pickup feasibility threshold;
- service completion time;
- vehicle release at destination;
- ODD-gated AV feasibility;
- HV compensation cost for high-stress residual assignments.

It no longer hard-codes match rate to 1 or cancellation to 0. In the current default fleet setting, the resulting match rate happens to be 1.0 because fleet capacity and pickup radius are generous.

## Baselines

The following baselines ran:

- Random
- Nearest
- GlobalMatch
- Cost-only
- Simple risk-penalty
- ODD-gated
- ODD-gated + HV compensation

## Main observations

With 250 AVs and 750 HVs:

- Random has much higher pickup time than spatially aware baselines.
- Simple risk-penalty assigns fewer orders to AVs and lowers AV mean stress exposure, but shifts more stress to HVs.
- ODD-gated policies avoid AV ODD violations by construction under the scenario profiles.
- HV compensation variants assign similar order volumes but explicitly expose the compensation cost needed when HVs retain high-stress residual orders.

The current run is a scenario mechanism test. It is not a calibrated real-world fleet deployment claim.

## Output files

- `stage4/output/dynamic_dispatch_v2/dynamic_dispatch_summary.csv`
- `stage4/output/dynamic_dispatch_v2/stage4_dynamic_dispatch_report.md`
