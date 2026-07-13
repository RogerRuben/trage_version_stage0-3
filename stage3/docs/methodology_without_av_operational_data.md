# Methodology Without Real AV Operational Data

## What this project can claim

This pipeline builds an order-level ODD-relevant environment representation from taxi trajectories, road networks, POI context, and historical traffic state. It predicts pre-dispatch operational stress and uncertainty, and provides a data-light interface for AV/HV capability-mapping scenarios.

## What it cannot claim

The current outputs are not real AV crash risk, disengagement probability, ADS failure probability, or verified AV safety outcomes. They should not be interpreted as evidence that a specific ADS will fail on a specific order.

## Where future AV data enters

Future AV operational data should calibrate the second layer:

- `P(remote assistance | condition vector)`
- `P(route rejection | condition vector)`
- `P(fallback | condition vector)`
- `P(abnormal delay | condition vector)`
- AV-specific operating cost
- AV-specific ODD thresholds

This calibration layer replaces or updates vehicle capability response functions. It does not require rebuilding the first-layer order environment representation or Stage3 prediction warehouse.

Current Stage4 vehicle profiles are scenario priors, not empirical AV capability estimates.

The first-layer condition vector is now frozen for the current Stage4
experiments. Stage4 changes vehicle-specific capability response functions,
pricing, compensation, and matching rules; it does not reinterpret Stage3
outputs as empirical AV safety outcomes.

## Structural vs numerical conclusions

The current framework can support structural counterfactuals: how ODD gates reshape assignment, whether AVs cream-skim lower-stress orders, how HV burden shifts, and how compensation/fairness constraints alter allocations.

It cannot yet support externally valid numerical claims about real crash-rate reduction, enterprise deployment profit, or specific ADS performance without AV operational calibration data.
