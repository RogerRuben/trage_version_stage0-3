# Stage4 Experiment Design

The current Stage4 experiment studies ODD-constrained pricing-and-dispatch
coordination for mixed AV/HV ride-hailing fleets.

## Folds and sample

The submitted run uses three Stage3 rolling test folds and a reproducible
1,500-order sample per fold for the full 69-scenario mechanism matrix. The full
15,000-order-per-fold input is available in `stage3/output/stage4_inputs_final`;
running with `--max-orders-per-fold 0` reproduces the same design at full fold
size but requires substantially longer runtime.

## Scenario axes

- Supply: abundant, moderate, tight
- AV penetration: 0%, 25%, 50%, 75%
- ODD profile: conservative, moderate, mature, intersection-sensitive,
  uncertainty-sensitive
- Pricing: uniform, platform-funded compensation, passenger-funded
  compensation, shared compensation, AV discount + HV compensation,
  three-stakeholder balanced
- Dispatch: Random, Nearest, GlobalMatch-MinPickup,
  GlobalMatch-MinOperatingCost, Cost-only, Simple Risk Penalty, ODD Gate Only,
  ODD-Gated Price-Aware Matching, Three-Stakeholder Balanced

All strategies within a fold use the same sampled order stream and the same
scenario parameter files.
