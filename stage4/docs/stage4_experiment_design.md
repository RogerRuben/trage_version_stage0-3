# Stage4 Experiment Design

The current Stage4 experiment studies ODD-constrained pricing-and-dispatch
coordination for mixed AV/HV ride-hailing fleets.

## Folds and sample

The submitted run uses three Stage3 rolling test folds and a reproducible
1,500-order sample per fold for the full 126-row mechanism matrix. A limited
15,000-order-per-fold robustness run is also included for five main mechanisms.

## Scenario axes

- Supply: abundant, moderate, tight
- AV penetration: 0%, 25%, 50%, 75%
- ODD profile: conservative, moderate, mature, intersection-sensitive,
  uncertainty-sensitive
- Pricing: uniform, platform-funded compensation, passenger-funded
  compensation, shared compensation, AV discount + HV compensation,
  three-stakeholder balanced
- Main mechanisms: B0-B6 from `selected_strategy_grid`
- One-factor analyses: supply, AV penetration, ODD profile, pricing
- Joint interaction: AV penetration × ODD profile under B3

The report distinguishes the 1,500-order multi-fold mechanism experiment from
the limited 15,000-order main-mechanism robustness experiment.

All strategies within a fold use the same sampled order stream and the same
scenario parameter files.
