# Demand-supply decoupling method

The counterfactual ABM separates demand and supply sources.

- Demand: all 114,356 observed served orders on 2016-10-23.
- Condition-known orders: 112,165 orders with Stage3 condition vectors.
- Unknown-condition orders: 2,191 orders kept in demand, HV-only for ODD
  purposes, with no stress fallback.
- HV supply: synthetic sessions generated from 20161019-20161022 empirical
  supply, not from 20161023 drivers.
- Day-type target curve: 20161022 Saturday is the weekend anchor with
  `w_weekend=0.7`; 20161019-20161021 weekday median contributes 0.3.

This environment is an observed served-order arrival stream, not a recovered
latent citywide demand model.

