# Stage4 Joint Pricing and Dynamic Matching Report

## Status update: single-day ABM supersedes synthetic-supply tests

The previous multi-fold synthetic-supply scenarios remain useful as regression
tests for pricing/accounting and matching logic. They are no longer the main
empirical Stage4 result. The current formal Stage4 base is the 2016-10-23
full-day agent-based simulation with full-day model predictions, observed HV
driver sessions, depot-based AV initialization, and dynamic search radius.
See `stage4_single_day_simulation_report.md` for the current result table.

## Scope

This stage studies ODD-constrained pricing-and-dispatch coordination for mixed
AV/HV ride-hailing fleets. Stage3 condition vectors remain frozen and are used
only as technology-neutral operational-stress inputs. They are not interpreted
as empirical AV crash risk, disengagement probability, or ADS failure
probability.

## Mechanism corrections in this revision

The Stage4 simulator was revised to address the credibility issues in the first
prototype:

1. Initial fleets are generated once per `(fold, supply, AV penetration, seed)`
   and saved as reusable snapshots with an `initial_fleet_hash`.
2. HV compensation is decomposed into base driver payout, service-time payout,
   pickup compensation, scarcity bonus, gross stress compensation,
   passenger-funded compensation, and platform-funded compensation.
3. Passenger generalized cost now uses accumulated waiting time plus pickup
   time. It no longer reuses pickup time as waiting time.
4. Candidate generation records spatial, ODD, passenger, and driver feasibility
   separately, so cancellation reasons are no longer collapsed into match rate.
5. `Weighted Stakeholder Heuristic` is separated from the true
   `Three-Stakeholder Balanced` mechanism.
6. `Three-Stakeholder Balanced` uses a lexicographic window-level matching
   approximation: maximize feasible service count, then maximize platform
   profit under passenger, driver, and AV ODD feasibility constraints.
7. AV capability cost, remote-assistance placeholder cost, fallback placeholder
   cost, and lost-demand cost are included in accounting.
8. Free relocation is disabled in the main experiment.

## Experiment design

The submitted sample experiment uses:

- 3 rolling test folds;
- 1,500 orders per fold;
- 126 scenario rows;
- main mechanisms B0-B6;
- one-factor supply / AV penetration / ODD profile / pricing sweeps;
- a true AV penetration × ODD profile interaction grid.

A limited full-fold robustness run was also completed:

- 15,000 orders per fold;
- 3 folds;
- moderate supply, 50% AV, moderate AV profile;
- mechanisms B0, B1, B2, B3, and B6 only.

## Main mechanisms

| ID | Dispatch | Pricing | ODD policy |
| --- | --- | --- | --- |
| B0 | GlobalMatch-MinPickup | P0 uniform | none |
| B1 | GlobalMatch-MinOperatingCost | P0 uniform | none |
| B2 | ODD Gate Only | P0 uniform | hard AV gate |
| B3 | ODD-Gated Price-Aware Matching | P3 shared compensation | hard AV gate |
| B4 | ODD-Gated Price-Aware Matching | P4 AV discount + HV compensation | hard AV gate |
| B5 | Weighted Stakeholder Heuristic | P5 balanced parameters | hard AV gate |
| B6 | Three-Stakeholder Balanced | P5 balanced parameters | hard AV gate |

## 1,500-order multi-fold results

Mean across three folds:

| Mechanism | Match | Cancel | Passenger accept | Driver accept | Net profit | HV income | AV share | AV ODD violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 MinPickup Uniform | 0.870 | 0.130 | 0.870 | 0.970 | 30,939 | 9,179 | 0.476 | 0.132 |
| B1 MinCost Uniform | 0.869 | 0.131 | 0.869 | 0.970 | 35,596 | 7,256 | 0.571 | 0.003 |
| B2 ODD Gate Uniform | 0.869 | 0.131 | 0.869 | 0.970 | 32,553 | 10,082 | 0.426 | 0.000 |
| B3 ODD Price Shared | 0.868 | 0.132 | 0.868 | 0.970 | 35,430 | 9,165 | 0.567 | 0.000 |
| B4 AV Discount + HV Comp | 0.873 | 0.127 | 0.874 | 0.970 | 33,685 | 9,333 | 0.582 | 0.000 |
| B5 Weighted Heuristic | 0.852 | 0.148 | 0.855 | 0.970 | 32,117 | 12,195 | 0.487 | 0.000 |
| B6 Three-Stakeholder Balanced | 0.852 | 0.148 | 0.854 | 0.970 | 33,211 | 9,253 | 0.581 | 0.000 |

Interpretation: ODD-gated mechanisms remove AV hard violations. B3 preserves
most of the B1 profit advantage while enforcing AV feasibility. B6 is more
conservative in service level than B3 in this parameterization but keeps ODD
violations at zero and uses a clean lexicographic objective rather than mixed
unit weights.

## Limited 15,000-order full-fold robustness

Mean across three full folds:

| Mechanism | Match | Cancel | Net profit | AV ODD violation |
| --- | ---: | ---: | ---: | ---: |
| B0 MinPickup Uniform | 0.827 | 0.173 | 365,234 | 0.133 |
| B1 MinCost Uniform | 0.821 | 0.179 | 397,166 | 0.012 |
| B2 ODD Gate Uniform | 0.826 | 0.174 | 384,359 | 0.000 |
| B3 ODD Price Shared | 0.819 | 0.181 | 394,355 | 0.000 |
| B6 Three-Stakeholder Balanced | 0.814 | 0.186 | 361,077 | 0.000 |

The full-fold run supports the same qualitative conclusion: ODD-gated
mechanisms enforce zero AV hard violations, and B3 remains close to the
operating-cost benchmark in platform net profit.

## Cancellation reasons

Across the 1,500-order multi-fold experiment:

| Reason | Orders |
| --- | ---: |
| passenger rejection | 19,384 |
| no available vehicle | 3,150 |
| no spatial candidate | 2,479 |
| no ODD-feasible AV | 1,990 |
| patience timeout | 559 |

`passenger_acceptance_rate` is no longer equal to match rate. It is computed
from candidate-edge acceptability before matching.

## Pricing and compensation

Passenger fare is:

```text
base fare + surge component + vehicle adjustment + passenger-funded compensation
```

HV driver total payout is:

```text
base driver payout
+ service-time payout
+ pickup compensation
+ scarcity bonus
+ gross stress compensation
```

Gross stress compensation is exactly:

```text
passenger-funded compensation + platform-funded compensation
```

Platform HV cost includes base driver components, platform-funded compensation,
and platform variable cost. The accounting audit verifies all identities.

## Capability and lost-demand accounting

AV edge cost includes pickup, service, energy, capability cost,
remote-assistance placeholder cost, and fallback expected cost. Non-ODD
baselines may assign ODD-infeasible AV edges, but those edges carry fallback
cost and are recorded as violations. ODD-gated mechanisms prohibit such edges.

Scenario net profit is:

```text
served-order profit - lost-demand penalty
```

## Audits

All audits passed for both the 1,500-order multi-fold experiment and the limited
15,000-order full-fold robustness experiment:

- Stage3 export audit
- capability mapping audit
- matching feasibility audit
- pricing accounting audit
- scenario comparability audit
- dynamic dispatch consistency audit

## Output files

Small committed result files live under `stage4/docs/results`. Figures live
under `stage4/docs/figures`.

## Limitations

1. Supply is a reconstructed scenario, not observed idle cruising behavior.
2. AV profiles are scenario priors, not empirical AV capability estimates.
3. The main multi-scenario comparison uses 1,500 orders/fold for tractability;
   the full-fold run is limited to five main mechanisms.
4. The current candidate builder is distance-filtered with zone statistics. It
   is not a full road-network travel-time candidate generator.
