# Stage2 route-conditioned contract

## Scope

This project now studies route-conditioned ODD-stress prediction:

```text
given route -> route-conditioned ODD-stress prediction -> calibrated stress vector -> AV/HV dispatch
```

The platform is assumed to have an assigned service route at dispatch time. In
the empirical implementation, the map-matched completed route is used as a
revealed proxy for that assigned/planned service route. Route generation, route
choice, and navigation routing are outside the main scope of this study.

In Chinese:

```text
本文不研究平台如何规划路线，而研究给定候选服务路线条件下的
operational-stress 预测。实证中使用 map-matched completed route 作为
assigned/planned route 的代理。
```

## Main and Appendix Branches

The main Stage2 branch is:

```text
matched actual route as assigned-route proxy
+ dispatch/departure-time proxy
+ estimated link entry time
+ strictly lagged traffic state
+ historical profile
+ road semantics and POI exposure
-> route-conditioned operational-stress prediction
```

The OD-based shortest-path and historical-fastest planned-route branch is
demoted to audit/appendix use. It remains valuable for showing route-choice
mismatch and label-observability limits, but planned-route common-link subsets
must not determine whether Stage3 is allowed to proceed.

## Time Modes

Stage2 keeps two route-conditioned products.

`route_conditioned_estimated_time` is the main deployable research setting. It
uses the matched route as the route proxy, but all time-dependent features are
indexed by estimated link entry time. Lagged features must satisfy:

```text
feature_availability_timestamp < estimated_link_entry_time
```

`route_conditioned_oracle_time` is an upper-bound diagnostic setting. It may
use actual link entry time as a feature to measure the cost of entry-time
uncertainty. It is not a valid Stage3 or Stage4 input.

## Allowed Model Inputs

The estimated-time main model may use:

```text
order_id, driver_id as identifiers or cautious fixed/context fields
dispatch_time or departure_time proxy
route_link_id and route_link_seq
route position and distance-to-destination features
road_class, area_grid, endpoint_degree, link_fragmentation, minor_road
POI exposure and activity intensity descriptors
route length and route link count
rolling historical profiles fitted only from past days/windows
strictly lagged link/area/network/upstream/downstream traffic state
estimated_link_entry_time and estimated_time_bin
feature availability timestamps and availability checks
```

The oracle-time diagnostic product additionally may expose:

```text
actual_link_entry_time
actual_time_bin
actual hour/weekend descriptors
```

## Forbidden Model Inputs

The main estimated-time model must not use:

```text
actual speed of this order
actual low-speed ratio of this order
actual stop, delay, or traversal behavior of this order
actual LCS, IIS, RTS, or PMIS labels as features
actual realized link travel time
actual link enter_time
post-trip realized stress
future profile
current-row self-included profile
```

Target columns may remain in the supervised dataset, but they must be consumed
only as labels or audit descriptors.

## Targets

Stage2 no longer treats percentile labels as the only output. For LCS, PMIS,
and RTS the target family is:

```text
expected raw stress
conditional percentile anomaly
calibrated tail probability
predictive uncertainty
```

IIS uses a movement-level two-stage contract:

```text
applicability head
severity head conditional on applicability
```

IIS missing severity is never filled with zero. PMIS is interpreted as a POI
exposure x behavioral-stress interaction, not as a fully independent fifth
risk type.

## Stage3 Admission

Stage3 may proceed only from route-conditioned estimated-time outputs, and only
after:

```text
three rolling folds are stable under the estimated-time setting
estimated-time vs oracle-time gaps are reported and interpretable
tail probabilities are validation-day calibrated
uncertainty intervals have reasonable coverage
IIS applicability/severity outputs are stable
order-level aggregation shows positive tail separation
the conclusion does not rely on OD-planned common-link subsets
```

When admitted, Stage3 becomes:

```text
route-conditioned calibrated ODD-stress vector
```

Stage4 remains paused until that vector exists.
