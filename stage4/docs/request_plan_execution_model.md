# Request-plan-execution model

Simulator v3 has three explicit layers.

## Request layer

`RequestState` owns the order lifecycle:

```text
UNREVEALED → PENDING → ASSIGNED → PICKUP_STARTED → BOARDED → IN_SERVICE → COMPLETED
```

Cancelled requests terminate in `CANCELLED` and cannot return to pending.

## Plan layer

`VehiclePlan` is an operator intention. It can contain pickup, dropoff, repositioning, rebalancing, or future reservation stops. A plan does not move a vehicle.

## Execution layer

`VehicleLeg` is a physical movement task derived from a plan stop. Only `VehicleExecutor.complete_current_leg` updates vehicle coordinates, current leg, execution status, and cumulative movement counters.

The Phase 1 smoke run confirms:

```text
completed requests = 657
vehicle legs = 1,314 = 2 × completed requests
plan revisions = 657
```

