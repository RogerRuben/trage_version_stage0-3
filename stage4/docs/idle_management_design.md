# Idle management design

Idle management is pending Phase 4.

Simulator v3 will represent idle movement as VehiclePlan stops and VehicleLeg execution:

```text
HV_REPOSITION PlanStop → HV_REPOSITION VehicleLeg
AV_REBALANCE PlanStop → AV_REBALANCE VehicleLeg
```

This prevents the v2 problem where idle policies directly teleported vehicle coordinates.

The Phase 1 smoke uses:

```text
operation = O0
idle movement = Stay
preassignment = Off
```

