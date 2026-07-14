# VehiclePlan and VehicleLeg definition

`VehiclePlan` represents future operator intent:

- `vehicle_id`
- `plan_version`
- `stops`
- `created_time`
- `trigger`
- `assigned_request_ids`
- `reserved_request_ids`

`VehicleLeg` represents physical execution:

- `leg_type`
- `request_id`
- start/end coordinates
- planned/actual times
- expected/realized travel time
- route source
- ODD feasibility

In Phase 1, each matched order generates:

```text
PICKUP leg → SERVICE leg
```

Repositioning, rebalancing, and preassignment stops are reserved for later phases.

