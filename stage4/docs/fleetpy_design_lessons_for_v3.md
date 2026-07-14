# FleetPy design lessons for Stage4 Simulator v3

FleetPy is used as an architectural reference, not as code to copy.

Inspected public sources:

- `TUM-VT/FleetPy` README: FleetPy is a modular agent-based fleet simulation framework for ride-sharing, autonomous mobility, and on-demand transport.
- `src/FleetSimulationBase.py`: the simulation base initializes demand, routing, operators, vehicles, and then updates vehicle state and operator feedback during the simulation loop.
- `src/simulation/Vehicles.py`: `SimulationVehicle` stores an `assigned_route` list of vehicle route legs; `start_next_leg`, `end_current_leg`, and `update_veh_state` handle physical vehicle progression and output records.

Design lessons adopted in our v3:

1. The simulator core should coordinate modules, not compute matching logic itself.
2. Fleet control should publish plans/routes, not directly mutate vehicle coordinates.
3. Physical vehicle state changes belong to the vehicle execution layer.
4. A vehicle's current/locked task must be preserved when future tasks are inserted.
5. Vehicle logs should be leg-based, because completed route legs are the natural auditable unit.
6. Demand/request logs and vehicle/task logs should remain separate and then be reconciled by audit.

Mapping to our implementation:

| FleetPy concept | Stage4 v3 concept |
| --- | --- |
| `FleetSimulationBase` | `SimulationEngine` |
| `SimulationVehicle` | `VehicleState` + `VehicleExecutor` |
| `assigned_route` | `VehiclePlan.stops` |
| `VehicleRouteLeg` | `VehicleLeg` |
| `update_veh_state` | event-driven `VehicleExecutor.complete_current_leg` |
| operator fleet control | `FleetController` |

Important boundary:

```text
FleetController can create or revise VehiclePlan.
VehicleExecutor alone turns a plan into physical VehicleLeg execution.
```

This is the key FleetPy-style correction relative to the v2 prototype.

