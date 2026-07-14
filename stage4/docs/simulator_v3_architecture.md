# Stage4 Simulator v3 architecture

Simulator v3 replaces the v2 discrete-time prototype with a request-plan-execution separated kernel.

The core rule is:

```text
FleetController may publish or revise VehiclePlan objects.
VehicleExecutor is the only component allowed to mutate vehicle physical state.
```

The design follows the same high-level separation used by FleetPy: the simulation
base coordinates demand/routing/operator/vehicles, fleet control publishes
vehicle tasks, and the simulation vehicle/executor advances route legs and
records completed legs.  We adapt the concept to our AV/HV ODD setting rather
than importing FleetPy code.

Implemented modules:

- `stage4/simulator_v3/entities.py`: `RequestState`, `VehicleState`, `VehiclePlan`, `PlanStop`, `VehicleLeg`.
- `stage4/simulator_v3/event_queue.py`: heap-backed event queue with deterministic event priority.
- `stage4/simulator_v3/simulation_engine.py`: event processing plus fixed 30-second decision epochs.
- `stage4/simulator_v3/fleet_controller.py`: candidate generation, validation, matching, and plan publication.
- `stage4/simulator_v3/vehicle_executor.py`: VehiclePlan to VehicleLeg conversion and physical state update.
- `stage4/simulator_v3/routing_engine.py`: final pickup/service ETA query facade with cache.
- `stage4/simulator_v3/plan_validator.py`: session, deadline, pickup ODD, and service ODD checks.

Current run status:

```text
Phase 1 regression smoke: PASS
Full-day replication 1: not yet run under v3
Preassignment: pending Phase 3
Idle management: pending Phase 4
Balanced formal constraints: pending Phase 5
FleetPy cross-validation: pending Phase 6
```
