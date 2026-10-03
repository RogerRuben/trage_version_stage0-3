# Stage4 S2 FleetPy Native Shell Summary

## Native FleetPy path

- Simulation class: `Stage4NativeImmediateDecisionsSimulation`
- Run method: `FleetSimulationBase.run`
- Step method: `ImmediateDecisionsSimulation.step`
- Fleet-control hook: `Stage4NativeFleetControl.time_trigger`
- Demand/request hook: `Demand.get_new_travelers`
- Broker: `BrokerBasic`
- Vehicle progression: `SimulationVehicle.update_veh_state`
- Network/routing hook: `Stage4ValhallaNetworkBridge.move_along_route`

FleetPy owns the simulation clock, request activation, vehicle route-leg progression, busy/available transitions, broker callbacks, and native leg logging.

## Reused upstream modules

- `FleetSimulationBase.run`
- `ImmediateDecisionsSimulation.step`
- `Demand.get_new_travelers`
- `BrokerBasic`
- `FleetControlBase` subclass contract
- `SimulationVehicle.update_veh_state`
- `VehicleRouteLeg` and `VRL_STATES`
- `NetworkBase` subclass contract

## Project adapters

- deterministic Test31 demand population
- HV/AV fleet-window eligibility
- WGS84 Valhalla plus frozen 15-minute beta network bridge
- realized occupied-service leg timing
- minimum-corrected-pickup-ETA fleet-control stub
- optional silent progress shim when the existing Valhalla environment lacks progress-only `tqdm`

## Availability semantics

- `HV = reconstructed S0 session window`
- `AV = full simulation horizon, unavailable only while busy`
- An admitted HV trip that realizes past session end is completed; the vehicle is then permanently ineligible for new assignments.
- The 40/10 fixture is not scientifically normalized.

## Native replay result

- Interval: `2016-10-31T08:00:00+08:00` to `2016-10-31T09:00:00+08:00`
- Native timestep: 60 s
- Requests loaded/activated: 200/200
- HV/AV fixture: 40/10
- Assigned/completed: 70/53
- HV/AV assignments: 45/25
- Activation lag mean/max (s): 28.235/59.000
- Corrected pickup ETA mean/max (s): 798.283/2063.535
- Candidate arc evaluations: 13285
- Valhalla calls/cache hits/failures: 13285/0/0
- HV session-end candidate exclusions: 12965
- HV realized overruns count/mean/max (s): 0/0.000/0.000
- AV availability violations: 0

No scientific service-rate conclusion is drawn from assigned/completed counts.

## Lifecycle reconciliation

- Position failures: 0
- Request-state failures: 0
- Vehicle-state failures: 0
- Native completed route-leg rows: 236

FleetPy request states, vehicle states, native route legs, assignments, completion positions, and availability policies reconcile when all failure counts are zero.

## S1 loop status

The S1 project-side event loop remains only as compatibility-spike lineage and is not part of the S2 formal native simulation path.

## FleetPy core changes

`NONE`

## Recommendation

`GO_NATIVE_FLEETPY_SHELL`

The shell proves native FleetPy ownership of the one-hour engineering replay. Full-day replay, scientific fleet normalization, candidate pruning, and the OR dispatch model remain unauthorized.
