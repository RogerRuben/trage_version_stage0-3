# Stage4 S1 FleetPy Compatibility Spike

## Upstream dependency

- Repository: `https://github.com/TUM-VT/FleetPy.git`
- Commit: `0379f9725a147ff33c674de4884cdf89fd787fa9`
- License: MIT
- Installation: pinned external shallow checkout; no FleetPy source vendored
- Runtime Python: `3.12.13`
- FleetPy upstream environment targets Python 3.10; the exercised pure-Python classes imported and ran in the existing Stage0-Valhalla environment.
- C++ router required: NO
- Gurobi required: NO
- OR-Tools required: NO

## Reused FleetPy modules

- `src.demand.TravelerModels.BasicRequest`
- `src.simulation.Vehicles.ExternallyMovingSimulationVehicle`
- `src.simulation.Legs.VehicleRouteLeg`
- `src.misc.globals.VRL_STATES`
- `FleetSimulationBase`, `ImmediateDecisionsSimulation`, `Demand`, `FleetControlBase`, and `NetworkBase` were inspected to verify the future subclass hooks; no upstream file was changed.

## Project-specific adapters

- Test31 demand selection and `BasicRequest` construction
- deterministic S0-derived 40-HV/10-AV fixture
- WGS84 Valhalla pickup callback with frozen 15-minute S0 beta
- external-arrival bridge into FleetPy's native vehicle-leg lifecycle
- transparent minimum-corrected-pickup-ETA dispatch stub

## Availability semantics

- `HV = reconstructed S0 session window`
- `AV = full simulation horizon / always available unless busy`
- AVs do not inherit source-session end times.
- AV fleet-count normalization is not frozen in this spike.
- The 40/10 split is an engineering fixture with no scientific interpretation.

## Tiny replay result

- Interval: `2016-10-31T08:00:00+08:00` to `2016-10-31T09:00:00+08:00`
- Requests: 200
- HV fixture: 40
- AV fixture: 10
- Assigned/completed: 69/69
- HV assignments: 45
- AV assignments: 24
- Corrected pickup ETA mean/max (s): 841.094/2063.535
- Vehicles activated: 50
- HV session-end exclusions: 98919
- AV availability violations: 0
- Position/timing failures: 0/0
- Valhalla calls/cache hits/failures: 17553/81682/0
- FleetPy native output reconciles with adapter logs: `True`

This is an engineering compatibility run only; no scientific fleet or policy conclusion is drawn from it.

## Compatibility matrix

| Check | Status |
|---|---|
| exact_historical_request_injection | `SUPPORTED_BY_THIN_ADAPTER` |
| custom_initial_vehicle_positions | `SUPPORTED_DIRECTLY` |
| hv_availability_windows | `SUPPORTED_BY_THIN_ADAPTER` |
| always_on_av_availability | `SUPPORTED_BY_THIN_ADAPTER` |
| fleet_control_waiting_available_time_access | `SUPPORTED_BY_SUBCLASS` |
| custom_assignment_submission | `SUPPORTED_DIRECTLY` |
| completed_service_position_update | `SUPPORTED_DIRECTLY` |
| vehicle_available_again | `SUPPORTED_DIRECTLY` |
| external_valhalla_travel_time | `SUPPORTED_BY_THIN_ADAPTER` |
| request_vehicle_kpi_logs | `SUPPORTED_BY_THIN_ADAPTER` |

## Core FleetPy modifications

`NONE`

## Recommendation

`GO_FLEETPY`

Pinned FleetPy request/vehicle/leg lifecycle and native logging worked with thin Test31, availability, and Valhalla adapters and no upstream modification.

## Limitations and stop condition

- The spike directly exercised FleetPy request objects, external vehicles, route legs, state transitions, and native leg logs. It did not run the full `FleetSimulationBase` scenario loader; the inspected subclass hooks are reserved for a separately authorized rolling replay.
- Request selection applied only a basic finite-coordinate and positive finite service-time smoke filter before stable-hash truncation; no missing predictor was imputed.
- Occupied progression used frozen realized service duration; predicted service time remained a separate decision-time field and was used only for the HV session-end admission check.
- No full-day replay, OR/Gamma/passenger model, repositioning, charging, or AV penetration scenario was run.

`ROLLING_DISPATCH_KERNEL = NOT STARTED`
