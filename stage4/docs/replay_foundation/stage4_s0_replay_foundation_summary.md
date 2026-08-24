# Stage4 S0 Replay Foundation Summary

## S0 correction comparison

| Metric | Before correction | After correction |
|---|---:|---:|
| Session count | 29,604 | 29,583 |
| Fleet size | 8,442 | 8,435 |
| Supply-fit MAE | 24.593750 | 24.593750 |
| Global beta | 1.905572 | 2.413763 |

The pre-correction beta is retained only as lineage and is scientifically invalid because the legacy Valhalla requests interpreted GCJ-02 as WGS84.

## Driver/session reconstruction

- Full Test31 source drivers: 17,341
- Drivers with valid reconstructed sessions: 17,337
- Sessions: 29,583
- 90-minute splits: 12,246
- Negative inter-order gaps recorded: 4,600
- Session duration (s): `{'p50': 3213.0, 'p90': 12545.8, 'p95': 16544.09999999998, 'p99': 25557.340000000004}`
- Orders/session: `{'p50': 3.0, 'p90': 7.0, 'p95': 10.0, 'p99': 15.0}`
- Inter-order gap (s): `{'p50': 703.0, 'p90': 7932.699999999997, 'p95': 15039.0, 'p99': 38389.0}`

Effective gaps use the previous running-maximum arrival time. Session end and final drop-off come from the order with the maximum arrival time. These sessions are effective observed service episodes, not true online or employment shifts.

## 15-minute fleet scaling

- Full demand total: 105,460
- Replay demand total: 30,000
- Full active supply: `{'p00': 136.0, 'p50': 2737.5, 'p100': 3646.0}`
- Target replay supply: `{'p00': 36.0, 'p50': 790.0, 'p100': 1090.0}`
- Selected replay fleet templates: 8,435
- Supply-fit MAE: 24.594
- Supply-fit maximum absolute error: 90.000
- Mean relative error where target > 0: 5.5420%
- Exactly matched bins: 3/96
- Top-5 absolute-error bins: `[{'time_bin_index': 82, 'time_bin_start': '2016-10-31T20:30:00+08:00', 'target_active_supply': 801, 'simulated_active_supply': 891, 'absolute_supply_error': 90.0, 'relative_supply_error': 0.11235955056179775}, {'time_bin_index': 42, 'time_bin_start': '2016-10-31T10:30:00+08:00', 'target_active_supply': 760, 'simulated_active_supply': 840, 'absolute_supply_error': 80.0, 'relative_supply_error': 0.10526315789473684}, {'time_bin_index': 91, 'time_bin_start': '2016-10-31T22:45:00+08:00', 'target_active_supply': 543, 'simulated_active_supply': 607, 'absolute_supply_error': 64.0, 'relative_supply_error': 0.11786372007366483}, {'time_bin_index': 53, 'time_bin_start': '2016-10-31T13:15:00+08:00', 'target_active_supply': 847, 'simulated_active_supply': 785, 'absolute_supply_error': 62.0, 'relative_supply_error': 0.07319952774498228}, {'time_bin_index': 51, 'time_bin_start': '2016-10-31T12:45:00+08:00', 'target_active_supply': 706, 'simulated_active_supply': 767, 'absolute_supply_error': 61.0, 'relative_supply_error': 0.08640226628895184}]`
- Top-5 relative-error bins: `[{'time_bin_index': 14, 'time_bin_start': '2016-10-31T03:30:00+08:00', 'target_active_supply': 43, 'simulated_active_supply': 55, 'absolute_supply_error': 12.0, 'relative_supply_error': 0.27906976744186046}, {'time_bin_index': 16, 'time_bin_start': '2016-10-31T04:00:00+08:00', 'target_active_supply': 37, 'simulated_active_supply': 45, 'absolute_supply_error': 8.0, 'relative_supply_error': 0.21621621621621623}, {'time_bin_index': 20, 'time_bin_start': '2016-10-31T05:00:00+08:00', 'target_active_supply': 44, 'simulated_active_supply': 36, 'absolute_supply_error': 8.0, 'relative_supply_error': 0.18181818181818182}, {'time_bin_index': 19, 'time_bin_start': '2016-10-31T04:45:00+08:00', 'target_active_supply': 46, 'simulated_active_supply': 39, 'absolute_supply_error': 7.0, 'relative_supply_error': 0.15217391304347827}, {'time_bin_index': 3, 'time_bin_start': '2016-10-31T00:45:00+08:00', 'target_active_supply': 171, 'simulated_active_supply': 146, 'absolute_supply_error': 25.0, 'relative_supply_error': 0.14619883040935672}]`

Target supply uses deterministic nearest-integer rounding (`floor(x + 0.5)`). Complete sessions are selected within start-time bins by a seed-bound SHA-256 priority; no fleet optimizer is used.

## Pickup ETA calibration

- Valid calibration orders: 30,000
- Invalid ATA rows: 0
- Invalid Valhalla ETA rows: 0
- ATA/Valhalla ratio: `{'p01': 1.03952259258331, 'p05': 1.3634701831757134, 'p50': 2.4137629875909457, 'p95': 4.822029643332529, 'p99': 7.367493537745861}`
- Global median ratio: 2.413763
- Selected multiplier min/p50/max: `{'p00': 1.3189033252684674, 'p50': 2.2521660084812534, 'p100': 3.693838257484858}`
- Fallback-bin counts: `{'BIN_MEDIAN': 87, 'HOUR_MEDIAN': 9}`

## Produced files

- `stage4/input/replay_foundation/full_test31_driver_sessions.parquet`
- `stage4/input/replay_foundation/replay_fleet_template.parquet`
- `stage4/input/replay_foundation/fleet_scaling_15min.parquet`
- `stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet`
- `stage4/input/replay_foundation/stage4_order_replay_base.parquet`
- `stage4/input/replay_foundation/stage4_s0_summary.json`
- `stage4/input/replay_foundation/historical_valhalla_auto_eta.parquet`

## Input selection

The Stage0 Test31 candidate manifest is the canonical full-order activity source because it contains all 105,460 source orders with driver, timestamps, and OD coordinates. Manifest GCJ-02 coordinates are preserved for lineage and converted with `stage0.v6.coordinates.gcj02_to_wgs84`; Valhalla and every future vehicle-to-pickup route must use only the explicit WGS84 fields. The Stage1 frozen Test31 order base defines the exact 30,000 replay orders. No independent canonical auto-route ETA product existed, so S0 computes exactly one deterministic Valhalla `auto` route per replay OD at request time using the same frozen config and tiles as Stage3. Stage1 trace-route elapsed values are not used because they inherit observed trajectory timing. Frozen Stage3 descriptors and the final Stage3→Stage4 interface supply decision-time service predictions and per-profile capability fields.

## Known limitations

- Sessions represent observed service episodes; unseen idle/online drivers are not inferred.
- Fleet scaling samples complete sessions by start-time bin and may leave a small 15-minute supply error.
- ETA calibration is a day-specific replay traffic calibration using Test31 aggregate time-of-day medians; it is not a strict out-of-sample decision-time ETA predictor.
- Small source-day timestamp spillover beyond local midnight is clipped to the final Test31 bin and counted in the local summary JSON.
- Existing legacy Stage4 dispatch/simulator code was not invoked or modified by S0.

`ROLLING_DISPATCH = NOT STARTED`
