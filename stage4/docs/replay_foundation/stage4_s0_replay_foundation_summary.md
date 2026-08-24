# Stage4 S0 Replay Foundation Summary

## Driver/session reconstruction

- Full Test31 source drivers: 17,341
- Drivers with valid reconstructed sessions: 17,337
- Sessions: 29,604
- 90-minute splits: 12,267
- Negative inter-order gaps recorded: 4,546
- Session duration (s): `{'p50': 3196.0, 'p90': 12506.100000000002, 'p95': 16488.0, 'p99': 25534.130000000034}`
- Orders/session: `{'p50': 3.0, 'p90': 7.0, 'p95': 10.0, 'p99': 15.0}`
- Inter-order gap (s): `{'p50': 712.0, 'p90': 7939.699999999997, 'p95': 15040.399999999965, 'p99': 38405.92000000004}`

These sessions are effective observed service episodes, not true online or employment shifts.

## 15-minute fleet scaling

- Full demand total: 105,460
- Replay demand total: 30,000
- Full active supply: `{'p00': 136.0, 'p50': 2730.5, 'p100': 3635.0}`
- Target replay supply: `{'p00': 36.0, 'p50': 789.0, 'p100': 1088.0}`
- Selected replay fleet templates: 8,442
- Supply-fit MAE: 24.594
- Supply-fit maximum absolute error: 91.000
- Mean relative error where target > 0: 5.5547%
- Exactly matched bins: 3/96

Target supply uses deterministic nearest-integer rounding (`floor(x + 0.5)`). Complete sessions are selected within start-time bins by a seed-bound SHA-256 priority; no fleet optimizer is used.

## Pickup ETA calibration

- Valid calibration orders: 30,000
- Invalid ATA rows: 0
- Invalid Valhalla ETA rows: 0
- ATA/Valhalla ratio: `{'p01': 0.5507413892045858, 'p05': 0.8278062719286721, 'p50': 1.9055717401529642, 'p95': 3.9341558875749807, 'p99': 5.862615064272282}`
- Global median ratio: 1.905572
- Selected multiplier min/p50/max: `{'p00': 1.0086380049842976, 'p50': 1.7638227954651395, 'p100': 2.901415429348088}`
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

The Stage0 Test31 candidate manifest is the canonical full-order activity source because it contains all 105,460 source orders with driver, timestamps, and OD coordinates. The Stage1 frozen Test31 order base defines the exact 30,000 replay orders. No independent canonical auto-route ETA product existed, so S0 computes exactly one deterministic Valhalla `auto` route per replay OD at request time using the same frozen config and tiles as Stage3. Stage1 trace-route elapsed values are not used because they inherit observed trajectory timing. Frozen Stage3 descriptors and the final Stage3→Stage4 interface supply decision-time service predictions and per-profile capability fields.

## Known limitations

- Sessions represent observed service episodes; unseen idle/online drivers are not inferred.
- Fleet scaling samples complete sessions by start-time bin and may leave a small 15-minute supply error.
- ETA calibration is a Test31 time-of-day median ratio, not a new route-level prediction model.
- Small source-day timestamp spillover beyond local midnight is clipped to the final Test31 bin and counted in the local summary JSON.
- Existing legacy Stage4 dispatch/simulator code was not invoked or modified by S0.

`ROLLING_DISPATCH = NOT STARTED`
