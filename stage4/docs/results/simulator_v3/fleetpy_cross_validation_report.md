# FleetPy cross-validation report

Overall status: **PASS**

Controlled requests: 500
Controlled vehicles: 699
Window: 2016-10-23T01:00:00+00:00 to 2016-10-23T02:00:00+00:00

Both engines used the frozen request/vehicle key hashes in the input manifest. This is a kernel consistency check (Stay, preassignment off, Safe/closest vehicle), not a validation of Stage4 ODD or pricing mechanisms.

| metric                |     stage4_value |    fleetpy_value |   relative_error |   threshold | status   |
|:----------------------|-----------------:|-----------------:|-----------------:|------------:|:---------|
| completed_orders      |    500           |    500           |      0           |        0.05 | PASS     |
| cancelled_orders      |      0           |      0           |      0           |        0.05 | PASS     |
| mean_pickup_time_sec  |      0           |      0           |      0           |        0.1  | PASS     |
| mean_waiting_time_sec |      0           |      0           |      0           |        0.1  | PASS     |
| vehicle_busy_time_sec | 258310           | 258310           |      2.34354e-14 |        0.1  | PASS     |
| pickup_distance_m     |      0           |      0           |      0           |        0.1  | PASS     |
| service_distance_m    |      2.06648e+06 |      2.06648e+06 |      2.2872e-14  |        0.1  | PASS     |

FleetPy source: https://github.com/TUM-VT/FleetPy
