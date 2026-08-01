# Stage 2 v5 Service-time Target Audit

## Conclusion

The frozen products do not contain a complete realized full-edge travel time for every route token. The reliable primary supervision is directly observed pace (`observed_sec_per_m`), calculated from directly timed GPS intervals. Stage 2 v5 therefore models positive pace as its primary distribution target and derives a full traversal service time from predicted pace × `allocated_distance_m`.

`travel_time_s`, `observed_travel_time_s`, and `direct_observed_time_s` are not independent labels: on direct rows they represent the same direct-interval timing evidence. Engine allocation is disabled, so interpolated and interval-supported tokens do not have individual-link service-time targets.

## Coverage

| Metric | Value |
|---|---:|
| Route tokens | 15,649,455 |
| Traversal labels | 5,775,530 |
| Direct time valid | 5,775,530 (36.91%) |
| Direct pace valid | 5,309,097 (33.93%) |
| RTS measurement available | 5,309,097 (33.93%) |
| Engine/interpolated time labels | 0 |

## Provenance and consistency

| Audit | Count |
|---|---:|
| Duplicate traversal keys | 0 |
| Duplicate label keys | 0 |
| Missing paired label rows | 9,873,925 |
| Direct time mismatch between input and label | 0 |
| Pace formula mismatch | 0 |
| Measurement-source mismatch | 0 |
| `travel_time_s` alias mismatch | 0 |
| `time_observation_valid=true` on non-direct rows | 9,873,925 |

The last row is an upstream flag-semantics defect: v5 never uses `time_observation_valid` alone. It requires a direct measurement source and finite positive time/distance.

## Distribution summary

| Quantity | Count | Mean | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| Direct observed time (s) | 5,775,530 | 10.0977 | 6.0000 | 21.0000 | 27.0000 | 39.0000 |
| Direct observed distance (m) | 5,762,834 | 83.1416 | 59.7779 | 183.3958 | 229.9107 | 340.7615 |
| Direct pace (s/m) | 5,309,097 | 0.134702 | 0.121026 | 0.209887 | 0.244726 | 0.296697 |
| Direct distance coverage | 5,775,530 | 0.6526 | 0.6780 | 0.8943 | 0.9293 | 0.9741 |

Quantiles use a deterministic bounded sample per partition; counts and moments are exact.

## Frozen v5 target contract

- Primary distribution target: positive direct-observed pace.
- Derived physical traversal time: predicted pace × allocated traversal distance.
- Direct observed time: auxiliary/high-coverage sensitivity target.
- Interpolated and interval-supported time: unavailable for link supervision in the frozen upstream.
- Missing targets remain `NaN` with explicit masks and source classes.
- RTS remains a secondary relative-delay target; LCS components remain auxiliary state targets.
