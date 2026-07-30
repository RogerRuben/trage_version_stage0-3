# Stage 1 v3 methodology

Stage 1 v3 derives retrospective realized-traffic labels from the frozen
`stage1/input_v1` product. It never modifies Stage 0 or the 220,000 accepted
orders. The primary supervision unit is `(order_id, traversal_id)`; order-level
values are auxiliary summaries.

## Evidence and identity

Only `link_interval_observations` rows with
`measurement_source == direct_observed` and `label_valid == true` contribute
dynamic evidence. Modeling uses `observed_directed_edge_uid`; the original
`canonical_edge_uid` remains physical lineage. Train alone fits the directed
catalog, support counts, reference pace, and LCS/RTS empirical CDFs.
Validation and Test never change a fitted object.

## LCS

For interval speed \(s\), with frozen thresholds \(s_{stop}=1\) m/s and
\(s_{low}=5\) m/s:

```text
stop  = s <= s_stop
crawl = s_stop < s < s_low
```

The states are mutually exclusive. `is_low_speed_total` is their union.
Traversal `stop_time_share` and `crawl_time_share` use direct-observed duration
as weights.

For adjacent direct intervals, acceleration uses the difference between their
midpoint timestamps:

```text
a_i = (v_i - v_(i-1)) / delta_midpoint_i
acceleration_rms = sqrt(sum(delta_midpoint_i * a_i^2)
                        / sum(delta_midpoint_i))
```

`maximum_absolute_acceleration_mps2` remains the unweighted maximum absolute
acceleration. LCS retains four separately usable components:

1. `crawl_time_share`
2. `stop_time_share`
3. `speed_cv_bounded`
4. `acceleration_rms_bounded`

`lcs_raw` is their equal-weight baseline, not a unique ground-truth construct.
`lcs_pct` is the row's conditional position in a Train-fitted empirical CDF.

## RTS

RTS compares direct realized seconds per metre with a Train-fitted conditional
reference:

```text
excess_time_ratio = max(observed_sec_per_m / reference_sec_per_m - 1, 0)
rts_raw = excess_time_ratio / (1 + excess_time_ratio)
```

`rts_measurement_available` describes upstream physical measurement
eligibility. `rts_available` additionally requires reference and CDF support.
Upstream missing reasons are preserved and never replaced by a downstream
support reason.

## Physical distance gate

If summed direct distance exceeds traversal allocated distance by more than
`direct.distance_identity_tolerance_m`, the traversal keeps lineage but emits
neither LCS nor RTS. Its fixed reason is
`DIRECT_DISTANCE_EXCEEDS_TRAVERSAL`; pace is missing, and the row cannot enter
reference or normalization fitting.

## Support

All road-type support uses `canonical_highway`. The legacy
`road_class_hour` scope is forbidden. Frozen Train-only scopes are `edge`,
`edge_hour`, `highway_hour`, `node_hour`, and `global_hour`.
`edge_time_bin_30m` is an additional diagnostic count and does not replace
hourly fallback. Fallback order is:

```text
edge_hour -> highway_hour -> spatial_neighbor -> global_hour -> unavailable
```

An edge unseen in Train has zero edge, edge-hour, and edge-30-minute support.

## Tail and missingness semantics

Traversal `lcs_tail_event` and `rts_tail_event`, and order
`*_tail_event_present`, are nullable booleans:

```text
unavailable dimension -> NA
available, no tail     -> false
available, tail        -> true
```

For an available order with no tail, tail mean remains missing and persistence
is zero. Missing values are never converted to zero.

## Scientific boundary

The engineering audit establishes formulas, keys, identities, leakage
controls, ranges, and missingness. The scientific review reports availability,
distribution shift, support fallback, and coverage. Neither report claims that
the equal-weight LCS scalar is theoretically unique or that retrospective
labels are permissible prediction-time features.
