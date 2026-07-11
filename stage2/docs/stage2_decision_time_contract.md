# Stage2 decision-time contract

This contract separates the existing oracle-route experiment from a genuinely
deployable pre-dispatch prediction problem. It supersedes earlier wording that
treated actual matched routes or actual link entry times as ordinary
pre-dispatch inputs.

## Decision event

The deployable decision time is platform dispatch time `t0`, after the platform
has received the order origin and destination but before assignment and before
any realized trajectory is observed.

The platform is assumed to know origin, destination, dispatch time, a route
produced by a policy available at `t0`, static road/POI/topology attributes, and
historical or lagged traffic summaries available strictly before prediction.
It is not assumed to know the driver's completed route, actual link entry time,
or any realized speed, stop or delay from the order being predicted.

## Two experiment tracks

### `oracle_route_upper_bound`

This track may use the actual map-matched route, actual link/movement sequence
and actual link `enter_time`. It estimates predictability conditional on knowing
the completed route and its realized timing. It is an upper-bound diagnostic,
not a deployable pre-dispatch experiment.

The current `stage2/output/link_dataset` belongs to this track because its
route sequence is reconstructed from the completed trip and its `hour` is
derived from actual `enter_time`.

### `deployable_predispatch`

This track may use only information available at `t0`:

- dispatch time, origin and destination;
- planned route and planned link/movement sequence;
- estimated link entry time from historical or strictly lagged travel times;
- rolling profiles built without current-row or future-day data;
- traffic state with `availability_timestamp < prediction_timestamp`;
- static road, geometry, topology, POI and land-use context.

The deployable link prediction timestamp is the estimated link entry time. A
departure-time approximation is allowed only as a recorded ablation.

## Field classes

| Class | Meaning | Examples |
|---|---|---|
| A | Deployable feature | dispatch time, planned link, static semantics, causal profile, causal lagged state |
| B | Oracle-only feature | actual matched route, actual `link_seq`, actual `enter_time`, actual movement sequence |
| C | Realized label | raw/percentile LCS, IIS, RTS and PMIS; tail events |
| D | Audit descriptor | matcher version, traversal quality, inferred/observed status, cohort provenance |
| E | Forbidden leakage | current-order speed/stop/delay, future-day or self-including profile |
| F | Validity/applicability | target masks, IIS applicability, coverage/missingness |

Every dynamic feature must carry or inherit:

```text
feature_timestamp
availability_timestamp
target_prediction_timestamp
```

and pass `availability_timestamp < target_prediction_timestamp`.

## Existing Stage2 field reclassification

| Field | Oracle track | Deployable track |
|---|---|---|
| actual `enter_time`, derived `hour/time_bin` | allowed | forbidden; replace with estimated entry time |
| actual matched `link_id/link_seq` | allowed | forbidden; replace with planned route |
| actual route length/position | allowed | forbidden; recompute on planned route |
| static road/POI/topology | allowed | allowed |
| train-only all-days profile | diagnostic only | forbidden for train rows; use rolling/OOF profile |
| rolling previous-day profile | allowed | allowed |
| lagged state before prediction time | allowed | allowed |
| realized travel time/speed/stop/delay | label/audit only | forbidden input |

## Output units

- link context: GNS and POI exposure descriptors;
- order-link traversal: LCS, RTS and PMIS;
- `from_link -> node -> to_link` movement: IIS applicability and conditional
  IIS severity;
- route/order: aggregation, uncertainty and auxiliary consistency targets.

## Naming conclusions

Current results must be called `oracle-route stress prediction prototype` or
`oracle-route upper-bound experiment`. Only experiments built from planned
routes, estimated link entry times and strictly causal profiles/state may be
called `deployable pre-dispatch stress prediction`.

