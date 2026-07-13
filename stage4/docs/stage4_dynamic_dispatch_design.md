# Stage4 Dynamic Dispatch Design

The formal Stage4 simulator is implemented in
`stage4/scripts/run_stage4_pricing_dispatch_experiments.py`.

## Dynamic loop

Each 120-second dispatch window executes:

1. update vehicle status;
2. release completed vehicles;
3. add new requests to the pending queue;
4. cancel orders exceeding passenger patience;
5. generate zone-based candidate vehicles;
6. construct price/utility/ODD-feasible edges;
7. solve matching using greedy, random, or Hungarian global matching;
8. update vehicle state, income, burden, and location;
9. write order/window/vehicle logs.

The pending queue processes all eligible pending orders in each window; it does
not block on `pending[0]`.

## Vehicle state

Vehicles have `OFFLINE`, `IDLE`, `BUSY`, and `NEAR_FREE` states. HV supply is a
reconstructed scenario with shifts and release locations; AV supply is
initialized from observable early-window demand distributions. The model does
not use future full-day demand to initialize fleet positions.

## Matching

GlobalMatch strategies use Hungarian assignment over a window-level candidate
matrix. Each order is matched at most once and each vehicle at most once per
window. Unserved orders remain pending until cancellation.

ODD feasibility only applies to AV edges. HVs are governed by driver utility and
compensation.
