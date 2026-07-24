# FleetPy adapter technical validation plan

## Scope

FleetPy is a candidate execution kernel, not yet the formal simulator. Validation
is restricted to 500--2,000 orders with Stay, preassignment off, one non-shared
request per vehicle, and precomputed HV/AV service routes. No result from this
test is eligible for paper inference.

## Required adapter capabilities

1. Read one explicit Stage 3.5 manifest and resolve routes by order, vehicle type
   and departure-time product version.
2. Represent HV/AV attributes and apply distinct HV response and AV ODD checks.
3. Execute pickup and service legs without online service-route replacement.
4. Accept a custom zonal sparse fleet controller.
5. Export request transitions, vehicle legs, route identity and economy fields.

## Baseline comparison

Run the same request table, initial fleet, patience, pickup inputs and service
durations through FleetPy and the retained Simulator v3 comparison harness.
Compare completed/cancelled counts, waiting and pickup time, busy time, pickup
distance, service distance and state continuity. Any discrepancy above the
registered tolerance must be traced to an explicit semantic difference or fixed.

## Zonal matching validation

Every 30 seconds, solve sparse within-zone graphs, then adjacent-zone additions,
then a bounded spillover pool. Report per-zone orders/vehicles/edges, local and
spillover match rates, peak memory and solve time. On a small common instance,
compare zonal and citywide sparse matching to quantify objective loss.

## Exit gate

FleetPy becomes the formal execution kernel only after the adapter, state,
precomputed-route, zonal-matching and cross-kernel audits pass. Until then,
formal full-day Stage 4 remains disabled in `config/pipeline_research_v3.yaml`.
