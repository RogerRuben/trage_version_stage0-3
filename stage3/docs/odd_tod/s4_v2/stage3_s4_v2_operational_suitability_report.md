# Stage 3 S4 v2 — AV Operational Suitability Interface

S4 v2 keeps the frozen Test31 historical route fixed and exposes three separate products: structural hard feasibility, continuous capability-envelope utilization, and diagnostic attribution.

It does not estimate AV safety, accident probability, legal certification, passenger choice, or an optimized route. No rerouting, fallback, dispatch, profile refit, CDF fit, or M3 retraining was performed.

## Population

- Test date: `20161031`
- Orders: 30,000
- Order-profile rows: 90,000
- Original route only: YES
- Passenger acceptance probability: reserved nullable Stage4 input; not modeled here

## Profile comparison

| Profile | Hard FEASIBLE | Hard UNKNOWN | Hard INFEASIBLE | rho overall p50 | p90 | p99 | near 1 | >1 | >2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 5,821 | 6,511 | 17,668 | 10.0000 | 34.4000 | 38.6000 | 68 | 29,825 | 27,930 |
| M | 9,258 | 4,920 | 15,822 | 2.9412 | 10.1176 | 11.3529 | 1,294 | 28,676 | 20,655 |
| A | 9,848 | 5,315 | 14,837 | 1.4400 | 4.7123 | 5.2877 | 5,767 | 22,294 | 12,391 |

`rho > 1` means the frozen route descriptor exceeds at least one frozen profile envelope. It is a capability-requirement signal, not an automatic impossibility or safety claim.

## Capability requirement distribution by family

| Profile | Static p50 / p90 | Dynamic p50 / p90 | Speed p50 / p90 | Overall p50 / p90 |
|---|---:|---:|---:|---:|
| C | 10.0000 / 34.4000 | 1.2800 / 2.3044 | 1.1667 / 1.1667 | 10.0000 / 34.4000 |
| M | 2.9412 / 10.1176 | 0.9891 / 1.5317 | 0.8750 / 0.8750 | 2.9412 / 10.1176 |
| A | 1.4400 / 4.7123 | 0.8488 / 1.0755 | 0.5833 / 0.5833 | 1.4400 / 4.7123 |

## Dominant utilization family

- C: DIRECTION=14,837, STATIC=14,378, DYNAMIC=546, SPEED=176, UNKNOWN_INCOMPLETE=63
- M: DIRECTION=14,837, STATIC=13,786, DYNAMIC=1,207, SPEED=107, UNKNOWN_INCOMPLETE=63
- A: DIRECTION=14,837, STATIC=12,830, DYNAMIC=2,269, UNKNOWN_INCOMPLETE=63, SPEED=1

Direction is reported as a structural hard constraint. Static, dynamic, and speed bottlenecks are determined by the non-weighted maximum utilization family. Missing critical evidence remains UNKNOWN and is never silently dropped from `rho_overall`.

## Stage4 contract

Stage4 should consume `hard_state + rho_* + vectors + reason_codes + passenger_acceptance_probability`. The passenger field is intentionally null in S4 v2.

S5_AUTHORIZED = NO
NEXT_PHASE_AUTHORIZED = NO
