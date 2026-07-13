# Stage4 Stakeholder Utility Model

## Passenger generalized cost

```text
GC_om = quoted_fare
      + value_of_wait_time * accumulated_waiting
      + value_of_pickup_time * pickup_time
      + vehicle_preference_adjustment
```

Accumulated waiting is recomputed in every dispatch window as
`current_window_time - order_decision_time`, so passenger generalized cost rises
while an order remains pending. The simulator supports deterministic threshold
acceptance. An order-vehicle edge is infeasible if passenger generalized cost
exceeds the scenario threshold.

## HV driver utility

```text
U_ov^HV = driver_payout
        - pickup_cost
        - service_cost
        - stress_disutility
```

HV payout is explicitly decomposed into base payout, service-time payout, pickup
compensation, scarcity bonus, and gross stress compensation. HV acceptance is
controlled by a minimum utility threshold.

## AV operating cost

AV cost includes pickup empty mileage, service distance/time, energy,
capability cost, and placeholder remote-assistance/fallback terms. AVs have no
driver payout. Non-ODD baselines may serve ODD-infeasible AV edges, but those
edges carry fallback cost and are recorded as ODD violations.

## Platform profit

```text
profit_HV = passenger_fare - driver_payout - platform_variable_cost
profit_AV = passenger_fare - AV_operating_cost
```

All served-order logs record fare, payout, platform cost, compensation source,
and profit. `audit_pricing_accounting.py` verifies these identities.
