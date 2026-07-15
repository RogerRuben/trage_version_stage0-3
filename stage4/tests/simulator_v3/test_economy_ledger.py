import pandas as pd

from stage4.simulator_v3.economy_ledger import (
    ScenarioCostParameters,
    assignment_ledger_row,
    audit_ledger,
    build_scenario_ledger,
)


def test_economy_ledger_recomputes_without_double_counting():
    served = [assignment_ledger_row("o1", "av1", "AV", {
        "fare_revenue": 20.0,
        "pickup_variable_cost": 2.0,
        "service_variable_cost": 3.0,
        "capability_cost": 1.0,
    })]
    legs = pd.DataFrame([
        {"leg_type": "AV_REBALANCE", "distance_m": 1000.0},
        {"leg_type": "PICKUP", "distance_m": 500.0},
    ])
    ledger = build_scenario_ledger(
        served,
        ["o2"],
        legs,
        av_vehicle_count=1,
        depot_count=1,
        preassignment_failure_count=1,
        params=ScenarioCostParameters(
            lost_demand_cost_per_order=5.0,
            av_fixed_daily_cost_per_vehicle=10.0,
            depot_daily_cost=2.0,
            av_rebalancing_cost_per_km=1.0,
            preassignment_failure_cost=3.0,
        ),
    )
    audit = audit_ledger(ledger)
    assert audit["economy_audit_status"] == "PASS"
    # 14 contribution - 5 lost - 10 fixed - 2 depot - 1 rebalance - 3 failure
    assert audit["scenario_net_profit"] == -7.0
