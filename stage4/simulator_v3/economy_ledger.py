"""Recomputable scenario ledger for Simulator v3.

Matching uses only ``marginal_operating_contribution``.  Daily deployment and
unserved-demand costs are deliberately appended here after simulation, so
they can never leak into an edge objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScenarioCostParameters:
    lost_demand_cost_per_order: float = 5.0
    av_fixed_daily_cost_per_vehicle: float = 80.0
    depot_daily_cost: float = 150.0
    hv_reposition_cost_per_km: float = 0.30
    av_rebalancing_cost_per_km: float = 0.30
    preassignment_failure_cost: float = 1.0


EDGE_COLUMNS = [
    "fare_revenue",
    "driver_payout",
    "pickup_cost",
    "service_cost",
    "capability_cost",
    "remote_assistance_cost",
    "platform_variable_cost",
]


def assignment_ledger_row(order_id: str, vehicle_id: str, vehicle_type: str, metadata: dict) -> dict:
    row = {
        "ledger_type": "served_order",
        "order_id": str(order_id),
        "vehicle_id": str(vehicle_id),
        "vehicle_type": str(vehicle_type),
        "fare_revenue": float(metadata.get("fare_revenue", 0.0)),
        "driver_payout": float(metadata.get("driver_payout", 0.0)),
        "pickup_cost": float(metadata.get("pickup_variable_cost", 0.0)),
        "service_cost": float(metadata.get("service_variable_cost", 0.0)),
        "capability_cost": float(metadata.get("capability_cost", 0.0)),
        "remote_assistance_cost": float(metadata.get("remote_assistance_cost", 0.0)),
        "platform_variable_cost": float(metadata.get("platform_variable_cost", 0.0)),
        "hv_repositioning_cost": 0.0,
        "av_rebalancing_cost": 0.0,
        "av_fixed_cost": 0.0,
        "depot_cost": 0.0,
        "lost_demand_cost": 0.0,
        "preassignment_failure_cost": 0.0,
    }
    row["operating_contribution"] = (
        row["fare_revenue"]
        - row["driver_payout"]
        - row["pickup_cost"]
        - row["service_cost"]
        - row["capability_cost"]
        - row["remote_assistance_cost"]
        - row["platform_variable_cost"]
    )
    row["scenario_net_profit_component"] = row["operating_contribution"]
    return row


def build_scenario_ledger(
    served_rows: list[dict],
    cancelled_order_ids: list[str],
    vehicle_legs: pd.DataFrame,
    av_vehicle_count: int,
    depot_count: int,
    preassignment_failure_count: int,
    params: ScenarioCostParameters | None = None,
) -> pd.DataFrame:
    params = params or ScenarioCostParameters()
    rows = list(served_rows)

    if cancelled_order_ids:
        rows.append({
            "ledger_type": "lost_demand",
            "order_id": "",
            "vehicle_id": "",
            "vehicle_type": "",
            **{c: 0.0 for c in EDGE_COLUMNS},
            "hv_repositioning_cost": 0.0,
            "av_rebalancing_cost": 0.0,
            "av_fixed_cost": 0.0,
            "depot_cost": 0.0,
            "lost_demand_cost": len(cancelled_order_ids) * params.lost_demand_cost_per_order,
            "preassignment_failure_cost": 0.0,
            "operating_contribution": 0.0,
            "scenario_net_profit_component": -len(cancelled_order_ids) * params.lost_demand_cost_per_order,
        })

    if vehicle_legs is None or vehicle_legs.empty:
        hv_km = av_km = 0.0
    else:
        distance = pd.to_numeric(vehicle_legs.get("distance_m", 0.0), errors="coerce").fillna(0.0)
        leg_type = vehicle_legs.get("leg_type", pd.Series("", index=vehicle_legs.index)).astype(str)
        hv_km = float(distance[leg_type.eq("HV_REPOSITION")].sum() / 1000.0)
        av_km = float(distance[leg_type.eq("AV_REBALANCE")].sum() / 1000.0)
    movement_cost = hv_km * params.hv_reposition_cost_per_km + av_km * params.av_rebalancing_cost_per_km
    rows.append({
        "ledger_type": "scenario_costs",
        "order_id": "",
        "vehicle_id": "",
        "vehicle_type": "",
        **{c: 0.0 for c in EDGE_COLUMNS},
        "hv_repositioning_cost": hv_km * params.hv_reposition_cost_per_km,
        "av_rebalancing_cost": av_km * params.av_rebalancing_cost_per_km,
        "av_fixed_cost": av_vehicle_count * params.av_fixed_daily_cost_per_vehicle,
        "depot_cost": depot_count * params.depot_daily_cost,
        "lost_demand_cost": 0.0,
        "preassignment_failure_cost": preassignment_failure_count * params.preassignment_failure_cost,
        "operating_contribution": 0.0,
        "scenario_net_profit_component": -(
            movement_cost
            + av_vehicle_count * params.av_fixed_daily_cost_per_vehicle
            + depot_count * params.depot_daily_cost
            + preassignment_failure_count * params.preassignment_failure_cost
        ),
    })
    return pd.DataFrame(rows)


def audit_ledger(ledger: pd.DataFrame, tolerance: float = 1e-6) -> dict:
    if ledger.empty:
        return {"ledger_rows": 0, "identity_error": None, "economy_audit_status": "FAIL"}
    served = ledger[ledger["ledger_type"].eq("served_order")].copy()
    recomputed = (
        served["fare_revenue"]
        - served["driver_payout"]
        - served["pickup_cost"]
        - served["service_cost"]
        - served["capability_cost"]
        - served["remote_assistance_cost"]
        - served["platform_variable_cost"]
    )
    edge_error = float((recomputed - served["operating_contribution"]).abs().max()) if len(served) else 0.0
    expected_net = float(
        ledger["operating_contribution"].sum()
        - ledger["lost_demand_cost"].sum()
        - ledger["av_fixed_cost"].sum()
        - ledger["depot_cost"].sum()
        - ledger["hv_repositioning_cost"].sum()
        - ledger["av_rebalancing_cost"].sum()
        - ledger["preassignment_failure_cost"].sum()
    )
    actual_net = float(ledger["scenario_net_profit_component"].sum())
    net_error = abs(expected_net - actual_net)
    duplicate_served = int(served["order_id"].duplicated().sum()) if len(served) else 0
    non_served = ledger[~ledger["ledger_type"].eq("served_order")]
    edge_cost_outside_served = float(
        non_served[[
            "fare_revenue", "driver_payout", "pickup_cost", "service_cost",
            "capability_cost", "remote_assistance_cost", "platform_variable_cost",
        ]].abs().sum().sum()
    )
    movement_cost_outside_scenario = float(
        ledger[~ledger["ledger_type"].eq("scenario_costs")][
            ["hv_repositioning_cost", "av_rebalancing_cost"]
        ].abs().sum().sum()
    )
    fixed_cost_in_served = float(
        served[["av_fixed_cost", "depot_cost", "lost_demand_cost", "preassignment_failure_cost"]]
        .abs().sum().sum()
    ) if len(served) else 0.0
    scenario_row_count = int(ledger["ledger_type"].eq("scenario_costs").sum())
    ok = (
        edge_error <= tolerance
        and net_error <= tolerance
        and duplicate_served == 0
        and edge_cost_outside_served <= tolerance
        and movement_cost_outside_scenario <= tolerance
        and fixed_cost_in_served <= tolerance
        and scenario_row_count == 1
    )
    return {
        "ledger_rows": int(len(ledger)),
        "served_ledger_rows": int(len(served)),
        "duplicate_served_ledger_rows": duplicate_served,
        "edge_operating_contribution_max_error": edge_error,
        "scenario_net_profit_recalculation_error": net_error,
        "edge_cost_outside_served_rows": edge_cost_outside_served,
        "movement_cost_outside_scenario_row": movement_cost_outside_scenario,
        "fixed_or_lost_cost_in_served_rows": fixed_cost_in_served,
        "scenario_cost_row_count": scenario_row_count,
        "operating_contribution": float(ledger["operating_contribution"].sum()),
        "scenario_net_profit": actual_net,
        "economy_audit_status": "PASS" if ok else "FAIL",
    }
