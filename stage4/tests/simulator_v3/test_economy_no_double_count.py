import pandas as pd

from stage4.simulator_v3.economics import EconomicsModel
from stage4.simulator_v3.entities import RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import VehicleExecutionStatus


def request() -> RequestState:
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    return RequestState("o1", t, t, 108.9, 34.2, "z0_0", 108.91, 34.21, "z0_0", t + pd.Timedelta(minutes=8), True, 600, 600, 3000, 0.5)


def vehicle(vehicle_type="HV") -> VehicleState:
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    return VehicleState("v1", vehicle_type, 108.9, 34.2, "z0_0", t, t + pd.Timedelta(hours=1), VehicleExecutionStatus.IDLE, None, VehiclePlan("v1", 0, [], t, "init", True, 0.0), 0)


def test_economics_has_single_marginal_contribution_identity_for_hv():
    econ = EconomicsModel().evaluate(request(), vehicle("HV"), 1000.0, 300.0, 60.0, "ODD-Gated Price-Aware")
    assert econ.driver_payout > 0
    assert econ.marginal_operating_contribution < econ.fare_revenue

