from types import SimpleNamespace
import pandas as pd
import pytest
from stage4.tests.test_fleetpy_native_shell import _bindings, _request, _fixture, START, END
from stage4.fleetpy_adapter.upstream import CoordinateRegistry
from stage4.fleetpy_adapter.test31_demand_adapter import attach_fleetpy_requests
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.native_demand import create_native_demand
from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
from stage4.fleetpy_adapter.native_fleet_control import create_native_fleet_control
from stage4.fleetpy_adapter.native_simulation import create_native_simulation
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate
from stage4.analysis.mechanism_validity import production_neutral_arcs
from stage4.dispatch.candidate_graph import SpatialVehicle


@pytest.mark.parametrize('overhead', [0,60,120])
def test_native_pickup_duration_and_lifecycle(tmp_path,overhead):
    b = _bindings()
    registry = CoordinateRegistry()
    request = _request(sim_time_s=0,request_time=START)
    attach_fleetpy_requests([request],b,registry)
    net = create_native_network(b,registry)
    demand = create_native_demand(b,[request],registry,net,tmp_path)
    vehicles, output = create_native_vehicles([_fixture('AV')],b,registry,demand.rq_db,
        tmp_path/'runtime',native_movement=True,routing_engine=net)
    adapter = SimpleNamespace(estimate=lambda *args:PickupEstimate(100,100,1000,1,0,False))
    control = create_native_fleet_control(b,vehicles,[request],demand,net,adapter,START,END)
    control.config = {'additional_pickup_overhead_s':overhead}
    sim = create_native_simulation(b,simulation_end_s=600,time_step_s=30,demand=demand,
        vehicles=[v.native_vehicle for v in vehicles],fleet_control=control,network=net,native_output=output)
    for t in range(0,601,30):
        sim.step(t)
    row = control.assignment_rows[0]
    assert row['pickup_time'] == START+pd.Timedelta(seconds=100)
    assert row['service_end_time'] == START+pd.Timedelta(seconds=100+overhead+321)
    assert row['completed']


def test_rolling_session_admission_counts_dwell_but_patience_is_arrival():
    r = _request(sim_time_s=0,request_time=START,predicted_service_time_s=280)
    v = SpatialVehicle('HV_000',0,'HV',108.89,34.19)
    fixture = _fixture(end=START+pd.Timedelta(seconds=400))
    adapter = SimpleNamespace(routing_time_s=0,estimate_many=lambda candidates,*args:
        {0:PickupEstimate(100,100,1000,1,0,False)})
    cfg = dict(candidate_top_k=20,search_radius_initial_m=2000,search_radius_step_m=1000,
        search_radius_cap_m=8000,max_pickup_wait_s=100)
    args = ([v],{0:fixture},[(r,0,False,False)],START,START)
    assert len(production_neutral_arcs(*args,cfg,adapter,neutral=False)) == 1
    assert not production_neutral_arcs(*args,{**cfg,'additional_pickup_overhead_s':60},adapter,neutral=False)
