from types import SimpleNamespace
import pandas as pd
from stage4.analysis.frozen_state_prediction_ablation import _vehicle_state, pre_decision_vehicle_state


def test_current_action_not_in_predecision_vehicle_state():
    t = pd.Timestamp('2016-10-31T10:30:00+08:00')
    fixtures = [SimpleNamespace(native_id=i, vehicle_id=str(i), vehicle_type='AV',
        availability_start_time=t-pd.Timedelta(hours=1), availability_end_time=t+pd.Timedelta(hours=1),
        initial_lon_wgs84=108.9, initial_lat_wgs84=34.2) for i in range(3)]
    a = pd.DataFrame([
        dict(vehicle_id='0',assignment_time=t,service_end_time=t+pd.Timedelta(minutes=10),completion_lon_wgs84=109.,completion_lat_wgs84=34.3),
        dict(vehicle_id='1',assignment_time=t-pd.Timedelta(minutes=5),service_end_time=t+pd.Timedelta(minutes=10),completion_lon_wgs84=109.,completion_lat_wgs84=34.3),
        dict(vehicle_id='2',assignment_time=t-pd.Timedelta(minutes=10),service_end_time=t,completion_lon_wgs84=109.,completion_lat_wgs84=34.3)])
    assert {v.vehicle_id for v in _vehicle_state(fixtures,a,t)} == {'2'}
    state = {v.vehicle_id:v for v in pre_decision_vehicle_state(fixtures,a,t)}
    assert set(state) == {'0','2'}
    assert state['0'].lon_wgs84 == 108.9
    assert state['2'].lon_wgs84 == 109.
