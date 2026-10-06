import numpy as np
from collections import Counter
from types import SimpleNamespace as NS
from stage4.dispatch.controlled_movement import ControlledGeometryBridge, CommonIdleMovementManager, ShapeProgress


def test_bent_shape_progress_is_not_od_chord():
    shape=ShapeProgress([(108.9,34.2),(108.91,34.2),(108.91,34.21)])
    p=shape.position(.5)
    assert np.isfinite(p).all()
    assert p[0]>108.908 or p[1]<34.201
    assert shape.position(0)==(108.9,34.2)
    assert shape.position(1)==(108.91,34.21)


def test_shape_bridge_exact_completion_and_partial_distance():
    coordinates={1:(108.9,34.2),2:(108.91,34.21)}
    network=NS(return_position_coordinates=lambda p:coordinates[p[0]],
        move_along_route=lambda *a,**kw:None,_metrics=lambda *a:(100.,1000.))
    bridge=ControlledGeometryBridge(network)
    bridge.register(7,(1,None,None),(2,None,None),[(108.9,34.2),(108.91,34.2),(108.91,34.21)])
    p,d,t,_,_=bridge.move([1,2],(1,None,None),30,(0,7),10)
    assert (d,t)==(300.,-1)
    lon,lat=network.return_position_coordinates(p)
    assert 108.9<lon<=108.91 and lat<34.201
    p,d,t,_,_=bridge.move([1,2],p,70,(0,7),40)
    assert p==(2,None,None) and d==700. and t==110.
    bridge.release(7)
    assert not bridge.shapes


def test_interrupt_records_only_actual_prefix_and_current_position():
    manager=CommonIdleMovementManager.__new__(CommonIdleMovementManager)
    manager.active={7:0}; manager.counts=Counter()
    manager.rows=[dict(start_time_s=10,planned_duration_s=100.,planned_distance_m=1000.,traffic_unknown_share=1.)]
    manager.routing_wall_time_s=manager.total_decision_time_s=0.
    observed=[]
    native=NS(pos=(1,2,.3,99),cl_driven_distance=300.,assign_vehicle_plan=lambda p,t:observed.append((p,t)))
    runtime=NS(fixture=NS(native_id=7),native_vehicle=native,active_order_id=None)
    manager.control=NS(fixture_windows_s={7:(0,100)},runtime_by_vid={7:runtime})
    manager.network=NS(return_position_coordinates=lambda p:(108.903,34.2),
        registry=NS(position_for=lambda lon,lat:(8,None,None)))
    manager.geometry=NS(release=lambda vid:observed.append(vid))
    manager.interrupt_for_assignment(runtime,40)
    row=manager.rows[0]
    assert observed==[([],40),7] and native.pos==(8,None,None)
    assert (row["actual_duration_s"],row["actual_distance_m"] )==(30.,300.)
    assert row["status"]=="INTERRUPTED_BY_CUSTOMER" and row["after_shift_time_s"]==0.
    assert (row["end_lon_wgs84"],row["end_lat_wgs84"])==(108.903,34.2)
    manager.finalize(40)
    assert manager.summary()["traffic_unknown_share"]==1.
