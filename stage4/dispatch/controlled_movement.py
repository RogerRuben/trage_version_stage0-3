"""Short, shared, interruptible empty movements; no future request access."""
from __future__ import annotations

from collections import Counter
from math import cos, radians
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .repositioning_policy import largest_remainder_quotas, time_bin_index


def segment_lengths(points):
    p=np.deg2rad(np.asarray(points,float))
    d=p[1:]-p[:-1]
    h=np.sin(d[:,1]/2)**2+np.cos(p[1:,1])*np.cos(p[:-1,1])*np.sin(d[:,0]/2)**2
    return 12_742_017.6*np.arcsin(np.sqrt(np.clip(h,0,1)))


class ShapeProgress:
    def __init__(self, points):
        cleaned=[tuple(map(float,points[0]))]
        for p in points[1:]:
            p=tuple(map(float,p))
            if p != cleaned[-1]: cleaned.append(p)
        self.points=np.asarray(cleaned,float)
        self.cumulative=np.r_[0.,np.cumsum(segment_lengths(self.points))]

    def position(self, fraction):
        if len(self.points)==1 or self.cumulative[-1] <= 0: return tuple(self.points[-1])
        distance=np.clip(float(fraction),0.,1.)*self.cumulative[-1]
        i=min(len(self.points)-2,max(0,int(np.searchsorted(self.cumulative,distance,side="right")-1)))
        length=self.cumulative[i+1]-self.cumulative[i]
        alpha=0. if length<=0 else (distance-self.cumulative[i])/length
        return tuple(self.points[i]*(1-alpha)+self.points[i+1]*alpha)


class ControlledGeometryBridge:
    """Retain old geometry for customer legs; shape-token ONLY on empty legs."""
    def __init__(self, network):
        self.network=network
        self.shapes={}
        self.by_vehicle={}
        self.next_token=1
        self.original_coords=network.return_position_coordinates
        self.original_move=network.move_along_route
        network.return_position_coordinates=self.coordinates
        network.move_along_route=self.move

    def register(self, vid, origin, destination, points):
        token=self.next_token; self.next_token+=1
        start=self.original_coords(origin); target=self.original_coords(destination)
        # Explicit observation-to-snapped-path connectors keep current state
        # continuous. Road identity/qualification still uses the scalar shape.
        shape=ShapeProgress([start,*points,target])
        self.shapes[token]=shape
        self.by_vehicle[int(vid)]=(int(origin[0]),int(destination[0]),token)
        return token

    def coordinates(self, position):
        if len(position)>3 and position[3] in self.shapes:
            return self.shapes[position[3]].position(position[2])
        return self.original_coords(position)

    def move(self, route, last_position, time_step, sim_vid_id=None, new_sim_time=None, record_node_times=False):
        vid=None if sim_vid_id is None else int(sim_vid_id[1])
        selected=self.by_vehicle.get(vid)
        if selected is None:
            return self.original_move(route,last_position,time_step,sim_vid_id,new_sim_time,record_node_times)
        origin,destination,token=selected
        if int(last_position[0])!=origin or not route or int(route[-1])!=destination:
            return self.original_move(route,last_position,time_step,sim_vid_id,new_sim_time,record_node_times)
        fraction=0. if last_position[1] is None else float(last_position[2])
        duration,distance=self.network._metrics(origin,destination,sim_vid_id)
        current=float(new_sim_time or 0.)
        remaining=duration*(1-fraction)
        if duration<=0 or float(time_step)+1e-9>=remaining:
            arrival=current+max(0.,remaining)
            return (destination,None,None),distance*(1-fraction),arrival,[destination],[arrival] if record_node_times else []
        delta=float(time_step)/duration
        return (origin,destination,min(1.,fraction+delta),token),distance*delta,-1,[],[]

    def release(self, vid):
        selected=self.by_vehicle.pop(int(vid),None)
        if selected is not None: self.shapes.pop(selected[2],None)


class CommonIdleMovementManager:
    def __init__(self, control, reference, empty_router, *, max_moves=50, radius_m=2000., max_eta_s=300., top_k=3, day_end_s=86400):
        self.control=control
        self.network=control.routing_engine
        self.bindings=control.bindings
        self.router=empty_router
        self.geometry=ControlledGeometryBridge(self.network)
        self.max_moves=int(max_moves); self.radius=float(radius_m); self.max_eta=float(max_eta_s); self.top_k=int(top_k)
        self.day_end_s=int(day_end_s)
        self.targets={int(k):g.sort_values("node_id",kind="mergesort").reset_index(drop=True) for k,g in reference.groupby("time_bin_index")}
        self.active={}; self.rows=[]; self.epochs=[]; self.counts=Counter()
        self.routing_wall_time_s=0.
        self.total_decision_time_s=0.
        old_assign=control._assign
        def assign(runtime, request, estimate, simulation_time):
            if self.is_interruptible(runtime,simulation_time): self.interrupt_for_assignment(runtime,simulation_time)
            old_assign(runtime,request,estimate,simulation_time)
        control._assign=assign
        old_status=control.receive_status_update
        def status(vid, simulation_time, list_finished_vrl, force_update=True):
            self._completed(vid,list_finished_vrl)
            old_status(vid,simulation_time,list_finished_vrl,force_update)
        control.receive_status_update=status

    def is_interruptible(self, runtime, sim_time):
        return (int(runtime.fixture.native_id) in self.active and runtime.active_order_id is None
            and self.control.fixture_windows_s[int(runtime.fixture.native_id)][0] <= sim_time
            < self.control.fixture_windows_s[int(runtime.fixture.native_id)][1])

    def _record_end(self, vid, timestamp, distance, status):
        index=self.active.pop(int(vid))
        row=self.rows[index]
        row.update(end_time_s=float(timestamp),actual_duration_s=float(timestamp)-row["start_time_s"],
            actual_distance_m=float(distance),status=status)
        lon,lat=self.network.return_position_coordinates(self.control.runtime_by_vid[int(vid)].native_vehicle.pos)
        row.update(end_lon_wgs84=float(lon),end_lat_wgs84=float(lat))
        end=self.control.fixture_windows_s[int(vid)][1]
        row["after_shift_time_s"]=max(0.,float(timestamp)-max(end,row["start_time_s"]))
        self.geometry.release(vid)
        self.counts[status]+=1

    def _completed(self, vid, finished):
        if int(vid) not in self.active: return
        runtime=self.control.runtime_by_vid[int(vid)]
        row=self.rows[self.active[int(vid)]]
        if not any(getattr(leg,"destination_pos",None)==row["destination_position"] for leg in finished): return
        output=runtime.native_vehicle.op_output
        record=output[-1]
        if int(record["vehicle_id"])!=int(vid): raise RuntimeError("empty completion event identity mismatch")
        self._record_end(vid,record["end_time"],record["driven_distance"],"COMPLETED")
        runtime.state="NATIVE_AVAILABLE"

    def interrupt_for_assignment(self, runtime, sim_time):
        vid=int(runtime.fixture.native_id)
        if not self.is_interruptible(runtime,sim_time): raise RuntimeError("only online uncommitted empty movements may be interrupted")
        native=runtime.native_vehicle
        lon,lat=self.network.return_position_coordinates(native.pos)
        distance=float(native.cl_driven_distance)
        # This records the partial native leg without moving its position.
        native.assign_vehicle_plan([],int(sim_time))
        native.pos=self.network.registry.position_for(lon,lat)
        self._record_end(vid,sim_time,distance,"INTERRUPTED_BY_CUSTOMER")
        runtime.state="NATIVE_AVAILABLE"

    def before_normal_dispatch(self, simulation_time):
        # Physical callbacks, not this polling clock, finalize completed legs.
        return None

    def after_normal_dispatch(self, control, simulation_time):
        if simulation_time%900 or not 0<=simulation_time<self.day_end_s: return
        decision_started=perf_counter()
        target=self.targets[time_bin_index(control._timestamp(simulation_time))]
        xy=target[["lon_wgs84","lat_wgs84"]].to_numpy(float)*np.array([111320*cos(radians(34.25)),110540])
        tree=cKDTree(xy)
        idle=[]; inflight=Counter()
        for vid in control.event_calendar.active_ids(simulation_time):
            r=control.runtime_by_vid[int(vid)]
            if vid in self.active:
                inflight[self.rows[self.active[vid]]["target_node_id"]]+=1
            elif control._available(r,simulation_time):
                lon,lat=self.network.return_position_coordinates(r.native_vehicle.pos)
                node_index=int(tree.query(np.array([lon,lat])*np.array([111320*cos(radians(34.25)),110540]))[1])
                idle.append((r,int(target.iloc[node_index].node_id),float(lon),float(lat)))
        quotas=largest_remainder_quotas(target.set_index("node_id").demand_share,len(idle)+sum(inflight.values()))
        current=Counter(x[1] for x in idle)+inflight
        deficits={int(k):max(0,int(q)-current[int(k)]) for k,q in quotas.items()}
        surplus=[]
        groups={}
        for x in idle: groups.setdefault(x[1],[]).append(x)
        for node,g in groups.items():
            g.sort(key=lambda x:x[0].fixture.vehicle_id)
            retained=max(0,int(quotas.loc[node])-inflight[node])
            surplus.extend(g[retained:])
        surplus.sort(key=lambda x:x[0].fixture.vehicle_id)
        started=attempts=0
        for runtime,node,lon,lat in surplus:
            if started>=self.max_moves: break
            nearby=tree.query_ball_point(np.array([lon,lat])*np.array([111320*cos(radians(34.25)),110540]),self.radius)
            nearby=[i for i in nearby if deficits[int(target.iloc[i].node_id)]>0 and int(target.iloc[i].node_id)!=node]
            nearby.sort(key=lambda i:(float(np.sum((xy[i]-np.array([lon,lat])*np.array([111320*cos(radians(34.25)),110540]))**2)),str(int(target.iloc[i].node_id))))
            for i in nearby[:self.top_k]:
                dest=target.iloc[i]; attempts+=1
                route_started=perf_counter()
                result=self.router.route(lon,lat,float(dest.lon_wgs84),float(dest.lat_wgs84),control._timestamp(simulation_time))
                self.routing_wall_time_s+=perf_counter()-route_started
                kind="HV" if runtime.fixture.vehicle_type=="HV" else control.config["profile_id"]
                if not result["common_supported"]:
                    self.counts["COMMON_ROUTE_UNSUPPORTED"]+=1; continue
                if kind not in result["compatible_profiles"]:
                    self.counts["PROFILE_REJECTED"]+=1; continue
                duration=float(result["duration_s"]); distance=float(result["distance_m"])
                if not np.isfinite(duration) or duration<=0 or distance<=0 or duration>self.max_eta:
                    self.counts["ETA_OR_ZERO_MOVE_REJECTED"]+=1; continue
                if simulation_time+duration>control.fixture_windows_s[int(runtime.fixture.native_id)][1]:
                    self.counts["OPTIONAL_MOVE_CROSSES_SHIFT"]+=1; continue
                origin=runtime.native_vehicle.pos
                destination=self.network.registry.position_for(float(dest.lon_wgs84),float(dest.lat_wgs84))
                self.network.register_vehicle_leg((0,int(runtime.fixture.native_id)),origin,destination,duration,distance)
                self.geometry.register(runtime.fixture.native_id,origin,destination,result["points"])
                row=dict(native_vehicle_id=int(runtime.fixture.native_id),vehicle_id=runtime.fixture.vehicle_id,
                    vehicle_type=runtime.fixture.vehicle_type,start_time_s=int(simulation_time),target_node_id=int(dest.node_id),
                    destination_position=destination,planned_duration_s=duration,planned_distance_m=distance,status="IN_PROGRESS",
                    actual_duration_s=None,actual_distance_m=None,traffic_unknown_share=float(result["traffic_unknown_share"]),
                    start_lon_wgs84=lon,start_lat_wgs84=lat,
                    shape_polyline6=result.get("shape_polyline6"),geometry_point_count=len(result["points"]),
                    snap_gap_origin_m=float(result.get("snap_gap_origin_m",0.)),snap_gap_target_m=float(result.get("snap_gap_target_m",0.)))
                self.active[int(runtime.fixture.native_id)]=len(self.rows); self.rows.append(row)
                leg=self.bindings.vehicle_route_leg(self.bindings.states.REPOSITION,destination,{})
                runtime.native_vehicle.assign_vehicle_plan([leg],int(simulation_time))
                runtime.state="NATIVE_REPOSITIONING"
                deficits[int(dest.node_id)]-=1; started+=1
                self.counts["STARTED"]+=1
                break
        self.epochs.append(dict(simulation_time_s=simulation_time,idle_after_dispatch=len(idle),routing_candidates=attempts,started=started))
        self.total_decision_time_s+=perf_counter()-decision_started

    def finalize(self, simulation_time):
        if self.active: raise RuntimeError("empty physical drain incomplete")
        for row in self.rows:
            if row["actual_duration_s"] < -1e-6 or row["actual_distance_m"] < -1e-6: raise RuntimeError("negative empty movement allocation")
            if row["actual_distance_m"] > row["planned_distance_m"]+1e-5: raise RuntimeError("duplicated empty distance")
            if row["status"]=="COMPLETED" and (abs(row["actual_duration_s"]-row["planned_duration_s"])>1e-5
                or abs(row["actual_distance_m"]-row["planned_distance_m"])>1e-5):
                raise RuntimeError("empty physical conservation failure")

    def summary(self):
        duration=sum(r["actual_duration_s"] or 0. for r in self.rows)
        unknown_time=sum((r["actual_duration_s"] or 0.)*r["traffic_unknown_share"] for r in self.rows)
        return dict(counts=dict(self.counts),movement_rows=len(self.rows),
            empty_time_s=duration,
            empty_distance_m=sum(r["actual_distance_m"] or 0. for r in self.rows),
            empty_after_shift_time_s=sum(r.get("after_shift_time_s",0.) for r in self.rows),
            traffic_unknown_share=unknown_time/duration if duration>0 else None,
            routing_wall_time_s=self.routing_wall_time_s,total_decision_time_s=self.total_decision_time_s,
            dynamic_new_route_prediction_fabricated=False,geometry_endpoint_connectors_reported=True)
