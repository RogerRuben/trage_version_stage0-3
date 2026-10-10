"""Reconstruct native state from recorded actions, not from idle vehicles alone.

Prehistory is forced canonical replay (no routing or optimization). Checkpoints
contain native route legs, passengers, demand, request metadata and Gamma state.
Only load these trusted, locally generated pickle files.
"""
import argparse
import gc
import json
import math
from pathlib import Path
from types import MethodType
import time

from joblib.externals import cloudpickle
import pandas as pd
import psutil

from stage4.analysis import frozen_state_prediction_ablation as f
from stage4.analysis import mechanism_validity as m
from stage4.dispatch.rolling_or_control import create_rolling_or_fleet_control
from stage4.dispatch.candidate_graph import search_radius_m
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings, CoordinateRegistry
from stage4.fleetpy_adapter.test31_demand_adapter import attach_fleetpy_requests
from stage4.fleetpy_adapter.native_demand import create_native_demand
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
from stage4.fleetpy_adapter.native_simulation import create_native_simulation
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate

OUT = Path('stage4/output/paper_enhancement/native_checkpoint_closure')
CUT = 37800


class CheckpointReached(Exception):
    pass


def forced_actions(self, simulation_time):
    """Replay recorded decisions while retaining FleetPy's actual motion lifecycle."""
    self.sim_time = int(simulation_time)
    waiting = [int(r) for r in self.demand.waiting_rq if r not in self.rid_to_assigned_vid]
    for rid in list(waiting):
        if simulation_time >= self.request_meta[rid]['pickup_deadline_s']:
            self._expire(rid, simulation_time)
            waiting.remove(rid)
    if simulation_time == CUT:
        raise CheckpointReached()
    for rid in waiting:
        meta = self.request_meta[rid]
        if meta['first_attempt_time'] is None:
            meta['first_attempt_time'] = self._timestamp(simulation_time)
        meta['attempt_count'] += 1
        meta['entered_critical'] |= 0 < meta['pickup_deadline_s']-simulation_time <= self.dispatch_interval_s
        meta['final_search_radius_m'] = search_radius_m(meta['failed_round_count'],
            self.config['search_radius_initial_m'], self.config['search_radius_step_m'], self.config['search_radius_cap_m'])
    selected = set()
    exposures = []
    for row in self.recorded_actions.get(simulation_time, []):
        rid, vid = int(row['native_request_id']), int(row['native_vehicle_id'])
        assert rid in waiting
        runtime = self.runtime_by_vid[vid]
        assert self._available(runtime, simulation_time)
        estimate = PickupEstimate(row['valhalla_time_s'], row['pickup_eta_s'],
            row['pickup_route_distance_m'], row['beta'], int(row['time_bin_index']), False)
        self._assign(runtime, self.request_by_rid[rid], estimate, simulation_time)
        if runtime.fixture.vehicle_type == 'AV':
            exposures.append(self.request_meta[rid]['exposure'])
        selected.add(rid)
    self.exposure_state.update(exposures)
    for rid in set(waiting)-selected:
        self.request_meta[rid]['carry_over_flag'] = True
        self.request_meta[rid]['failed_round_count'] += 1


def signature(sim):
    control = sim.operators[0]
    return dict(vehicles=[(vid, str(v.status), v.pos, v.soc,
        [(str(l.status), l.destination_pos, l.duration,
          {k: [r.get_rid_struct() for r in rs] for k,rs in l.rq_dict.items()}) for l in v.assigned_route])
        for (_,vid),v in sorted(sim.sim_vehicles.items())],
        active=control.rid_to_assigned_vid, completed=sorted(control.completed_rids),
        waiting=sorted(control.demand.waiting_rq), expired=sorted(control.expired_rids),
        metadata=control.request_meta, exposure=control.exposure_state.__dict__)


def run(root, fleetpy_root, attempt=None):
    root = Path(root).resolve()
    assert json.loads((root/'stage4/output/paper_enhancement/strict_state_recalculation/status.json').read_text())['status'] == 'COMPLETE'
    output = root/OUT
    if attempt is not None:
        if not attempt.isalnum():
            raise ValueError('attempt must be alphanumeric')
        output = output/attempt
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bindings = load_fleetpy_bindings(fleetpy_root)
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    reports = []
    for q, scenario in ((.5, f.SCENARIO_ID), (.75, 'MAIN_Q75_M_P70')):
        directory = output/f'q{int(q*100)}'
        directory.mkdir()
        source = root/'stage4/output/final_experiments'/scenario
        config = json.loads((source/'scenario_config.json').read_text())['runtime_configuration']
        assert not config.get('repositioning_enabled', False)
        actions = pd.read_parquet(source/'assignment_log.parquet')
        requests = f.load_all_test31_requests(root,start=start,end=start+pd.Timedelta(days=1,seconds=60),profile_id='M')
        step = int(config['dispatch_interval_s'])
        end_s = int(config['matching_end_s']) + math.ceil(max(r.realized_service_time_s for r in requests)/step)*step+step
        end = start+pd.Timedelta(seconds=end_s)
        fleet = f.build_fleet_scenario(root,benchmark_start=start,simulation_end=end,requested_q_a=q,
            seed=config['fleet_sampling_seed'],max_hv_hour_error_pct=config['max_hv_vehicle_hour_error_pct'])
        registry = CoordinateRegistry()
        attach_fleetpy_requests(requests,bindings,registry)
        network = create_native_network(bindings,registry)
        demand = create_native_demand(bindings,requests,registry,network,directory)
        vehicles, native_output = create_native_vehicles(fleet.native_fixtures,bindings,registry,demand.rq_db,
            directory/'runtime',native_movement=True,routing_engine=network)
        control = create_rolling_or_fleet_control(bindings,vehicles,requests,demand,network,None,start,end,config)
        control.recorded_actions = {int(t): g.to_dict('records') for t,g in actions[actions.simulation_time_s<CUT].groupby('simulation_time_s')}
        control.time_trigger = MethodType(forced_actions,control)
        sim = create_native_simulation(bindings,simulation_end_s=end_s,time_step_s=step,demand=demand,
            vehicles=[v.native_vehicle for v in vehicles],fleet_control=control,network=network,native_output=native_output)
        try:
            for tick in range(0,CUT+step,step):
                sim.step(tick)
        except CheckpointReached:
            pass
        else:
            raise AssertionError('Did not reach pre-decision checkpoint')
        del control.time_trigger
        del control.recorded_actions
        strict = f.pre_decision_vehicle_state(fleet.native_fixtures,actions,start+pd.Timedelta(seconds=CUT))
        actual = [control._spatial_vehicle(v) for v in vehicles if control._available(v,CUT)]
        assert f._sha([v.__dict__ for v in sorted(actual,key=lambda v:v.vehicle_id)]) == f._sha([v.__dict__ for v in sorted(strict,key=lambda v:v.vehicle_id)])
        expected_busy = actions[(actions.simulation_time_s<CUT)&(pd.to_datetime(actions.service_end_time)>start+pd.Timedelta(seconds=CUT))]
        assert set(control.rid_to_assigned_vid)-control.completed_rids == set(expected_busy.native_request_id.astype(int))
        reference_exposure = f._exposure_before(pd.read_parquet(source/'exposure_state.parquet'),CUT)
        assert all(abs(v-reference_exposure.__dict__[k]) < 1e-7 for k,v in control.exposure_state.__dict__.items())
        checkpoint = directory/'native_checkpoint.pkl'
        with checkpoint.with_suffix('.tmp').open('wb') as stream:
            cloudpickle.dump(sim,stream)
        checkpoint.with_suffix('.tmp').replace(checkpoint)
        with checkpoint.open('rb') as stream:
            restored = cloudpickle.load(stream)
        assert f._sha(signature(sim)) == f._sha(signature(restored))
        # Advance existing tasks only: both native objects must complete identically,
        # at canonical timestamps, without dispatch or new demand activation.
        drain_end = math.ceil((pd.to_datetime(expected_busy.service_end_time).max()-start).total_seconds()/step)*step+step
        for tick in range(CUT+step,drain_end+step,step):
            sim.update_sim_state_fleets(tick-step,tick)
            restored.update_sim_state_fleets(tick-step,tick)
        assert f._sha(signature(sim)) == f._sha(signature(restored))
        assert not (set(control.rid_to_assigned_vid)-control.completed_rids)
        observed = pd.DataFrame(control.assignment_rows).set_index('native_request_id')
        prior = actions[actions.simulation_time_s<CUT].set_index('native_request_id')
        errors = {}
        for col in ('pickup_time','service_end_time'):
            errors[col] = float((pd.to_datetime(observed.loc[prior.index,col])-pd.to_datetime(prior[col])).dt.total_seconds().abs().max())
            assert errors[col] < 1e-6, errors
        reports.append(dict(q_A=q,available=len(actual),busy=len(expected_busy),
            prior_assignments=len(prior),roundtrip_equal=True,busy_drain_equal=True,
            timestamp_max_error_s=errors,checkpoint=str(checkpoint.relative_to(root)),checkpoint_bytes=checkpoint.stat().st_size))
        (output/'progress.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
        print(json.dumps(reports[-1]),flush=True)
        del sim,restored,control,vehicles,requests,demand,network,registry
        gc.collect()
    mem = psutil.Process().memory_info()
    result = dict(status='NATIVE_CHECKPOINT_VERIFIED',rows=reports,runtime_s=time.perf_counter()-started,
        peak_rss_mib=getattr(mem,'peak_wset',mem.rss)/2**20,full_day_rerun=False,
        construction='forced recorded actions through 10:30; native motion; no prehistory routing/solver',
        continuation_boundary='call control.time_trigger(37800) once, then sim.step from 37830')
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--fleetpy-root',type=Path,required=True)
    parser.add_argument('--attempt')
    args = parser.parse_args()
    run(args.root,args.fleetpy_root,args.attempt)
