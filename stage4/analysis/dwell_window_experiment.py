"""Frozen six-condition deterministic-routing continuation, not canonical reproduction."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from joblib.externals import cloudpickle
import pandas as pd
import psutil

from stage4.analysis import mechanism_validity as m
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings

CONFIG = Path('stage4/config/dwell_deterministic_window.json')
OUT = Path('stage4/output/paper_enhancement/dwell_deterministic_window')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(value,path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf8')
    temporary.replace(path)


def run_condition(root, cfg, q, overhead, directory):
    started = time.perf_counter()
    checkpoint = root/cfg['checkpoint_source']/f'q{q}'/'native_checkpoint.pkl'
    checkpoint_sha = digest(checkpoint)
    with checkpoint.open('rb') as stream:
        sim = cloudpickle.load(stream)
    c = sim.operators[0]
    c.config = {**c.config,'additional_pickup_overhead_s':overhead}
    assert c.max_pickup_wait_s == cfg['pickup_patience_s']
    assert c.dispatch_interval_s == cfg['dispatch_interval_s']
    c.eta_adapter = m.ArcDeterministicValhallaAdapter(root,routing_mode=cfg['routing_mode'])
    c.run_started_perf = time.perf_counter()
    c.runtime_guard_s = cfg['scenario_timeout_s']
    c.matching_end_s = cfg['last_dispatch_s']
    sim.demand.future_requests = {t:v for t,v in sim.demand.future_requests.items() if t<cfg['measurement_end_s']}
    cohort = {rid:r for rid,r in c.request_by_rid.items()
              if cfg['measurement_start_s'] <= r.sim_time_s < cfg['measurement_end_s']}
    step = cfg['dispatch_interval_s']
    drain = math.ceil((cfg['last_dispatch_s']+cfg['pickup_patience_s']+overhead+
        max(r.realized_service_time_s for r in c.request_by_rid.values()))/step)*step+step
    for tick in range(cfg['checkpoint_s'],drain+step,step):
        if time.perf_counter()-started > cfg['scenario_timeout_s']:
            raise TimeoutError(f"Per-condition {cfg['scenario_timeout_s']}-second budget exhausted")
        if tick == cfg['checkpoint_s']:
            c.time_trigger(tick)
        else:
            sim.step(tick)
        c.eta_adapter.cache.clear()
        if tick % 900 == 0:
            rss = psutil.Process().memory_info().rss/2**20
            print(json.dumps(dict(q=q,overhead=overhead,tick=tick,rss_mib=rss,
                resource_warning=rss>cfg['rss_warning_mib'])),flush=True)
        if tick>cfg['last_dispatch_s'] and not(set(c.rid_to_assigned_vid)-c.completed_rids):
            break
    c.reconcile()
    assert not(set(c.rid_to_assigned_vid)-c.completed_rids), 'Incomplete task at drain end'
    assert not any((c.position_reconciliation_failures,c.request_state_reconciliation_failures,
                   c.vehicle_state_reconciliation_failures,c.av_availability_violations))
    assignments = pd.DataFrame(c.assignment_rows)
    new = assignments[assignments.simulation_time_s>=cfg['checkpoint_s']].copy()
    new['additional_pickup_overhead_s'] = overhead
    new['vehicle_arrival_time'] = new.pickup_time
    new['service_start_time'] = pd.to_datetime(new.pickup_time)+pd.to_timedelta(overhead,unit='s')
    pickup_error = (pd.to_datetime(new.pickup_time)-pd.to_datetime(new.assignment_time)).dt.total_seconds()-new.pickup_eta_s
    service_error = (pd.to_datetime(new.service_end_time)-pd.to_datetime(new.pickup_time)).dt.total_seconds()-new.realized_service_time_s-overhead
    assert pickup_error.abs().max()<1e-6
    assert service_error.abs().max()<1e-6
    hv = new[new.vehicle_type=='HV']
    planned_end = pd.to_datetime(hv.assignment_time)+pd.to_timedelta(hv.pickup_eta_s+hv.predicted_service_time_s+overhead,unit='s')
    assert ((planned_end-pd.to_datetime(hv.hv_session_end_time)).dt.total_seconds()<=1e-6).all()
    by_rid = new.set_index('native_request_id')
    outcomes = []
    for rid,r in cohort.items():
        matched = rid in by_rid.index
        assert matched or rid in c.expired_rids
        record = dict(native_request_id=rid,order_id=r.order_id,release_time=r.request_time,matched=matched,
                      expired=not matched,vehicle_type=None,request_to_arrival_s=None,request_to_service_start_s=None)
        if matched:
            a = by_rid.loc[rid]
            wait = (pd.Timestamp(a.pickup_time)-r.request_time).total_seconds()
            assert wait<=cfg['pickup_patience_s']+1e-6
            record.update(vehicle_type=a.vehicle_type,request_to_arrival_s=wait,request_to_service_start_s=wait+overhead)
        outcomes.append(record)
    outcome = pd.DataFrame(outcomes)
    # Exact online/idle hours in the observation window, including inherited tasks.
    t0 = c.start+pd.Timedelta(seconds=cfg['measurement_start_s'])
    t1 = c.start+pd.Timedelta(seconds=cfg['measurement_end_s'])
    hours = {k:dict(online_s=0.,busy_s=0.) for k in ('HV','AV')}
    grouped = {int(vid):g.sort_values('assignment_time') for vid,g in assignments.groupby('native_vehicle_id')}
    for vid,runtime in c.runtime_by_vid.items():
        fixture = runtime.fixture
        lo,hi = max(t0,fixture.availability_start_time),min(t1,fixture.availability_end_time)
        if hi<=lo:
            continue
        values = hours[fixture.vehicle_type]
        values['online_s'] += (hi-lo).total_seconds()
        previous_end = None
        for a in grouped.get(vid,pd.DataFrame()).itertuples():
            begin,end = pd.Timestamp(a.assignment_time),pd.Timestamp(a.service_end_time)
            assert previous_end is None or begin>=previous_end-pd.Timedelta(microseconds=1)
            previous_end = end
            values['busy_s'] += max(0.,(min(hi,end)-max(lo,begin)).total_seconds())
    for values in hours.values():
        assert values['busy_s']<=values['online_s']+1e-6
        values['available_vehicle_hours'] = (values['online_s']-values['busy_s'])/3600
    if q==50 and overhead==0:
        previous = pd.read_parquet(root/'stage4/output/paper_enhancement/native_zero_continuation/q50_assignments.parquet')
        keys = ['simulation_time_s','native_request_id','native_vehicle_id','pickup_eta_s']
        a = new[new.simulation_time_s<=41460][keys].sort_values(keys[:3]).reset_index(drop=True)
        b = previous[keys].sort_values(keys[:3]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    new.to_parquet(directory/'assignments.parquet',index=False)
    outcome.to_parquet(directory/'cohort_outcomes.parquet',index=False)
    pd.DataFrame(c.epoch_rows).to_parquet(directory/'epochs.parquet',index=False)
    pd.DataFrame(c.exposure_rows).to_parquet(directory/'exposure.parquet',index=False)
    assert digest(checkpoint)==checkpoint_sha
    mem = psutil.Process().memory_info()
    result = dict(q_A=q/100,additional_pickup_overhead_s=overhead,status='COMPLETE',
        cohort_orders=len(cohort),matched=int(outcome.matched.sum()),expired=int(outcome.expired.sum()),
        matched_AV=int((outcome.vehicle_type=='AV').sum()),matched_HV=int((outcome.vehicle_type=='HV').sum()),
        mean_request_to_arrival_s=float(outcome.request_to_arrival_s.mean()),
        mean_request_to_service_start_s=float(outcome.request_to_service_start_s.mean()),
        assignments_after_checkpoint=len(new),hours=hours,pickup_time_max_error_s=float(pickup_error.abs().max()),
        service_time_max_error_s=float(service_error.abs().max()),
        hv_realized_overrun_count=int((pd.to_datetime(hv.service_end_time)>pd.to_datetime(hv.hv_session_end_time)).sum()),
        routing_arcs=c.eta_adapter.routing_arc_evaluations,routing_failures=c.eta_adapter.routing_failures,
        runtime_s=time.perf_counter()-started,peak_rss_mib=getattr(mem,'peak_wset',mem.rss)/2**20,
        drain_end_s=tick,checkpoint_sha256=checkpoint_sha)
    atomic_json(result,directory/'summary.json')
    return result


def run(root,fleetpy_root):
    root = Path(root).resolve()
    cfg = json.loads((root/CONFIG).read_text())
    load_fleetpy_bindings(fleetpy_root)
    assert json.loads((root/cfg['checkpoint_source']/'summary.json').read_text())['status']=='NATIVE_CHECKPOINT_VERIFIED'
    output = root/OUT
    output.mkdir(parents=True,exist_ok=False)
    status = dict(status='RUNNING',config_sha256=digest(root/CONFIG),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline='NEW_DETERMINISTIC_ROUTING_CONDITIONAL_CONTINUATION',rows=[])
    atomic_json(status,output/'summary.json')
    for overhead in cfg['additional_pickup_overhead_s']:
        for q in (50,75):
            name = f'q{q}_dwell{overhead}'
            directory = output/name
            directory.mkdir()
            status['active'] = name
            atomic_json(status,output/'summary.json')
            try:
                result = run_condition(root,cfg,q,overhead,directory)
            except Exception as error:
                status.update(status='STOPPED',error=repr(error))
                atomic_json(status,output/'summary.json')
                raise
            status['rows'].append(result)
            atomic_json(status,output/'summary.json')
            print(json.dumps(result),flush=True)
            gc.collect()
    assert digest(root/CONFIG)==status['config_sha256']
    status.update(status='COMPLETE',active=None)
    atomic_json(status,output/'summary.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--fleetpy-root',type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.fleetpy_root)
