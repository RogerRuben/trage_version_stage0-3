"""Zero-overhead native continuation gate before any dwell treatment."""
import argparse
import gc
import json
from pathlib import Path
import time

from joblib.externals import cloudpickle
import pandas as pd
import psutil

from stage4.analysis import frozen_state_prediction_ablation as f
from stage4.analysis import mechanism_validity as m
from stage4.analysis.native_checkpoint_closure import CUT, OUT
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings


def run(root, fleetpy_root):
    root = Path(root).resolve()
    load_fleetpy_bindings(fleetpy_root)
    checkpoint_root = root/OUT/'retry1'
    summary = json.loads((checkpoint_root/'summary.json').read_text())
    assert summary['status'] == 'NATIVE_CHECKPOINT_VERIFIED'
    output = root/'stage4/output/paper_enhancement/native_zero_continuation'
    output.mkdir(parents=True,exist_ok=False)
    reports = []
    for q, scenario in ((50,f.SCENARIO_ID),(75,'MAIN_Q75_M_P70')):
        started = time.perf_counter()
        with (checkpoint_root/f'q{q}'/'native_checkpoint.pkl').open('rb') as stream:
            sim = cloudpickle.load(stream)
        control = sim.operators[0]
        control.eta_adapter = m.ArcDeterministicValhallaAdapter(root,routing_mode=m.SINGLE_SOURCE_MATRIX)
        control.run_started_perf = time.perf_counter()
        control.runtime_guard_s = 1800
        control.matching_end_s = 43500
        sim.demand.future_requests = {t:v for t,v in sim.demand.future_requests.items() if t<43200}
        source = root/'stage4/output/final_experiments'/scenario
        actions = pd.read_parquet(source/'assignment_log.parquet')
        expected = actions[(actions.simulation_time_s>=CUT)&(actions.simulation_time_s<43200)]
        expected_by_tick = {int(t):set(zip(g.native_request_id.astype(int),g.native_vehicle_id.astype(int)))
                            for t,g in expected.groupby('simulation_time_s')}
        mismatches = []
        for tick in range(CUT,43200,30):
            if time.perf_counter()-started > 1800:
                raise TimeoutError('Zero-control validation exceeded 30 minutes')
            previous = len(control.assignment_rows)
            if tick == CUT:
                control.time_trigger(tick)
            else:
                sim.step(tick)
            actual = {(int(r['native_request_id']),int(r['native_vehicle_id']))
                      for r in control.assignment_rows[previous:]}
            reference = expected_by_tick.get(tick,set())
            if actual != reference:
                mismatches.append(dict(simulation_time_s=tick,actual=len(actual),reference=len(reference),
                    symmetric_difference=len(actual^reference)))
                # Stop at first divergence; later differences would reflect different history.
                break
            if tick % 900 == 0:
                print(json.dumps(dict(q=q,tick=tick,matched_epoch=len(actual))),flush=True)
            control.eta_adapter.cache.clear()
        new = pd.DataFrame(control.assignment_rows)
        new = new[new.simulation_time_s>=CUT]
        new.to_parquet(output/f'q{q}_assignments.parquet',index=False)
        pd.DataFrame(control.epoch_rows).to_parquet(output/f'q{q}_epochs.parquet',index=False)
        mem = psutil.Process().memory_info()
        row = dict(q_A=q/100,status='PASS' if not mismatches else 'BASELINE_DIVERGENCE',
            compared_through_s=tick,first_mismatch=mismatches[0] if mismatches else None,
            routing_mode=m.SINGLE_SOURCE_MATRIX,routing_failures=control.eta_adapter.routing_failures,
            runtime_s=time.perf_counter()-started,peak_rss_mib=getattr(mem,'peak_wset',mem.rss)/2**20,
            checkpoint_state_verified=True,dwell_treatments_started=False)
        reports.append(row)
        result = dict(status='BLOCKED_BASELINE_DIVERGENCE' if mismatches else 'RUNNING',rows=reports)
        (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)
        if mismatches:
            return result
        del sim,control,new
        gc.collect()
    result = dict(status='ZERO_CONTINUATION_VERIFIED',rows=reports)
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--fleetpy-root',type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.fleetpy_root)
