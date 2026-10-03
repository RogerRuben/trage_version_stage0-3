"""Two-epoch, zero-overhead reconstruction check; no vehicle progression."""
import argparse
import json
from pathlib import Path
import time

import pandas as pd
import psutil

from stage4.analysis import frozen_state_prediction_ablation as f
from stage4.analysis import mechanism_validity as m
from stage4.dispatch.exposure import parse_gammas
from stage4.dispatch.solver import solve_lexicographic


def main(strict_pre_epoch=False):
    root = Path.cwd()
    output = root/'stage4/output/paper_enhancement/dwell_checkpoint_preflight'
    if strict_pre_epoch:
        output = output/'strict_pre_epoch'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise RuntimeError('Refuse to overwrite completed preflight')
    started = time.perf_counter()
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    ts = start+pd.Timedelta(hours=10,minutes=30)
    sim_s = 37800
    requests = f.load_all_test31_requests(root,start=start,end=start+pd.Timedelta(days=1,seconds=60),profile_id='M')
    rows = []
    for q,scenario in ((.5,f.SCENARIO_ID),(.75,'MAIN_Q75_M_P70')):
        d = root/'stage4/output/final_experiments'/scenario
        cfg = json.loads((d/'scenario_config.json').read_text())['runtime_configuration']
        assignment = pd.read_parquet(d/'assignment_log.parquet')
        for col in ['assignment_time','pickup_time','service_end_time']:
            assignment[col] = pd.to_datetime(assignment[col],utc=True).dt.tz_convert('Asia/Shanghai')
        fleet = f.build_fleet_scenario(root,benchmark_start=start,
            simulation_end=start+pd.Timedelta(days=1,seconds=max(r.realized_service_time_s for r in requests)+60),
            requested_q_a=q,seed=cfg['fleet_sampling_seed'],max_hv_hour_error_pct=cfg['max_hv_vehicle_hour_error_pct'])
        fixtures = {v.native_id:v for v in fleet.native_fixtures}
        restore = f.pre_decision_vehicle_state if strict_pre_epoch else f._vehicle_state
        vehicles = restore(fleet.native_fixtures,assignment,ts)
        waiting = f._waiting_requests(requests,assignment,sim_s)
        exposure = f._exposure_before(pd.read_parquet(d/'exposure_state.parquet'),sim_s)
        adapter = m.ArcDeterministicValhallaAdapter(root,routing_mode=m.SINGLE_SOURCE_MATRIX)
        arcs = m.production_neutral_arcs(vehicles,fixtures,waiting,ts,start,cfg,adapter,
            neutral=False,integer_deadlines=True)
        solved = solve_lexicographic(arcs,exposure_state=exposure,gammas=parse_gammas(cfg),
            cost_level_enabled=False,pickup_cost_epsilon=0.,numerical_tolerance=cfg['solver_numerical_tolerance'])
        chosen = {(arcs[i].request_id,arcs[i].vehicle_id) for i in solved.selected_indices}
        expected = assignment.loc[assignment.simulation_time_s.eq(sim_s)]
        reference = set(zip(expected.native_request_id.astype(int),expected.native_vehicle_id.astype(int)))
        epoch = pd.read_parquet(d/'epoch_stats.parquet')
        epoch = epoch.loc[epoch.simulation_time_s.eq(sim_s)].iloc[0]
        busy = assignment.loc[(assignment.assignment_time<ts)&(assignment.service_end_time>ts)]
        row = dict(q_A=q,scenario=scenario,checkpoint_time=ts.isoformat(),
            strict_pre_epoch=strict_pre_epoch,
            total_fixtures=len(fixtures),available_reconstructed=len(vehicles),available_reference=int(epoch.available_vehicles),
            waiting_reconstructed=len(waiting),waiting_reference=int(epoch.waiting_orders),
            in_progress_assignments=len(busy),in_progress_pickups=int((busy.pickup_time>ts).sum()),
            in_progress_services=int((busy.pickup_time<=ts).sum()),
            captured_arcs=len(arcs),reference_arcs=int(epoch.valid_or_arcs),
            selected_count=len(chosen),reference_selected_count=len(reference),selected_identity_equal=chosen==reference,
            selected_symmetric_difference=len(chosen^reference),
            pickup_objective_s=float(solved.pickup_eta_optimum_s),reference_pickup_objective_s=float(expected.pickup_eta_s.sum()),
            routing_arcs=adapter.routing_arc_evaluations,routing_failures=adapter.routing_failures,
            gamma_state=exposure.__dict__,native_checkpoint_restored=False)
        rows.append(row)
        print(json.dumps(row),flush=True)
    result = dict(status='EPOCH_PREFLIGHT_COMPLETE_NOT_NATIVE_CHECKPOINT',rows=rows,
        runtime_s=time.perf_counter()-started,peak_working_set_mib=psutil.Process().memory_info().peak_wset/2**20,
        routing_mode='SINGLE_SOURCE_MATRIX',full_simulation_run=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-pre-epoch', action='store_true')
    main(parser.parse_args().strict_pre_epoch)
