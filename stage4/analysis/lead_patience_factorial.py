"""Predefined 4-state x 4-lead x 3-patience conditional sparse analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import psutil

from stage4.analysis import frozen_state_prediction_ablation as f
from stage4.analysis import mechanism_validity as m
from stage4.analysis.rt_patience_sensitivity import request_variants
from stage4.analysis.recover_rt_parameters import sha256

OUT = Path('stage4/output/paper_enhancement/lead_patience_factorial')


def run(root, attempt=None, *, strict_pre_epoch=False):
    started = time.perf_counter()
    out = root / f.state_output(OUT, strict_pre_epoch)
    if attempt is not None:
        if not attempt.isalnum():
            raise ValueError('attempt must be alphanumeric')
        out = out / attempt
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'cells.csv').exists() or (out / 'summary.json').exists():
        raise RuntimeError('Output exists; preserve it and use an explicitly authorized new attempt')
    recovered_path = root / 'stage4/docs/paper_redesign/recovered_rt_environment_parameters.json'
    recovered = json.loads(recovered_path.read_text())
    assert recovered['status'] == 'RECOVERED_FINGERPRINT_VERIFIED'
    generator = 'stage4/scripts/build_decoupled_abm_environment.py'
    assert sha256(root / generator) == recovered['code_sources'][generator]
    old_path = root / f.state_output(m.OUT, strict_pre_epoch) / 'request_time_sensitivity.csv'
    old = pd.read_csv(old_path)
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests = f.load_all_test31_requests(root, start=start,
        end=start + pd.Timedelta(days=1, seconds=60), profile_id='M')
    assert len(requests) == 30000
    variants, _ = request_variants(requests, recovered['request_time_chain_stats'], recovered['parameters'], start)
    adapter = m.ArcDeterministicValhallaAdapter(root, routing_mode=m.SINGLE_SOURCE_MATRIX)
    rows, sources = [], {str(old_path.relative_to(root)): sha256(old_path),
                         str(recovered_path.relative_to(root)): sha256(recovered_path)}
    baseline_passes = 0
    # All sixteen 300s cells must pass before testing other patience levels.
    for phase in ((300,), (180, 600)):
        for q, scenario in ((.5, f.SCENARIO_ID), (.75, 'MAIN_Q75_M_P70')):
            directory = root / 'stage4/output/final_experiments' / scenario
            config_path, assignment_path = directory/'scenario_config.json', directory/'assignment_log.parquet'
            for path in (config_path, assignment_path):
                sources[str(path.relative_to(root))] = sha256(path)
            config = json.loads(config_path.read_text())['runtime_configuration']
            assert config['max_pickup_wait_s'] == 300 and config['candidate_top_k'] == 20
            assignments = pd.read_parquet(assignment_path)
            for col in ('assignment_time', 'service_end_time'):
                assignments[col] = pd.to_datetime(assignments[col], utc=True).dt.tz_convert('Asia/Shanghai')
            fleet = f.build_fleet_scenario(root, benchmark_start=start,
                simulation_end=start+pd.Timedelta(days=1, seconds=max(r.realized_service_time_s for r in requests)+60),
                requested_q_a=q, seed=config['fleet_sampling_seed'],
                max_hv_hour_error_pct=config['max_hv_vehicle_hour_error_pct'])
            fixtures = {v.native_id: v for v in fleet.native_fixtures}
            for clock in ('12:00', '17:30'):
                ts = pd.Timestamp(f'2016-10-31T{clock}:00+08:00')
                sim_s = int((ts-start).total_seconds())
                vehicles = f.restore_state(fleet.native_fixtures, assignments, ts, strict_pre_epoch)
                physical_hash = f._sha([v.__dict__ for v in vehicles])
                for name, timed in variants.items():
                    common = {r.native_id for r,*_ in f._waiting_requests(timed, assignments, sim_s, 180)}
                    for patience in phase:
                        if time.perf_counter()-started > 900:
                            raise TimeoutError('900s budget; partial output is not a completed experiment')
                        waiting = f._waiting_requests(timed, assignments, sim_s, patience)
                        graphs = m.gate_graphs(vehicles, waiting, ts, start, config, adapter, 20, patience)
                        metrics = {g: m.graph_metrics(p) for g,p in graphs.items()}
                        reference = old[(old.q_A == q) & (old.timestamp == ts.isoformat()) & (old.rt_variant == name)].iloc[0]
                        assert reference.physical_state_sha256 == physical_hash
                        if patience == 300:
                            assert len(waiting) == reference.waiting_orders
                            for g, values in metrics.items():
                                for key, value in values.items():
                                    assert value == reference[f'{g}_{key}'], (q, clock, name, g, key, value)
                            baseline_passes += 1
                        arcs = m.production_neutral_arcs(vehicles, fixtures, waiting, ts, start,
                            {**config, 'max_pickup_wait_s': patience}, adapter, neutral=False, integer_deadlines=True)
                        mixed = [(a.request_id, a.vehicle_id) for a in arcs]
                        av_actual = {(a.request_id,a.vehicle_id) for a in arcs if a.vehicle_type == 'AV'}
                        # Legacy AV analysis retains microseconds; production floors deadlines.
                        # Integer timing can only remove otherwise identical feasible AV arcs.
                        assert av_actual <= set(graphs['G5_SOLVER']), 'Non-timing AV mirror difference'
                        assert physical_hash == f._sha([v.__dict__ for v in vehicles])
                        row = dict(q_A=q, timestamp=ts.isoformat(), rt_variant=name, patience_s=patience,
                            waiting_orders=len(waiting), retained_180s_cohort=len(common),
                            additional_waiting_orders=len(waiting)-len(common),
                            critical_orders=sum(bool(x[3]) for x in waiting), physical_state_sha256=physical_hash,
                            production_subsecond_expired_orders=sum(int(r.sim_time_s+patience) <= sim_s for r,*_ in waiting),
                            av_arcs_lost_to_integer_deadline=len(set(graphs['G5_SOLVER'])-av_actual),
                            **{f'{g}_{k}':v for g,values in metrics.items() for k,v in values.items()})
                        for label,pairs in [('mixed',mixed), ('production_av',list(av_actual)), ('retained_mixed', [p for p in mixed if p[0] in common]),
                            ('retained_av',[p for p in av_actual if p[0] in common]),
                            ('additional_mixed',[p for p in mixed if p[0] not in common])]:
                            row.update({f'{label}_{k}':v for k,v in m.graph_metrics(pairs).items()})
                        # Disjoint cohort capacities are not additive under shared vehicles.
                        rows.append(row)
                        f._atomic_csv(pd.DataFrame(rows), out/'cells.csv')
                        print(json.dumps({k:row[k] for k in ['q_A','timestamp','rt_variant','patience_s','waiting_orders','G5_SOLVER_M','mixed_M']}), flush=True)
                adapter.cache.clear()
        if phase == (300,):
            assert baseline_passes == 16
            print('BASELINE_300_PASS 16/16', flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 48
    for _, group in frame.groupby(['q_A','timestamp','rt_variant']):
        ordered = group.sort_values('patience_s')
        for col in ['retained_av_M','retained_mixed_M','mixed_M','G5_SOLVER_M']:
            assert ordered[col].is_monotonic_increasing, col
    assert all(sha256(root/p) == h for p,h in sources.items())
    mem = psutil.Process().memory_info()
    summary = dict(status='COMPLETE', cells=48, baseline_300_passes=baseline_passes,
        runtime_s=time.perf_counter()-started, peak_rss_mib=getattr(mem,'peak_wset',mem.rss)/2**20,
        routing_arcs=adapter.routing_arc_evaluations, routing_failures=adapter.routing_failures,
        sources=sources, no_full_day_simulation=True, no_prediction_reinference=True,
        capacity_scope='solver-input graph maximum matching, before cumulative Gamma and lexicographic selection',
        clock_scope='legacy AV gate metrics retain microseconds; production_av and mixed metrics use production integer deadlines; discrepancies explicitly counted',
        conditioning='canonical prior assignments and positions; release-specific waiting cohorts; frozen descriptors',
        aggregate=frame.groupby(['rt_variant','patience_s'])[['waiting_orders','G5_SOLVER_M','mixed_M','retained_av_M','retained_mixed_M']].sum().reset_index().to_dict('records'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--attempt')
    args = parser.parse_args()
    run(args.root, args.attempt)
