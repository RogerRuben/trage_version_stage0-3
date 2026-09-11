"""Four conditional frozen states; change RT release, never progress the fleet."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import pandas as pd

from stage4.analysis import frozen_state_prediction_ablation as frozen
from stage4.analysis import mechanism_validity as mvc
from stage4.analysis.recover_rt_parameters import fingerprint, sha256
from stage4.scripts import build_decoupled_abm_environment as original


def request_variants(requests, stats, parameters, start):
    demand = pd.DataFrame([dict(order_id=r.order_id,
        observed_boarding_time=r.request_time.tz_convert('UTC'),
        origin_lon=r.pickup_lon_wgs84, origin_lat=r.pickup_lat_wgs84,
        destination_lon=r.dropoff_lon_wgs84, destination_lat=r.dropoff_lat_wgs84)
        for r in requests])
    tables = original.attach_request_times(demand, stats,
        argparse.Namespace(**{**parameters, 'date': '20161031'}))
    variants = {'CANONICAL_ZERO_LEAD': requests}
    for name, frame in tables.items():
        stamps = frame.set_index('order_id').simulated_request_time
        variants[name] = [replace(r, request_time=stamps[r.order_id].tz_convert('Asia/Shanghai'),
            sim_time_s=float((stamps[r.order_id] - start).total_seconds())) for r in requests]
    return variants, fingerprint(tables)


def run(root: Path):
    started = time.perf_counter()
    recovered_path = root / 'stage4/docs/paper_redesign/recovered_rt_environment_parameters.json'
    recovered = json.loads(recovered_path.read_text())
    if recovered['status'] != 'RECOVERED_FINGERPRINT_VERIFIED':
        raise RuntimeError('RT fingerprint recovery must pass first')
    generator = 'stage4/scripts/build_decoupled_abm_environment.py'
    if sha256(root/generator) != recovered['code_sources'][generator]:
        raise RuntimeError('Original RT transform changed after recovery')
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests = frozen.load_all_test31_requests(root, start=start,
        end=start+pd.Timedelta(days=1, seconds=60), profile_id='M')
    assert len(requests) == 30000
    variants, lead_summary = request_variants(requests, recovered['request_time_chain_stats'],
                                             recovered['parameters'], start)
    output = root / mvc.OUT
    frozen._atomic_csv(lead_summary, output / 'test31_rt_lead_summary.csv')
    adapter = mvc.ArcDeterministicValhallaAdapter(root, routing_mode=mvc.SINGLE_SOURCE_MATRIX)
    rows, sources = [], []
    # Fixed before results: one normal and one evening clock, both already used
    # by the existing ten-state analysis. q75 uses its own frozen assignment log.
    for q, scenario_id in ((.5, frozen.SCENARIO_ID), (.75, 'MAIN_Q75_M_P70')):
        directory = root / 'stage4/output/final_experiments' / scenario_id
        config_path = directory / 'scenario_config.json'
        assignment_path = directory / 'assignment_log.parquet'
        config = json.loads(config_path.read_text())['runtime_configuration']
        assert config['max_pickup_wait_s'] == 300 and config['candidate_top_k'] == 20
        assignments = pd.read_parquet(assignment_path)
        for col in ('assignment_time', 'service_end_time'):
            assignments[col] = pd.to_datetime(assignments[col], utc=True).dt.tz_convert('Asia/Shanghai')
        # Exact restoration of the existing canonical fixture selection; no new
        # fleet definition, resampling protocol, simulation or availability rule.
        fleet = frozen.build_fleet_scenario(root, benchmark_start=start,
            simulation_end=start+pd.Timedelta(days=1, seconds=max(r.realized_service_time_s for r in requests)+60),
            requested_q_a=q, seed=config['fleet_sampling_seed'],
            max_hv_hour_error_pct=config['max_hv_vehicle_hour_error_pct'])
        sources.append(dict(scenario_id=scenario_id,
            config_sha256=sha256(config_path), assignment_log_sha256=sha256(assignment_path)))
        for clock, period in (('12:00', 'NORMAL'), ('17:30', 'EVENING')):
            ts = pd.Timestamp(f'2016-10-31T{clock}:00+08:00')
            sim_s = int((ts-start).total_seconds())
            vehicles = frozen._vehicle_state(fleet.native_fixtures, assignments, ts)
            physical_hash = frozen._sha([v.__dict__ for v in vehicles])
            for name, timed_requests in variants.items():
                waiting = frozen._waiting_requests(timed_requests, assignments, sim_s)
                graphs = mvc.gate_graphs(vehicles, waiting, ts, start, config, adapter, 20)
                metrics = {gate: mvc.graph_metrics(pairs) for gate, pairs in graphs.items()}
                assert physical_hash == frozen._sha([v.__dict__ for v in vehicles])
                before, after = metrics['ROUTE_RETURNED'], metrics['G5_SOLVER']
                row = dict(q_A=q, scenario_id=scenario_id, timestamp=ts.isoformat(), period=period,
                    rt_variant=name, physical_state_sha256=physical_hash,
                    available_vehicles=len(vehicles), available_AV=sum(v.vehicle_type=='AV' for v in vehicles),
                    waiting_orders=len(waiting), imminently_expiring_orders=sum(bool(x[3]) for x in waiting),
                    remaining_patience_mean_s=(sum(r.sim_time_s+300-sim_s for r,*_ in waiting)/len(waiting) if waiting else None),
                    av_option_orders=after['U'], maximum_AV_matching=after['M'],
                    patience_M_retention=(after['M']/before['M'] if before['M'] else None),
                    patience_U_retention=(after['U']/before['U'] if before['U'] else None),
                    patience_E_retention=(after['E']/before['E'] if before['E'] else None),
                    selected_assignment_count=None, expired_orders=None,
                    **{f'{g}_{m}': value for g, result in metrics.items() for m, value in result.items()})
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('q_A','timestamp','rt_variant','waiting_orders','maximum_AV_matching','patience_M_retention')}), flush=True)
                frozen._atomic_csv(pd.DataFrame(rows), output / 'request_time_sensitivity.csv')
            adapter.cache.clear()  # bounded to one physical state's sparse arcs
    summary = dict(status='FIXED_STATE_RT_COMPLETE', physical_states=4, variants=4,
        comparisons=len(rows), root_request_count=len(requests), recovered_parameters_sha256=sha256(recovered_path),
        canonical_sources=sources, full_day_simulation=False, vehicle_progression=False,
        prediction_reinference=False, solver_executed=False,
        conditioning='canonical prior assignments and physical vehicle state retained; no counterfactual history',
        descriptor_caveat='Frozen route descriptors are held constant, not asserted to be available at the earlier RT release.',
        carry_over='waiting membership and failed-round proxy are derived from release using the unchanged 30s/300s rules',
        expired_orders='NOT_ESTIMATED: canonical assignment history is not an RT counterfactual history',
        timestamp_precision='original microsecond request timestamp retained; fixed-state analysis does not quantize to integer seconds',
        routing_arcs=adapter.routing_arc_evaluations, routing_failures=adapter.routing_failures,
        runtime_s=time.perf_counter()-started)
    (output/'request_time_sensitivity_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    print(json.dumps(run(parser.parse_args().root), indent=2), flush=True)
