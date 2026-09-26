"""Read-only timestamp audit; no inference, routing or vehicle progression."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

from stage4.analysis import frozen_state_prediction_ablation as frozen
from stage4.analysis.rt_patience_sensitivity import request_variants

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'stage4/docs/wuhan_robotaxi_review/lifecycle_audit_summary.json'


def main():
    started = time.perf_counter()
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests = frozen.load_all_test31_requests(ROOT, start=start,
        end=start+pd.Timedelta(days=1, seconds=60), profile_id='M')
    recovered = json.loads((ROOT/'stage4/docs/paper_redesign/recovered_rt_environment_parameters.json').read_text())
    variants, _ = request_variants(requests, recovered['request_time_chain_stats'], recovered['parameters'], start)
    releases = {name: {r.order_id:r.request_time.timestamp() for r in rows} for name,rows in variants.items()}
    input_rel = 'stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day=20161031.parquet'
    counts = {name:dict(tokens=0, missing_availability=0, history_at_or_after_release=0, decision_after_release=0) for name in releases}
    affected = {name:set() for name in releases}
    mismatch = 0
    for batch in pq.ParquetFile(ROOT/input_rel).iter_batches(batch_size=65536,
            columns=['order_id','decision_time','availability_timestamp','feature_age_s']):
        d = batch.to_pandas()
        order = d.order_id.astype(str)
        decision = pd.to_numeric(d.decision_time).to_numpy(float)
        available = pd.to_numeric(d.availability_timestamp).to_numpy(float)
        age = pd.to_numeric(d.feature_age_s).to_numpy(float)
        valid = np.isfinite(available) & np.isfinite(age)
        mismatch += int((np.abs(decision[valid]-age[valid]-available[valid]) > 1e-5).sum())
        for name, lookup in releases.items():
            release = order.map(lookup).to_numpy(float)
            assert np.isfinite(release).all()
            late = np.isfinite(available) & (available >= release)
            counts[name]['tokens'] += len(d)
            counts[name]['missing_availability'] += int((~np.isfinite(available)).sum())
            counts[name]['history_at_or_after_release'] += int(late.sum())
            counts[name]['decision_after_release'] += int((decision > release+1e-5).sum())
            affected[name].update(order[late])
    for name in counts:
        counts[name]['orders_with_history_at_or_after_release'] = len(affected[name])
    accounting = []
    for scenario in ['ODD_Q50_M_P70_REFERENCE','MAIN_Q75_M_P70']:
        path = ROOT/'stage4/output/final_experiments'/scenario/'assignment_log.parquet'
        d = pd.read_parquet(path, columns=['assignment_time','pickup_time','service_end_time','pickup_eta_s','realized_service_time_s','completed'])
        for c in ['assignment_time','pickup_time','service_end_time']:
            d[c] = pd.to_datetime(d[c],utc=True)
        pickup_error = (d.pickup_time-d.assignment_time).dt.total_seconds()-d.pickup_eta_s
        service_error = (d.service_end_time-d.pickup_time).dt.total_seconds()-d.realized_service_time_s
        accounting.append(dict(scenario=scenario, rows=len(d), completed=int(d.completed.sum()),
            missing_time_rows=int(d[['assignment_time','pickup_time','service_end_time']].isna().any(axis=1).sum()),
            pickup_error_abs_max_s=float(pickup_error.abs().max()),
            service_error_abs_max_s=float(service_error.abs().max()),
            service_error_abs_above_1s=int((service_error.abs()>1).sum()),
            service_error_quantiles_s=service_error.quantile([0,.5,.9,1]).to_dict()))
    result = dict(status='READ_ONLY_AUDIT_COMPLETE', route_source=input_rel, order_count=len(requests),
        feature_age_reconstruction_mismatches=mismatch, release_comparisons=counts, assignment_accounting=accounting,
        interpretation='timestamp compatibility, not proof of all feature provenance; missing history is unknown',
        runtime_s=time.perf_counter()-started, peak_working_set_mib=psutil.Process().memory_info().peak_wset/2**20)
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
