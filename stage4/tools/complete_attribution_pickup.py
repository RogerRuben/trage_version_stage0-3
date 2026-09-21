"""Bounded pickup-only completion of the frozen Phase0 union; fail closed.

Detailed routing receipts remain local. No service routing, inference, or simulation.
"""
import json
import math
import os
from pathlib import Path
import time

import pandas as pd
import psutil

from stage4.tools.control_freedom_diagnostic import sha, write_json
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX

OUT = Path('stage4/output/paper_enhancement/stage3_action_space_attribution_v1')
PHASE0 = Path('stage4/output/paper_enhancement/stage3_attribution_phase0')


def key(row):
    return json.dumps([row['timestamp'], *(round(row[n], 7) for n in
                     ('source_lon', 'source_lat', 'pickup_lon', 'pickup_lat'))], separators=(',', ':'))


def main():
    start = time.perf_counter()
    root = Path.cwd()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    packed = json.loads((root/PHASE0/'packed_artifacts.json').read_text())
    for name, meta in packed.items():
        assert sha(root/PHASE0/(name+'.json')) == meta['sha256']
    provenance = json.loads((root/'stage4/output/paper_enhancement/control_freedom_structure/provenance.json').read_text())
    assert sha(Path(provenance['routing_config_path'])) == provenance['routing_config_sha256']
    assert sha(root/'stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet') == provenance['beta_sha256']
    config = json.loads((root/'stage3/config/stage3_finalization.json').read_text())
    assert Path(config['valhalla_config']).resolve() == Path(provenance['routing_config_path']).resolve()
    union = json.loads((root/PHASE0/'required_pickup_union.json').read_text())
    representatives, cache = {}, {}
    for row in union:
        k = key(row)
        representatives.setdefault(k, row)
        if row['eta_archived']:
            eta = row['archived_eta_s']
            assert math.isfinite(eta) and eta >= 0
            if k in cache:
                assert abs(cache[k]['eta_s']-eta) <= 1e-7
            cache[k] = {'status': 'VALID_ETA', 'eta_s': eta, 'source': 'FROZEN_ARCHIVE'}
    assert len(representatives) == 16386 and len(cache) == 125
    journal = out/'pickup_receipts.jsonl'
    attempts = set()
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            k = item.pop('key')
            assert k in representatives and k not in attempts and k not in cache
            attempts.add(k)
            cache[k] = item
    actor = ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX)
    peak = 0
    stop = None
    calls = 0
    try:
        with journal.open('a', encoding='utf-8') as stream:
            for k, row in representatives.items():
                if k in cache:
                    continue
                rss = psutil.Process().memory_info().rss
                peak = max(peak, rss)
                if rss > 2*1024**3 or time.perf_counter()-start > 600:
                    stop = 'RESOURCE_LIMIT'
                    break
                assert len(attempts) < 16261
                local = pd.Timestamp(row['timestamp']).tz_convert('Asia/Shanghai')
                bin_index, beta = actor.beta_for(local)
                request = {'sources': [{'lon': row['source_lon'], 'lat': row['source_lat']}],
                           'targets': [{'lon': row['pickup_lon'], 'lat': row['pickup_lat']}],
                           'costing': 'auto', 'units': 'kilometers',
                           'date_time': {'type': 1, 'value': local.strftime('%Y-%m-%dT%H:%M')}}
                record = {'status': 'UNRESOLVED', 'source': SINGLE_SOURCE_MATRIX,
                          'beta': beta, 'bin_index': bin_index}
                try:
                    response = actor.actor.matrix(request)
                    record['response'] = response
                    cell = response['sources_to_targets'][0][0]
                    raw, distance = float(cell['time']), float(cell['distance'])
                    if not (math.isfinite(raw) and raw >= 0 and math.isfinite(distance) and distance >= 0):
                        raise ValueError('Invalid matrix time/distance')
                    eta = raw * beta
                    assert math.isfinite(eta) and eta >= 0
                    record.update(status='VALID_ETA', eta_s=eta)
                except Exception as exc:
                    # Generic errors/null cells are NOT certified unreachable routes.
                    record['error'] = f'{type(exc).__name__}: {exc}'
                attempts.add(k)
                calls += 1
                cache[k] = record
                stream.write(json.dumps({'key': k, **record}, allow_nan=False)+'\n')
                stream.flush()
                if calls % 250 == 0:
                    os.fsync(stream.fileno())
                    print(f'pickup calls={calls} keys={len(cache)}/16386 rss_mib={rss/1024**2:.1f}', flush=True)
            os.fsync(stream.fileno())
    finally:
        valid = sum(r['status'] == 'VALID_ETA' for r in cache.values())
        certified = sum(r['status'] == 'CERTIFIED_ROUTING_FAILURE' for r in cache.values())
        unresolved = 16386-valid-certified
        result = {'status': 'PASS' if unresolved == 0 else 'STOP', 'required_keys': 16386,
                  'valid_eta': valid, 'certified_routing_failure': certified,
                  'unresolved': unresolved, 'calls_this_run': calls,
                  'new_key_attempts_total': len(attempts), 'archive_keys': 125,
                  'stop_reason': stop, 'runtime_s': time.perf_counter()-start,
                  'internally_sampled_peak_rss_mib': peak/1024**2,
                  'routing_mode': SINGLE_SOURCE_MATRIX, 'scientific_conditions_run': 0,
                  'union_sha256': packed['required_pickup_union']['sha256'],
                  'journal_sha256': sha(journal) if journal.exists() else None}
        write_json(out/'routing_qa.json', result)
        write_json(out/'pickup_cache.json', cache)
        print(json.dumps(result), flush=True)
    if unresolved:
        raise SystemExit('STOP: unresolved routing keys; scientific conditions not authorized to run')


if __name__ == '__main__':
    main()
