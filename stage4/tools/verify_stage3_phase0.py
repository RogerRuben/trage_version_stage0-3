"""Read-only scientific QA plus compact packaging; no routing or optimization."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from stage4.tools.stage3_attribution_phase0 import OUT, route_variant, REVERSE, UTURN
from stage4.tools.control_freedom_diagnostic import write_json, sha, TOL


def main():
    root = Path.cwd()
    out = root/OUT
    desc = SimpleNamespace(unresolved_token_count=1)
    orig = SimpleNamespace(hard_reason_codes=json.dumps([REVERSE]), unknown_reason_codes='[]', rho_static=1., rho_dynamic=1., rho_speed=1.)
    assert route_variant(orig, desc, None, REVERSE)['hard_state'] == 'UNKNOWN'
    desc.unresolved_token_count = 0
    assert route_variant(orig, desc, None, REVERSE)['av_eligible']
    orig.hard_reason_codes = json.dumps([REVERSE, UTURN])
    assert not route_variant(orig, desc, None, REVERSE)['av_eligible']
    union = json.loads((out/'required_pickup_union.json').read_text())
    rows = json.loads((out/'condition_input_counts.json').read_text())
    q0 = json.loads((out/'q0.json').read_text())
    assert q0['status'] == 'PASS' and len(q0['states']) == 10 and len(rows) == 50
    assert len({(r['epoch_id'],r['native_request_id'],r['native_vehicle_id']) for r in union}) == len(union)
    assert all(r['required'] == r['archived']+r['missing'] for r in rows)
    # Exactly the frozen adapter's coordinate rounding and minute timestamp key.
    # All rows within a timestamp use the same frozen calibration bin/mode.
    def key(r):
        return (r['timestamp'], *(round(r[n],7) for n in ('source_lon','source_lat','pickup_lon','pickup_lat')))
    known = {}
    for r in union:
        if r['eta_archived']:
            k = key(r)
            if k in known:
                assert abs(known[k]-r['archived_eta_s']) <= TOL
            known[k] = r['archived_eta_s']
    provenance = json.loads((root/'stage4/output/paper_enhancement/control_freedom_structure/provenance.json').read_text())
    assert sha(Path(provenance['routing_config_path'])) == provenance['routing_config_sha256']
    assert sha(root/'stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet') == provenance['beta_sha256']
    result = {'status': 'PASS', 'route_semantics_fixture_assertions': 3, 'condition_input_rows': 50,
              'required_state_arcs': len(union), 'directly_archived_state_arcs': sum(r['eta_archived'] for r in union),
              'archive_key_reusable_state_arcs': sum(key(r) in known for r in union),
              'unique_required_routing_keys': len({key(r) for r in union}),
              'unique_missing_routing_keys': len({key(r) for r in union if key(r) not in known}),
              'routing_calls': 0, 'scientific_MILPs': 0,
              'claim': 'Exact production-cache-key reuse; no spatial interpolation'}
    write_json(out/'verification_and_cache_closure.json', result)
    packed = {}
    for name in ('required_pickup_union', 'route_counterfactuals', 'fallback_phase0_provenance'):
        source = out/(name+'.json')
        target = out/(name+'.json.gz')
        data = source.read_bytes()
        compressed = gzip.compress(data, mtime=0)
        temp = target.with_suffix('.gz.tmp')
        temp.write_bytes(compressed)
        temp.replace(target)
        assert gzip.decompress(target.read_bytes()) == data
        packed[name] = {'sha256': sha(source), 'gzip_sha256': sha(target), 'bytes': len(compressed)}
    write_json(out/'packed_artifacts.json', packed)
    print(json.dumps(result, indent=2))
    print('PACKED_BYTES',sum(v['bytes'] for v in packed.values()))


if __name__ == '__main__':
    main()
