"""Independent production D0 graph parity and small mathematical fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stage4.analysis import mechanism_validity as mv
from stage4.dispatch.solver import AssignmentArc
from stage4.tools.complete_attribution_pickup import OUT, key
from stage4.tools.control_freedom_diagnostic import write_json
from stage4.tools.run_stage3_attribution import solve_condition


def main():
    root = Path.cwd()
    out = root/OUT
    assert json.loads((out/'experiment_summary.json').read_text())['status'] == 'PASS'
    empty = solve_condition([])
    assert all(a['status'] == 'INFEASIBLE' for a in empty['alternatives'])
    swap = solve_condition([AssignmentArc(1, 1, 10., False, False, vehicle_type='HV'),
                            AssignmentArc(2, 1, 10.1, False, False, vehicle_type='AV')])
    assert swap['alternatives'][0]['same_order_swap_count'] == 1
    assert abs(swap['alternatives'][0]['gap_s']-.1) < 1e-7
    assert not swap['alternatives'][0]['budget_exists']['0.005']
    assert swap['alternatives'][0]['budget_exists']['0.01']
    different_orders = solve_condition([AssignmentArc(1, 1, 10., False, False, vehicle_type='HV'),
                                       AssignmentArc(1, 2, 11., False, False, vehicle_type='HV'),
                                       AssignmentArc(2, 2, 9., False, False, vehicle_type='AV')])
    assert different_orders['counts'][1] == 2
    cache = json.loads((out/'pickup_cache.json').read_text())

    class CachedAdapter:
        routing_time_s = 0.
        def estimate_many(self, candidates, lon, lat, timestamp):
            found = {}
            for v in candidates:
                row = {'timestamp': str(timestamp), 'source_lon': v.lon_wgs84, 'source_lat': v.lat_wgs84,
                       'pickup_lon': lon, 'pickup_lat': lat}
                receipt = cache[key(row)]
                if receipt['status'] == 'VALID_ETA':
                    found[v.native_vehicle_id] = SimpleNamespace(corrected_pickup_eta_s=receipt['eta_s'])
            return found

    records = []
    for registry, vehicles, waiting, fixtures, config, start in mv.load_states(root):
        production = mv.production_neutral_arcs(vehicles, fixtures, waiting, pd.Timestamp(registry.timestamp),
                                              start, config, CachedAdapter(), neutral=False)
        private = json.loads((out/f'private_{registry.epoch_id}_D0.json').read_text())
        expected = {(a['request_id'], a['vehicle_id']): a for a in private['arcs']}
        actual = {(a.request_id, a.vehicle_id): a for a in production}
        assert set(actual) == set(expected)
        for pair, a in actual.items():
            b = expected[pair]
            assert abs(a.pickup_eta_s-b['pickup_eta_s']) <= 1e-7
            assert (a.critical, a.carry_over, a.vehicle_type) == (b['critical'], b['carry_over'], b['vehicle_type'])
        records.append({'epoch_id': registry.epoch_id, 'arcs': len(actual), 'status': 'PASS'})
    write_json(out/'independent_qa.json', {'status': 'PASS', 'production_D0_graph_parity': records,
                                        'mathematical_fixture_count': 3, 'additional_routing_calls': 0})
    print('Independent QA PASS', flush=True)


if __name__ == '__main__':
    main()
