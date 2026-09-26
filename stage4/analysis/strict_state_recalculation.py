"""Single-process correction; never overwrites the inclusive-state analyses."""
import argparse
import gc
import json
from pathlib import Path
import time

from stage4.analysis import frozen_state_prediction_ablation as prediction
from stage4.analysis import mechanism_validity as mechanism
from stage4.analysis import rt_patience_sensitivity as timing
from stage4.analysis import lead_patience_factorial as factorial


def run(root):
    root = Path(root).resolve()
    output = root / 'stage4/output/paper_enhancement/strict_state_recalculation'
    output.mkdir(parents=True, exist_ok=False)
    status = {'state_boundary': 'assignment_time < decision_time',
              'full_day_simulation': False, 'completed': [], 'status': 'RUNNING'}
    path = output / 'status.json'
    started = time.perf_counter()
    for name, module in [('prediction', prediction), ('mechanism', mechanism),
                         ('timing', timing), ('factorial', factorial)]:
        status['active'] = name
        path.write_text(json.dumps(status, indent=2), encoding='utf8')
        print('START ' + name, flush=True)
        try:
            result = module.run(root, strict_pre_epoch=True)
            if name == 'mechanism' and result['neutral_classification'] != 'PASS_IDENTITY':
                raise RuntimeError('Neutral identity failed; downstream stages not started')
        except Exception as error:
            status.update(status='STOPPED', error=repr(error))
            path.write_text(json.dumps(status, indent=2), encoding='utf8')
            raise
        status['completed'].append(name)
        print('COMPLETE ' + name, flush=True)
        gc.collect()
    status.update(status='COMPLETE', active=None, runtime_s=time.perf_counter()-started)
    path.write_text(json.dumps(status, indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    run(parser.parse_args().root)
