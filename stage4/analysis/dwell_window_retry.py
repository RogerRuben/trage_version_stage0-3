"""Authorized q75/120-only retry; preserve the original five results and timeout."""
import argparse
import json
from pathlib import Path
import subprocess

from stage4.analysis.dwell_window_experiment import CONFIG, OUT, digest, atomic_json, run_condition
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings


def run(root, fleetpy_root):
    root = Path(root).resolve()
    cfg = json.loads((root/CONFIG).read_text())
    original = root/OUT
    previous = json.loads((original/'summary.json').read_text())
    assert previous['status']=='STOPPED' and previous['active']=='q75_dwell120'
    assert previous['config_sha256']==digest(root/CONFIG)
    expected = {(q,d) for q in (.5,.75) for d in (0,60,120)}-{(.75,120)}
    assert {(r['q_A'],r['additional_pickup_overhead_s']) for r in previous['rows']}==expected
    assert all(r['status']=='COMPLETE' for r in previous['rows'])
    preserved = {str(p.relative_to(root)):digest(p) for p in original.rglob('*') if p.is_file()}
    cfg = {**cfg,'scenario_timeout_s':3600}
    output = original.parent/'dwell_deterministic_window_retry1'
    output.mkdir(exist_ok=False)
    status = dict(status='RUNNING',original_config_sha256=digest(root/CONFIG),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        amended_execution_budget_s=3600,scientific_parameters_changed=False,
        retry_condition=dict(q_A=.75,additional_pickup_overhead_s=120),preserved_files=preserved)
    atomic_json(status,output/'attempt.json')
    load_fleetpy_bindings(fleetpy_root)
    try:
        result = run_condition(root,cfg,75,120,output)
        assert all(digest(root/p)==sha for p,sha in preserved.items())
        assert digest(root/CONFIG)==status['original_config_sha256']
    except Exception as error:
        status.update(status='STOPPED',error=repr(error))
        atomic_json(status,output/'attempt.json')
        raise
    status.update(status='COMPLETE',result=result)
    atomic_json(status,output/'attempt.json')
    combined = dict(status='COMPLETE',baseline=previous['baseline'],
        original_code_sha=previous['code_sha'],retry_code_sha=status['code_sha'],
        config_sha256=previous['config_sha256'],retry_budget_s=3600,
        original_timeout_preserved=True,rows=previous['rows']+[result])
    atomic_json(combined,output/'combined_summary.json')
    print(json.dumps(combined,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--fleetpy-root',type=Path,required=True)
    args = parser.parse_args()
    run(args.root,args.fleetpy_root)
