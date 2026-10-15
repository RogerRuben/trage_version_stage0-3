"""Two sequential 00:00--00:40 native prefixes, never a full-day experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter,sleep

from . import bottleneck_whole_method as protocol
from .scheme_a_two_scale_check import _atomic_json,_run_native,FLEETPY
from .scheme_a_two_scale_experiment import _BoundedLog


CONFIG=Path("stage4/config/bottleneck_whole_method_v3.json")


def worker(root,group):
    cfg,_,_=protocol.load(root,CONFIG)
    if cfg["version"]!=protocol.V3 or group not in protocol.ARMS[1:]:
        raise ValueError("only the two V3 joint prefixes are authorized here")
    cfg=protocol.group_config(cfg,group)
    cfg["diagnostic_stop_tick_s"]=2400
    directory=root/protocol.output_root(cfg)/"preflight"/group
    cfg["disk_route_cache"]=(directory.relative_to(root)/"routing/cache.sqlite3").as_posix()
    return _run_native(root,FLEETPY,CONFIG,cfg,directory=directory,
        doc_summary_path=directory/"prefix_receipt.json",completion_status="NOT_FULL_DAY",
        group=group,execution_binding=protocol.binding_for(root,CONFIG,cfg))


def run(root):
    cfg,_,_=protocol.load(root,CONFIG)
    directory=root/protocol.output_root(cfg)/"preflight"
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("prior preflight retained; no automatic overwrite/retry")
    binding=protocol.code_sha(root); results={}
    with protocol.execution_lock(root,protocol.output_root(cfg)):
        for group in protocol.ARMS[1:]:
            if protocol.code_sha(root)!=binding: raise ValueError("code changed between prefixes")
            command=[sys.executable,"-B","-u","-m",__name__.replace("__main__", "stage4.analysis.bottleneck_v3_preflight"),
                "--worker",group,"--execute"]
            env=dict(os.environ); n=int(env.get("GIT_CONFIG_COUNT","0"))
            for path in (FLEETPY,root):
                env[f"GIT_CONFIG_KEY_{n}"]="safe.directory"; env[f"GIT_CONFIG_VALUE_{n}"]=str(path); n+=1
            env["GIT_CONFIG_COUNT"]=str(n)
            for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): env[key]="1"
            env["CUDA_VISIBLE_DEVICES"]="-1"
            log=_BoundedLog(directory/"logs"/f"{group}.log",group,cfg)
            child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            started=perf_counter(); reader=threading.Thread(target=log.drain,args=(child.stdout,),daemon=True)
            reader.start(); timeout=False
            _atomic_json(directory/"active.json",dict(group=group,child_pid=child.pid,code_sha=binding))
            try:
                while child.poll() is None:
                    if perf_counter()-started>=600 or log.error is not None:
                        timeout=perf_counter()-started>=600
                        print(json.dumps(dict(action="STOP_PREFIX_AT_ADMIN_LIMIT_OR_LOG_ERROR",group=group)),flush=True)
                        child.terminate(); break
                    sleep(.2)
                code=child.wait(); reader.join(10)
            finally:
                if child.poll() is None: child.terminate(); child.wait()
            _atomic_json(directory/"receipts"/f"{group}.json",dict(exit_code=code,
                wall_time_s=perf_counter()-started,hard_timeout=timeout,log_error=log.error,code_sha=binding))
            if code or timeout or log.error is not None or reader.is_alive():
                raise RuntimeError("prefix failed; remaining work stopped, no automatic retry")
            summary=json.loads((directory/group/"summary.json").read_text(encoding="utf-8"))
            if summary["status"]!="JOINT_PERFORMANCE_PREFIX_COMPLETE":
                raise ValueError("prefix did not complete")
            results[group]={k:summary[k] for k in ("total_runner_wall_s","setup_time_s",
                "simulation_loop_wall_s","sum_complete_step_wall_s","peak_rss_mib","step_count")}
        _atomic_json(directory/"status.json",dict(status="TWO_BOUNDED_PREFIXES_COMPLETE",code_sha=binding,
            groups=results,full_day_experiment_started=False))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--execute",action="store_true")
    p.add_argument("--worker",choices=protocol.ARMS[1:]); args=p.parse_args()
    if not args.execute: p.error("explicit --execute is required")
    result=worker(Path.cwd(),args.worker) if args.worker else run(Path.cwd())
    print(json.dumps({"result":result} if not args.worker else {"status":result["status"]}),flush=True)


if __name__=="__main__": main()
