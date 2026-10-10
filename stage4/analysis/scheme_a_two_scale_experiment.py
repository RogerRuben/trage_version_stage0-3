"""Fixed three-group full-day comparison of the two-scale Scheme-A method.

The lightweight coordinator launches exactly one fresh child at a time so native
state and memory are released between groups. Complete identical groups alone
may be skipped with --resume. Failed/running output is never overwritten or
retried. Physical execution and accounting reuse the short-check runner.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from time import perf_counter, sleep

from .scheme_a_two_scale_check import (
    FLEETPY, _atomic_json, _run_native, _sha, load_frozen_layout,
    load_protocol as load_check_protocol,
)


CONFIG = Path("stage4/config/scheme_a_two_scale_city_v2.json")
OUT = Path("stage4/output/capability_chain_planning/two_scale_city_v2")
DOC = Path("stage4/docs/capability_chain_planning/two_scale_city_v2")
GROUPS = ("HOTSPOT_BASELINE", "CHAIN_LAYOUT_BASELINE", "CHAIN_LAYOUT_JOINT")
GROUP_RULES = dict(zip(GROUPS, (("hotspot", "SERVICE_PRESERVING"),
    ("joint", "SERVICE_PRESERVING"), ("joint", "CHAIN_DEFER"))))
COMPLETE = "TWO_SCALE_CITY_GROUP_COMPLETE"
PRODUCTS = ("assignments.parquet", "cohort_outcomes.parquet", "solver_trace.parquet",
    "epochs.parquet", "empty_movements.parquet", "step_timings.parquet", "coarse_refreshes.json")
_FIXED = dict(version="scheme_a_two_scale_city_v2", window_start_s=0,
    measurement_end_s=86434, admission_end_s=86760, last_dispatch_s=86760,
    physical_drain_limit_s=100800, carry_in_lookback_s=0, include_carry_in=False,
    scenario_timeout_s=10800, progress_interval_s=300, log_limit_mib=16,
    log_tail_kib=64, full_day=True, native_execution=True, automatic_retry=False,
    method_scope="COARSE_APPROXIMATE_SERVICE_VALUE_AND_EXACT_CURRENT_ACTION_FLOW",
    experiment_scope="FIXED_THREE_GROUP_FULL_DAY_LAYOUT_AND_CONTROL_COMPARISON")


def load_protocol(root, path=CONFIG):
    root, path = Path(root), Path(path)
    overrides = json.loads((root / path).read_text(encoding="utf-8"))
    base_path = Path("stage4/config/scheme_a_two_scale_v2.json")
    if (Path(overrides["base_config"]) != base_path
            or _sha(root / base_path) != overrides["base_config_sha256"]):
        raise ValueError("full-day comparison must inherit the unchanged validated two-scale algorithm")
    allowed = set(_FIXED) | {"base_config", "base_config_sha256", "groups", "method_scope", "experiment_scope"}
    if set(overrides) - allowed:
        raise ValueError("full-day configuration cannot change inherited scientific or algorithm parameters")
    if (any(overrides.get(key) != value for key, value in _FIXED.items())
            or tuple(overrides["groups"]) != GROUPS):
        raise ValueError("outside the fixed three-group full-day protocol")
    cfg = dict(load_check_protocol(root, base_path), **overrides)
    # These are derived per group, not old short-check semantics or outputs.
    for key in ("policy", "layout_mode", "disk_route_cache", "check_scope"):
        cfg.pop(key, None)
    return cfg


def group_configuration(cfg, group):
    if group not in GROUP_RULES:
        raise ValueError("only the three original full-day groups are authorized")
    layout, policy = GROUP_RULES[group]
    return dict(cfg, group=group, layout_mode=layout, policy=policy,
        disk_route_cache=(OUT / group / "routing/cache.sqlite3").as_posix())


def input_binding(root, cfg, config=CONFIG):
    root, config = Path(root), Path(config)
    _, protected = load_frozen_layout(root, cfg)
    protected.update({config.as_posix(): _sha(root / config),
        cfg["base_config"]: _sha(root / cfg["base_config"]),
        cfg["source_layout"]: _sha(root / cfg["source_layout"])})
    code = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
        "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return dict(code_sha=code, inputs_sha256=protected,
        config_sha256=_sha(root / config), layout_sha256=cfg["source_layout_sha256"])


def _group_binding(binding, cfg, group):
    chosen = group_configuration(cfg, group)
    return dict(binding, group=group, policy=chosen["policy"],
        layout_mode=chosen["layout_mode"], disk_route_cache=chosen["disk_route_cache"])


def completed_group(directory, expected):
    """Only a fully written same-code/config/input/layout group can be reused."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != COMPLETE or not summary.get("full_day")
            or summary.get("execution_binding") != expected
            or summary.get("code_sha") != expected["code_sha"]
            or summary.get("inputs_sha256") != expected["inputs_sha256"]
            or summary.get("layout_sha256") != expected["layout_sha256"]
            or (summary.get("group"), summary.get("layout_mode"), summary.get("policy"))
                != (expected["group"], expected["layout_mode"], expected["policy"])
            or any(not (directory / product).is_file() for product in PRODUCTS)):
        raise ValueError("existing group is not a complete identical full-day group; no automatic retry")
    return summary


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _process_running(pid, created):
    import psutil
    if pid is None:
        return False
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and (created is None or abs(process.create_time() - float(created)) < .01)
    except psutil.NoSuchProcess:
        return False


class _BoundedLog:
    """Drain child output without retaining unbounded native INFO messages."""
    def __init__(self, path, group, cfg):
        self.path, self.group = Path(path), group
        self.limit = cfg["log_limit_mib"] * 2**20
        self.tail_limit = cfg["log_tail_kib"] * 1024
        self.tail, self.tail_bytes = deque(), 0
        self.stats = dict(bytes_received=0, bytes_written=0, suppressed_info_lines=0,
            bytes_over_log_limit=0, log_limit_mib=cfg["log_limit_mib"])
        self.error = None

    def drain(self, stream):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("xb") as output:
                while True:
                    line = stream.readline(65536)
                    if not line:
                        break
                    self.stats["bytes_received"] += len(line)
                    self.tail.append(line)
                    self.tail_bytes += len(line)
                    while self.tail_bytes > self.tail_limit and len(self.tail) > 1:
                        self.tail_bytes -= len(self.tail.popleft())
                    if re.search(rb"(?:\[[ \t]*INFO[ \t]*\]|\bINFO[ \t]*:)", line):
                        self.stats["suppressed_info_lines"] += 1
                        continue
                    room = max(0, self.limit - self.stats["bytes_written"])
                    output.write(line[:room])
                    output.flush()
                    self.stats["bytes_written"] += min(room, len(line))
                    self.stats["bytes_over_log_limit"] += max(0, len(line) - room)
                    if line.startswith(b"{"):
                        try:
                            progress = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if "tick" in progress:
                            visible = {key: progress[key] for key in ("tick", "runtime_s", "matched",
                                "completed", "expired", "rss_mib", "last_step_wall_s") if key in progress}
                            print(json.dumps(dict(group=self.group, **visible)), flush=True)
                tail_path = self.path.with_name(self.path.stem + "_tail.log")
                tail_path.write_bytes(b"".join(self.tail)[-self.tail_limit:])
        except Exception as error:
            self.error = repr(error)
        finally:
            stream.close()


def _run_child(root, fleetpy, config, cfg, group, binding):
    import psutil
    directory = root / OUT / group
    receipt_path = root / OUT / "execution_receipts" / f"{group}.json"
    command = [sys.executable, "-u", "-m", "stage4.analysis.scheme_a_two_scale_experiment",
        "--root", str(root), "--fleetpy-root", str(fleetpy), "--config", str(config), "--_group", group]
    started = perf_counter()
    receipt = dict(status="LAUNCHING", group=group, child_pid=None, child_create_time=None, parent_pid=os.getpid(),
        started_utc=_utc(), command=command, execution_binding=_group_binding(binding, cfg, group),
        automatic_retry=False)
    _atomic_json(receipt_path, receipt)
    try:
        child = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except Exception as error:
        _atomic_json(receipt_path, dict(receipt, status="LAUNCH_FAILED", error=repr(error), exit_code=None,
            child_wall_time_s=perf_counter() - started, ended_utc=_utc()))
        raise
    try:
        child_created = psutil.Process(child.pid).create_time()
    except psutil.NoSuchProcess:
        child_created = None
    receipt.update(status="RUNNING", child_pid=child.pid, child_create_time=child_created)
    _atomic_json(receipt_path, receipt)
    log = _BoundedLog(root / OUT / "logs" / f"{group}.log", group, cfg)
    reader = threading.Thread(target=log.drain, args=(child.stdout,), daemon=True)
    reader.start()
    hard_timeout = False
    while child.poll() is None:
        if perf_counter() - started >= cfg["scenario_timeout_s"]:
            hard_timeout = True
            print(json.dumps(dict(group=group, action="TERMINATE_OWN_CHILD_AT_FIXED_HARD_LIMIT",
                pid=child.pid, hard_limit_s=cfg["scenario_timeout_s"])), flush=True)
            child.terminate()
            break
        sleep(.2)
    exit_code = child.wait()
    reader.join(timeout=10.)
    receipt.update(status="EXITED", exit_code=exit_code, ended_utc=_utc(),
        child_wall_time_s=perf_counter() - started, parent_hard_timeout=hard_timeout,
        log_statistics=log.stats, log_capture_error=log.error)
    _atomic_json(receipt_path, receipt)
    if hard_timeout:
        last = json.loads((directory / "summary.json").read_text()) if (directory / "summary.json").is_file() else {}
        marker = dict(status="STOPPED_HARD_TIMEOUT", pid=child.pid, hard_limit_s=cfg["scenario_timeout_s"],
            runtime_s=receipt["child_wall_time_s"], final_service_result_available=False,
            current_call_may_not_have_saved_partial_state=True, automatic_retry=False)
        _atomic_json(directory / "hard_timeout_marker.json", marker)
        _atomic_json(directory / "summary.json", dict(last, **marker))
    if exit_code != 0:
        raise RuntimeError(f"{group} child exited {exit_code}; remaining groups are not started")
    if reader.is_alive() or log.error is not None:
        raise RuntimeError(f"{group} log capture did not finish cleanly; remaining groups are not started")
    return completed_group(directory, _group_binding(binding, cfg, group))


def run(root, fleetpy=FLEETPY, config=CONFIG, *, resume=False):
    root, fleetpy, config = Path(root).resolve(), Path(fleetpy), Path(config)
    cfg = load_protocol(root, config)
    binding = input_binding(root, cfg, config)
    directory = root / OUT
    binding_path, status_path = directory / "input_binding.json", directory / "status.json"
    if binding_path.exists():
        if not resume or json.loads(binding_path.read_text()) != binding:
            raise ValueError("existing experiment requires explicit identical-input --resume; no overwrite")
    existing = {}
    if status_path.is_file():
        old = json.loads(status_path.read_text())
        if (old.get("pid") != os.getpid() and old.get("process_create_time") is not None
                and _process_running(old["pid"], old["process_create_time"])):
            raise ValueError("the existing coordinator is still running; no second experiment process")
    for group in GROUPS:
        result_dir = directory / group
        receipt_path = directory / "execution_receipts" / f"{group}.json"
        if receipt_path.is_file():
            launch = json.loads(receipt_path.read_text())
            if _process_running(launch["child_pid"], launch["child_create_time"]):
                raise ValueError(f"{group} child is still running; RUNNING state is not reusable")
        if (result_dir / "summary.json").is_file():
            if not resume:
                raise ValueError("existing group output is retained; use --resume only for complete identical groups")
            existing[group] = completed_group(result_dir, _group_binding(binding, cfg, group))
        elif (receipt_path.exists() or (result_dir.exists() and any(result_dir.iterdir()))
                or (directory / "logs" / f"{group}.log").exists()
                or (directory / "logs" / f"{group}_tail.log").exists()):
            raise ValueError(f"{group} has an incomplete prior launch/output; no automatic retry")
    import psutil
    state = dict(status="RUNNING", pid=os.getpid(), process_create_time=psutil.Process().create_time(),
        started_utc=_utc(), execution_binding=binding, group_order=list(GROUPS),
        completed_groups=list(existing), completed_group_count=len(existing),
        sequential_single_child=True, gpu_used=False, dense_matrix=False, automatic_retry=False)
    if not binding_path.exists():
        _atomic_json(binding_path, binding)
    _atomic_json(status_path, state)
    try:
        for group in GROUPS:
            if group in existing:
                print(json.dumps(dict(group=group, reused_complete_group=True, new_native_run=False)), flush=True)
                continue
            state["active_group"] = group
            _atomic_json(status_path, state)
            existing[group] = _run_child(root, fleetpy, config, cfg, group, binding)
            state.update(completed_groups=[g for g in GROUPS if g in existing], completed_group_count=len(existing))
            _atomic_json(status_path, state)
            print(json.dumps(dict(group=group, status=COMPLETE,
                total_runner_wall_s=existing[group]["total_runner_wall_s"],
                completed_orders=existing[group]["completed_orders"])), flush=True)
        state.update(status="TWO_SCALE_CITY_EXPERIMENT_COMPLETE", active_group=None, ended_utc=_utc())
        _atomic_json(status_path, state)
        return state
    except Exception as error:
        state.update(status="STOPPED", error=repr(error), ended_utc=_utc(),
            remaining_groups_not_started=[g for g in GROUPS if g not in existing and g != state.get("active_group")])
        _atomic_json(status_path, state)
        raise


def _child_group(root, fleetpy, config, group):
    started = perf_counter()
    root = Path(root).resolve()
    cfg = load_protocol(root, config)
    expected = json.loads((root / OUT / "input_binding.json").read_text(encoding="utf-8"))
    if input_binding(root, cfg, config) != expected:
        raise ValueError("child code/configuration/inputs changed after coordinator preparation")
    chosen = group_configuration(cfg, group)
    return _run_native(root, fleetpy, config, chosen,
        directory=root / OUT / group, doc_summary_path=root / DOC / f"{group.lower()}_summary.json",
        completion_status=COMPLETE, group=group, execution_binding=_group_binding(expected, cfg, group),
        started=started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fleetpy-root", type=Path, default=FLEETPY)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_group", choices=GROUPS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._group:
        if args.resume:
            parser.error("internal children never resume or retry an existing group")
        _child_group(args.root, args.fleetpy_root, args.config, args._group)
    else:
        run(args.root, args.fleetpy_root, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
