"""Reconstruct only the original failure prefix and compare its frozen model.

No full-window policy comparison, no new scenario, no parameter search. The
original pipeline/source is loaded from its Git commit into isolated modules;
only output paths, observation hooks and the early-stop limit are redirected.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import ModuleType

import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import atomic_json, sha
from stage4.dispatch import city_defer_native as fixed_numeric

ORIGINAL_SHA = "aeec12ad84d666190182fe0675197367afc2e7d0"
OUT = Path("stage4/output/city_numeric_diagnostic_v1")
DOC = Path("stage4/docs/capability_chain_planning/city_numeric_diagnostic_v1")
MAX_EPOCHS = 10
MAX_RUNTIME_S = 300
FLEETPY = Path("D:/pycodes/didi_xian_raw/.external/FleetPy")


class DiagnosticPrefixLimit(RuntimeError):
    """Expected stop: no original failure within the declared prefix."""


def load_frozen_module(root, path, name, *, redirect=None):
    source = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
        "-C", str(root), "show", f"{ORIGINAL_SHA}:{path}"], text=True, encoding="utf-8")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if redirect is not None:
        before, after = redirect
        if source.count(before) != 1:
            raise ValueError("frozen native numeric import is not uniquely redirectable")
        source = source.replace(before, after)
    module = ModuleType(name)
    module.__file__ = str(root / path)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module  # dataclasses resolve their owning module here.
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.frozen_source_sha256 = source_sha
    return module


def bounded_prefix(start, stop, step):
    for ordinal, tick in enumerate(builtins.range(start, stop, step)):
        if ordinal >= MAX_EPOCHS:
            raise DiagnosticPrefixLimit("no original integer exception within the fixed ten-epoch prefix")
        yield tick


def run(root, fleetpy=FLEETPY):
    from stage4.analysis.city_numeric_probe import LegacyNumericProbe, compare_saved_problem
    root = Path(root).resolve()
    destination = root / OUT
    receipt_path = destination / "diagnostic_summary.json"
    if receipt_path.exists():
        raise ValueError("diagnostic already has a receipt; no implicit rerun or overwrite")
    started = perf_counter()
    numeric_name = "stage4.dispatch._city_numeric_aeec12a_diagnostic"
    legacy = load_frozen_module(root, "stage4/dispatch/city_defer_native.py", numeric_name)
    runner = load_frozen_module(root, "stage4/analysis/city_joint_validation.py",
        "stage4.analysis._city_native_aeec12a_diagnostic", redirect=(
            "from stage4.dispatch.city_defer_native import CityDeferNativeAdapter",
            f"from {numeric_name} import CityDeferNativeAdapter"))
    cfg, reference, protected = runner.load_protocol(root)
    original_receipt = json.loads((root / "stage4/output/city_joint_validation_v1/LOCATION_AWARE_DEFER/summary.json").read_text(encoding="utf-8"))
    if original_receipt["code_sha"] != ORIGINAL_SHA or original_receipt["inputs_sha256"] != protected:
        raise ValueError("original failure source/config/input receipt does not match current frozen inputs")
    runner.resource_guard(root, cfg, initial=True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="DIAGNOSTIC_PREFIX_RUNNING", original_code_sha=ORIGINAL_SHA,
        original_numeric_source_sha256=legacy.frozen_source_sha256,
        original_runner_source_sha256=runner.frozen_source_sha256,
        maximum_replayed_epochs=MAX_EPOCHS, diagnostic_wall_limit_s=MAX_RUNTIME_S,
        input_hashes_equal_to_original_failure=True, inputs_sha256=protected,
        reconstruction_kind="SAME_FROZEN_CODE_INPUT_RECONSTRUCTION_NOT_RECOVERED_ORIGINAL_VECTOR",
        original_first_action_pair_hash_available=False, full_window_runs=0,
        generic_control_runs=0, data_or_scientific_policy_changed=False,
        gpu_used=False, dense_matrix=False, native_epochs=[])
    atomic_json(receipt_path, receipt)
    probe = LegacyNumericProbe(legacy, destination / "failed_model")
    original_adapter = legacy.CityDeferNativeAdapter
    class ObservedLegacyAdapter(original_adapter):
        def _problem(self, c, arcs, waiting_ids, now):
            problem = super()._problem(c, arcs, waiting_ids, now)
            probe.observe_problem(problem, now)
            return problem

        def solve(self, c, original_arcs, waiting_ids, now):
            if perf_counter() - started > MAX_RUNTIME_S:
                raise TimeoutError("fixed numeric diagnostic wall budget exceeded")
            epoch = dict(now_s=now, past_committed_count=len(c.assignment_rows),
                         original_current_arc_count=len(original_arcs), waiting_count=len(waiting_ids))
            receipt["native_epochs"].append(epoch)
            atomic_json(receipt_path, receipt)
            result = super().solve(c, original_arcs, waiting_ids, now)
            selected = sorted((original_arcs[i].vehicle_id, original_arcs[i].request_id)
                              for i in result.selected_indices)
            epoch.update(selected_count=len(selected), decision_pairs_sha256=hashlib.sha256(
                json.dumps(selected,separators=(",", ":")).encode()).hexdigest())
            atomic_json(receipt_path, receipt)
            return result
    legacy.CityDeferNativeAdapter = ObservedLegacyAdapter
    # Keep all original physical/demand/forecast parameters. Only limit the
    # native loop and put diagnostic output in a new, non-overwriting folder.
    runner.OUT = OUT / "reconstructed_prefix"
    runner.DOC = DOC / "private_prefix_receipt"
    runner.range = bounded_prefix
    saved_guard = runner.resource_guard
    def guard(*args, **kwargs):
        if perf_counter() - started > MAX_RUNTIME_S:
            raise TimeoutError("fixed numeric diagnostic wall budget exceeded")
        return saved_guard(*args, **kwargs)
    runner.resource_guard = guard
    probe.install()
    expected_failure = None
    try:
        runner.run_policy(root, Path(fleetpy), cfg, reference, protected, "LOCATION_AWARE_DEFER")
    except DiagnosticPrefixLimit as error:
        receipt.update(status="ORIGINAL_INTEGER_ERROR_NOT_REPRODUCED_WITHIN_PREFIX", error=str(error))
    except Exception as error:
        expected_failure = str(error)
        if probe.last_failure_path is None:
            receipt.update(status="DIAGNOSTIC_INTERRUPTED_BY_DIFFERENT_ERROR", error=repr(error))
            atomic_json(receipt_path,receipt)
            raise
        receipt.update(status="ORIGINAL_INTEGER_ERROR_REPRODUCED_IN_RECONSTRUCTED_STATE",
                       original_exception=str(error))
    finally:
        probe.close()
        legacy.CityDeferNativeAdapter = original_adapter
    receipt["legacy_solver_stage_summaries"] = probe.stage_summaries
    if expected_failure is not None and probe.last_failure_path is not None:
        model_path = Path(probe.last_failure_path).parent / "problem.json"
        receipt["same_problem_comparison"] = compare_saved_problem(model_path, legacy, fixed_numeric)
    if protected != {path:sha(root / path) for path in protected}:
        raise RuntimeError("frozen diagnostic inputs changed")
    receipt.update(runtime_s=perf_counter()-started,
                   peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20,
                   original_failed_output_preserved=True)
    atomic_json(receipt_path, receipt)
    atomic_json(root / DOC / "summary.json", receipt)
    print(json.dumps(dict(status=receipt["status"], runtime_s=receipt["runtime_s"],
                         peak_rss_mib=receipt["peak_rss_mib"])),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--fleetpy-root",type=Path,default=FLEETPY)
    args=parser.parse_args()
    run(args.root,args.fleetpy_root)


if __name__ == "__main__":
    main()
