"""Fixed six-case design and a cheap supply check, never a native launcher.

The three pairs reuse the same common control. Only birth layout changes within
a pair. A new label seed at q=.10 and q=.05/.20 at the reference seed are the
entire bounded confirmation; no seed/parameter search or upstream production.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

VERSION = "scheme_a_layout_confirmation_v1"
CONFIG = Path(f"stage4/config/{VERSION}.json")
BASE_CONFIG = Path("stage4/config/scheme_a_two_scale_city_v5_common.json")
OUT = Path("stage4/output/capability_chain_planning/layout_confirmation_v1")
DOC = Path("stage4/docs/capability_chain_planning/layout_confirmation_v1")
REFERENCE = Path("stage4/docs/capability_chain_planning/layout_comparison_300_600_v1/comparison.json")
GROUPS = ("COMPATIBLE_HOT_COMMON_CONTROL", "OPPORTUNITY_JOINT_COMMON_CONTROL")
MODES = ("compatible_hot", "opportunity_joint")
CONDITIONS = (
    ("Q10_LABEL_SEED2", .1, 20261013),
    ("Q05_REFERENCE_LABELS", .05, 20260824),
    ("Q20_REFERENCE_LABELS", .2, 20260824),
)


def load_design(root, config=CONFIG):
    from .scheme_a_two_scale_check import _sha
    root, config = Path(root), Path(config)
    data = json.loads((root / config).read_text(encoding="utf-8"))
    fixed = dict(version=VERSION, base_config=BASE_CONFIG.as_posix(),
        groups=list(GROUPS), layout_generators=list(MODES),
        layout_method="SHARED_SCENARIO_SPARSE_LP_INTEGER_PLACEMENT_RECOVERY",
        layout_bin_wall_limit_s=30., layout_preparation_timeout_s=900,
        layout_preparation_rss_mib=512, planned_native_groups=6,
        automatic_retry=False, parameter_search=False, model_refit=False,
        dense_matrix=False, gpu_used=False)
    rows = [dict(condition_id=name, requested_q_a=q, supply_seed=seed) for name, q, seed in CONDITIONS]
    if (set(data) != set(fixed) | {"base_config_sha256", "conditions"}
            or any(data.get(key) != value for key, value in fixed.items())
            or data.get("conditions") != rows
            or _sha(root / BASE_CONFIG) != data["base_config_sha256"]):
        raise ValueError("outside the fixed six-case C confirmation design")
    return data


def condition_configuration(root, config, condition):
    design = load_design(root, config)
    rows = {row["condition_id"]: row for row in design["conditions"]}
    if condition not in rows:
        raise ValueError("select one explicit condition from the six-case design")
    result = dict(design, **rows[condition])
    result.pop("conditions")
    result.update(confirmation_condition=condition, layout_product="PREPARED_300_600",
        source_layout=(OUT / condition / "layout/placements.json").as_posix(),
        experiment_scope="FIXED_C_TWO_PREPARED_LAYOUTS_IDENTICAL_COMMON_CONTROL")
    return result


def preparation_output_root(cfg):
    return OUT / cfg["confirmation_condition"] / "layout"


def preparation_document_root(cfg):
    return DOC / cfg["confirmation_condition"] / "layout"


def experiment_output_root(cfg):
    return OUT / cfg["confirmation_condition"] / "native"


def experiment_document_root(cfg):
    return DOC / cfg["confirmation_condition"] / "native"


def check_design(root, config=CONFIG):
    """Read the small common fleet four times sequentially; no routes or M3."""
    import pandas as pd
    import psutil
    from .capability_chain_instances import atomic_json
    from .c_operations_diagnostic import START
    from .scheme_a_two_scale_check import _sha
    from .scheme_a_two_scale_experiment import load_protocol
    from stage4.dispatch.controlled_fleet import FLEET_REL, build_controlled_fleet

    started = perf_counter()
    root = Path(root).resolve()
    design = load_design(root, config)
    base = load_protocol(root, BASE_CONFIG)
    target = root / DOC / "design_check.json"
    if target.exists():
        raise ValueError("existing design check retained; no silent overwrite")
    inputs = {Path(config).as_posix(): _sha(root / config), FLEET_REL.as_posix(): _sha(root / FLEET_REL),
        REFERENCE.as_posix(): _sha(root / REFERENCE)}
    prior = json.loads((root / REFERENCE).read_text(encoding="utf-8"))
    if prior["execution_binding"]["inputs_sha256"][FLEET_REL.as_posix()] != inputs[FLEET_REL.as_posix()]:
        raise ValueError("common fleet template changed since the existing reference")
    observed, reference_ids, label_sets, original_rows = {}, None, {}, None
    entries = [("Q10_REFERENCE_EXISTING_NOT_RERUN", .1, 20260824), *CONDITIONS]
    for name, q, seed in entries:
        fleet = build_controlled_fleet(root, benchmark_start=START,
            simulation_end=START + pd.Timedelta(seconds=base["physical_drain_limit_s"]), requested_q_a=q, seed=seed)
        frame = fleet.scenario_fleet
        invariant = frame[["vehicle_id", "native_id", "availability_start_time", "availability_end_time",
            "initial_lon_wgs84", "initial_lat_wgs84", "availability_policy"]]
        if original_rows is None:
            original_rows = invariant.copy()
        elif not invariant.equals(original_rows):
            raise RuntimeError("label conditions changed physical common supply rows")
        av = frame.loc[frame.vehicle_type.eq("AV")]
        labels = set(av.vehicle_id.astype(str))
        label_sets[name] = labels
        if reference_ids is None:
            reference_ids = labels
        bins = int(((av.availability_start_time - START).dt.total_seconds() // 1800).nunique())
        accounting = fleet.accounting
        if name == "Q10_REFERENCE_EXISTING_NOT_RERUN" and accounting["type_label_sha256"] != prior["type_label_sha256"]:
            raise ValueError("reference supply labels do not reproduce the existing reference")
        observed[name] = {key: accounting[key] for key in ("seed", "requested_q_a", "q_a_realized", "q_a_error_pp",
            "h_base_day_exact", "achieved_av_vehicle_hours", "achieved_hv_vehicle_hours", "slot_count", "av_count",
            "selected_hv_session_count", "type_label_sha256", "total_supply_error_nanoseconds")}
        observed[name].update(expected_layout_bins=bins, intersection_with_existing_q10_AV_slots=len(labels & reference_ids),
            physical_windows_and_initial_template_positions_unchanged=True)
        if accounting["total_supply_error_nanoseconds"]:
            raise RuntimeError("common admission hours are not conserved")
        del fleet, frame, av, invariant
        gc.collect()
    nested = (label_sets["Q05_REFERENCE_LABELS"] <= label_sets["Q10_REFERENCE_EXISTING_NOT_RERUN"]
        <= label_sets["Q20_REFERENCE_LABELS"])
    if not nested or label_sets["Q10_LABEL_SEED2"] == reference_ids:
        raise RuntimeError("scale labels must be nested and the second seed must really change identities")
    if any(_sha(root / path) != digest for path, digest in inputs.items()):
        raise RuntimeError("a design-check source changed during reading")
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) / 2**20
    if peak > 512:
        raise MemoryError("cheap design check exceeded 512 MiB")
    commands = []
    python = "D:/anaconda/envs/stage0-valhalla/python.exe"
    for name, _, _ in CONDITIONS:
        common = ["--root", ".", "--config", Path(config).as_posix(), "--condition", name]
        commands.append(dict(condition_id=name,
            prepare=[python, "-u", "-m", "stage4.analysis.scheme_a_layout_prepare", *common],
            native_pair=[python, "-u", "-m", "stage4.analysis.scheme_a_two_scale_experiment", *common],
            analyze=[python, "-m", "stage4.analysis.scheme_a_layout_results", *common],
            group_order=list(GROUPS)))
    result = dict(status="SIX_CASE_DESIGN_AND_SUPPLY_CHECK_COMPLETE_NOT_EXECUTED", origin_skill="academic-research-suite",
        inputs_sha256=inputs, existing_reference_is_not_a_new_run=True,
        existing_reference_fleet_and_labels_reproduced=True,
        conditions=observed, scale_AV_label_sets_nested=nested, planned_native_groups=6,
        commands_in_condition_order=commands, primary_metric="C_WINDOW_LOADED_HOURS_PER_C_ADMISSION_HOUR",
        native_simulation_started=False, layout_optimization_started=False, routing_started=False,
        inference_started=False, gpu_used=False, dense_matrix=False, individual_identifiers_exported=False,
        runtime_s=perf_counter() - started, peak_rss_mib=peak)
    atomic_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = check_design(args.root, args.config)
    print(json.dumps({key: result[key] for key in ("status", "planned_native_groups", "runtime_s", "peak_rss_mib")}), flush=True)


if __name__ == "__main__":
    main()
