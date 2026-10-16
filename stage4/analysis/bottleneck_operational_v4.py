"""One opt-in operational upgrade, never a new scenario/parameter sweep."""
import argparse
import json
from pathlib import Path

from . import bottleneck_whole_method as previous
from .scheme_a_two_scale_check import _atomic_json,_run_native,_sha,FLEETPY


CONFIG=Path("stage4/config/bottleneck_operational_v4.json")
OUT=Path("stage4/output/capability_chain_planning/bottleneck_operational_v4")
DOC=Path("stage4/docs/capability_chain_planning/bottleneck_operational_v4")


def load(root):
    root=Path(root)
    declared=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    source=Path(declared["base_config"])
    if (source!=Path("stage4/config/bottleneck_whole_method_v3.json")
            or _sha(root/source)!=declared["base_config_sha256"]):
        raise ValueError("V4 requires the unchanged completed V3 protocol")
    base,_,_=previous.load(root,source)
    expected=dict(version="bottleneck_operational_v4",group="ADAPTIVE_SHARED_PLAN",
        retained_layout_mode="bottleneck_joint",joint_execution_mode="INCREMENTAL_V3",
        joint_neighborhood_strategy="ADAPTIVE_SHARED",joint_move_validation_mode="LAZY_SELECTED_V4",
        joint_move_target_mode="PROFILE_HISTORY",joint_exact_warm_reuse=True,default_action="prepare")
    if any(declared.get(k)!=v for k,v in expected.items()) or any(declared.get(k) is not False for k in (
            "automatic_retry","parameter_search","model_refit","dense_matrix","gpu_used",
            "forced_C_priority","idle_utilization_reward","change_supply_calendar","recompute_birth_layout")):
        raise ValueError("outside the one fixed operational V4 method")
    cfg=dict(previous.group_config(base,previous.ARMS[2]),**{
        k:declared[k] for k in ("version","group","joint_execution_mode","joint_neighborhood_strategy",
            "joint_move_validation_mode","joint_move_target_mode","joint_exact_warm_reuse",
            "typed_reposition_reference","typed_reposition_manifest")})
    cfg.update(base_config=source.as_posix(),groups=[cfg["group"]],
        disk_route_cache=(OUT/"native"/cfg["group"]/"routing/cache.sqlite3").as_posix(),
        experiment_scope="ONE_FIXED_OPERATIONAL_METHOD_CONFIRMATION_NOT_PARAMETER_SEARCH")
    return cfg


def prepare(root):
    cfg=load(root)
    binding=previous.binding_for(Path(root),CONFIG,cfg)
    result=dict(status="OPERATIONAL_V4_PREPARED_NOT_EXECUTED",binding=binding,
        group=cfg["group"],source_layout=cfg["source_layout"],layout_mode=cfg["layout_mode"],
        source_history_dates=cfg["forecast_train_dates"],scientific_conditions_changed=False,
        supply_or_birth_layout_changed=False,objective_tiers_changed=False,
        current_move_cap=50,maximum_move_route_calls_per_epoch=50,
        shared_current_deadline_s=10,probe_budget_fraction=.4,certification_end_fraction=.7,
        method_is_finite_budget_heuristic=True,global_optimality_claimed=False,
        no_full_day_result_or_speedup_claim=True)
    _atomic_json(Path(root)/DOC/"preparation.json",result)
    return result


def run(root,*,prefix=False):
    root=Path(root); cfg=load(root)
    phase="prefix" if prefix else "native"
    directory=root/OUT/phase/cfg["group"]
    if prefix:
        cfg.update(diagnostic_stop_tick_s=2400,scenario_timeout_s=600,
            disk_route_cache=(OUT/phase/cfg["group"]/"routing/cache.sqlite3").as_posix())
    binding=previous.binding_for(root,CONFIG,cfg)
    with previous.execution_lock(root,OUT):
        return _run_native(root,FLEETPY,CONFIG,cfg,directory=directory,
            doc_summary_path=root/DOC/("prefix_summary.json" if prefix else "native_summary.json"),
            completion_status="OPERATIONAL_V4_NATIVE_COMPLETE",group=cfg["group"],execution_binding=binding)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","prefix","run"),nargs="?",default="prepare")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.action!="prepare" and not args.execute:
        parser.error("simulation needs explicit --execute; default only prepares metadata")
    result=prepare(Path.cwd()) if args.action=="prepare" else run(Path.cwd(),prefix=args.action=="prefix")
    print(json.dumps({k:result[k] for k in ("status","group") if k in result}),flush=True)


if __name__=="__main__": main()
