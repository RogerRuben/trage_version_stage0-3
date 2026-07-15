"""Single entrypoint for the Stage 0-4 canonical rebaseline smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from canonical_pipeline.manifest import config_sha256, load_yaml
from canonical_pipeline.preflight import STAGES, validate_config, validate_declared_inputs, validate_field_registry
from canonical_pipeline.registry import RunRegistry, combined_manifest_hash, git_commit, new_run_id, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "audit", "check-manifests"], required=True)
    parser.add_argument("--resume-from", choices=STAGES, default="stage0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    config_path = (workspace / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_yaml(config_path)
    cfg_hash = config_sha256(config_path)
    errors = validate_config(config, workspace)
    field_registry = workspace / "docs/pipeline_contract/field_availability_registry.csv"
    errors.extend(validate_field_registry(field_registry))
    schema_path = workspace / config["manifest_schema"]
    manifest_errors, manifest_hashes = validate_declared_inputs(
        config, workspace, schema_path, args.resume_from
    )
    if args.mode in {"smoke", "check-manifests"}:
        errors.extend(manifest_errors)
    audit = {
        "pipeline_version": config.get("pipeline_version"),
        "config_hash": cfg_hash,
        "mode": args.mode,
        "resume_from": args.resume_from,
        "contract_preflight": "PASS" if not validate_config(config, workspace) else "FAIL",
        "field_availability_registry": "PASS" if not validate_field_registry(field_registry) else "FAIL",
        "manifest_preflight": "PASS" if not manifest_errors else "FAIL",
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    audit_path = workspace / "docs/pipeline_rebaseline/rebaseline_preflight_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.mode != "smoke" or args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        if errors:
            raise SystemExit(2)
        return

    if errors:
        print(json.dumps(audit, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    seed = int(config["smoke"]["seed"])
    run_id = new_run_id(config["pipeline_version"], seed)
    registry = RunRegistry(workspace / config["run_registry"])
    registry.assert_unique_canonical_success("end_to_end", cfg_hash)
    started = now_iso()
    # Stage execution is deliberately gated until every canonical input
    # manifest exists.  Individual v2 stage runners are registered during
    # Milestones 3-7; no legacy-directory fallback is permitted here.
    row = {
        "run_id": run_id,
        "stage": "end_to_end",
        "commit": git_commit(),
        "config_hash": cfg_hash,
        "input_manifest_hash": combined_manifest_hash(manifest_hashes),
        "seed": seed,
        "started_at": started,
        "finished_at": now_iso(),
        "status": "FAILED",
        "canonical": "false",
        "supersedes_run_id": "",
        "audit_status": "FAIL",
        "output_path": config["smoke"]["output_root"],
    }
    registry.append(row)
    raise RuntimeError("Canonical stage runners are not yet registered; legacy fallback is forbidden")


if __name__ == "__main__":
    main()

