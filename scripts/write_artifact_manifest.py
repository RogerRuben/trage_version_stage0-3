"""Write a strict artifact manifest from explicit role=path arguments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import config_sha256, file_record, sha256_file, write_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--schema-version", required=True)
    parser.add_argument("--stage", required=True, choices=["raw", "stage0", "stage1", "stage2", "stage3", "stage4", "end_to_end"])
    parser.add_argument("--status", default="canonical", choices=["canonical", "exploratory", "deprecated"])
    parser.add_argument("--input-artifact-id", action="append", default=[])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-dates", default="")
    parser.add_argument("--target-dates", default="")
    parser.add_argument("--decision-time-contract", required=True)
    parser.add_argument("--file", action="append", default=[], help="role=workspace-relative-path")
    parser.add_argument("--file-list-json", type=Path, help="producer JSON containing explicit files[{role,path}]")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-status", required=True, choices=["PASS", "FAIL", "NOT_RUN"])
    parser.add_argument("--known-limitation", action="append", default=[])
    parser.add_argument("--field-role-json", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("config/artifact_manifest.schema.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_dates(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = arguments()
    workspace = Path.cwd().resolve()
    file_items = list(args.file)
    if args.file_list_json:
        producer = json.loads(args.file_list_json.read_text(encoding="utf-8"))
        file_items.extend(f"{item['role']}={item['path']}" for item in producer["files"])
    records = []
    for item in file_items:
        if "=" not in item:
            raise ValueError(f"--file requires role=path, got {item!r}")
        role, raw_path = item.split("=", 1)
        records.append(file_record(Path(raw_path), role, workspace))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    audit_relative = args.audit.resolve().relative_to(workspace).as_posix()
    data = {
        "manifest_version": "1.0",
        "artifact_id": args.artifact_id,
        "schema_version": args.schema_version,
        "stage": args.stage,
        "status": args.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by_commit": commit,
        "input_artifact_ids": args.input_artifact_id,
        "config_hash": config_sha256(args.config),
        "fit_dates": split_dates(args.fit_dates),
        "target_dates": split_dates(args.target_dates),
        "decision_time_contract": args.decision_time_contract,
        "files": records,
        "audit": {
            "status": args.audit_status,
            "path": audit_relative,
            "sha256": sha256_file(args.audit),
        },
        "known_limitations": args.known_limitation,
    }
    if args.field_role_json:
        data["field_roles"] = json.loads(args.field_role_json.read_text(encoding="utf-8"))
    write_manifest(args.output, data, args.schema)
    if not args.quiet:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
