"""Build an explicit inventory of pre-rebaseline artifacts.

This script is governance-only. It inventories declared legacy roots; canonical
pipeline runners are forbidden from using directory discovery.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOTS = {
    "stage0": Path("stage0/output"),
    "stage1": Path("stage1/output"),
    "stage2": Path("stage2/output"),
    "stage3": Path("stage3/output"),
    "stage4": Path("stage4/output"),
}

KNOWN_ISSUES = {
    "stage0": "direction topology, parallel-edge, interval allocation, and IIS influence-area re-audit required",
    "stage1": "v1 partition median, CDF empty-bin, PMIS weighting, and missing-mask semantics",
    "stage2": "dispatch-time versus estimated-link-entry availability requires re-audit",
    "stage3": "legacy Stage2 inputs, calibration isolation, overall semantics, and full-day scale require rebuild",
    "stage4": "legacy upstream inputs; historical duration/test-calibrated capability contamination risk",
}

FIELDS = [
    "artifact_id", "stage", "path", "description", "created_by_commit",
    "input_source", "status", "used_by", "known_issue", "replacement_artifact",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/pipeline_rebaseline/legacy_artifact_inventory.csv"))
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    commit = git_commit()
    rows: list[dict[str, str]] = []
    for stage, root in ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if path.name.startswith("."):
                continue
            artifact_id = f"legacy_{stage}_{path.name}".replace(" ", "_")
            rows.append({
                "artifact_id": artifact_id,
                "stage": stage,
                "path": path.as_posix(),
                "description": "legacy directory" if path.is_dir() else "legacy file",
                "created_by_commit": commit,
                "input_source": "legacy_unfrozen_lineage",
                "status": "exploratory",
                "used_by": "legacy_only",
                "known_issue": KNOWN_ISSUES[stage],
                "replacement_artifact": f"canonical_{stage}_v2_pending",
            })
    rows.append({
        "artifact_id": "legacy_stage4_full_day_calibrated_profile",
        "stage": "stage4",
        "path": "stage4/config/vehicle_capability_profiles.json#full_day_calibrated",
        "description": "test-population calibrated capability sensitivity profile",
        "created_by_commit": commit,
        "input_source": "20161023 full-day distribution",
        "status": "deprecated",
        "used_by": "sensitivity_only",
        "known_issue": "test-day population calibration; prohibited in canonical counterfactual runs",
        "replacement_artifact": "pre_registered_exogenous_capability_profile_v2",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} legacy artifact records to {args.output}")


if __name__ == "__main__":
    main()

