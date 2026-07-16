"""Audit governance gates for the staged paper pipeline without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


STAGES = ("stage0", "stage1", "stage2", "stage3", "stage35")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_research_v3.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pipeline_rebaseline/paper_pipeline_gate_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required_files = [
        config[stage]["contract"] for stage in STAGES
    ] + [
        config["stage0"]["route_quality_config"],
        config["stage3"]["odd_scenarios"],
        config["stage35"]["route_config"],
        config["stage4"]["contract"],
    ]
    missing_definitions = [path for path in required_files if not Path(path).is_file()]
    manifest_state = {
        stage: {
            "path": config[stage]["output_manifest"],
            "exists": Path(config[stage]["output_manifest"]).is_file(),
        }
        for stage in STAGES
    }
    first_blocking_stage = next(
        (stage for stage in STAGES if not manifest_state[stage]["exists"]), None
    )
    checks = {
        "definition_files_exist": not missing_definitions,
        "formal_stage4_disabled": config["formal_stage4_enabled"] is False,
        "formal_stage4_gate_is_hold": config["stage4"]["formal_full_day_gate"] == "HOLD",
        "stage4_reads_stage35_manifest": (
            config["stage4"]["input_manifest"] == config["stage35"]["output_manifest"]
        ),
        "fleetpy_validation_is_bounded": (
            500 <= int(config["stage4"]["maximum_technical_validation_orders"]) <= 2000
        ),
        "test_date_not_in_upstream_fit_or_train": not (
            set(config["dates"]["test"])
            & (set(config["dates"]["upstream_fit"]) | set(config["dates"]["train"]))
        ),
        "test_date_not_in_validation": not (
            set(config["dates"]["test"]) & set(config["dates"]["validation"])
        ),
    }
    governance_status = "PASS" if all(checks.values()) else "FAIL"
    promotion_status = "READY" if first_blocking_stage is None else "HOLD"
    result = {
        "status": governance_status,
        "pipeline_version": config["pipeline_version"],
        "promotion_status": promotion_status,
        "first_blocking_stage": first_blocking_stage,
        "checks": checks,
        "missing_definition_files": missing_definitions,
        "canonical_manifest_state": manifest_state,
        "interpretation": (
            "Governance PASS confirms that downstream execution is blocked correctly; "
            "it does not promote missing stage artifacts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if governance_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
