"""Evaluate Stage0 v4 promotion gates without manufacturing a canonical PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-audit", type=Path, required=True)
    parser.add_argument("--comparison-audit", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--connector-audit", type=Path, required=True)
    parser.add_argument("--conservation-audit", type=Path, required=True)
    parser.add_argument("--full-date-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    network = read(args.network_audit)
    comparison = read(args.comparison_audit)
    manual = read(args.manual_audit)
    connector = read(args.connector_audit)
    conservation = read(args.conservation_audit)
    gates = {
        "network_topology_diagnostic": network.get("status") == "DIAGNOSTIC_PASS",
        "grade_separation_not_silently_noded": network.get("grade_separated_pairs_not_noded", 0) > 0,
        "fixed_1000_comparison_reviewed": comparison.get("status") in {"DIAGNOSTIC_PASS", "STOP_FOR_MANUAL_REVIEW"},
        "manual_truth": manual.get("status") == "PASS",
        "connector_human_review": connector.get("status") == "PASS",
        "diagnostic_conservation": conservation.get("status") == "PASS",
        "full_date_chain": bool(args.full_date_audit and args.full_date_audit.is_file()),
    }
    if gates["full_date_chain"]:
        gates["full_date_chain"] = read(args.full_date_audit).get("status") == "PASS"
    passed = all(gates.values())
    result = {
        "status": "PASS" if passed else "HOLD",
        "canonical_promotion_allowed": passed,
        "gates": gates,
        "stop_rule_active": comparison.get("status") == "STOP_FOR_MANUAL_REVIEW",
        "blockers": [name for name, value in gates.items() if not value],
        "canonical_manifest": (
            "artifacts/canonical/stage0_v4/stage0_v4.manifest.json" if passed else None
        ),
        "downstream_stage_status": "OPEN" if passed else "HOLD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
