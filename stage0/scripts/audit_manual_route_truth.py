"""Audit completed independent route reviews and enforce the promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.manual_truth import review_metrics, validate_review_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = pd.read_csv(args.primary)
    secondary = pd.read_csv(args.secondary)
    errors = validate_review_schema(primary) + validate_review_schema(secondary)
    complete_primary = primary.review_status.eq("completed")
    core_ids = primary.loc[primary.route_quality_class.eq("core"), "order_id"].astype(str)
    metrics = review_metrics(primary, core_ids) if not errors else {}
    first = primary.loc[complete_primary, ["order_id", "route_correct"]]
    second = secondary.loc[secondary.review_status.eq("completed"), ["order_id", "route_correct"]]
    paired = first.merge(second, on="order_id", suffixes=("_primary", "_secondary"))
    agreement = float(
        paired.route_correct_primary.astype(bool).eq(paired.route_correct_secondary.astype(bool)).mean()
    ) if len(paired) else None
    pass_gate = bool(
        not errors
        and metrics.get("completed_reviews", 0) >= 300
        and metrics.get("core_precision") is not None
        and metrics["core_precision"] >= 0.90
        and len(paired) > 0
    )
    result = {
        "status": "PASS" if pass_gate else "HOLD",
        "schema_errors": errors,
        **metrics,
        "double_review_pairs": int(len(paired)),
        "double_review_agreement": agreement,
        "canonical_promotion_gate": "OPEN" if pass_gate else "HOLD",
        "blocker": None if pass_gate else "Complete independent review and adjudication first.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
