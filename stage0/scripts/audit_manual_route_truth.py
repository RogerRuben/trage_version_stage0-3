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


def passes_review_gate(metrics: dict, paired_reviews: int, agreement: float | None) -> bool:
    return bool(
        metrics.get("completed_reviews", 0) >= 120
        and metrics.get("core_major_error_rate") is not None
        and metrics["core_major_error_rate"] <= 0.15
        and metrics["core_wrong_direction_rate"] <= 0.05
        and metrics["core_wrong_road_level_rate"] <= 0.05
        and metrics["core_unreasonable_detour_rate"] <= 0.10
        and paired_reviews >= 30
        and agreement is not None
        and agreement >= 0.80
    )


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
    first = primary.loc[complete_primary, ["order_id", "review_class"]]
    second = secondary.loc[secondary.review_status.eq("completed"), ["order_id", "review_class"]]
    paired = first.merge(second, on="order_id", suffixes=("_primary", "_secondary"))
    agreement = float(
        paired.review_class_primary.astype(str).eq(paired.review_class_secondary.astype(str)).mean()
    ) if len(paired) else None
    pass_gate = not errors and passes_review_gate(metrics, len(paired), agreement)
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
