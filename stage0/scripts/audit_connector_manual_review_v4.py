"""Evaluate actual human judgments for the registered v4 connector sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "link_id", "reviewer_id", "review_status", "direction_correct",
    "level_transition_correct", "major_error", "data_limitation",
}


def connector_review_metrics(frame: pd.DataFrame) -> dict[str, object]:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        return {"schema_errors": [f"missing_columns:{','.join(missing)}"]}
    completed = frame.loc[frame.review_status.eq("completed")].copy()
    for column in ["direction_correct", "level_transition_correct", "major_error"]:
        completed[column] = completed[column].astype("boolean")
    return {
        "schema_errors": [],
        "sampled_connectors": int(len(frame)),
        "completed_connectors": int(len(completed)),
        "major_error_rate": float(completed.major_error.mean()) if len(completed) else None,
        "wrong_direction_rate": float((~completed.direction_correct).mean()) if len(completed) else None,
        "wrong_level_transition_rate": (
            float((~completed.level_transition_correct).mean()) if len(completed) else None
        ),
        "data_limitation_rate": (
            float(completed.data_limitation.astype("boolean").mean()) if len(completed) else None
        ),
    }


def connector_review_pass(metrics: dict[str, object]) -> bool:
    return bool(
        not metrics.get("schema_errors")
        and int(metrics.get("completed_connectors", 0)) >= 30
        and metrics.get("major_error_rate") is not None
        and float(metrics["major_error_rate"]) <= 0.15
        and float(metrics["wrong_direction_rate"]) <= 0.05
        and float(metrics["wrong_level_transition_rate"]) <= 0.05
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = connector_review_metrics(pd.read_csv(args.reviews))
    passed = connector_review_pass(metrics)
    result = {
        "status": "PASS" if passed else "HOLD",
        **metrics,
        "requirements": {
            "minimum_completed_connectors": 30,
            "maximum_major_error_rate": 0.15,
            "maximum_wrong_direction_rate": 0.05,
            "maximum_wrong_level_transition_rate": 0.05,
        },
        "blocker": None if passed else "Complete targeted independent connector review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
