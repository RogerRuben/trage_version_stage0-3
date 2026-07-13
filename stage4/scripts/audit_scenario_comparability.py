"""Audit that Stage4 scenarios use comparable order streams and configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("stage4/output/pricing_dispatch/scenario_summary.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/pricing_dispatch/audits"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    rows = []
    errors = []
    for fold, group in summary.groupby("fold"):
        order_counts = sorted(group["orders"].dropna().unique().tolist())
        rows.append({"fold": fold, "scenario_count": len(group), "order_counts": ";".join(map(str, order_counts)), "comparable": len(order_counts) == 1})
        if len(order_counts) != 1:
            errors.append(f"fold={fold}: scenario order counts differ {order_counts}")
    required = {"strategy", "supply", "av_penetration", "odd_profile", "pricing"}
    present = set(summary["experiment_family"].dropna().unique())
    missing_families = sorted(required - present)
    if missing_families:
        errors.append(f"missing experiment families: {missing_families}")
    pd.DataFrame(rows).to_csv(args.output_root / "scenario_comparability_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "scenario_comparability_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
