"""Audit that Stage4 scenarios use comparable order streams and configs."""

from __future__ import annotations

import argparse
import json
import hashlib
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
        fleet_hash_issues = 0
        for _, hash_group in group.groupby(["supply_scenario", "av_penetration"], dropna=False):
            if hash_group["initial_fleet_hash"].nunique() > 1:
                fleet_hash_issues += 1
        rows.append({"fold": fold, "scenario_count": len(group), "order_counts": ";".join(map(str, order_counts)), "comparable": len(order_counts) == 1 and fleet_hash_issues == 0, "fleet_hash_issue_groups": fleet_hash_issues})
        if len(order_counts) != 1:
            errors.append(f"fold={fold}: scenario order counts differ {order_counts}")
        if fleet_hash_issues:
            errors.append(f"fold={fold}: initial fleet hash differs within {fleet_hash_issues} supply/penetration groups")
    present = set(summary["experiment_family"].dropna().unique())
    if present == {"main_mechanism"}:
        required = {"main_mechanism"}
    else:
        required = {"main_mechanism", "supply", "av_penetration", "odd_profile", "pricing", "penetration_odd_interaction"}
    missing_families = sorted(required - present)
    if missing_families:
        errors.append(f"missing experiment families: {missing_families}")
    order_hash_rows = []
    for path in sorted(args.summary.parent.parent.parent.glob("output/pricing_dispatch/fold=*/exp=*/order_log.parquet")):
        pass
    # Prefer order logs colocated with the summary when available.
    run_root = args.summary.parents[2] / "output" / "pricing_dispatch"
    if not run_root.exists():
        run_root = Path("stage4/output/pricing_dispatch")
    fold_hashes: dict[str, str] = {}
    for path in sorted(run_root.glob("fold=*/exp=*/order_log.parquet")):
        fold = path.parts[-3].split("=", 1)[-1]
        frame = pd.read_parquet(path, columns=["order_id"])
        digest = hashlib.sha256("\n".join(sorted(frame["order_id"].astype(str).unique())).encode("utf-8")).hexdigest()
        order_hash_rows.append({"fold": fold, "experiment": path.parts[-2], "order_set_hash": digest})
        if fold not in fold_hashes:
            fold_hashes[fold] = digest
        elif fold_hashes[fold] != digest:
            errors.append(f"fold={fold}: order_id set differs across scenarios")
    pd.DataFrame(rows).to_csv(args.output_root / "scenario_comparability_audit.csv", index=False)
    if order_hash_rows:
        pd.DataFrame(order_hash_rows).to_csv(args.output_root / "scenario_order_hashes.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "scenario_comparability_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
