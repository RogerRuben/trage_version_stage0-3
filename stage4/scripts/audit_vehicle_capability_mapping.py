"""Audit vehicle capability mapping outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-root", type=Path, default=Path("stage4/output/capability_mapping"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/capability_mapping/audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for path in sorted(args.mapping_root.glob("fold=*/vehicle_capability_mapping.parquet")):
        fold = path.parent.name.split("=", 1)[-1]
        frame = pd.read_parquet(path)
        duplicate = int(frame.duplicated(["order_id", "vehicle_profile"]).sum())
        if duplicate:
            errors.append(f"fold={fold} duplicate order-profile rows={duplicate}")
        rows.append({"fold": fold, "rows": len(frame), "orders": int(frame["order_id"].nunique()), "profiles": int(frame["vehicle_profile"].nunique()), "feasible_rate": float(frame["service_feasible"].mean())})
    pd.DataFrame(rows).to_csv(args.output_root / "capability_mapping_audit.csv", index=False)
    result = {"status": "PASS" if not errors and rows else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "capability_mapping_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
