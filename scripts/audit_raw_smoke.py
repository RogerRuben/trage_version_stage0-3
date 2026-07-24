"""Aggregate deterministic raw-smoke extraction audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    days = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    result = {
        "status": "PASS" if all(item.get("status") == "PASS" for item in days) else "FAIL",
        "sampling_contract": "complete orders selected by stable SHA256 rank; no row sampling",
        "days": days,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
