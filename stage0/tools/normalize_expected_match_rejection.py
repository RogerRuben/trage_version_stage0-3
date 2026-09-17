"""Normalize expected Valhalla no-path outcomes in completed Stage 1 buckets.

The production runner originally counted a Valhalla ``map_snap`` no-path
outcome as an engineering exception.  It is instead a normal candidate
rejection: no accepted product rows are changed and the next stable-hash
candidate was already processed to fill the daily quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_PREFIX = (
    "PROCESSING_EXCEPTION:RuntimeError:Map Match algorithm failed to find path:"
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def normalize(root: Path) -> dict:
    changed_rows = 0
    changed_buckets: list[str] = []
    changed_daily: set[tuple[str, str]] = set()
    for rejection_path in sorted(
        root.glob("rejections/split=*/date=*/bucket=*.parquet")
    ):
        frame = pd.read_parquet(rejection_path)
        expected = frame.rejection_reason.astype(str).str.startswith(
            EXPECTED_PREFIX
        )
        count = int(expected.sum())
        if not count:
            continue
        frame.loc[expected, "rejection_reason"] = (
            frame.loc[expected, "rejection_reason"]
            .astype(str)
            .str.replace(
                "PROCESSING_EXCEPTION:",
                "MATCH_REJECTION:",
                n=1,
                regex=False,
            )
        )
        _atomic_parquet(rejection_path, frame)

        split = rejection_path.parents[1].name.removeprefix("split=")
        date = rejection_path.parent.name.removeprefix("date=")
        bucket = rejection_path.stem.removeprefix("bucket=")
        manifest_path = (
            root
            / f"split={split}"
            / f"date={date}"
            / f"bucket={bucket}"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["processing_exception_count"]) < count:
            raise RuntimeError(
                f"manifest exception underflow: {manifest_path}"
            )
        manifest["processing_exception_count"] -= count
        manifest["rejected_count"] += count
        manifest["status"] = (
            "PASS"
            if int(manifest["processing_exception_count"]) == 0
            else "FAIL"
        )
        _atomic_json(manifest_path, manifest)
        changed_rows += count
        changed_buckets.append(
            f"{split}/{date}/bucket={bucket}"
        )
        changed_daily.add((split, date))

    for split, date in sorted(changed_daily):
        daily_path = (
            root / "manifests" / f"split={split}" / f"date={date}.json"
        )
        daily = json.loads(daily_path.read_text(encoding="utf-8"))
        bucket_exception_count = 0
        for path in sorted(
            (root / f"split={split}" / f"date={date}").glob(
                "bucket=*/manifest.json"
            )
        ):
            bucket_exception_count += int(
                json.loads(path.read_text(encoding="utf-8"))[
                    "processing_exception_count"
                ]
            )
        daily["processing_exception_count"] = bucket_exception_count
        _atomic_json(daily_path, daily)

    return {
        "status": "PASS",
        "normalized_match_rejection_count": changed_rows,
        "changed_buckets": changed_buckets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            normalize(args.input.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
