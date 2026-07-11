"""Create and audit rolling Stage2 train/validation/test fold definitions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-root", type=Path, default=Path("stage1/output/prediction_split/primitives"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/rolling_oof_eval"))
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--validation-days", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--minimum-folds", type=int, default=3)
    return parser.parse_args()


def available_dates(root: Path) -> list[str]:
    return sorted(path.name.split("=", 1)[1] for path in root.glob("day=*") if path.is_dir())


def consecutive(values: list[str]) -> bool:
    parsed = [datetime.strptime(value, "%Y%m%d").date() for value in values]
    return all(right - left == timedelta(days=1) for left, right in zip(parsed[:-1], parsed[1:]))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = available_dates(args.primitive_root)
    width = args.train_days + args.validation_days + args.test_days
    folds = []
    for start in range(max(0, len(dates) - width + 1)):
        window = dates[start:start + width]
        if len(window) != width or not consecutive(window):
            continue
        train_end = args.train_days
        validation_end = train_end + args.validation_days
        folds.append({
            "fold": len(folds) + 1,
            "train_dates": window[:train_end],
            "validation_dates": window[train_end:validation_end],
            "test_dates": window[validation_end:],
        })
    ready = len(folds) >= args.minimum_folds
    manifest = {
        "available_dates": dates,
        "train_days": args.train_days,
        "validation_days": args.validation_days,
        "test_days": args.test_days,
        "minimum_required_folds": args.minimum_folds,
        "available_folds": len(folds),
        "ready_for_multi_day_rolling_oof": ready,
        "folds": folds,
        "blockers": [] if ready else [
            f"need at least {args.minimum_folds} consecutive {width}-day windows; current retained data supports {len(folds)}"
        ],
    }
    (args.output_root / "rolling_fold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Rolling/OOF readiness", "",
        f"Status: **{'READY' if ready else 'NOT READY'}**", "",
        f"Available consecutive dates: {', '.join(dates)}", "",
        f"Required folds: {args.minimum_folds}; available folds: {len(folds)}.", "",
    ]
    if not ready:
        lines.append("Additional compact Stage0/Stage1 days are required before publication-grade rolling/OOF model comparison.")
    (args.output_root / "rolling_oof_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

