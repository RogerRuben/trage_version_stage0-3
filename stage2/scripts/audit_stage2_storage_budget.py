"""Audit disk budget before large Stage2/Stage3 jobs.

The report is intentionally conservative: it measures retained products that
matter to the next experiments and estimates future jobs from already built
100k/15k artifacts when available.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


GB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", default="D:\\")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/storage_audit"))
    parser.add_argument("--extension-days", type=int, default=4)
    parser.add_argument("--scaling-factor", type=float, default=3.0)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--min-free-fraction", type=float, default=0.20)
    return parser.parse_args()


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def size_group(root: Path, paths: list[str]) -> dict:
    rows = []
    total = 0
    for rel in paths:
        path = root / rel
        size = dir_size(path)
        total += size
        rows.append({"path": rel, "exists": path.exists(), "size_bytes": size, "size_gb": size / GB})
    return {"total_bytes": total, "total_gb": total / GB, "items": rows}


def average_daily_size(path: Path, pattern: str = "day=*.parquet") -> float:
    files = sorted(path.glob(pattern)) if path.exists() else []
    if not files:
        return 0.0
    return sum(file.stat().st_size for file in files if file.is_file()) / len(files)


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Stage2 Storage Budget Audit",
        "",
        f"- D drive total: {payload['disk']['total_gb']:.2f} GB",
        f"- D drive free: {payload['disk']['free_gb']:.2f} GB ({payload['disk']['free_fraction']:.2%})",
        f"- Required free budget: {payload['policy']['required_free_gb']:.2f} GB",
        f"- Project measured footprint: {payload['project_measured_total_gb']:.2f} GB",
        f"- Estimated next jobs: {payload['estimates']['total_estimated_gb']:.2f} GB",
        f"- Free after estimates: {payload['decision']['free_after_estimates_gb']:.2f} GB",
        f"- Status: {payload['decision']['status']}",
        "",
        "## Measured Groups",
    ]
    for name, group in payload["groups"].items():
        lines.append(f"- {name}: {group['total_gb']:.2f} GB")
    lines += [
        "",
        "## Estimates",
    ]
    for key, value in payload["estimates"].items():
        if key.endswith("_gb"):
            lines.append(f"- {key}: {value:.2f} GB")
    lines += [
        "",
        "## Notes",
        "- Large jobs should pass explicit --output-root, --temp-root, and --skip-existing/--overwrite flags.",
        "- Reproducible caches may be pruned only after manifest and readiness audits pass.",
        "- Raw data, compact datasets, formal checkpoints, held-out predictions, manifests, metrics, and reports are retained.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.drive)

    groups = {
        "stage0_output": size_group(root, ["stage0/output"]),
        "stage1_output": size_group(root, ["stage1/output"]),
        "stage2_tensor_shards": size_group(root, [
            "stage2/output/deep_v3_tensor_shards_100k",
            "stage2/output/deep_v3_tensor_shards_5k",
            "stage2/output/deep_v3_scaling_300k",
        ]),
        "stage2_predictions": size_group(root, [
            "stage2/output/deep_v3",
            "stage2/output/deep_v3_100k",
            "stage2/output/deep_v3_5k",
        ]),
        "stage2_compact_datasets": size_group(root, [
            "stage2/output/route_conditioned_dataset_15k",
            "stage2/output/iis_movement_causal_dataset",
            "stage2/output/strict_targets",
            "stage2/output/lagged_state_store",
        ]),
        "stage3_warehouse": size_group(root, ["stage3/output"]),
        "temporary_or_smoke_outputs": size_group(root, [
            "stage2/output/deep_v3_tensor_shards_smoke",
            "stage2/output/deep_v3_tensor_shards_smoke_predictions",
            "stage2/output/deep_v3_tensor_shards_smoke_train",
            "stage2/output/planned_route_causal_smoke",
            "stage2/output/strict_targets_smoke",
        ]),
    }
    measured_total = sum(group["total_bytes"] for group in groups.values())

    shard_100k = dir_size(root / "stage2/output/deep_v3_tensor_shards_100k")
    route_daily = average_daily_size(root / "stage2/output/route_conditioned_dataset_15k/estimated_time_daily")
    iis_daily = average_daily_size(root / "stage2/output/iis_movement_causal_dataset")
    prediction_daily = 0.0
    warehouse_link = root / "stage3/output/stage2_prediction_warehouse/link_predictions"
    if warehouse_link.exists():
        files = list(warehouse_link.rglob("day=*.parquet"))
        prediction_daily = sum(file.stat().st_size for file in files) / max(1, len(files))

    estimated_300k = shard_100k * args.scaling_factor * 1.20
    estimated_extension = args.extension_days * (route_daily + iis_daily + prediction_daily) * 1.35
    estimated_temp = max(estimated_300k, estimated_extension) * 0.35
    estimated_total = estimated_300k + estimated_extension + estimated_temp
    required_free = max(args.min_free_gb * GB, disk.total * args.min_free_fraction)
    free_after = disk.free - estimated_total
    status = "PASS" if free_after >= required_free else "WARN"

    payload = {
        "disk": {
            "drive": args.drive,
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "total_gb": disk.total / GB,
            "used_gb": disk.used / GB,
            "free_gb": disk.free / GB,
            "free_fraction": disk.free / disk.total,
        },
        "policy": {
            "min_free_gb": args.min_free_gb,
            "min_free_fraction": args.min_free_fraction,
            "required_free_bytes": required_free,
            "required_free_gb": required_free / GB,
        },
        "groups": groups,
        "project_measured_total_bytes": measured_total,
        "project_measured_total_gb": measured_total / GB,
        "estimates": {
            "scaling_factor": args.scaling_factor,
            "extension_days": args.extension_days,
            "estimated_300k_build_gb": estimated_300k / GB,
            "estimated_extension_build_gb": estimated_extension / GB,
            "estimated_temp_gb": estimated_temp / GB,
            "total_estimated_gb": estimated_total / GB,
        },
        "decision": {
            "status": status,
            "free_after_estimates_bytes": free_after,
            "free_after_estimates_gb": free_after / GB,
            "message": "Budget satisfies the retained-space policy." if status == "PASS" else "Budget is below the retained-space policy; prune reproducible caches or reduce job scope first.",
        },
    }
    (args.output_root / "storage_budget.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.output_root / "storage_budget_report.md")
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
