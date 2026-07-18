"""Safe compact-retention gate. Deletion is explicit and dry-run by default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Stage0Config


REQUIRED = ("order_base", "link_traversals", "turn_movements", "route_parts", "route_quality", "performance")


def prune_point_work(config: Stage0Config, repo: Path, execute: bool = False) -> dict[str, Any]:
    output, work = config.path("output", repo), config.path("work", repo)
    missing = [name for name in REQUIRED if not (output / name).exists()]
    if not any((output / "case_traces").glob("*/day=*/case_trace_index.parquet")):
        missing.append("case_traces/*/day=*/case_trace_index.parquet")
    if not any((output / "reports").glob("stage0_v5_run_summary__*.json")):
        missing.append("reports/stage0_v5_run_summary__*.json")
    candidates = sorted((work / "sampled_points").glob("*/day=*/part=*/*.parquet"))
    candidates.extend(sorted((work / "matched_diagnostics").glob("*/day=*/*.parquet")))
    if execute and missing:
        raise RuntimeError(f"compact pruning blocked; missing products: {missing}")
    if execute:
        for path in candidates:
            path.unlink()
    return {"dry_run": not execute, "candidate_files": len(candidates), "missing_prerequisites": missing, "deleted_files": len(candidates) if execute else 0}
