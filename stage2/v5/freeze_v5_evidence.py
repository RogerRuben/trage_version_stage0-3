"""Freeze hashes for the immutable Stage 2 v5 failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EVIDENCE = (
    "stage2/docs/v5/stage2_v5_rolling_origin_summary.json",
    "stage2/docs/v5/stage2_v5_fold3_tail_diagnostic.md",
    "stage2/docs/v5/stage2_v5_fold3_tail_diagnostic.json",
    "stage2/docs/v5/protocols/fold_3/service_time_metrics.csv",
    "stage2/docs/v5/protocols/fold_3/scenario_coverage.csv",
    "stage2/docs/v5/stage2_v5_final_verification.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files = []
    for relative in EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    rolling = json.loads((root / EVIDENCE[0]).read_text(encoding="utf-8"))
    tail = json.loads((root / EVIDENCE[2]).read_text(encoding="utf-8"))
    day = next(item for item in tail["dates"] if item["date"] == "20161026")
    result = {
        "schema_version": "stage2_v5_1_frozen_v5_evidence.1",
        "base_commit": "3b581c5e6d108441ca623bcc03923f5893de7ea9",
        "status": "FROZEN_IMMUTABLE",
        "files": files,
        "frozen_failure_facts": {
            "20161026_frozen_pace_mean_max_s_per_m": day["maximum_prediction_sec_per_m"],
            "rolling_v5_aggregate_mae": rolling["v5_aggregate_mae"],
            "rolling_tree_aggregate_mae": rolling["tree_aggregate_mae"],
            "rolling_relative_degradation": rolling["aggregate_relative_mae_change"],
            "rolling_scientific_status": rolling["scientific_status"],
        },
        "prohibited_reinterpretations": [
            "no_trimming_or_row_deletion",
            "no_p50_substitution_for_frozen_mean_metric",
            "no_rolling_status_rewrite",
        ],
    }
    output = root / "stage2/docs/v5_1/stage2_v5_1_frozen_v5_evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(freeze(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
