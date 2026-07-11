"""Summarize Stage2 Deep v3 artifacts into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-v3-root", type=Path, default=Path("stage2/output/deep_v3"))
    parser.add_argument("--output", type=Path, default=Path("stage2/output/deep_v3/deep_v3_report.md"))
    parser.add_argument("--tensor-shard-root", type=Path, default=None)
    return parser.parse_args()


def read_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"_Missing: `{path}`_\n"


def main() -> None:
    args = parse_args()
    candidates = [
        args.deep_v3_root / "formal_rolling" / "rc_mstnet" / "rc_mstnet_report.md",
        args.deep_v3_root / "feasibility_100k" / "rc_mstnet" / "rc_mstnet_report.md",
        args.deep_v3_root / "feasibility" / "rc_mstnet" / "rc_mstnet_report.md",
    ]
    rc_report = next((path for path in candidates if path.exists()), candidates[0])
    manifest_path = rc_report.parent / "rc_mstnet_manifest.json"
    scale_note = "Training scale is unavailable."
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        effective = [fold.get("effective_train_orders") for fold in payload.get("folds", [])]
        if effective:
            scale_note = f"Effective train orders by fold: `{effective}`."
    shard_audit = (
        args.tensor_shard_root / "audit_report.json"
        if args.tensor_shard_root is not None else args.deep_v3_root / "tensor_shard_audit_report.json"
    )
    sections = [
        "# Stage2 Deep v3 report",
        "",
        "Deep v3 tests RC-MSTNet under the final route-conditioned estimated-time protocol. "
        "The report derives its scale claim from the saved fold manifests rather than the requested CLI budget. " + scale_note,
        "",
        "## Tensor shard audit",
        "",
        read_if_exists(shard_audit)[:4000],
        "",
        "## Sequence manifest",
        "",
        read_if_exists(args.deep_v3_root / "data_manifests" / "deep_v3_sequence_manifest.json")[:4000],
        "",
        "## RC-MSTNet",
        "",
        read_if_exists(rc_report),
        "",
        "## Evaluation",
        "",
        read_if_exists(args.deep_v3_root / "metrics" / "deep_v3_eval_report.md"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
