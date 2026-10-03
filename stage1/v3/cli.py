"""Command-line entry point for the isolated Stage 1 v3 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audit import verify_stage1_v3
from .config import load_config, validate_config
from .io import atomic_write_json
from .pipeline import fit_stage1_v3, transform_stage1_v3
from .preflight import run_global_preflight
from .scientific_review import run_scientific_review


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage1 v3 direct-GPS label pipeline. The default review-candidate "
            "configuration is intentionally non-executable without an explicit override."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit train-only reference/CDF models")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--model-root", type=Path, required=True)
    fit.add_argument("--stage0-freeze-manifest", type=Path, required=True)
    fit.add_argument(
        "--validated-preflight",
        type=Path,
        required=True,
        help="PASS global preflight report bound to the exact fit inputs",
    )
    fit.add_argument(
        "--stage1-code-sha",
        help="optional expected content identity; actual v3 sources are always hashed",
    )
    fit.add_argument("--no-resume", action="store_true")
    fit.add_argument("--allow-review-candidate", action="store_true")

    transform = subparsers.add_parser(
        "transform",
        help="apply frozen train models to all input buckets",
    )
    transform.add_argument("--input", type=Path, required=True)
    transform.add_argument("--model-root", type=Path, required=True)
    transform.add_argument("--output", type=Path, required=True)
    transform.add_argument("--stage0-freeze-manifest", type=Path, required=True)
    transform.add_argument(
        "--stage1-code-sha",
        help="optional expected content identity; actual v3 sources are always hashed",
    )
    transform.add_argument("--no-resume", action="store_true")
    transform.add_argument("--allow-review-candidate", action="store_true")

    verify = subparsers.add_parser(
        "verify",
        help="verify engineering correctness without claiming scientific validity",
    )
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--model-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--stage0-freeze-manifest", type=Path, required=True)
    verify.add_argument(
        "--stage1-code-sha",
        help="optional expected content identity; actual v3 sources are always hashed",
    )
    verify.add_argument("--report", type=Path)

    preflight = subparsers.add_parser(
        "preflight",
        help="read-only global validation of the frozen 220k Stage1 input",
    )
    preflight.add_argument("--input", type=Path, required=True)
    preflight.add_argument("--report", type=Path)
    scientific = subparsers.add_parser(
        "scientific-review",
        help="stream descriptive scientific diagnostics over all output buckets",
    )
    scientific.add_argument("--input", type=Path, required=True)
    scientific.add_argument("--output", type=Path, required=True)
    scientific.add_argument("--report-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    if args.command == "fit":
        result = fit_stage1_v3(
            args.input,
            args.model_root,
            args.stage0_freeze_manifest,
            args.validated_preflight,
            config,
            stage1_code_sha=args.stage1_code_sha,
            resume=not args.no_resume,
            allow_review_candidate=args.allow_review_candidate,
        )
    elif args.command == "transform":
        result = transform_stage1_v3(
            args.input,
            args.model_root,
            args.output,
            args.stage0_freeze_manifest,
            config,
            stage1_code_sha=args.stage1_code_sha,
            resume=not args.no_resume,
            allow_review_candidate=args.allow_review_candidate,
        )
    elif args.command == "verify":
        result = verify_stage1_v3(
            args.input,
            args.model_root,
            args.output,
            args.stage0_freeze_manifest,
            config,
            stage1_code_sha=args.stage1_code_sha,
        )
        if args.report:
            atomic_write_json(args.report, result)
    elif args.command == "preflight":
        result = run_global_preflight(args.input, config)
        if args.report:
            atomic_write_json(args.report, result)
    else:
        result = run_scientific_review(
            args.input, args.output, args.report_dir, config
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("engineering_status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
