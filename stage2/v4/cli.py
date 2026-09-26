"""Command line entry point for Stage 2 v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .contracts import Stage2V4ContractError
from .io import atomic_write_json


DEFAULT_STAGE1_RELEASE = Path("stage1/docs/stage1_v3_release_manifest.json")
DEFAULT_STAGE1_INPUT = Path("stage1/input_v1")
DEFAULT_STAGE1_MODELS = Path("stage1/models/stage1_v3_final")


def _add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing incompatible output",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 v4 decision-time route-conditioned modelling pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="bind the frozen Stage 1 release and audit route/traversal alignment",
    )
    _add_runtime_flags(preflight)
    preflight.add_argument("--stage1-release", type=Path, required=True)
    preflight.add_argument("--stage1-output", type=Path, required=True)
    preflight.add_argument("--stage1-input", type=Path, required=True)
    preflight.add_argument(
        "--stage1-models",
        type=Path,
        default=Path("stage1/models/stage1_v3_final"),
    )
    preflight.add_argument(
        "--alignment-output",
        type=Path,
        default=Path(
            "stage2/output_v4/preflight/traversal_route_alignment.parquet"
        ),
    )
    preflight.add_argument("--report", type=Path, required=True)

    history = subparsers.add_parser(
        "build-history",
        help="stream Stage 1 traversal labels into timestamped daily history events",
    )
    _add_runtime_flags(history)
    history.add_argument("--stage1-output", type=Path, required=True)
    history.add_argument("--output", type=Path, required=True)
    history.add_argument("--stage1-release", type=Path, default=DEFAULT_STAGE1_RELEASE)
    history.add_argument("--stage1-input", type=Path, default=DEFAULT_STAGE1_INPUT)
    history.add_argument("--stage1-models", type=Path, default=DEFAULT_STAGE1_MODELS)

    dataset = subparsers.add_parser(
        "build-dataset",
        help="build daily revealed-route and isolated oracle-timing products",
    )
    _add_runtime_flags(dataset)
    dataset.add_argument("--stage1-output", type=Path, required=True)
    dataset.add_argument("--stage1-input", type=Path, required=True)
    dataset.add_argument("--history-root", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument(
        "--tracks",
        required=True,
        help="comma-separated tracks: revealed_route_proxy,oracle_timing",
    )
    dataset.add_argument("--stage1-release", type=Path, default=DEFAULT_STAGE1_RELEASE)
    dataset.add_argument("--stage1-models", type=Path, default=DEFAULT_STAGE1_MODELS)

    baselines = subparsers.add_parser(
        "train-baselines",
        help="fit causal profile/statistical/tree baselines on Train only",
    )
    _add_runtime_flags(baselines)
    baselines.add_argument("--dataset", type=Path, required=True)
    baselines.add_argument("--output", type=Path, required=True)

    shards = subparsers.add_parser(
        "build-shards",
        help="fit Train-only feature artifacts and build continuous route chunks",
    )
    _add_runtime_flags(shards)
    shards.add_argument("--dataset", type=Path, required=True)
    shards.add_argument("--output", type=Path, required=True)

    deep = subparsers.add_parser(
        "train-deep",
        help="train RC-MSTNet v4 with Validation-model early stopping",
    )
    _add_runtime_flags(deep)
    deep.add_argument("--tensor-root", type=Path, required=True)
    deep.add_argument("--output", type=Path, required=True)
    deep.add_argument("--prediction-root", type=Path, required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="fit tail calibration and conformal intervals on 20161027 only",
    )
    _add_runtime_flags(calibrate)
    calibrate.add_argument("--model-root", type=Path, required=True)
    calibrate.add_argument("--calibration-date", required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run the frozen model once on held-out 20161031 and evaluate",
    )
    _add_runtime_flags(evaluate)
    evaluate.add_argument("--test-date", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("stage2/output_v4/tensor_shards"),
    )
    evaluate.add_argument(
        "--model-root",
        type=Path,
        default=Path("stage2/output_v4/models/rc_mstnet_v4"),
    )
    evaluate.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("stage2/output_v4/models/calibration"),
    )
    evaluate.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("stage2/output_v4/predictions/uncalibrated"),
    )
    evaluate.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("stage2/output_v4/route_conditioned_dataset"),
    )

    verify = subparsers.add_parser(
        "verify",
        help="verify every Stage 2 v4 product and temporal contract",
    )
    _add_runtime_flags(verify)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="freeze a verified Stage 2 v4 release manifest",
    )
    _add_runtime_flags(freeze)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument(
        "--verification",
        type=Path,
        default=Path("stage2/docs/v4/stage2_v4_final_verification.json"),
    )
    freeze.add_argument(
        "--release",
        type=Path,
        default=Path("stage2/docs/v4/stage2_v4_release_manifest.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        from .preflight import run_preflight

        if args.resume and args.report.is_file() and args.alignment_output.is_file():
            existing = json.loads(args.report.read_text(encoding="utf-8"))
            if (
                existing.get("engineering_status") == "PASS"
                and existing.get("stage2_config_sha256") == config.digest
            ):
                print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
                return 0
        if (args.report.exists() or args.alignment_output.exists()) and not args.force:
            raise Stage2V4ContractError(
                "preflight output already exists; use --resume or --force explicitly"
            )
        result = run_preflight(
            config,
            stage1_release=args.stage1_release,
            stage1_output=args.stage1_output,
            stage1_input=args.stage1_input,
            stage1_models=args.stage1_models,
            alignment_output=args.alignment_output,
        )
        atomic_write_json(args.report, result)
    elif args.command == "build-history":
        from .history_store import build_history_store
        from .release import bind_stage1_release

        bind_stage1_release(
            args.stage1_release,
            args.stage1_output,
            args.stage1_input,
            args.stage1_models,
            config,
        )
        result = build_history_store(
            args.stage1_output,
            args.output,
            config,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "build-dataset":
        from .dataset_builder import build_route_conditioned_dataset
        from .release import bind_stage1_release

        bind_stage1_release(
            args.stage1_release,
            args.stage1_output,
            args.stage1_input,
            args.stage1_models,
            config,
        )
        tracks = tuple(
            item.strip() for item in args.tracks.split(",") if item.strip()
        )
        result = build_route_conditioned_dataset(
            args.stage1_output,
            args.stage1_input,
            args.history_root,
            args.output,
            config,
            tracks=tracks,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "train-baselines":
        from .models.baselines import train_baselines

        result = train_baselines(
            args.dataset,
            args.output,
            config,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "build-shards":
        from .models.datasets import build_tensor_shards

        result = build_tensor_shards(
            args.dataset,
            args.output,
            config,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "train-deep":
        from .models.trainer import train_deep_model

        result = train_deep_model(
            args.config,
            args.tensor_root,
            args.output,
            args.prediction_root,
            config,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "calibrate":
        from .calibration import calibrate_predictions

        result = calibrate_predictions(
            args.model_root,
            args.calibration_date,
            args.output,
            config,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_test

        result = evaluate_test(
            args.config,
            args.test_date,
            args.output,
            config,
            tensor_root=args.tensor_root,
            model_root=args.model_root,
            calibration_root=args.calibration_root,
            prediction_root=args.prediction_root,
            dataset_root=args.dataset_root,
            resume=args.resume,
            force=args.force,
        )
    elif args.command == "verify":
        from .verify import verify_stage2_v4

        if args.report.exists() and not (args.resume or args.force):
            raise Stage2V4ContractError(
                "verification report exists; use --resume or --force"
            )
        if args.resume and args.report.is_file():
            existing = json.loads(args.report.read_text(encoding="utf-8"))
            if (
                existing.get("engineering_status") == "PASS"
                and existing.get("stage2_config_sha256") == config.digest
            ):
                result = existing
            else:
                result = verify_stage2_v4(args.output_root, config)
                atomic_write_json(args.report, result)
        else:
            result = verify_stage2_v4(args.output_root, config)
            atomic_write_json(args.report, result)
    elif args.command == "freeze":
        from .verify import freeze_stage2_v4

        if args.release.exists() and not (args.resume or args.force):
            raise Stage2V4ContractError("release manifest exists; use --resume or --force")
        if args.resume and args.release.is_file():
            result = json.loads(args.release.read_text(encoding="utf-8"))
        else:
            result = freeze_stage2_v4(
                args.output_root,
                args.verification,
                args.release,
                config,
            )
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0
        if result.get("engineering_status") in {"PASS", "ENGINEERING_PASS"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
