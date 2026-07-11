"""Train a diagnostic hybrid fusion model from tabular features plus deep scores.

This script is intentionally conservative: by default it requires train-split
deep predictions. For publication-grade fusion, those train predictions should
come from OOF / rolling models rather than in-sample deep predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_model_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    ID_COLUMNS,
    TARGETS,
    TARGET_ORDER,
    evaluate_predictions,
    prepare_tabular_features,
    safe_json_float,
    unique_existing_columns,
)


PRED_COLUMNS = ["pred_lcs", "pred_iis", "pred_rts", "pred_pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--deep-prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines/hybrid_fusion"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions_hybrid"))
    parser.add_argument("--prediction-prefix", default="", help="optional filename prefix, e.g. dual_graph_")
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-in-sample-train-predictions", action="store_true", help="metadata flag only; train prediction file is still required")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def prediction_file(root: Path, prefix: str, split: str) -> Path:
    candidates = [
        root / f"{prefix}{split}.parquet",
        root / f"{prefix}_{split}.parquet",
        root / f"{split}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_split(dataset_root: Path, pred_root: Path, prefix: str, split: str) -> pd.DataFrame:
    dataset_path = dataset_root / f"{split}.parquet"
    pred_path = prediction_file(pred_root, prefix, split)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    data_cols = unique_existing_columns(dataset_path, ID_COLUMNS + FEATURE_COLUMNS + [column for triple in TARGETS.values() for column in triple])
    data = pd.read_parquet(dataset_path, columns=data_cols)
    pred = pd.read_parquet(pred_path)
    keep_pred = [column for column in ID_COLUMNS + PRED_COLUMNS if column in pred.columns]
    return data.merge(pred[keep_pred], on=[column for column in ID_COLUMNS if column in data.columns and column in pred.columns], how="left")


def main() -> None:
    args = parse_args()
    train_pred_path = prediction_file(args.deep_prediction_root, args.prediction_prefix, "train")
    if not train_pred_path.exists():
        raise SystemExit(
            "No train deep prediction file found. Hybrid fusion should use OOF/rolling train predictions; "
            "produce train-split deep scores first. Use --allow-in-sample-train-predictions only to mark diagnostic, non-publication runs."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)

    train = read_split(args.dataset_root, args.deep_prediction_root, args.prediction_prefix, "train")
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=args.seed)
    validation = read_split(args.dataset_root, args.deep_prediction_root, args.prediction_prefix, "validation")
    test = read_split(args.dataset_root, args.deep_prediction_root, args.prediction_prefix, "test")

    features = FEATURE_COLUMNS + [column for column in PRED_COLUMNS if column in train.columns]
    metrics_rows = []
    manifest = {
        "model": "stage2_hybrid_fusion_lgbm",
        "deep_prediction_root": str(args.deep_prediction_root),
        "prediction_prefix": args.prediction_prefix,
        "max_train_rows": args.max_train_rows,
        "features": features,
        "warning": "Use OOF / rolling train deep predictions for publication-grade fusion.",
        "allow_in_sample_train_predictions": args.allow_in_sample_train_predictions,
        "targets": {},
    }
    predictions = {"validation": validation[[column for column in ID_COLUMNS if column in validation.columns]].copy(),
                   "test": test[[column for column in ID_COLUMNS if column in test.columns]].copy()}

    for target_name in TARGET_ORDER:
        target_col, mask_col, high_col = TARGETS[target_name]
        valid_train = train[mask_col].fillna(False) & train[target_col].notna()
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=80,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_samples=120,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            prepare_tabular_features(train.loc[valid_train], features),
            train.loc[valid_train, target_col].astype(float).clip(0, 1),
        )
        target_metrics = {"train_rows": int(valid_train.sum()), "splits": {}}
        for split, frame in [("validation", validation), ("test", test)]:
            pred = np.clip(model.predict(prepare_tabular_features(frame, features)), 0, 1).astype("float32")
            valid = frame[mask_col].fillna(False) & frame[target_col].notna()
            metrics = evaluate_predictions(
                frame.loc[valid, target_col].to_numpy(dtype=float),
                pred[valid.to_numpy()],
                frame.loc[valid, high_col].to_numpy(dtype=bool),
            )
            metrics.update({"target": target_name, "split": split})
            metrics_rows.append(metrics)
            target_metrics["splits"][split] = metrics
            predictions[split][f"pred_{target_name.lower()}"] = pred
            predictions[split][f"target_{target_name.lower()}"] = frame[target_col].to_numpy()
            predictions[split][f"{target_name.lower()}_valid"] = frame[mask_col].to_numpy()
        manifest["targets"][target_name] = target_metrics

    pd.DataFrame(metrics_rows).to_csv(args.output_root / "hybrid_fusion_metrics_by_target.csv", index=False)
    (args.output_root / "hybrid_fusion_metrics.json").write_text(
        json.dumps(safe_json_float(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split, frame in predictions.items():
        frame.to_parquet(args.prediction_root / f"hybrid_fusion_{split}.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
