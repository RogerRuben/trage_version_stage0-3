"""Order-level LightGBM using only aggregated Stage2 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

STAGE2_SCRIPTS = Path(__file__).resolve().parents[2] / "stage2" / "scripts"
if str(STAGE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE2_SCRIPTS))
from stage2_deep_v3_utils import metric_dict  # noqa: E402


TARGETS = ["lcs", "pmis", "rts"]
FEATURE_SETS = ["rc_mstnet", "lightgbm", "combined"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/tabular_predictor"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(root: Path, targets: Path, split: str) -> pd.DataFrame:
    x = pd.read_parquet(root / f"split={split}" / "order_features.parquet")
    y = pd.read_parquet(targets / f"split={split}" / "order_targets.parquet")
    return x.merge(y, on=["order_id", "fold_id", "split", "date"], validate="one_to_one")


def main() -> None:
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.feature_root / "manifest.json").read_text(encoding="utf-8"))
    data = {split: load(args.feature_root, args.target_root, split) for split in ["train", "validation", "test"]}
    metric_rows = []; prediction_rows = []
    for feature_set in FEATURE_SETS:
        columns = [column for column in manifest["feature_sets"][feature_set] if column in data["train"] and pd.api.types.is_numeric_dtype(data["train"][column])]
        medians = data["train"][columns].median(numeric_only=True)
        x = {split: frame[columns].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0) for split, frame in data.items()}
        for target in TARGETS + ["overall"]:
            raw_column = f"order_{target}_raw" if target != "overall" else None
            tail_column = f"order_{target}_tail" if target != "overall" else "order_overall_high_stress"
            train_mask = data["train"][tail_column].notna()
            if raw_column:
                reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=.04, num_leaves=31, subsample=.85, colsample_bytree=.85, random_state=args.seed, verbosity=-1)
                reg.fit(x["train"].loc[train_mask], data["train"].loc[train_mask, raw_column])
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.04, num_leaves=31, subsample=.85, colsample_bytree=.85, random_state=args.seed, verbosity=-1)
            clf.fit(x["train"].loc[train_mask], data["train"].loc[train_mask, tail_column].astype(int))
            for split in ["validation", "test"]:
                if raw_column:
                    pred_raw = np.clip(reg.predict(x[split]), 0, 1)
                    truth = data[split][raw_column].to_numpy(float)
                else:
                    truth = data[split][[f"order_{name}_raw" for name in TARGETS]].max(axis=1).to_numpy(float)
                    pred_raw = clf.predict_proba(x[split])[:, 1]
                probability = clf.predict_proba(x[split])[:, 1]
                high = data[split][tail_column].to_numpy(bool)
                metrics = metric_dict(truth, pred_raw, probability, high)
                metric_rows.append({"model": "stage3_lightgbm", "feature_set": feature_set, "features": len(columns), "split": split, "target": target.upper(), **metrics})
                prediction_rows.append(pd.DataFrame({"order_id": data[split]["order_id"], "split": split, "target": target.upper(), "feature_set": feature_set, "true_raw": truth, "true_tail": high, "pred_raw": pred_raw, "pred_probability": probability}))
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(args.output_root / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(args.output_root / "predictions.parquet", index=False, compression="zstd")
    test = metrics[metrics["split"].eq("test")]
    (args.output_root / "report.md").write_text("# Stage3 tabular predictor\n\n" + test.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
