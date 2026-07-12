"""Transparent Stage3 order aggregation baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE2_SCRIPTS = Path(__file__).resolve().parents[2] / "stage2" / "scripts"
if str(STAGE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE2_SCRIPTS))
from stage2_deep_v3_utils import metric_dict  # noqa: E402


TARGETS = ["lcs", "pmis", "rts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/rule_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []; predictions = []
    for split in ["validation", "test"]:
        feature = pd.read_parquet(args.feature_root / f"split={split}" / "order_features.parquet")
        target = pd.read_parquet(args.target_root / f"split={split}" / "order_targets.parquet")
        data = feature.merge(target, on=["order_id", "fold_id", "split", "date"], validate="one_to_one")
        overall_probabilities = []
        for name in TARGETS:
            pred_raw = data[f"rc_{name}_q90"].clip(0, 1).to_numpy(float)
            pred_prob = data[f"rc_{name}_tail_prob_q90"].clip(0, 1).to_numpy(float)
            truth = data[f"order_{name}_raw"].to_numpy(float)
            high = data[f"order_{name}_tail"].to_numpy(bool)
            metrics = metric_dict(truth, pred_raw, pred_prob, high)
            rows.append({"model": "rule_q90", "split": split, "target": name.upper(), **metrics})
            predictions.append(pd.DataFrame({"order_id": data["order_id"], "split": split, "target": name.upper(), "true_raw": truth, "true_tail": high, "pred_raw": pred_raw, "pred_probability": pred_prob}))
            overall_probabilities.append(pred_prob)
        overall_prob = np.max(np.stack(overall_probabilities), axis=0)
        overall_high = data["order_overall_high_stress"].to_numpy(bool)
        overall_raw = data[[f"order_{name}_raw" for name in TARGETS]].max(axis=1).to_numpy(float)
        metrics = metric_dict(overall_raw, overall_prob, overall_prob, overall_high)
        rows.append({"model": "rule_q90", "split": split, "target": "OVERALL", **metrics})
        predictions.append(pd.DataFrame({"order_id": data["order_id"], "split": split, "target": "OVERALL", "true_raw": overall_raw, "true_tail": overall_high, "pred_raw": overall_prob, "pred_probability": overall_prob}))
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output_root / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(args.output_root / "predictions.parquet", index=False, compression="zstd")
    (args.output_root / "report.md").write_text("# Stage3 rule baseline\n\n" + metrics.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
