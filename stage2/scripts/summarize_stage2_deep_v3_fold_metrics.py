"""Create fair fold-level LightGBM versus RC-MSTNet comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import LINK_TARGETS, metric_dict, order_level_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-prediction-root", type=Path, required=True)
    parser.add_argument("--lightgbm-oof", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/metrics"))
    parser.add_argument("--lightgbm-ablation", default="static_rolling_dynamic_topology_route")
    return parser.parse_args()


def aligned_fold(deep_path: Path, lightgbm: pd.DataFrame, fold: int, target: str, ablation: str) -> pd.DataFrame:
    deep = pd.read_parquet(deep_path)
    deep = deep[deep[f"{target}_valid"]].rename(columns={
        "route_link_id": "planned_link_id", "route_link_seq": "planned_link_seq",
        f"target_{target}_raw": "true_raw_deep", f"target_{target}_tail": "true_tail_deep",
        f"pred_{target}_raw": "deep_raw", f"pred_{target}_tail_prob": "deep_prob",
    })
    lgbm = lightgbm[
        lightgbm["fold"].eq(fold) & lightgbm["target"].eq(target) & lightgbm["ablation"].eq(ablation)
    ].rename(columns={"pred_raw": "lgbm_raw", "pred_tail_probability": "lgbm_prob"})
    keys = ["order_id", "date", "planned_link_id", "planned_link_seq"]
    keep_deep = keys + ["true_raw_deep", "true_tail_deep", "deep_raw", "deep_prob"]
    keep_lgbm = keys + ["true_raw", "true_tail", "lgbm_raw", "lgbm_prob"]
    merged = deep[keep_deep].merge(lgbm[keep_lgbm], on=keys, how="inner", validate="one_to_one")
    merged["true_raw"] = merged["true_raw"].fillna(merged["true_raw_deep"])
    merged["true_tail"] = merged["true_tail"].fillna(merged["true_tail_deep"]).astype(bool)
    return merged


def model_metrics(data: pd.DataFrame, prefix: str, target: str) -> dict:
    result = metric_dict(
        data["true_raw"].to_numpy(float), data[f"{prefix}_raw"].to_numpy(float),
        data[f"{prefix}_prob"].to_numpy(float), data["true_tail"].to_numpy(bool),
    )
    wide = pd.DataFrame({
        "order_id": data["order_id"], f"{target}_valid": True,
        f"target_{target}_raw": data["true_raw"], f"pred_{target}_raw": data[f"{prefix}_raw"],
        f"pred_{target}_tail_prob": data[f"{prefix}_prob"],
    })
    result.update(order_level_metrics(wide, target))
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lightgbm = pd.read_parquet(args.lightgbm_oof)
    rows = []
    for fold_root in sorted(args.deep_prediction_root.glob("fold=*")):
        fold = int(fold_root.name.split("=", 1)[-1])
        for target in LINK_TARGETS:
            aligned = aligned_fold(fold_root / "test_predictions.parquet", lightgbm, fold, target, args.lightgbm_ablation)
            deep_metrics = model_metrics(aligned, "deep", target)
            lgbm_metrics = model_metrics(aligned, "lgbm", target)
            rows.append({"fold": fold, "target": target.upper(), "model": "LightGBM", "common_rows": len(aligned), "common_orders": aligned["order_id"].nunique(), **lgbm_metrics})
            rows.append({"fold": fold, "target": target.upper(), "model": "RC-MSTNet", "common_rows": len(aligned), "common_orders": aligned["order_id"].nunique(), **deep_metrics})
            delta = {key: deep_metrics[key] - lgbm_metrics[key] for key in deep_metrics if isinstance(deep_metrics[key], (int, float, np.integer, np.floating)) and key in lgbm_metrics}
            rows.append({"fold": fold, "target": target.upper(), "model": "RC-MSTNet - LightGBM", "common_rows": len(aligned), "common_orders": aligned["order_id"].nunique(), **delta})
    result = pd.DataFrame(rows)
    result.to_csv(args.output_root / "fold_level_metrics.csv", index=False)
    columns = [
        "fold", "target", "model", "common_rows", "common_orders", "auc", "ap", "spearman", "pearson",
        "mae", "rmse", "lift_top5", "lift_top10", "order_q90_auc", "order_q90_ap", "order_q90_lift_top10",
    ]
    view = result[[column for column in columns if column in result]].copy()
    report = [
        "# Stage2 Deep v3 fold-level fair comparison", "",
        "All model metrics use identical aligned prediction rows within each fold/target. Order tails are empirical top-decile events on the common orders.", "",
        view.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    (args.output_root / "fold_level_comparison.md").write_text("\n".join(report), encoding="utf-8")
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
