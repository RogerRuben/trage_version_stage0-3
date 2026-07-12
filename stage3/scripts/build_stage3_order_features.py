"""Aggregate Stage2 held-out link/movement predictions into order inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "pmis", "rts"]
SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/order_features"))
    return parser.parse_args()


def weighted(value: pd.Series, weight: pd.Series) -> float:
    mask = value.notna() & weight.notna() & weight.gt(0)
    return float(np.average(value[mask], weights=weight[mask])) if mask.any() else np.nan


def max_probability_run(group: pd.DataFrame, probability: str, threshold: float = 0.5) -> tuple[int, float]:
    best = current = 0
    best_m = current_m = 0.0
    for high, length in zip(group[probability].fillna(0).ge(threshold), group["route_link_length_m"].fillna(0)):
        if high:
            current += 1; current_m += float(length); best = max(best, current); best_m = max(best_m, current_m)
        else:
            current = 0; current_m = 0.0
    return best, best_m


def aggregate_link(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order_id, group in frame.sort_values(["order_id", "route_link_seq"]).groupby("order_id", sort=False):
        row = {"order_id": order_id, "fold_id": int(group["fold_id"].iloc[0]), "split": group["split"].iloc[0], "date": str(group["date"].iloc[0]), "route_length_m": float(group["route_link_length_m"].sum()), "link_count": len(group), "lightgbm_link_coverage": float(group["lightgbm_prediction_available"].mean())}
        for target in TARGETS:
            raw = group[f"{target}_raw_pred"]
            probability = group[f"{target}_tail_prob_calibrated"]
            uncertainty = group[f"{target}_uncertainty"]
            row.update({
                f"rc_{target}_mean": float(raw.mean()), f"rc_{target}_weighted_mean": weighted(raw, group["route_link_length_m"]),
                f"rc_{target}_q90": float(raw.quantile(.90)), f"rc_{target}_q95": float(raw.quantile(.95)), f"rc_{target}_max": float(raw.max()), f"rc_{target}_std": float(raw.std()),
                f"rc_{target}_tail_prob_mean": float(probability.mean()), f"rc_{target}_tail_prob_q90": float(probability.quantile(.90)), f"rc_{target}_tail_prob_max": float(probability.max()), f"rc_{target}_tail_share_50": float(probability.ge(.5).mean()),
                f"rc_{target}_uncertainty_mean": float(uncertainty.mean()), f"rc_{target}_uncertainty_q90": float(uncertainty.quantile(.90)), f"rc_{target}_uncertainty_max": float(uncertainty.max()),
            })
            count, distance = max_probability_run(group, f"{target}_tail_prob_calibrated")
            row[f"rc_{target}_consecutive_high_links"] = count; row[f"rc_{target}_consecutive_high_m"] = distance
            for label, low, high in [("early", 0, 1/3), ("middle", 1/3, 2/3), ("late", 2/3, 1.01)]:
                section = group[group["position_ratio"].ge(low) & group["position_ratio"].lt(high)]
                row[f"rc_{target}_{label}_mean"] = float(section[f"{target}_raw_pred"].mean()) if len(section) else np.nan
            lgbm_raw = f"lgbm_{target}_raw_pred"
            lgbm_prob = f"lgbm_{target}_tail_prob"
            row[f"lgbm_{target}_mean"] = float(group[lgbm_raw].mean()) if lgbm_raw in group else np.nan
            row[f"lgbm_{target}_q90"] = float(group[lgbm_raw].quantile(.90)) if lgbm_raw in group else np.nan
            row[f"lgbm_{target}_max"] = float(group[lgbm_raw].max()) if lgbm_raw in group else np.nan
            row[f"lgbm_{target}_tail_prob_q90"] = float(group[lgbm_prob].quantile(.90)) if lgbm_prob in group else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_movement(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.groupby("order_id", as_index=False).agg(
        movement_count=("movement_seq", "size"),
        iis_applicability_mean=("iis_applicability_prob", "mean"),
        iis_applicability_q90=("iis_applicability_prob", lambda value: value.quantile(.90)),
        iis_predicted_applicable_share=("iis_valid_prediction_flag", "mean"),
        iis_severity_mean=("iis_severity_pred", "mean"),
        iis_severity_q90=("iis_severity_pred", lambda value: value.quantile(.90)),
        iis_severity_max=("iis_severity_pred", "max"),
        iis_tail_prob_mean=("iis_tail_prob", "mean"),
        iis_tail_prob_q90=("iis_tail_prob", lambda value: value.quantile(.90)),
        iis_tail_share_50=("iis_tail_prob", lambda value: value.ge(.5).mean()),
    )
    output["iis_availability"] = True
    return output


def main() -> None:
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"feature_sets": {}, "splits": {}}
    for split in SPLITS:
        link_path = next((args.warehouse_root / "link_predictions" / f"split={split}").glob("*.parquet"))
        movement_path = next((args.warehouse_root / "movement_predictions" / f"split={split}").glob("*.parquet"))
        features = aggregate_link(pd.read_parquet(link_path))
        movement = aggregate_movement(pd.read_parquet(movement_path))
        features = features.merge(movement, on="order_id", how="left", validate="one_to_one")
        features["iis_prediction_available"] = features["movement_count"].notna()
        features["iis_availability"] = features["iis_availability"].fillna(False)
        features["movement_count"] = features["movement_count"].fillna(0)
        features["modality_coverage_score"] = (
            0.75
            + 0.25 * features["iis_prediction_available"].astype(float)
        )
        features["route_prediction_confidence"] = (
            features["modality_coverage_score"]
            - features[["rc_lcs_uncertainty_q90", "rc_pmis_uncertainty_q90", "rc_rts_uncertainty_q90"]].mean(axis=1).fillna(0).clip(0, 1) * 0.25
        ).clip(0, 1)
        split_root = args.output_root / f"split={split}"; split_root.mkdir(parents=True, exist_ok=True)
        features.to_parquet(split_root / "order_features.parquet", index=False, compression="zstd")
        manifest["splits"][split] = {"orders": len(features), "iis_coverage": float(features["iis_prediction_available"].mean()), "lightgbm_order_coverage": float(features["lightgbm_link_coverage"].gt(0).mean())}
    sample = features.columns.tolist()
    manifest["feature_sets"] = {
        "rc_mstnet": [column for column in sample if column.startswith("rc_") or column in {"route_length_m", "link_count"} or column.startswith("iis_")],
        "lightgbm": [column for column in sample if column.startswith("lgbm_") or column in {"route_length_m", "link_count", "lightgbm_link_coverage"}],
        "combined": [column for column in sample if column not in {"order_id", "fold_id", "split", "date"}],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
