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
    parser.add_argument("--fold", type=int, default=None, help="Optional Stage3 rolling fold id.")
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
    fold_column = "fold_id_stage3" if "fold_id_stage3" in frame.columns else "fold_id"
    split_column = "stage3_split" if "stage3_split" in frame.columns else "split"
    base = frame.groupby("order_id", as_index=False).agg(
        fold_id=(fold_column, "first"),
        split=(split_column, "first"),
        date=("date", "first"),
        route_length_m=("route_link_length_m", "sum"),
        link_count=("route_link_seq", "size"),
        lightgbm_link_coverage=("lightgbm_prediction_available", "mean"),
    )
    for target in TARGETS:
        raw_col = f"{target}_raw_pred"
        prob_col = f"{target}_tail_prob_calibrated"
        unc_col = f"{target}_uncertainty"
        grouped = frame.groupby("order_id", sort=False)
        agg = grouped[raw_col].agg(
            **{
                f"rc_{target}_mean": "mean",
                f"rc_{target}_q90": lambda value: value.quantile(.90),
                f"rc_{target}_q95": lambda value: value.quantile(.95),
                f"rc_{target}_max": "max",
                f"rc_{target}_std": "std",
            }
        ).reset_index()
        weighted_frame = frame.assign(_wx=frame[raw_col] * frame["route_link_length_m"].fillna(0))
        weighted_frame = weighted_frame.groupby("order_id", as_index=False).agg(_wx=("_wx", "sum"), _w=("route_link_length_m", "sum"))
        weighted_frame[f"rc_{target}_weighted_mean"] = weighted_frame["_wx"] / weighted_frame["_w"].replace(0, np.nan)
        prob = grouped[prob_col].agg(
            **{
                f"rc_{target}_tail_prob_mean": "mean",
                f"rc_{target}_tail_prob_q90": lambda value: value.quantile(.90),
                f"rc_{target}_tail_prob_max": "max",
                f"rc_{target}_tail_share_50": lambda value: value.ge(.5).mean(),
            }
        ).reset_index()
        unc = grouped[unc_col].agg(
            **{
                f"rc_{target}_uncertainty_mean": "mean",
                f"rc_{target}_uncertainty_q90": lambda value: value.quantile(.90),
                f"rc_{target}_uncertainty_max": "max",
            }
        ).reset_index()
        agg = agg.merge(weighted_frame[["order_id", f"rc_{target}_weighted_mean"]], on="order_id", how="left")
        agg = agg.merge(prob, on="order_id", how="left").merge(unc, on="order_id", how="left")
        for label, low, high in [("early", 0, 1/3), ("middle", 1/3, 2/3), ("late", 2/3, 1.01)]:
            section = frame[frame["position_ratio"].ge(low) & frame["position_ratio"].lt(high)]
            section_mean = section.groupby("order_id", sort=False)[raw_col].mean().rename(f"rc_{target}_{label}_mean").reset_index()
            agg = agg.merge(section_mean, on="order_id", how="left")
        agg[f"rc_{target}_consecutive_high_links"] = np.nan
        agg[f"rc_{target}_consecutive_high_m"] = np.nan
        lgbm_raw = f"lgbm_{target}_raw_pred"
        lgbm_prob = f"lgbm_{target}_tail_prob"
        if lgbm_raw in frame:
            lgbm = grouped[lgbm_raw].agg(
                **{f"lgbm_{target}_mean": "mean", f"lgbm_{target}_q90": lambda value: value.quantile(.90), f"lgbm_{target}_max": "max"}
            ).reset_index()
            agg = agg.merge(lgbm, on="order_id", how="left")
        else:
            agg[f"lgbm_{target}_mean"] = np.nan
            agg[f"lgbm_{target}_q90"] = np.nan
            agg[f"lgbm_{target}_max"] = np.nan
        if lgbm_prob in frame:
            lgbm_prob_frame = grouped[lgbm_prob].quantile(.90).rename(f"lgbm_{target}_tail_prob_q90").reset_index()
            agg = agg.merge(lgbm_prob_frame, on="order_id", how="left")
        else:
            agg[f"lgbm_{target}_tail_prob_q90"] = np.nan
        base = base.merge(agg, on="order_id", how="left")
    return base


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
        if args.fold is None:
            link_base = args.warehouse_root / "link_predictions" / f"split={split}"
            movement_base = args.warehouse_root / "movement_predictions" / f"split={split}"
        else:
            link_base = args.warehouse_root / "link_predictions" / f"fold={args.fold}" / f"split={split}"
            movement_base = args.warehouse_root / "movement_predictions" / f"fold={args.fold}" / f"split={split}"
        link_paths = sorted(link_base.glob("*.parquet"))
        movement_paths = sorted(movement_base.glob("*.parquet"))
        if not link_paths:
            raise FileNotFoundError(f"No link predictions under {link_base}")
        if not movement_paths:
            raise FileNotFoundError(f"No movement predictions under {movement_base}")
        features = aggregate_link(pd.concat([pd.read_parquet(path) for path in link_paths], ignore_index=True))
        movement = aggregate_movement(pd.concat([pd.read_parquet(path) for path in movement_paths], ignore_index=True))
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
