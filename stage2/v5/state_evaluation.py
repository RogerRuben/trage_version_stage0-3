"""Same-row v4/v5 evaluation for auxiliary continuous and tail targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2.v4.metrics import binary_metrics
from stage2.v4.prediction_io import merge_chunk_predictions

from .baselines import _paired_bootstrap, continuous_metrics
from .config import load_config


CONTINUOUS = (
    ("crawl_time_share", "crawl_target_valid", "pred_crawl_time_share"),
    ("stop_time_share", "stop_target_valid", "pred_stop_time_share"),
    ("speed_cv_bounded", "speed_cv_target_valid", "pred_speed_cv_bounded"),
    ("acceleration_rms_bounded", "acceleration_rms_target_valid", "pred_acceleration_rms_bounded"),
    ("rts_raw", "rts_target_valid", "pred_rts_raw"),
    ("lcs_raw", "lcs_target_valid", "pred_lcs_raw"),
)
TAILS = (
    ("lcs_tail_event", "lcs_target_valid", "lcs_tail_score"),
    ("rts_tail_event", "rts_target_valid", "rts_tail_score"),
)


def _identity(frame: pd.DataFrame) -> np.ndarray:
    return frame[["order_id", "traversal_id"]].astype(str).to_numpy()


def evaluate_state_targets(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_config(root / "stage2/config/stage2_v5.json")
    date_pairs = [("validation_model", date) for date in config.section("split")["validation_model_dates"]]
    date_pairs += [("calibration", date) for date in config.section("split")["calibration_dates"]]
    seed = int(config.section("runtime")["random_seed"])
    continuous_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for split, date in date_pairs:
        v4 = merge_chunk_predictions(
            root / "stage2/output_v4/predictions/uncalibrated",
            split=split,
            date=date,
        ).sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
        v5 = pd.read_parquet(
            root / "stage2/output_v5/predictions" / f"split={split}" / f"date={date}" / "traversal_predictions.parquet"
        ).sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
        if len(v4) != len(v5) or not np.array_equal(_identity(v4), _identity(v5)):
            raise ValueError(f"v4/v5 traversal identity mismatch on {date}")

        for target, mask_name, prediction_name in CONTINUOUS:
            truth = v5[target].to_numpy(float)
            valid = v5[mask_name].to_numpy(bool) & np.isfinite(truth)
            v4_valid = v4[mask_name].to_numpy(bool)
            if not np.array_equal(valid, v4_valid & np.isfinite(v4[target].to_numpy(float))):
                raise ValueError(f"v4/v5 {target} mask mismatch on {date}")
            if not np.allclose(truth[valid], v4[target].to_numpy(float)[valid]):
                raise ValueError(f"v4/v5 {target} target mismatch on {date}")
            predictions = {
                "rc_mstnet_v4": v4[prediction_name].to_numpy(float),
                "rc_mstnet_v5": v5[prediction_name].to_numpy(float),
            }
            for model, prediction in predictions.items():
                continuous_rows.append({
                    "split": split,
                    "date": date,
                    "target": target,
                    "model": model,
                    **continuous_metrics(np.where(valid, truth, np.nan), prediction),
                })
            compare = pd.DataFrame({
                "order_id": v5["order_id"].astype(str),
                "truth": np.where(valid, truth, np.nan),
                **predictions,
            })
            paired_rows.append({
                "split": split,
                "date": date,
                "target": target,
                "left_model": "rc_mstnet_v5",
                "right_model": "rc_mstnet_v4",
                **_paired_bootstrap(compare, "rc_mstnet_v5", "rc_mstnet_v4", seed=seed),
            })

        for target, mask_name, score_name in TAILS:
            truth = v5[target].to_numpy(float)
            valid = v5[mask_name].to_numpy(bool) & np.isfinite(truth)
            v4_valid = v4[mask_name.replace("_target_valid", "_tail_valid")].to_numpy(bool)
            if not np.array_equal(valid, v4_valid & np.isfinite(v4[target].to_numpy(float))):
                raise ValueError(f"v4/v5 {target} mask mismatch on {date}")
            if not np.allclose(truth[valid], v4[target].to_numpy(float)[valid]):
                raise ValueError(f"v4/v5 {target} target mismatch on {date}")
            for model, frame in (("rc_mstnet_v4", v4), ("rc_mstnet_v5", v5)):
                tail_rows.append({
                    "split": split,
                    "date": date,
                    "target": target,
                    "model": model,
                    **binary_metrics(truth, frame[score_name].to_numpy(float), valid),
                })

    continuous_frame = pd.DataFrame(continuous_rows)
    tail_frame = pd.DataFrame(tail_rows)
    paired_frame = pd.DataFrame(paired_rows)
    report_root = root / "stage2/docs/v5"
    continuous_frame.to_csv(report_root / "state_target_metrics.csv", index=False)
    tail_frame.to_csv(report_root / "state_tail_metrics.csv", index=False)
    paired_frame.to_csv(report_root / "state_paired_error_bootstrap.csv", index=False)

    validation = continuous_frame[continuous_frame["split"].eq("validation_model")].copy()
    validation["weighted_error"] = validation["mae"] * validation["count"]
    totals = validation.groupby(["target", "model"], sort=False, observed=True)[["weighted_error", "count"]].sum()
    totals["mae"] = totals["weighted_error"] / totals["count"]
    wins: dict[str, str] = {}
    for target in validation["target"].unique():
        local = totals.loc[target, "mae"]
        wins[str(target)] = str(local.idxmin())
    summary = {
        "schema_version": "stage2_v5_state_target_evaluation.1",
        "evaluation_rule": "v4 and v5 use identical physical traversal rows and masks",
        "validation_winner_by_continuous_target": wins,
        "continuous_target_count": len(CONTINUOUS),
        "tail_target_count": len(TAILS),
        "new_final_test_consumed": False,
        "status": "PASS",
    }
    (report_root / "stage2_v5_state_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(evaluate_state_targets(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
