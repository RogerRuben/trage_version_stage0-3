"""Bootstrap confidence intervals for Stage2 slice metrics.

This is intended for checking whether apparent slice gains (for example peak
RTS improvements) are stable or just probe noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score


TARGETS = {
    "LCS": ("target_lcs", "pred_lcs", "lcs_valid"),
    "IIS": ("target_iis", "pred_iis", "iis_valid"),
    "RTS": ("target_rts", "pred_rts", "rts_valid"),
    "PMIS": ("target_pmis", "pred_pmis", "pmis_valid"),
}

ID_COLUMNS = ["order_id", "driver_id", "date", "link_id", "link_seq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_joined(prediction_file: Path, dataset_file: Path, max_rows: int, seed: int) -> pd.DataFrame:
    pred = pd.read_parquet(prediction_file)
    needed = ID_COLUMNS + ["peak_offpeak", "endpoint_degree", "route_link_count", "position_ratio"]
    available = set(pq.ParquetFile(dataset_file).schema_arrow.names)
    data = pd.read_parquet(dataset_file, columns=[column for column in needed if column in available])
    frame = pred.merge(data, on=[column for column in ID_COLUMNS if column in pred.columns and column in data.columns], how="left")
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    return frame


def slice_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {"all": pd.Series(True, index=frame.index)}
    if "peak_offpeak" in frame.columns:
        masks["peak"] = frame["peak_offpeak"].astype(str).str.lower().eq("peak")
        masks["offpeak"] = ~masks["peak"]
    if "endpoint_degree" in frame.columns:
        threshold = frame["endpoint_degree"].quantile(0.75)
        masks["high_endpoint_degree"] = frame["endpoint_degree"].ge(threshold)
    if "route_link_count" in frame.columns:
        threshold = frame["route_link_count"].quantile(0.75)
        masks["long_route"] = frame["route_link_count"].ge(threshold)
    return masks


def top_lift(y: np.ndarray, pred: np.ndarray, share: float) -> float:
    high = y >= 0.90
    base = high.mean()
    if base <= 0:
        return float("nan")
    n = max(1, int(len(y) * share))
    selected = np.argsort(-pred)[:n]
    return float(high[selected].mean() / base)


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    high = y >= 0.90
    result = {
        "rows": int(len(y)),
        "top10_lift": top_lift(y, pred, 0.10),
        "top5_lift": top_lift(y, pred, 0.05),
    }
    if high.any() and (~high).any():
        result["auc"] = float(roc_auc_score(high, pred))
        result["ap"] = float(average_precision_score(high, pred))
    else:
        result["auc"] = float("nan")
        result["ap"] = float("nan")
    return result


def bootstrap(frame: pd.DataFrame, y_col: str, pred_col: str, n_bootstrap: int, seed: int) -> dict[str, float]:
    valid = frame[[y_col, pred_col]].dropna()
    if len(valid) < 100:
        return {"rows": int(len(valid))}
    y = valid[y_col].to_numpy(dtype=float)
    pred = valid[pred_col].to_numpy(dtype=float)
    point = point_metrics(y, pred)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in point if key != "rows"}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        metrics = point_metrics(y[idx], pred[idx])
        for key in samples:
            samples[key].append(metrics[key])
    result = dict(point)
    for key, values in samples.items():
        arr = np.array(values, dtype=float)
        result[f"{key}_lo"] = float(np.nanquantile(arr, 0.025))
        result[f"{key}_hi"] = float(np.nanquantile(arr, 0.975))
    return result


def main() -> None:
    args = parse_args()
    frame = read_joined(args.prediction_file, args.dataset_file, args.max_rows, args.seed)
    rows = []
    for slice_name, mask in slice_masks(frame).items():
        sliced = frame[mask.fillna(False)]
        for target_name, (target_col, pred_col, valid_col) in TARGETS.items():
            if target_col not in sliced.columns or pred_col not in sliced.columns:
                continue
            target_slice = sliced[sliced[valid_col].fillna(False)] if valid_col in sliced.columns else sliced
            metrics = bootstrap(target_slice, target_col, pred_col, args.n_bootstrap, args.seed)
            metrics.update({"model": args.model_name, "split": args.split, "slice": slice_name, "target": target_name})
            rows.append(metrics)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
