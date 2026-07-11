"""Audit Stage2 link-level labels and validity masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - optional dependency
    ks_2samp = None


TARGETS = {
    "LCS": ("target_lcs_pct", "lcs_valid"),
    "IIS": ("target_iis_pct", "iis_valid"),
    "RTS": ("target_rts_pct", "rts_valid"),
    "PMIS": ("target_pmis_pct", "pmis_valid"),
}
SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/label_audit"))
    parser.add_argument("--sample-per-split", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=250_000)
    return parser.parse_args()


def empty_valid_accumulator(group_name: str) -> dict:
    return {"group": group_name, "total": 0, **{target: 0 for target in TARGETS}}


def update_group_counts(store: dict[tuple, dict], frame: pd.DataFrame, group_cols: list[str]) -> None:
    grouped = frame.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = store.setdefault(key, {"total": 0, **{target: 0 for target in TARGETS}})
        row["total"] += len(group)
        for target, (_, mask_col) in TARGETS.items():
            row[target] += int(group[mask_col].fillna(False).sum())


def group_store_to_frame(store: dict[tuple, dict], columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, counts in store.items():
        row = {column: value for column, value in zip(columns, key)}
        row["rows"] = counts["total"]
        for target in TARGETS:
            row[f"{target.lower()}_valid_ratio"] = counts[target] / counts["total"] if counts["total"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def take_sample(existing: list[pd.DataFrame], frame: pd.DataFrame, max_rows: int, seed: int) -> list[pd.DataFrame]:
    if frame.empty:
        return existing
    current = sum(len(item) for item in existing)
    if current < max_rows:
        remaining = max_rows - current
        existing.append(frame.sample(n=min(remaining, len(frame)), random_state=seed) if len(frame) > remaining else frame)
    return existing


def load_split_batches(path: Path, columns: list[str], batch_size: int):
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected = [column for column in columns if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield batch.to_pandas()


def add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    frame["route_position_bin"] = pd.cut(
        pd.to_numeric(frame["position_ratio"], errors="coerce"),
        bins=[-0.001, 0.1, 0.33, 0.66, 0.9, 1.001],
        labels=["pickup_side", "early", "middle", "late", "dropoff_side"],
    )
    if "endpoint_degree" in frame.columns:
        frame["intersection_proxy"] = np.where(
            pd.to_numeric(frame["endpoint_degree"], errors="coerce").fillna(0).ge(3),
            "degree_ge_3",
            "degree_lt_3",
        )
    else:
        frame["intersection_proxy"] = "missing"
    return frame


def distribution_rows(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in samples.items():
        for target, (column, mask_col) in TARGETS.items():
            values = frame.loc[frame[mask_col].fillna(False), column].dropna()
            if values.empty:
                rows.append({"split": split, "target": target, "count": 0})
                continue
            quantiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            row = {
                "split": split,
                "target": target,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for q, value in quantiles.items():
                row[f"q{int(q * 100):02d}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def corr_matrix(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(samples.values(), ignore_index=True)
    columns = [column for column, _ in TARGETS.values()]
    return frame[columns].corr(method="spearman", min_periods=1)


def distribution_comparison(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    train = samples.get("train", pd.DataFrame())
    for split in ["validation", "test"]:
        target_frame = samples.get(split, pd.DataFrame())
        for target, (column, mask_col) in TARGETS.items():
            train_values = train.loc[train[mask_col].fillna(False), column].dropna()
            split_values = target_frame.loc[target_frame[mask_col].fillna(False), column].dropna()
            row = {"target_split": split, "target": target}
            if train_values.empty or split_values.empty:
                row.update({"status": "missing"})
            else:
                row.update({
                    "status": "complete",
                    "train_mean": float(train_values.mean()),
                    "target_mean": float(split_values.mean()),
                    "mean_diff": float(split_values.mean() - train_values.mean()),
                    "train_q90": float(train_values.quantile(0.90)),
                    "target_q90": float(split_values.quantile(0.90)),
                    "q90_diff": float(split_values.quantile(0.90) - train_values.quantile(0.90)),
                })
                if ks_2samp is not None:
                    ks = ks_2samp(train_values.to_numpy(), split_values.to_numpy())
                    row["ks_statistic"] = float(ks.statistic)
                    row["ks_pvalue"] = float(ks.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def decile_stats(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in samples.items():
        for target, (column, mask_col) in TARGETS.items():
            values = frame.loc[frame[mask_col].fillna(False), column].dropna()
            if values.empty:
                continue
            decile = np.minimum((values.rank(pct=True) * 10).astype(int), 9) + 1
            table = pd.DataFrame({"decile": decile, "value": values}).groupby("decile").value.agg(
                ["count", "mean", "min", "max"]
            ).reset_index()
            table["split"] = split
            table["target"] = target
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def tail_behavior(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in samples.items():
        for target, (column, mask_col) in TARGETS.items():
            values = frame.loc[frame[mask_col].fillna(False), column].dropna()
            row = {"split": split, "target": target, "valid_sample_rows": len(values)}
            for cutoff in [0.85, 0.90, 0.95]:
                row[f"share_ge_{int(cutoff * 100)}"] = float(values.ge(cutoff).mean()) if len(values) else np.nan
                row[f"mean_ge_{int(cutoff * 100)}"] = float(values[values.ge(cutoff)].mean()) if values.ge(cutoff).any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def iis_missingness(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in samples.items():
        valid = frame["iis_valid"].fillna(False)
        invalid = ~valid
        rows.append({
            "split": split,
            "rows": len(frame),
            "iis_valid_ratio": float(valid.mean()),
            "iis_missing_ratio": float(invalid.mean()),
            "iis_invalid_target_null_ratio": float(frame.loc[invalid, "target_iis_pct"].isna().mean()) if invalid.any() else np.nan,
            "iis_invalid_target_zero_ratio": float(frame.loc[invalid, "target_iis_pct"].eq(0).mean()) if invalid.any() else np.nan,
            "iis_valid_endpoint_degree_mean": float(pd.to_numeric(frame.loc[valid, "endpoint_degree"], errors="coerce").mean()) if valid.any() else np.nan,
            "iis_missing_endpoint_degree_mean": float(pd.to_numeric(frame.loc[invalid, "endpoint_degree"], errors="coerce").mean()) if invalid.any() else np.nan,
            "iis_valid_intersection_proxy_ge3_share": float(frame.loc[valid, "intersection_proxy"].eq("degree_ge_3").mean()) if valid.any() else np.nan,
            "iis_missing_intersection_proxy_ge3_share": float(frame.loc[invalid, "intersection_proxy"].eq("degree_ge_3").mean()) if invalid.any() else np.nan,
        })
    return pd.DataFrame(rows)


def plot_outputs(samples: dict[str, pd.DataFrame], valid_by_day: pd.DataFrame, corr: pd.DataFrame, deciles: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for target, (column, mask_col) in TARGETS.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for split, frame in samples.items():
            values = frame.loc[frame[mask_col].fillna(False), column].dropna()
            if len(values):
                ax.hist(values, bins=40, alpha=0.35, density=True, label=split)
        ax.set_title(f"{target} distribution")
        ax.set_xlabel(column)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{target.lower()}_distribution.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for target in TARGETS:
        column = f"{target.lower()}_valid_ratio"
        if column in valid_by_day.columns:
            ax.plot(valid_by_day["date"].astype(str), valid_by_day[column], marker="o", label=target)
    ax.set_ylim(0, 1.05)
    ax.set_title("Valid ratio by day")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "valid_ratio_by_day.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Target Spearman correlation")
    fig.tight_layout()
    fig.savefig(figures / "target_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = deciles[deciles.target.eq(target)]
        for split, shown in data.groupby("split"):
            ax.plot(shown.decile, shown["mean"], marker="o", label=split)
        ax.set_title(f"{target} decile means")
        ax.set_xlabel("within-split label decile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{target.lower()}_decile_stats.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures = args.output_root / "figures"

    columns = [
        "date", "time_bin", "peak_offpeak", "road_class", "position_ratio", "endpoint_degree",
        "target_lcs_pct", "target_iis_pct", "target_rts_pct", "target_pmis_pct",
        "lcs_valid", "iis_valid", "rts_valid", "pmis_valid",
    ]

    split_counts = {split: {"total": 0, **{target: 0 for target in TARGETS}} for split in SPLITS}
    by_day: dict[tuple, dict] = {}
    by_time: dict[tuple, dict] = {}
    by_road: dict[tuple, dict] = {}
    by_position: dict[tuple, dict] = {}
    samples: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        path = args.dataset_root / f"{split}.parquet"
        split_samples: list[pd.DataFrame] = []
        for batch_no, frame in enumerate(load_split_batches(path, columns, args.batch_size), start=1):
            frame = add_bins(frame)
            split_counts[split]["total"] += len(frame)
            for target, (_, mask_col) in TARGETS.items():
                split_counts[split][target] += int(frame[mask_col].fillna(False).sum())
            frame["split"] = split
            update_group_counts(by_day, frame, ["split", "date"])
            update_group_counts(by_time, frame, ["split", "peak_offpeak"])
            update_group_counts(by_road, frame, ["split", "road_class"])
            update_group_counts(by_position, frame, ["split", "route_position_bin"])
            sample_cols = columns + ["split", "route_position_bin", "intersection_proxy"]
            split_samples = take_sample(split_samples, frame[sample_cols], args.sample_per_split, seed=batch_no)
        samples[split] = pd.concat(split_samples, ignore_index=True) if split_samples else pd.DataFrame(columns=columns)

    valid_by_target_rows = []
    for split, counts in split_counts.items():
        for target in TARGETS:
            valid_by_target_rows.append({
                "split": split,
                "target": target,
                "rows": counts["total"],
                "valid_rows": counts[target],
                "valid_ratio": counts[target] / counts["total"] if counts["total"] else np.nan,
                "missing_ratio": 1 - counts[target] / counts["total"] if counts["total"] else np.nan,
            })
    valid_by_target = pd.DataFrame(valid_by_target_rows)
    valid_by_day = group_store_to_frame(by_day, ["split", "date"]).sort_values(["split", "date"])
    valid_by_time = group_store_to_frame(by_time, ["split", "peak_offpeak"]).sort_values(["split", "peak_offpeak"])
    valid_by_road = group_store_to_frame(by_road, ["split", "road_class"]).sort_values(["split", "road_class"])
    valid_by_position = group_store_to_frame(by_position, ["split", "route_position_bin"]).sort_values(["split", "route_position_bin"])
    quantiles = distribution_rows(samples)
    corr = corr_matrix(samples)
    comparison = distribution_comparison(samples)
    deciles = decile_stats(samples)
    tails = tail_behavior(samples)
    iis = iis_missingness(samples)

    valid_by_target.to_csv(args.output_root / "label_valid_ratio_by_target.csv", index=False)
    valid_by_day.to_csv(args.output_root / "label_valid_ratio_by_day.csv", index=False)
    valid_by_time.to_csv(args.output_root / "label_valid_ratio_by_time_period.csv", index=False)
    valid_by_road.to_csv(args.output_root / "label_valid_ratio_by_road_class.csv", index=False)
    valid_by_position.to_csv(args.output_root / "label_valid_ratio_by_route_position.csv", index=False)
    quantiles.to_csv(args.output_root / "label_quantile_summary.csv", index=False)
    quantiles.to_csv(args.output_root / "label_distribution_by_split.csv", index=False)
    corr.to_csv(args.output_root / "label_corr_matrix.csv")
    comparison.to_csv(args.output_root / "label_distribution_shift.csv", index=False)
    deciles.to_csv(args.output_root / "label_decile_statistics.csv", index=False)
    tails.to_csv(args.output_root / "label_tail_behavior.csv", index=False)
    iis.to_csv(args.output_root / "iis_missingness_audit.csv", index=False)
    plot_outputs(samples, valid_by_day, corr, deciles, figures)

    summary = {
        "dataset_root": str(args.dataset_root),
        "sample_per_split": args.sample_per_split,
        "valid_ratio_by_target": valid_by_target.to_dict(orient="records"),
        "distribution_shift": comparison.to_dict(orient="records"),
        "iis_missingness": iis.to_dict(orient="records"),
        "outputs": sorted(path.name for path in args.output_root.glob("*.csv")),
    }
    (args.output_root / "label_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
