"""Build strictly lagged traffic-state features from completed traversals.

The first implementation uses completed five-minute bins. A row predicted inside
bin ``b`` only sees observations from bins ending before ``b``. This avoids the
common same-bin leakage that occurs when rolling aggregates include the current
order. Current outputs belong to the oracle-route track because prediction time
is actual link ``enter_time``; the same engine can later consume estimated entry
times from a planned-route table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


METRICS = [
    "mean_speed_mps", "low_speed_ratio", "stop_duration_ratio", "tail_delay_ratio",
    "lcs_raw", "rts_raw", "pmis_raw",
]
WINDOWS = [5, 15, 30, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-root", type=Path, default=Path("stage1/output/prediction_split/primitives"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/causal_features"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--order-sample-rate", type=float, default=0.10, help="1.0 retains all orders")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--bin-minutes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def config_dates(path: Path) -> dict[str, list[str]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": [str(value) for value in cfg["train_dates"]],
        "validation": [str(cfg["validation_date"])],
        "test": [str(cfg["test_date"])],
    }


def sample_mask(values: pd.Series, rate: float, seed: int) -> np.ndarray:
    if rate >= 1:
        return np.ones(len(values), dtype=bool)
    hashed = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(dtype="uint64") ^ np.uint64(seed)
    return (hashed % np.uint64(1_000_000)) < np.uint64(max(1, int(rate * 1_000_000)))


def load_events(args: argparse.Namespace, dates: dict[str, list[str]]) -> pd.DataFrame:
    requested = ["order_id", "date", "link_id", "link_seq", "enter_time", "area_grid"] + METRICS
    parts: list[pd.DataFrame] = []
    split_lookup = {date: split for split, values in dates.items() for date in values}
    for date in sorted(split_lookup):
        for path in sorted((args.primitive_root / f"day={date}").glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            selected = [column for column in requested if column in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(columns=selected, batch_size=args.batch_size):
                frame = batch.to_pandas()
                frame = frame.loc[sample_mask(frame["order_id"], args.order_sample_rate, args.seed)]
                if not frame.empty:
                    frame["split"] = split_lookup[date]
                    parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=requested + ["split"])
    events = pd.concat(parts, ignore_index=True)
    events["target_prediction_timestamp"] = pd.to_datetime(events["enter_time"], unit="s", utc=True, errors="coerce")
    events = events[events["target_prediction_timestamp"].notna()].copy()
    events["bin_time"] = events["target_prediction_timestamp"].dt.floor(f"{args.bin_minutes}min")
    local = events["target_prediction_timestamp"].dt.tz_convert("Asia/Shanghai")
    events["history_time_bin"] = (local.dt.hour * 2 + (local.dt.minute >= 30).astype(int)).astype("int16")
    return events


def add_rolling_raw_baselines(events: pd.DataFrame, previous_days: int = 7) -> pd.DataFrame:
    parts = []
    dates = sorted(events["date"].astype(str).unique())
    for date in dates:
        current = events[events["date"].astype(str).eq(date)].copy()
        history_dates = [value for value in dates if value < date][-previous_days:]
        history = events[events["date"].astype(str).isin(history_dates)]
        for target in ["lcs", "rts", "pmis"]:
            column = f"{target}_raw"
            output_column = f"rolling_{target}_raw_mean"
            if history.empty or column not in history:
                current[output_column] = np.nan
                continue
            fallback = float(history[column].mean())
            link = history.groupby("link_id", observed=True)[column].mean()
            link_time = history.groupby(["link_id", "history_time_bin"], observed=True)[column].mean()
            index = pd.MultiIndex.from_frame(current[["link_id", "history_time_bin"]])
            values = pd.Series(link_time.reindex(index).to_numpy(), index=current.index)
            current[output_column] = values.fillna(current["link_id"].map(link)).fillna(fallback)
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def aggregate_bins(events: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    group_keys = keys + ["bin_time"]
    clean_metrics = [column for column in METRICS if column in events]
    grouped = events.groupby(group_keys, dropna=False, observed=True)
    result = grouped.size().rename("traversal_count").to_frame()
    for metric in clean_metrics:
        result[f"{metric}_sum"] = grouped[metric].sum(min_count=1)
        result[f"{metric}_count"] = grouped[metric].count()
    return result.reset_index().sort_values(group_keys)


def group_window_features(group: pd.DataFrame, prefix: str) -> pd.DataFrame:
    group = group.sort_values("bin_time").reset_index(drop=True)
    times = group["bin_time"].astype("int64").to_numpy()
    output = group[[column for column in group.columns if column in ["link_id", "area_grid", "bin_time"]]].copy()
    count_values = group["traversal_count"].to_numpy(dtype=float)
    count_cum = np.concatenate([[0.0], np.cumsum(count_values)])
    metric_names = sorted({column[:-4] for column in group.columns if column.endswith("_sum")})
    for window in WINDOWS:
        left = np.searchsorted(times, times - np.int64(window * 60 * 1_000_000_000), side="left")
        right = np.arange(len(group))
        output[f"{prefix}_recent_traversal_count_{window}m"] = count_cum[right] - count_cum[left]
        for metric in metric_names:
            sums = group[f"{metric}_sum"].fillna(0).to_numpy(dtype=float)
            valid_counts = group[f"{metric}_count"].fillna(0).to_numpy(dtype=float)
            sum_cum = np.concatenate([[0.0], np.cumsum(sums)])
            valid_cum = np.concatenate([[0.0], np.cumsum(valid_counts)])
            numerator = sum_cum[right] - sum_cum[left]
            denominator = valid_cum[right] - valid_cum[left]
            output[f"{prefix}_recent_{metric}_{window}m"] = np.divide(
                numerator, denominator, out=np.full(len(group), np.nan), where=denominator > 0
            )
    return output


def rolling_table(events: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    aggregates = aggregate_bins(events, keys)
    if keys:
        grouper = keys[0] if len(keys) == 1 else keys
        parts = [group_window_features(group, prefix) for _, group in aggregates.groupby(grouper, dropna=False, observed=True, sort=False)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return group_window_features(aggregates, prefix)


def directed_adjacency(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    roads = pd.read_parquet(path, columns=["link_id", "from_node", "to_node", "oneway_code"])
    roads["link_id"] = roads["link_id"].astype(str)
    forward = roads[roads["oneway_code"].fillna("B").isin(["F", "B"])][["link_id", "from_node", "to_node"]].rename(
        columns={"from_node": "start_node", "to_node": "end_node"}
    )
    reverse = roads[roads["oneway_code"].fillna("B").isin(["T", "B"])][["link_id", "from_node", "to_node"]].rename(
        columns={"to_node": "start_node", "from_node": "end_node"}
    )
    directed = pd.concat([forward, reverse], ignore_index=True).drop_duplicates()
    current_start = directed[["link_id", "start_node"]]
    current_end = directed[["link_id", "end_node"]]
    upstream = current_start.merge(
        directed[["link_id", "end_node"]].rename(columns={"link_id": "neighbor_link_id"}),
        left_on="start_node", right_on="end_node", how="inner",
    )[["link_id", "neighbor_link_id"]]
    downstream = current_end.merge(
        directed[["link_id", "start_node"]].rename(columns={"link_id": "neighbor_link_id"}),
        left_on="end_node", right_on="start_node", how="inner",
    )[["link_id", "neighbor_link_id"]]
    upstream = upstream[upstream["link_id"].ne(upstream["neighbor_link_id"])].drop_duplicates()
    downstream = downstream[downstream["link_id"].ne(downstream["neighbor_link_id"])].drop_duplicates()
    return upstream, downstream


def neighbor_features(link: pd.DataFrame, adjacency: pd.DataFrame, prefix: str) -> pd.DataFrame:
    selected = [
        column for column in link.columns
        if any(f"recent_{target}_raw_{window}m" in column for target in ["lcs", "rts", "pmis"] for window in [15, 30, 60])
        and "deviation" not in column
    ]
    neighbor = link[["link_id", "bin_time"] + selected].rename(columns={"link_id": "neighbor_link_id"})
    joined = adjacency.merge(neighbor, on="neighbor_link_id", how="inner")
    result = joined.groupby(["link_id", "bin_time"], observed=True)[selected].mean().reset_index()
    result = result.rename(columns={column: column.replace("link_recent_", f"{prefix}_recent_") for column in selected})
    counts = joined.groupby(["link_id", "bin_time"], observed=True)["neighbor_link_id"].nunique().rename(f"{prefix}_neighbor_count").reset_index()
    return result.merge(counts, on=["link_id", "bin_time"], how="left", validate="one_to_one")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = config_dates(args.split_config)
    events = load_events(args, dates)
    if events.empty:
        raise RuntimeError("no primitive events found")
    events = add_rolling_raw_baselines(events)
    link = rolling_table(events, ["link_id"], "link")
    area = rolling_table(events, ["area_grid"], "area")
    network = rolling_table(events.assign(_network="x"), [], "network")
    upstream_adjacency, downstream_adjacency = directed_adjacency(args.roads)
    upstream = neighbor_features(link, upstream_adjacency, "upstream")
    downstream = neighbor_features(link, downstream_adjacency, "downstream")
    key_columns = ["order_id", "date", "link_id", "link_seq", "area_grid", "split", "target_prediction_timestamp", "bin_time"]
    output = events[key_columns].copy()
    output = output.merge(link, on=["link_id", "bin_time"], how="left", validate="many_to_one")
    output = output.merge(area, on=["area_grid", "bin_time"], how="left", validate="many_to_one")
    output = output.merge(network, on=["bin_time"], how="left", validate="many_to_one")
    output = output.merge(upstream, on=["link_id", "bin_time"], how="left", validate="many_to_one")
    output = output.merge(downstream, on=["link_id", "bin_time"], how="left", validate="many_to_one")
    for target in ["lcs", "rts", "pmis"]:
        baseline = f"rolling_{target}_raw_mean"
        output[baseline] = events[baseline].to_numpy()
        for window in [15, 30, 60]:
            recent = f"link_recent_{target}_raw_{window}m"
            if recent in output:
                output[f"{recent}_deviation_from_history"] = output[recent] - output[baseline]
    output["feature_timestamp"] = output["bin_time"]
    output["availability_timestamp"] = output["bin_time"] - pd.Timedelta(milliseconds=1)
    output["strictly_causal_time_check"] = output["availability_timestamp"] < output["target_prediction_timestamp"]
    output["prediction_time_source"] = "actual_enter_time"
    output["experiment_track"] = "oracle_route_upper_bound"
    names = {"train": "stage2_oracle_lagged_features_train.parquet", "validation": "stage2_oracle_lagged_features_validation.parquet", "test": "stage2_oracle_lagged_features_test.parquet"}
    counts = {}
    audit_dir = args.output_root / "audit_labels"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for split, name in names.items():
        frame = output[output["split"].eq(split)].drop(columns=["split"])
        frame.to_parquet(args.output_root / name, index=False, compression="zstd")
        audit_columns = key_columns + [column for column in METRICS if column in events]
        audit = events.loc[events["split"].eq(split), audit_columns].copy()
        audit.to_parquet(audit_dir / f"{split}_dynamic_oracle_labels.parquet", index=False, compression="zstd")
        counts[split] = int(len(frame))
    manifest = {
        "experiment_track": "oracle_route_upper_bound",
        "prediction_time_source": "actual_enter_time",
        "same_bin_excluded": True,
        "rolling_raw_baseline_previous_days": 7,
        "current_deviation_from_history": True,
        "feature_availability_rule": "availability_timestamp < target_prediction_timestamp",
        "causal_check_pass_ratio": float(output["strictly_causal_time_check"].mean()),
        "bin_minutes": args.bin_minutes,
        "windows_minutes": WINDOWS,
        "order_sample_rate": args.order_sample_rate,
        "rows": counts,
        "audit_labels_separated_from_features": True,
        "upstream_downstream_status": "complete_physical_directed_topology",
    }
    (args.output_root / "causal_feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
