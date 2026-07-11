"""Utilities for Stage2 Deep v3 route-conditioned experiments."""

from __future__ import annotations

import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import Dataset, Sampler


LINK_TARGETS = ["lcs", "pmis", "rts"]
WINDOWS = ["5m", "15m", "30m", "60m"]
DYNAMIC_METRICS = [
    "traversal_count",
    "lcs_raw",
    "pmis_raw",
    "rts_raw",
    "low_speed_ratio",
    "mean_speed_mps",
    "stop_duration_ratio",
    "tail_delay_ratio",
]
CATEGORICAL_COLUMNS = ["route_link_id", "road_class", "area_grid", "prediction_time_bin"]
ID_COLUMNS = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"]
FORBIDDEN_SUBSTRINGS = [
    "actual_link_entry_time",
    "actual_link_exit_time",
    "travel_time_sec",
    "mean_speed_mps_current_order",
    "low_speed_time_sec",
    "low_speed_ratio_on_poi_link",
    "stop_time_on_poi_link",
    "delay_on_poi_link",
]


def load_fold_config(path: Path) -> list[dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return config["folds"]


def existing_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def unique_existing(path: Path, columns: Iterable[str]) -> list[str]:
    available = set(existing_columns(path))
    result = []
    for column in columns:
        if column in available and column not in result:
            result.append(column)
    return result


def default_static_numeric_columns(columns: Iterable[str]) -> list[str]:
    candidates = [
        "prediction_hour",
        "prediction_weekday",
        "prediction_is_weekend",
        "position_ratio",
        "distance_to_destination_ratio",
        "route_link_seq",
        "route_link_count",
        "route_link_length_m",
        "estimated_link_travel_time_sec",
        "endpoint_degree",
        "link_fragmentation",
        "minor_road",
        "activity_intensity_index",
        "upstream_neighbor_count",
        "downstream_neighbor_count",
        "rolling_lcs_raw_mean",
        "rolling_lcs_raw_std",
        "rolling_lcs_history_count",
        "rolling_pmis_raw_mean",
        "rolling_pmis_raw_std",
        "rolling_pmis_history_count",
        "rolling_rts_raw_mean",
        "rolling_rts_raw_std",
        "rolling_rts_history_count",
    ]
    candidates += [column for column in columns if column.startswith("poi_density_100m_")]
    return [column for column in candidates if column in columns and not any(bad in column for bad in FORBIDDEN_SUBSTRINGS)]


def dynamic_feature_columns(columns: Iterable[str], scopes: Iterable[str] = ("link", "area", "network")) -> dict[str, list[str]]:
    available = set(columns)
    by_window: dict[str, list[str]] = {}
    for window in WINDOWS:
        names = []
        for scope in scopes:
            for metric in DYNAMIC_METRICS:
                column = f"{scope}_recent_{metric}_{window}"
                if column in available:
                    names.append(column)
        by_window[window] = names
    return by_window


def dynamic_channel_names(scopes: Iterable[str] = ("link", "area", "network")) -> list[str]:
    """Return window-independent channel names in a stable order.

    The parquet columns include the lookback window in their names.  Keeping
    the union of those names used to expand every window to 96 columns, 75% of
    which were structural zeros.  The model only needs the 24 aligned
    scope/metric channels at each of the four lookback windows.
    """
    return [f"{scope}_{metric}" for scope in scopes for metric in DYNAMIC_METRICS]


def add_route_position_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    position = pd.to_numeric(frame.get("position_ratio"), errors="coerce").fillna(0)
    frame["route_position_bucket"] = pd.cut(
        position,
        bins=[-0.01, 0.2, 0.45, 0.7, 1.01],
        labels=["early", "early_middle", "late_middle", "late"],
    ).astype("string")
    count = pd.to_numeric(frame.get("route_link_count"), errors="coerce")
    frame["route_length_bucket"] = pd.cut(
        count,
        bins=[-1, 20, 60, 10_000],
        labels=["short", "medium", "long"],
    ).astype("string")
    return frame


def read_filtered_day(path: Path, columns: list[str], order_ids: np.ndarray | None) -> pd.DataFrame:
    if order_ids is None:
        return pd.read_parquet(path, columns=unique_existing(path, columns))
    dataset = ds.dataset(path, format="parquet")
    order_values = [str(value) for value in order_ids]
    table = dataset.to_table(columns=unique_existing(path, columns), filter=ds.field("order_id").isin(order_values))
    return table.to_pandas()


def _balanced_order_budget(order_counts: dict[str, int], max_orders: int | None) -> dict[str, int]:
    if max_orders is None:
        return order_counts.copy()
    target = min(int(max_orders), sum(order_counts.values()))
    dates = list(order_counts)
    allocation = {date: 0 for date in dates}
    remaining = target
    active = dates.copy()
    # Equal allocation across dates, redistributing spare capacity from dates
    # with fewer orders.  This prevents a scaling probe from silently using
    # only the first one or two days in a rolling training window.
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        progressed = 0
        next_active = []
        for index, date in enumerate(active):
            request = share + (1 if index < extra else 0)
            capacity = order_counts[date] - allocation[date]
            take = min(request, capacity)
            allocation[date] += take
            remaining -= take
            progressed += take
            if allocation[date] < order_counts[date]:
                next_active.append(date)
        if progressed == 0:
            break
        active = next_active
    return allocation


def _truncate_routes(frame: pd.DataFrame, max_seq_len: int) -> pd.DataFrame:
    if not max_seq_len:
        return frame
    head_count = (max_seq_len + 1) // 2
    tail_count = max_seq_len // 2
    grouped = frame.groupby("order_id", sort=False)
    size = grouped["order_id"].transform("size")
    forward = grouped.cumcount()
    reverse = grouped.cumcount(ascending=False)
    # Preserve both pickup- and dropoff-side context.  The old head()
    # truncation systematically removed the destination side.
    keep = size.le(max_seq_len) | forward.lt(head_count) | reverse.lt(tail_count)
    return frame.loc[keep].reset_index(drop=True)


def read_dates(dataset_root: Path, dates: list[str], max_orders: int | None, seed: int, max_seq_len: int) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(seed)
    order_ids_by_date: dict[str, np.ndarray] = {}
    for date in dates:
        path = dataset_root / f"day={date}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        order_ids_by_date[date] = pd.read_parquet(path, columns=["order_id"])["order_id"].astype(str).drop_duplicates().to_numpy()
    allocation = _balanced_order_budget({date: len(values) for date, values in order_ids_by_date.items()}, max_orders)
    for date in dates:
        path = dataset_root / f"day={date}.parquet"
        columns = existing_columns(path)
        needed = list(dict.fromkeys(
            ID_COLUMNS
            + CATEGORICAL_COLUMNS
            + ["route_position_bucket", "route_length_bucket"]
            + default_static_numeric_columns(columns)
            + [column for cols in dynamic_feature_columns(columns).values() for column in cols]
            + [f"target_{target}_raw" for target in LINK_TARGETS]
            + [f"target_{target}_tail90_raw" for target in LINK_TARGETS]
            + [f"target_{target}_valid" for target in LINK_TARGETS]
        ))
        needed = unique_existing(path, needed)
        order_ids = order_ids_by_date[date]
        take = allocation[date]
        if take <= 0:
            continue
        selected_orders = None if take == len(order_ids) else rng.choice(order_ids, size=take, replace=False)
        frame = read_filtered_day(path, needed, selected_orders)
        frame = add_route_position_buckets(frame)
        frame = frame.sort_values(["order_id", "route_link_seq"], kind="mergesort")
        frame = _truncate_routes(frame, max_seq_len)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@dataclass
class DeepV3Metadata:
    static_numeric_columns: list[str]
    dynamic_columns_by_window: dict[str, list[str]]
    dynamic_channel_names: list[str]
    categorical_columns: list[str]
    category_maps: dict[str, dict[str, int]]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    dynamic_mean: dict[str, float]
    dynamic_std: dict[str, float]
    max_seq_len: int
    link_id_min_count: int

    def to_json(self) -> dict:
        return {
            "static_numeric_columns": self.static_numeric_columns,
            "dynamic_columns_by_window": self.dynamic_columns_by_window,
            "dynamic_channel_names": self.dynamic_channel_names,
            "categorical_columns": self.categorical_columns,
            "category_maps": self.category_maps,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "dynamic_mean": self.dynamic_mean,
            "dynamic_std": self.dynamic_std,
            "max_seq_len": self.max_seq_len,
            "link_id_min_count": self.link_id_min_count,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "DeepV3Metadata":
        return cls(
            static_numeric_columns=list(payload["static_numeric_columns"]),
            dynamic_columns_by_window={key: list(value) for key, value in payload["dynamic_columns_by_window"].items()},
            dynamic_channel_names=list(payload.get("dynamic_channel_names", dynamic_channel_names())),
            categorical_columns=list(payload["categorical_columns"]),
            category_maps={column: {str(key): int(value) for key, value in mapping.items()} for column, mapping in payload["category_maps"].items()},
            numeric_mean={str(key): float(value) for key, value in payload["numeric_mean"].items()},
            numeric_std={str(key): float(value) for key, value in payload["numeric_std"].items()},
            dynamic_mean={str(key): float(value) for key, value in payload["dynamic_mean"].items()},
            dynamic_std={str(key): float(value) for key, value in payload["dynamic_std"].items()},
            max_seq_len=int(payload["max_seq_len"]),
            link_id_min_count=int(payload["link_id_min_count"]),
        )


def build_metadata(train: pd.DataFrame, max_seq_len: int, link_id_min_count: int) -> DeepV3Metadata:
    columns = train.columns
    static_numeric = default_static_numeric_columns(columns)
    dyn_by_window = dynamic_feature_columns(columns)
    dyn_channels = dynamic_channel_names()
    categorical = [column for column in CATEGORICAL_COLUMNS + ["route_position_bucket", "route_length_bucket"] if column in columns]
    category_maps: dict[str, dict[str, int]] = {}
    for column in categorical:
        values = train[column].astype("string").fillna("__MISSING__")
        if column == "route_link_id":
            counts = values.value_counts()
            keep = counts[counts >= link_id_min_count].index
            values = values.where(values.isin(keep), "__RARE_LINK__")
        category_maps[column] = {value: i + 1 for i, value in enumerate(sorted(values.dropna().unique()))}
    numeric = train[static_numeric].apply(pd.to_numeric, errors="coerce")
    dyn_columns = sorted({column for cols in dyn_by_window.values() for column in cols})
    dynamic = train[dyn_columns].apply(pd.to_numeric, errors="coerce") if dyn_columns else pd.DataFrame(index=train.index)
    return DeepV3Metadata(
        static_numeric_columns=static_numeric,
        dynamic_columns_by_window=dyn_by_window,
        dynamic_channel_names=dyn_channels,
        categorical_columns=categorical,
        category_maps=category_maps,
        numeric_mean=numeric.mean().fillna(0).to_dict(),
        numeric_std=numeric.std().replace(0, 1).fillna(1).to_dict(),
        dynamic_mean=dynamic.mean().fillna(0).to_dict(),
        dynamic_std=dynamic.std().replace(0, 1).fillna(1).to_dict(),
        max_seq_len=max_seq_len,
        link_id_min_count=link_id_min_count,
    )


class RouteConditionedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, metadata: DeepV3Metadata):
        self.metadata = metadata
        self.emit_ids = False
        frame = frame.sort_values(["order_id", "route_link_seq"], kind="mergesort").reset_index(drop=True)
        order_values = frame["order_id"].astype(str).to_numpy()
        starts = np.r_[0, np.flatnonzero(order_values[1:] != order_values[:-1]) + 1]
        self.offsets = np.r_[starts, len(frame)].astype("int64")
        self.lengths = np.diff(self.offsets).astype("int32")
        self.static_numeric = self._encode_numeric(frame)
        self.dynamic = self._encode_dynamic(frame)
        self.categorical = self._encode_categorical(frame)
        self.target = np.nan_to_num(np.stack(
            [pd.to_numeric(frame[f"target_{name}_raw"], errors="coerce").to_numpy(dtype="float32") for name in LINK_TARGETS],
            axis=1,
        ), nan=0.0)
        self.tail = np.stack(
            [pd.to_numeric(frame[f"target_{name}_tail90_raw"], errors="coerce").fillna(0).to_numpy(dtype="float32") for name in LINK_TARGETS],
            axis=1,
        )
        self.mask = np.stack(
            [frame[f"target_{name}_valid"].fillna(False).to_numpy(dtype=bool) for name in LINK_TARGETS],
            axis=1,
        ).astype("float32")
        self.ids = {column: frame[column].to_numpy(copy=True) for column in ID_COLUMNS}

    def __len__(self) -> int:
        return len(self.lengths)

    def orders_by_date(self) -> dict[str, int]:
        dates = pd.Series(self.ids["date"][self.offsets[:-1]]).astype(str)
        return {str(key): int(value) for key, value in dates.value_counts().sort_index().items()}

    def encoded_bytes(self) -> int:
        arrays = [self.static_numeric, self.dynamic, self.categorical, self.target, self.tail, self.mask, self.offsets, self.lengths]
        return int(sum(array.nbytes for array in arrays))

    def _encode_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.metadata.static_numeric_columns].apply(pd.to_numeric, errors="coerce")
        for column in self.metadata.static_numeric_columns:
            values[column] = (values[column] - self.metadata.numeric_mean.get(column, 0.0)) / self.metadata.numeric_std.get(column, 1.0)
        return values.fillna(0).to_numpy(dtype="float32")

    def _encode_dynamic(self, frame: pd.DataFrame) -> np.ndarray:
        per_window = []
        for window in WINDOWS:
            available = set(self.metadata.dynamic_columns_by_window.get(window, []))
            values = np.zeros((len(frame), len(self.metadata.dynamic_channel_names)), dtype="float32")
            for index, channel in enumerate(self.metadata.dynamic_channel_names):
                scope, metric = channel.split("_", 1)
                column = f"{scope}_recent_{metric}_{window}"
                if column in available:
                    series = pd.to_numeric(frame[column], errors="coerce")
                    normalized = (series - self.metadata.dynamic_mean.get(column, 0.0)) / self.metadata.dynamic_std.get(column, 1.0)
                    values[:, index] = normalized.fillna(0).to_numpy(dtype="float32")
            per_window.append(values)
        if not per_window:
            return np.zeros((len(frame), len(WINDOWS), 0), dtype="float32")
        return np.stack(per_window, axis=1)

    def _encode_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        cats = []
        for column in self.metadata.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            if column == "route_link_id":
                mapping = self.metadata.category_maps[column]
                values = values.where(values.isin(mapping), "__RARE_LINK__")
            cats.append(values.map(self.metadata.category_maps[column]).fillna(0).to_numpy(dtype="int64"))
        return np.stack(cats, axis=1) if cats else np.zeros((len(frame), 0), dtype="int64")

    def __getitem__(self, index: int) -> dict:
        start, end = self.offsets[index:index + 2]
        return {
            "static_numeric": torch.from_numpy(self.static_numeric[start:end]),
            "dynamic": torch.from_numpy(self.dynamic[start:end]),
            "categorical": torch.from_numpy(self.categorical[start:end]),
            "target": torch.from_numpy(self.target[start:end]),
            "tail": torch.from_numpy(self.tail[start:end]),
            "mask": torch.from_numpy(self.mask[start:end]),
            "ids": (
                [{column: self.ids[column][row] for column in ID_COLUMNS} for row in range(start, end)]
                if self.emit_ids else None
            ),
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle orders within coarse length buckets to reduce padding waste."""

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, shuffle: bool = True, bucket_multiplier: int = 20):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = np.argsort(self.lengths, kind="stable")
        buckets = [indices[start:start + self.bucket_size].copy() for start in range(0, len(indices), self.bucket_size)]
        batches = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            batches.extend(bucket[start:start + self.batch_size].tolist() for start in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class MemmapRouteDataset(Dataset):
    """Order-indexed dataset backed by daily NumPy mmap shards."""

    ARRAY_NAMES = ("static_numeric", "dynamic", "categorical", "target", "tail", "mask", "offsets", "lengths")

    def __init__(self, split_root: Path):
        self.split_root = Path(split_root)
        self.emit_ids = False
        self.shards = []
        self._order_ends = []
        self._ids_cache: dict[int, dict[str, np.ndarray]] = {}
        total_orders = 0
        for day_dir in sorted(self.split_root.glob("day=*")):
            manifest_path = day_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrays = {name: np.load(day_dir / f"{name}.npy", mmap_mode="c") for name in self.ARRAY_NAMES}
            shard = {"root": day_dir, "manifest": manifest, "arrays": arrays}
            self.shards.append(shard)
            total_orders += int(len(arrays["lengths"]))
            self._order_ends.append(total_orders)
        if not self.shards:
            raise FileNotFoundError(f"No tensor shards found under {self.split_root}")
        self.lengths = np.concatenate([np.asarray(shard["arrays"]["lengths"]) for shard in self.shards]).astype("int32", copy=False)

    def __len__(self) -> int:
        return int(self._order_ends[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._order_ends, index)
        previous = self._order_ends[shard_index - 1] if shard_index else 0
        return shard_index, index - previous

    def _ids(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index not in self._ids_cache:
            frame = pd.read_parquet(self.shards[shard_index]["root"] / "ids.parquet", columns=ID_COLUMNS)
            self._ids_cache[shard_index] = {column: frame[column].to_numpy(copy=True) for column in ID_COLUMNS}
        return self._ids_cache[shard_index]

    def __getitem__(self, index: int) -> dict:
        shard_index, local_order = self._locate(index)
        arrays = self.shards[shard_index]["arrays"]
        start, end = np.asarray(arrays["offsets"][local_order:local_order + 2], dtype="int64")
        ids = None
        if self.emit_ids:
            id_arrays = self._ids(shard_index)
            ids = [{column: id_arrays[column][row] for column in ID_COLUMNS} for row in range(start, end)]
        return {
            "static_numeric": torch.from_numpy(np.asarray(arrays["static_numeric"][start:end])),
            "dynamic": torch.from_numpy(np.asarray(arrays["dynamic"][start:end])),
            "categorical": torch.from_numpy(np.asarray(arrays["categorical"][start:end])),
            "target": torch.from_numpy(np.asarray(arrays["target"][start:end])),
            "tail": torch.from_numpy(np.asarray(arrays["tail"][start:end])),
            "mask": torch.from_numpy(np.asarray(arrays["mask"][start:end])),
            "ids": ids,
        }

    def orders_by_date(self) -> dict[str, int]:
        return {
            str(shard["manifest"]["date"]): int(shard["manifest"]["orders"])
            for shard in self.shards
        }

    def encoded_bytes(self) -> int:
        return int(sum((shard["root"] / f"{name}.npy").stat().st_size for shard in self.shards for name in self.ARRAY_NAMES))

    @property
    def offsets(self) -> np.ndarray:
        # Compatibility with manifest reporting; only the last value is used.
        return np.array([0, sum(int(shard["manifest"]["rows"]) for shard in self.shards)], dtype="int64")


def collate_routes(batch: list[dict]) -> dict:
    size = len(batch)
    max_len = max(item["static_numeric"].shape[0] for item in batch)
    n_static = batch[0]["static_numeric"].shape[1]
    n_windows = batch[0]["dynamic"].shape[1]
    n_dynamic = batch[0]["dynamic"].shape[2]
    n_cat = batch[0]["categorical"].shape[1]
    n_targets = len(LINK_TARGETS)
    static_numeric = torch.zeros(size, max_len, n_static)
    dynamic = torch.zeros(size, max_len, n_windows, n_dynamic)
    categorical = torch.zeros(size, max_len, n_cat, dtype=torch.long)
    target = torch.zeros(size, max_len, n_targets)
    tail = torch.zeros(size, max_len, n_targets)
    mask = torch.zeros(size, max_len, n_targets)
    pad_mask = torch.ones(size, max_len, dtype=torch.bool)
    ids = []
    for i, item in enumerate(batch):
        length = item["static_numeric"].shape[0]
        static_numeric[i, :length] = item["static_numeric"]
        dynamic[i, :length] = item["dynamic"]
        categorical[i, :length] = item["categorical"]
        target[i, :length] = item["target"]
        tail[i, :length] = item["tail"]
        mask[i, :length] = item["mask"]
        pad_mask[i, :length] = False
        ids.append(item["ids"])
    return {
        "static_numeric": static_numeric,
        "dynamic": dynamic,
        "categorical": categorical,
        "target": target,
        "tail": tail,
        "mask": mask,
        "pad_mask": pad_mask,
        "ids": ids,
    }


def safe_float(value):
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {key: safe_float(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_float(item) for item in value]
    return value


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & ((y_prob <= high) if high == 1 else (y_prob < high))
        if mask.any():
            value += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(value)


def ndcg_at_fraction(high: np.ndarray, score: np.ndarray, fraction: float) -> float:
    if len(high) == 0:
        return float("nan")
    k = max(1, int(len(high) * fraction))
    order = np.argsort(-score)[:k]
    gains = high[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(high.astype(float))[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def metric_dict(y: np.ndarray, raw_pred: np.ndarray, prob: np.ndarray, high: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(raw_pred) & np.isfinite(prob)
    y = y[valid]
    raw_pred = np.clip(raw_pred[valid], 0, 1)
    prob = np.clip(prob[valid], 0, 1)
    high = high[valid].astype(bool)
    result = {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, raw_pred)) if len(y) else float("nan"),
        "rmse": float(mean_squared_error(y, raw_pred, squared=False)) if len(y) else float("nan"),
        "pearson": float(pd.Series(y).corr(pd.Series(raw_pred), method="pearson")) if len(y) > 1 else float("nan"),
        "spearman": float(pd.Series(y).corr(pd.Series(raw_pred), method="spearman")) if len(y) > 1 else float("nan"),
        "ndcg_top5": ndcg_at_fraction(high, prob, 0.05),
        "ndcg_top10": ndcg_at_fraction(high, prob, 0.10),
    }
    if high.any() and (~high).any():
        result["auc"] = float(roc_auc_score(high, prob))
        result["ap"] = float(average_precision_score(high, prob))
        result["brier"] = float(brier_score_loss(high, prob))
        result["ece"] = ece_score(high.astype(float), prob)
    else:
        result.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan"), "ece": float("nan")})
    base = high.mean() if len(high) else float("nan")
    for fraction, label in [(0.05, "top5"), (0.10, "top10")]:
        k = max(1, int(len(high) * fraction)) if len(high) else 0
        idx = np.argsort(-prob)[:k]
        precision = float(high[idx].mean()) if k else float("nan")
        recall = float(high[idx].sum() / max(high.sum(), 1)) if k else float("nan")
        result[f"precision_{label}"] = precision
        result[f"recall_{label}"] = recall
        result[f"lift_{label}"] = precision / base if base and base > 0 else float("nan")
    return result


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        self.seconds = time.time() - self.start


def order_level_metrics(predictions: pd.DataFrame, target: str) -> dict[str, float]:
    data = predictions[predictions[f"{target}_valid"]].copy()
    if data.empty:
        return {}
    grouped = data.groupby("order_id").agg(
        true_mean=(f"target_{target}_raw", "mean"),
        true_q90=(f"target_{target}_raw", lambda value: float(np.nanquantile(value, 0.90))),
        true_max=(f"target_{target}_raw", "max"),
        pred_raw_mean=(f"pred_{target}_raw", "mean"),
        pred_raw_q90=(f"pred_{target}_raw", lambda value: float(np.nanquantile(value, 0.90))),
        pred_raw_max=(f"pred_{target}_raw", "max"),
        pred_tail_mean=(f"pred_{target}_tail_prob", "mean"),
        pred_tail_q90=(f"pred_{target}_tail_prob", lambda value: float(np.nanquantile(value, 0.90))),
        pred_tail_max=(f"pred_{target}_tail_prob", "max"),
    )
    result = {}
    for aggregation in ["mean", "q90", "max"]:
        truth = grouped[f"true_{aggregation}"].to_numpy(dtype=float)
        raw = grouped[f"pred_raw_{aggregation}"].to_numpy(dtype=float)
        probability = grouped[f"pred_tail_{aggregation}"].to_numpy(dtype=float)
        threshold = float(np.nanquantile(truth, 0.90))
        high = truth >= threshold
        metrics = metric_dict(truth, raw, probability, high)
        result.update({f"order_{aggregation}_{key}": value for key, value in metrics.items()})
        if aggregation == "q90":
            # Backward-compatible aliases now use a stable empirical top-decile
            # order definition rather than the nearly empty absolute q90 >= .90 event.
            result.update({f"order_{key}": value for key, value in metrics.items()})
    result["order_tail_definition"] = "empirical_top_decile_within_evaluation_split"
    return result
