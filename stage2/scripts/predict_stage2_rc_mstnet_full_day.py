"""Inference-only RC-MSTNet prediction for a full held-out day.

This script deliberately does not sample orders.  It loads an existing
Stage2 RC-MSTNet checkpoint and applies it to a deployable
route-conditioned estimated-entry dataset, writing link-level predictions
incrementally so a full day can be processed without materializing a giant
prediction DataFrame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    LINK_TARGETS,
    DeepV3Metadata,
    LengthBucketBatchSampler,
    RouteConditionedDataset,
    _truncate_routes,
    add_route_position_buckets,
    collate_routes,
    default_static_numeric_columns,
    dynamic_feature_columns,
    existing_columns,
    read_filtered_day,
    read_dates,
    unique_existing,
)
from train_stage2_rc_mstnet import RC_MSTNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-orders", type=int, default=5000, help="0 keeps legacy full-frame loading; positive values stream order chunks.")
    parser.add_argument("--bucket-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_order_ids(path: Path) -> np.ndarray:
    return pd.read_parquet(path, columns=["order_id"])["order_id"].astype(str).drop_duplicates().to_numpy()


def _read_date_subset(path: Path, order_ids: np.ndarray, metadata: DeepV3Metadata) -> pd.DataFrame:
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
    frame = read_filtered_day(path, unique_existing(path, needed), order_ids)
    frame = add_route_position_buckets(frame)
    frame = frame.sort_values(["order_id", "route_link_seq"], kind="mergesort")
    return _truncate_routes(frame, metadata.max_seq_len)


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[RC_MSTNet, DeepV3Metadata, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = DeepV3Metadata.from_json(json.loads(checkpoint["metadata"]) if isinstance(checkpoint["metadata"], str) else checkpoint["metadata"])
    model_args = checkpoint.get("args", {})
    model = RC_MSTNet(
        n_static=len(metadata.static_numeric_columns),
        n_dynamic=len(metadata.dynamic_channel_names),
        category_sizes=[len(metadata.category_maps[column]) + 1 for column in metadata.categorical_columns],
        hidden_dim=int(model_args.get("hidden_dim", 128)),
        cat_emb_dim=int(model_args.get("cat_emb_dim", 12)),
        layers=int(model_args.get("layers", 3)),
        heads=int(model_args.get("heads", 4)),
        dropout=float(model_args.get("dropout", 0.15)),
        use_dynamic=bool(model_args.get("dynamic_encoder", True)),
        use_local_route=bool(model_args.get("local_route_encoder", True)),
        use_transformer=bool(model_args.get("route_transformer", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, metadata, model_args


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _make_loader(dataset: RouteConditionedDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    batch_sampler = LengthBucketBatchSampler(
        dataset.lengths,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        bucket_multiplier=args.bucket_multiplier,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_routes,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
    )


def _batch_to_frame(batch: dict, raw: np.ndarray, tail_prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    target = batch["target"].numpy()
    tail = batch["tail"].numpy().astype(bool)
    mask = batch["mask"].numpy().astype(bool)
    for i, id_rows in enumerate(batch["ids"]):
        for j, id_row in enumerate(id_rows):
            row = {column: id_row.get(column) for column in ID_COLUMNS}
            for k, target_name in enumerate(LINK_TARGETS):
                row[f"pred_{target_name}_raw"] = float(raw[i, j, k])
                row[f"pred_{target_name}_tail_prob"] = float(tail_prob[i, j, k])
                row[f"target_{target_name}_raw"] = float(target[i, j, k]) if mask[i, j, k] else np.nan
                row[f"target_{target_name}_tail"] = bool(tail[i, j, k]) if mask[i, j, k] else False
                row[f"{target_name}_valid"] = bool(mask[i, j, k])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, model_args = _load_model(args.checkpoint, device)
    started = time.time()

    amp_enabled = bool(args.amp and device.type == "cuda")
    writer: pq.ParquetWriter | None = None
    output_rows = 0
    output_orders: set[str] = set()
    input_rows = 0
    day_path = args.dataset_root / f"day={args.date}.parquet"
    all_order_ids = _read_order_ids(day_path)
    orders = int(len(all_order_ids))

    def iter_frames():
        if args.chunk_orders and args.chunk_orders > 0:
            for start in range(0, len(all_order_ids), args.chunk_orders):
                selected = all_order_ids[start:start + args.chunk_orders]
                yield start // args.chunk_orders, _read_date_subset(day_path, selected, metadata)
        else:
            yield 0, read_dates(args.dataset_root, [args.date], None, args.seed, metadata.max_seq_len)

    try:
        with torch.no_grad():
            for chunk_index, frame in iter_frames():
                input_rows += int(len(frame))
                dataset = RouteConditionedDataset(frame, metadata)
                dataset.emit_ids = True
                loader = _make_loader(dataset, args, device)
                for batch_index, batch in enumerate(loader):
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        raw_tensor, tail_logits, _ = model(
                            batch["static_numeric"].to(device, non_blocking=True),
                            batch["dynamic"].to(device, non_blocking=True),
                            batch["categorical"].to(device, non_blocking=True),
                            batch["pad_mask"].to(device, non_blocking=True),
                        )
                    raw = raw_tensor.cpu().numpy()
                    tail_prob = torch.sigmoid(tail_logits).cpu().numpy()
                    part = _batch_to_frame(batch, raw, tail_prob)
                    output_rows += int(len(part))
                    output_orders.update(part["order_id"].astype(str).unique().tolist())
                    table = pa.Table.from_pandas(part, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
                    else:
                        table = table.cast(writer.schema)
                    writer.write_table(table)
                print(f"chunk={chunk_index} rows={output_rows} orders={len(output_orders)}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "date": args.date,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "device": str(device),
        "input_rows": input_rows,
        "input_orders": orders,
        "output_rows": output_rows,
        "output_orders": len(output_orders),
        "metadata_max_seq_len": metadata.max_seq_len,
        "model_args": _json_safe(model_args),
        "runtime_sec": time.time() - started,
        "status": "PASS" if output_orders == orders and output_rows == input_rows else "WARN",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
