"""Train RC-MSTNet for route-conditioned Stage2 link stress prediction."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import (  # noqa: E402
    LINK_TARGETS,
    LengthBucketBatchSampler,
    MemmapRouteDataset,
    RouteConditionedDataset,
    Timer,
    DeepV3Metadata,
    build_metadata,
    collate_routes,
    load_fold_config,
    metric_dict,
    order_level_metrics,
    read_dates,
    safe_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset/estimated_time_daily"))
    parser.add_argument("--tensor-shard-root", type=Path, default=None, help="Optional fold-aware mmap shards built by build_stage2_deep_v3_tensor_shards.py.")
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/feasibility_100k/rc_mstnet"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_v3/rolling_predictions/rc_mstnet"))
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--max-train-orders", type=int, default=100_000)
    parser.add_argument("--max-eval-orders", type=int, default=0, help="0 means full validation/test orders.")
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--link-id-min-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cat-emb-dim", type=int, default=12)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--tail-loss-weight", type=float, default=0.5)
    parser.add_argument("--route-aux-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows/low-RAM hosts; encoded arrays make workers optional.")
    parser.add_argument("--bucket-multiplier", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class RC_MSTNet(nn.Module):
    def __init__(self, n_static: int, n_dynamic: int, category_sizes: list[int], hidden_dim: int, cat_emb_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, cat_emb_dim, padding_idx=0) for size in category_sizes])
        cat_dim = len(category_sizes) * cat_emb_dim
        self.static_mlp = nn.Sequential(nn.Linear(n_static + cat_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.dynamic_in = nn.Linear(max(n_dynamic, 1), hidden_dim)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.local_route = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.route_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.raw_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, len(LINK_TARGETS)))
        self.tail_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, len(LINK_TARGETS)))
        self.route_tail_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(LINK_TARGETS)))

    def forward(self, static_numeric: torch.Tensor, dynamic: torch.Tensor, categorical: torch.Tensor, pad_mask: torch.Tensor):
        embeddings = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        static = torch.cat([static_numeric] + embeddings, dim=-1) if embeddings else static_numeric
        h_static = self.static_mlp(static)
        if dynamic.shape[-1] == 0:
            h_dynamic = torch.zeros_like(h_static)
        else:
            b, l, w, d = dynamic.shape
            h = self.dynamic_in(dynamic.reshape(b * l, w, d)).transpose(1, 2)
            h = self.temporal_conv(h).mean(dim=2).reshape(b, l, -1)
            h_dynamic = h
        h = h_static + h_dynamic
        h = h + self.local_route(h.transpose(1, 2)).transpose(1, 2)
        h = self.route_encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return torch.sigmoid(self.raw_head(h)), self.tail_head(h), self.route_tail_head(pooled)


def loss_fn(raw_pred, tail_logits, route_tail_logits, target, tail, mask, tail_weight: float, route_weight: float) -> torch.Tensor:
    valid = mask.sum().clamp_min(1.0)
    huber = nn.functional.huber_loss(raw_pred, target, reduction="none", delta=0.08)
    raw_loss = (huber * mask).sum() / valid
    bce = nn.functional.binary_cross_entropy_with_logits(tail_logits, tail, reduction="none")
    tail_loss = (bce * mask).sum() / valid
    route_tail = (tail * mask).amax(dim=1)
    route_mask = mask.amax(dim=1).clamp_max(1.0)
    route_bce = nn.functional.binary_cross_entropy_with_logits(route_tail_logits, route_tail, reduction="none")
    route_loss = (route_bce * route_mask).sum() / route_mask.sum().clamp_min(1.0)
    return raw_loss + tail_weight * tail_loss + route_weight * route_loss


def predict(model: nn.Module, loader: DataLoader, device: torch.device, *, materialize: bool, amp_enabled: bool) -> tuple[pd.DataFrame, dict]:
    rows = []
    y_by_target = {target: [] for target in LINK_TARGETS}
    raw_by_target = {target: [] for target in LINK_TARGETS}
    prob_by_target = {target: [] for target in LINK_TARGETS}
    high_by_target = {target: [] for target in LINK_TARGETS}
    loader.dataset.emit_ids = materialize
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                raw, tail_logits, _ = model(
                    batch["static_numeric"].to(device, non_blocking=True),
                    batch["dynamic"].to(device, non_blocking=True),
                    batch["categorical"].to(device, non_blocking=True),
                    batch["pad_mask"].to(device, non_blocking=True),
                )
            raw = raw.cpu().numpy()
            tail_pred = torch.sigmoid(tail_logits).cpu().numpy()
            target = batch["target"].numpy()
            high = batch["tail"].numpy().astype(bool)
            mask = batch["mask"].numpy().astype(bool)
            for k, target_name in enumerate(LINK_TARGETS):
                valid = mask[:, :, k]
                y_by_target[target_name].extend(target[:, :, k][valid].tolist())
                raw_by_target[target_name].extend(raw[:, :, k][valid].tolist())
                prob_by_target[target_name].extend(tail_pred[:, :, k][valid].tolist())
                high_by_target[target_name].extend(high[:, :, k][valid].tolist())
            if materialize:
                for i, id_rows in enumerate(batch["ids"]):
                    for j, id_row in enumerate(id_rows):
                        row = dict(id_row)
                        for k, target_name in enumerate(LINK_TARGETS):
                            row[f"pred_{target_name}_raw"] = float(raw[i, j, k])
                            row[f"pred_{target_name}_tail_prob"] = float(tail_pred[i, j, k])
                            row[f"target_{target_name}_raw"] = float(target[i, j, k]) if mask[i, j, k] else np.nan
                            row[f"target_{target_name}_tail"] = bool(high[i, j, k]) if mask[i, j, k] else False
                            row[f"{target_name}_valid"] = bool(mask[i, j, k])
                        rows.append(row)
    predictions = pd.DataFrame(rows)
    metrics = {}
    for target_name in LINK_TARGETS:
        metrics[target_name] = metric_dict(
            np.array(y_by_target[target_name]),
            np.array(raw_by_target[target_name]),
            np.array(prob_by_target[target_name]),
            np.array(high_by_target[target_name], dtype=bool),
        )
        if materialize:
            metrics[target_name].update(order_level_metrics(predictions, target_name))
    return predictions, metrics


def make_loader(dataset: RouteConditionedDataset | MemmapRouteDataset, args: argparse.Namespace, seed: int, shuffle: bool, device: torch.device) -> DataLoader:
    batch_sampler = LengthBucketBatchSampler(
        dataset.lengths,
        batch_size=args.batch_size,
        seed=seed,
        shuffle=shuffle,
        bucket_multiplier=args.bucket_multiplier,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_routes,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )


def run_fold(args: argparse.Namespace, fold: dict, device: torch.device) -> dict:
    fold_id = int(fold["fold"])
    seed = args.seed + fold_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    with Timer() as prep_timer:
        if args.tensor_shard_root is not None:
            shard_fold_root = args.tensor_shard_root / f"fold={fold_id}"
            metadata = DeepV3Metadata.from_json(json.loads((shard_fold_root / "metadata.json").read_text(encoding="utf-8")))
            datasets = {
                split: MemmapRouteDataset(shard_fold_root / split)
                for split in ["train", "validation", "test"]
            }
        else:
            train = read_dates(args.dataset_root, fold["train_dates"], args.max_train_orders, seed, args.max_seq_len)
            eval_limit = None if args.max_eval_orders <= 0 else args.max_eval_orders
            validation = read_dates(args.dataset_root, [fold["validation_date"]], eval_limit, seed, args.max_seq_len)
            test = read_dates(args.dataset_root, [fold["test_date"]], eval_limit, seed, args.max_seq_len)
            metadata = build_metadata(train, args.max_seq_len, args.link_id_min_count)
            datasets = {
                "train": RouteConditionedDataset(train, metadata),
                "validation": RouteConditionedDataset(validation, metadata),
                "test": RouteConditionedDataset(test, metadata),
            }
            del train, validation, test
    gc.collect()
    loaders = {
        "train": make_loader(datasets["train"], args, seed, True, device),
        "validation": make_loader(datasets["validation"], args, seed, False, device),
        "test": make_loader(datasets["test"], args, seed, False, device),
    }
    model = RC_MSTNet(
        n_static=len(metadata.static_numeric_columns),
        n_dynamic=len(metadata.dynamic_channel_names),
        category_sizes=[len(metadata.category_maps[column]) + 1 for column in metadata.categorical_columns],
        hidden_dim=args.hidden_dim,
        cat_emb_dim=args.cat_emb_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best_score = -np.inf
    best_state = None
    training_seconds = 0.0
    training_tokens = 0
    padded_tokens = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        with Timer() as timer:
            for batch in loaders["train"]:
                training_tokens += int((~batch["pad_mask"]).sum())
                padded_tokens += int(batch["pad_mask"].numel())
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    raw, tail_logits, route_tail_logits = model(
                        batch["static_numeric"].to(device, non_blocking=True),
                        batch["dynamic"].to(device, non_blocking=True),
                        batch["categorical"].to(device, non_blocking=True),
                        batch["pad_mask"].to(device, non_blocking=True),
                    )
                    loss = loss_fn(
                        raw, tail_logits, route_tail_logits,
                        batch["target"].to(device, non_blocking=True),
                        batch["tail"].to(device, non_blocking=True),
                        batch["mask"].to(device, non_blocking=True),
                        args.tail_loss_weight, args.route_aux_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
        training_seconds += timer.seconds
        train_loss = float(np.mean(losses))
        print(f"fold={fold_id} epoch={epoch} train_loss={train_loss:.5f} train_seconds={timer.seconds:.1f}", flush=True)
        _, val_metrics = predict(model, loaders["validation"], device, materialize=False, amp_enabled=amp_enabled)
        score = float(np.nanmean([val_metrics[target]["ap"] for target in LINK_TARGETS]))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_mean_ap": score, "epoch_seconds": timer.seconds})
        print(f"fold={fold_id} epoch={epoch} validation_mean_ap={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_predictions, val_metrics = predict(model, loaders["validation"], device, materialize=True, amp_enabled=amp_enabled)
    test_predictions, test_metrics = predict(model, loaders["test"], device, materialize=True, amp_enabled=amp_enabled)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    fold_pred_root = args.prediction_root / f"fold={fold_id}"
    fold_pred_root.mkdir(parents=True, exist_ok=True)
    val_predictions.to_parquet(fold_pred_root / "validation_predictions.parquet", index=False, compression="zstd")
    test_predictions.to_parquet(fold_pred_root / "test_predictions.parquet", index=False, compression="zstd")
    fold_root = args.output_root / f"fold={fold_id}"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata.to_json(), "args": vars(args)}, fold_root / "rc_mstnet.pt")
    fold_manifest = {
        "fold": fold_id,
        "train_dates": fold["train_dates"],
        "validation_date": fold["validation_date"],
        "test_date": fold["test_date"],
        "device": str(device),
        "data_source": "tensor_shards" if args.tensor_shard_root is not None else "daily_parquet_in_memory",
        "amp_enabled": amp_enabled,
        "data_preparation_seconds": prep_timer.seconds,
        "requested_max_train_orders": args.max_train_orders,
        "requested_max_eval_orders": args.max_eval_orders,
        "effective_train_orders": len(datasets["train"]),
        "train_orders_by_date": datasets["train"].orders_by_date(),
        "validation_orders": len(datasets["validation"]),
        "test_orders": len(datasets["test"]),
        "effective_train_rows": int(datasets["train"].offsets[-1]),
        "validation_rows": int(datasets["validation"].offsets[-1]),
        "test_rows": int(datasets["test"].offsets[-1]),
        "encoded_dataset_bytes": {name: dataset.encoded_bytes() for name, dataset in datasets.items()},
        "train_route_length_p50": float(np.quantile(datasets["train"].lengths, 0.50)),
        "train_route_length_p90": float(np.quantile(datasets["train"].lengths, 0.90)),
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": training_seconds,
        "training_orders_per_second": float(len(datasets["train"]) * args.epochs / max(training_seconds, 1e-9)),
        "training_links_per_second": float(training_tokens / max(training_seconds, 1e-9)),
        "padding_efficiency": float(training_tokens / max(padded_tokens, 1)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "metadata": metadata.to_json(),
    }
    (fold_root / "manifest.json").write_text(json.dumps(safe_float(fold_manifest), indent=2), encoding="utf-8")
    return fold_manifest


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    folds = load_fold_config(args.fold_config)
    selected = {int(part.strip()) for part in args.folds.split(",") if part.strip()}
    manifests = []
    for fold in folds:
        if int(fold["fold"]) in selected:
            manifests.append(run_fold(args, fold, device))
    rows = []
    for manifest in manifests:
        for split_name, metrics in [("validation", manifest["validation_metrics"]), ("test", manifest["test_metrics"])]:
            for target, metric in metrics.items():
                rows.append({"fold": manifest["fold"], "split": split_name, "target": target, **metric})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_root / "rc_mstnet_metrics_by_fold.csv", index=False)
    summary = metrics.groupby(["split", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        spearman_mean=("spearman", "mean"),
        lift_top5_mean=("lift_top5", "mean"),
        order_lift_top10_mean=("order_lift_top10", "mean"),
    )
    summary.to_csv(args.output_root / "rc_mstnet_summary.csv", index=False)
    report = [
        "# RC-MSTNet Deep v3",
        "",
        f"Device: `{device}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Data-scale note",
        "",
        "The current route-conditioned estimated-time product contains about 1000 sampled orders per day. "
        "If `effective_train_orders` is far below the requested 100k, this run is a protocol/feasibility run, not the formal 100k+ deep comparison.",
    ]
    (args.output_root / "rc_mstnet_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "rc_mstnet_manifest.json").write_text(json.dumps(safe_float({"folds": manifests, "summary": summary.to_dict(orient="records")}), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
