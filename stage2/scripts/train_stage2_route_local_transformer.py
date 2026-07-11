"""RouteLocalTransformer for Stage2 deep upper-bound modeling v2.

The model is inspired by DeepTTE-style local/whole-path multitask learning and
ConSTGAT-style local route context modeling:

- local route convolution before attention;
- local-window Transformer attention along the ordered route;
- link-level LCS/IIS/RTS/PMIS heads with valid-mask losses;
- order-level high-stress auxiliary head;
- optional route contrastive pretraining with two noisy route views.

It uses only pre-dispatch features from the Stage2 data contract.
"""

from __future__ import annotations

import argparse
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

from stage2_model_utils import CATEGORICAL_COLUMNS, ID_COLUMNS, NUMERIC_COLUMNS, TARGET_ORDER, evaluate_predictions, safe_json_float  # noqa: E402
from train_stage2_sequence_model import (  # noqa: E402
    TARGETS,
    RouteDataset,
    build_metadata,
    collect_orders,
    collate,
    load_profiles,
    max_eval_orders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines_v2/route_local_transformer"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions_v2"))
    parser.add_argument("--profile-path", type=Path, default=Path("stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet"))
    parser.add_argument("--max-train-orders", type=int, default=60_000)
    parser.add_argument("--max-eval-orders", default="30000")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cat-emb-dim", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--local-kernel", type=int, default=5)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--tail-loss-weight", type=float, default=0.35)
    parser.add_argument("--order-aux-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def profile_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("profile_") and not column.endswith("_count")]


class RouteLocalTransformer(nn.Module):
    def __init__(self, num_numeric: int, cat_sizes: list[int], hidden_dim: int, cat_emb_dim: int, layers: int, heads: int, local_kernel: int, local_window: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, cat_emb_dim, padding_idx=0) for size in cat_sizes])
        input_dim = num_numeric + len(cat_sizes) * cat_emb_dim
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=local_kernel, padding=local_kernel // 2, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.link_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(TARGET_ORDER)))
        self.order_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(TARGET_ORDER)))
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.local_window = local_window

    def local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        return (idx[None, :] - idx[:, None]).abs() > self.local_window

    def encode(self, numeric: torch.Tensor, categorical: torch.Tensor, pad_mask: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        if noise > 0:
            numeric = numeric + torch.randn_like(numeric) * noise
        embeddings = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([numeric] + embeddings, dim=-1) if embeddings else numeric
        x = self.input(x)
        x = x + self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        mask = self.local_mask(x.shape[1], x.device)
        return self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)

    def pooled(self, hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).float().unsqueeze(-1)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor, pad_mask: torch.Tensor, noise: float = 0.0):
        hidden = self.encode(numeric, categorical, pad_mask, noise=noise)
        pooled = self.pooled(hidden, pad_mask)
        return torch.sigmoid(self.link_head(hidden)), torch.sigmoid(self.order_head(pooled)), self.projection(pooled)


def masked_multitask_loss(pred, target, high, mask, order_pred, tail_weight: float, order_weight: float):
    denominator = mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask).sum() / denominator
    bce = nn.functional.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), high, reduction="none")
    bce = (bce * mask).sum() / denominator
    order_target = (high * mask).amax(dim=1)
    order_mask = mask.amax(dim=1).clamp_max(1.0)
    aux = nn.functional.binary_cross_entropy(order_pred.clamp(1e-5, 1 - 1e-5), order_target, reduction="none")
    aux = (aux * order_mask).sum() / order_mask.sum().clamp_min(1.0)
    return mse + tail_weight * bce + order_weight * aux


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2


def predict(model, loader, device):
    rows = []
    ys = {target: [] for target in TARGET_ORDER}
    preds = {target: [] for target in TARGET_ORDER}
    highs = {target: [] for target in TARGET_ORDER}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output, _, _ = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["pad_mask"].to(device))
            output = output.cpu().numpy()
            target = batch["target"].numpy()
            mask = batch["mask"].numpy().astype(bool)
            high = batch["high"].numpy().astype(bool)
            for i, id_list in enumerate(batch["ids"]):
                for j, id_row in enumerate(id_list):
                    row = dict(id_row)
                    for k, target_name in enumerate(TARGET_ORDER):
                        row[f"pred_{target_name.lower()}"] = float(output[i, j, k])
                        row[f"target_{target_name.lower()}"] = float(target[i, j, k]) if mask[i, j, k] else np.nan
                        row[f"{target_name.lower()}_valid"] = bool(mask[i, j, k])
                        if mask[i, j, k]:
                            ys[target_name].append(float(target[i, j, k]))
                            preds[target_name].append(float(output[i, j, k]))
                            highs[target_name].append(bool(high[i, j, k]))
                    rows.append(row)
    metrics = {target: evaluate_predictions(np.array(ys[target]), np.array(preds[target]), np.array(highs[target], dtype=bool)) for target in TARGET_ORDER}
    return pd.DataFrame(rows), metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profile_path)
    train = collect_orders(args.dataset_root / "train.parquet", profiles, args.max_train_orders, args.max_seq_len, args.seed)
    profile_cols = profile_numeric_columns(train)
    numeric_columns = [column for column in NUMERIC_COLUMNS + profile_cols if column in train.columns]
    cat_maps, numeric_mean, numeric_std = build_metadata(train, numeric_columns)
    validation = collect_orders(args.dataset_root / "validation.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    test = collect_orders(args.dataset_root / "test.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    train_dataset = RouteDataset(train, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, numeric_mean, numeric_std)
    val_dataset = RouteDataset(validation, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, numeric_mean, numeric_std)
    test_dataset = RouteDataset(test, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, numeric_mean, numeric_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = RouteLocalTransformer(
        len(numeric_columns),
        [len(cat_maps[column]) + 1 for column in CATEGORICAL_COLUMNS],
        args.hidden_dim,
        args.cat_emb_dim,
        args.layers,
        args.heads,
        args.local_kernel,
        args.local_window,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            _, _, z1 = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["pad_mask"].to(device), noise=0.03)
            _, _, z2 = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["pad_mask"].to(device), noise=0.06)
            loss = contrastive_loss(z1, z2, args.contrastive_temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"phase": "contrastive_pretrain", "epoch": epoch, "loss": float(np.mean(losses))})
        print(f"pretrain epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred, order_pred, _ = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["pad_mask"].to(device))
            loss = masked_multitask_loss(
                pred,
                batch["target"].to(device),
                batch["high"].to(device),
                batch["mask"].to(device),
                order_pred,
                args.tail_loss_weight,
                args.order_aux_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"phase": "supervised", "epoch": epoch, "loss": float(np.mean(losses))})
        print(f"supervised epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    val_pred, val_metrics = predict(model, val_loader, device)
    test_pred, test_metrics = predict(model, test_loader, device)
    val_pred.to_parquet(args.prediction_root / "route_local_transformer_validation.parquet", index=False, compression="zstd")
    test_pred.to_parquet(args.prediction_root / "route_local_transformer_test.parquet", index=False, compression="zstd")
    rows = []
    for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        for target, row in metrics.items():
            rows.append({"split": split, "target": target, **row})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_root / "route_local_transformer_metrics_by_target.csv", index=False)
    manifest = {
        "model": "RouteLocalTransformer",
        "device": str(device),
        "max_train_orders": args.max_train_orders,
        "max_eval_orders": args.max_eval_orders,
        "max_seq_len": args.max_seq_len,
        "local_window": args.local_window,
        "train_orders": len(train_dataset),
        "validation_orders": len(val_dataset),
        "test_orders": len(test_dataset),
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "history": history,
        "metrics": rows,
    }
    (args.output_root / "route_local_transformer_metrics.json").write_text(json.dumps(safe_json_float(manifest), indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "manifest": manifest}, args.output_root / "route_local_transformer.pt")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
