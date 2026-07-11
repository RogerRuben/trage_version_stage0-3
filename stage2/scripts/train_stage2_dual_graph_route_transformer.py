"""DualGraphRouteTransformer for Stage2 deep modeling v2.

Model ideas represented here:

- STDGNN-style dual graph: a physical consecutive-link graph plus an
  order co-occurrence / stress-propagation graph.
- Spatial-topological dual-view Transformer: route features and dual graph
  link embeddings are fused before route attention.
- Intersection interaction gate: endpoint/intersection context gets a separate
  gated channel, especially useful for IIS.
- Masked multi-task link heads and order-level auxiliary heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from train_stage2_gnn_model import GNNRouteDataset, build_link_graph, collate  # noqa: E402
from train_stage2_route_local_transformer import masked_multitask_loss  # noqa: E402
from train_stage2_sequence_model import TARGETS, build_metadata, collect_orders, load_profiles, max_eval_orders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines_v2/dual_graph_route_transformer"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions_v2"))
    parser.add_argument("--profile-path", type=Path, default=Path("stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet"))
    parser.add_argument("--max-train-orders", type=int, default=60_000)
    parser.add_argument("--max-eval-orders", default="30000")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--cooccur-window", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cat-emb-dim", type=int, default=12)
    parser.add_argument("--link-emb-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--tail-loss-weight", type=float, default=0.35)
    parser.add_argument("--order-aux-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def profile_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("profile_") and not column.endswith("_count")]


def build_cooccurrence_graph(train: pd.DataFrame, link_map: dict[str, int], max_neighbors: int, window: int) -> torch.Tensor:
    counters: dict[int, Counter] = defaultdict(Counter)
    for _, group in train.sort_values(["order_id", "link_seq"]).groupby("order_id", sort=False):
        ids = [link_map.get(str(link), 0) for link in group.link_id.tolist()]
        for i, src in enumerate(ids):
            if src == 0:
                continue
            low = max(0, i - window)
            high = min(len(ids), i + window + 1)
            for j in range(low, high):
                dst = ids[j]
                if dst and dst != src:
                    counters[src][dst] += 1
    matrix = np.zeros((len(link_map) + 1, max_neighbors), dtype="int64")
    for src, counter in counters.items():
        neighbors = [dst for dst, _ in counter.most_common(max_neighbors)]
        matrix[src, : len(neighbors)] = neighbors
    return torch.from_numpy(matrix)


class DualGraphRouteTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_sizes: list[int],
        physical_neighbors: torch.Tensor,
        cooccur_neighbors: torch.Tensor,
        hidden_dim: int,
        cat_emb_dim: int,
        link_emb_dim: int,
        layers: int,
        heads: int,
        local_window: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, cat_emb_dim, padding_idx=0) for size in cat_sizes])
        self.physical_embedding = nn.Embedding(physical_neighbors.shape[0], link_emb_dim, padding_idx=0)
        self.cooccur_embedding = nn.Embedding(cooccur_neighbors.shape[0], link_emb_dim, padding_idx=0)
        self.register_buffer("physical_neighbors", physical_neighbors)
        self.register_buffer("cooccur_neighbors", cooccur_neighbors)
        self.physical_proj = nn.Sequential(nn.Linear(link_emb_dim * 2, link_emb_dim), nn.ReLU(), nn.LayerNorm(link_emb_dim))
        self.cooccur_proj = nn.Sequential(nn.Linear(link_emb_dim * 2, link_emb_dim), nn.ReLU(), nn.LayerNorm(link_emb_dim))
        input_dim = num_numeric + len(cat_sizes) * cat_emb_dim + link_emb_dim * 2
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.local_conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2), nn.GELU())
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
        self.intersection_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.shared_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.link_head = nn.Linear(hidden_dim, len(TARGET_ORDER))
        self.iis_head = nn.Linear(hidden_dim, 1)
        self.order_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(TARGET_ORDER)))
        self.local_window = local_window

    def local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        return (idx[None, :] - idx[:, None]).abs() > self.local_window

    def aggregate_graph(self, embedding: nn.Embedding, neighbors: torch.Tensor, link_idx: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        self_emb = embedding(link_idx)
        neighbor_ids = neighbors[link_idx]
        neighbor_emb = embedding(neighbor_ids)
        mask = neighbor_ids.ne(0).unsqueeze(-1)
        neighbor_mean = (neighbor_emb * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        return projection(torch.cat([self_emb, neighbor_mean], dim=-1))

    def forward(self, numeric, categorical, link_idx, pad_mask):
        embeddings = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        physical = self.aggregate_graph(self.physical_embedding, self.physical_neighbors, link_idx, self.physical_proj)
        cooccur = self.aggregate_graph(self.cooccur_embedding, self.cooccur_neighbors, link_idx, self.cooccur_proj)
        x = torch.cat([numeric] + embeddings + [physical, cooccur], dim=-1) if embeddings else torch.cat([numeric, physical, cooccur], dim=-1)
        x = self.input(x)
        x = x + self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x, mask=self.local_mask(x.shape[1], x.device), src_key_padding_mask=pad_mask)
        gate = self.intersection_gate(x)
        shared = self.shared_head(x)
        logits = self.link_head(shared)
        logits[:, :, 1:2] = logits[:, :, 1:2] + self.iis_head(shared * gate)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return torch.sigmoid(logits), torch.sigmoid(self.order_head(pooled))


def predict(model, loader, device):
    rows = []
    ys = {target: [] for target in TARGET_ORDER}
    preds = {target: [] for target in TARGET_ORDER}
    highs = {target: [] for target in TARGET_ORDER}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output, _ = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["link_idx"].to(device), batch["pad_mask"].to(device))
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
    link_map, physical_neighbors = build_link_graph(train, args.max_neighbors, args.seed)
    cooccur_neighbors = build_cooccurrence_graph(train, link_map, args.max_neighbors, args.cooccur_window)
    validation = collect_orders(args.dataset_root / "validation.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    test = collect_orders(args.dataset_root / "test.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    train_dataset = GNNRouteDataset(train, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    val_dataset = GNNRouteDataset(validation, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    test_dataset = GNNRouteDataset(test, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = DualGraphRouteTransformer(
        len(numeric_columns),
        [len(cat_maps[column]) + 1 for column in CATEGORICAL_COLUMNS],
        physical_neighbors.to(device),
        cooccur_neighbors.to(device),
        args.hidden_dim,
        args.cat_emb_dim,
        args.link_emb_dim,
        args.layers,
        args.heads,
        args.local_window,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred, order_pred = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["link_idx"].to(device), batch["pad_mask"].to(device))
            loss = masked_multitask_loss(pred, batch["target"].to(device), batch["high"].to(device), batch["mask"].to(device), order_pred, args.tail_loss_weight, args.order_aux_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"epoch={epoch} train_loss={history[-1]['loss']:.5f}", flush=True)
    val_pred, val_metrics = predict(model, val_loader, device)
    test_pred, test_metrics = predict(model, test_loader, device)
    val_pred.to_parquet(args.prediction_root / "dual_graph_route_transformer_validation.parquet", index=False, compression="zstd")
    test_pred.to_parquet(args.prediction_root / "dual_graph_route_transformer_test.parquet", index=False, compression="zstd")
    rows = []
    for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        for target, row in metrics.items():
            rows.append({"split": split, "target": target, **row})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_root / "dual_graph_route_transformer_metrics_by_target.csv", index=False)
    manifest = {
        "model": "DualGraphRouteTransformer",
        "device": str(device),
        "max_train_orders": args.max_train_orders,
        "max_eval_orders": args.max_eval_orders,
        "max_seq_len": args.max_seq_len,
        "link_nodes": len(link_map),
        "cooccur_window": args.cooccur_window,
        "history": history,
        "metrics": rows,
    }
    (args.output_root / "dual_graph_route_transformer_metrics.json").write_text(json.dumps(safe_json_float(manifest), indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "manifest": manifest, "link_map": link_map}, args.output_root / "dual_graph_route_transformer.pt")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
