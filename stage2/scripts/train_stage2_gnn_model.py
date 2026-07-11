"""Train topology-aware GraphSAGE-style route sequence models for Stage2.

This implementation avoids PyG/DGL dependencies. It builds a link adjacency
graph from train route sequences, learns link embeddings with neighbor mean
aggregation, and feeds graph-aware link representations into a route sequence
encoder with masked multi-task heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_model_utils import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_ORDER,
    evaluate_predictions,
    safe_json_float,
)
from train_stage2_sequence_model import (  # noqa: E402
    TARGETS,
    build_metadata,
    collect_orders,
    load_profiles,
    masked_loss,
    max_eval_orders,
    parse_args as _sequence_parse_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines/gnn_model"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions"))
    parser.add_argument("--profile-path", type=Path, default=Path("stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet"))
    parser.add_argument("--model-type", choices=["bigru", "transformer"], default="bigru")
    parser.add_argument("--max-train-orders", type=int, default=80_000)
    parser.add_argument("--max-eval-orders", default="all")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--cat-emb-dim", type=int, default=12)
    parser.add_argument("--link-emb-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tail-loss-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def build_link_graph(train: pd.DataFrame, max_neighbors: int, seed: int) -> tuple[dict, torch.Tensor]:
    link_values = train.link_id.astype(str).drop_duplicates().tolist()
    link_map = {link_id: idx + 1 for idx, link_id in enumerate(link_values)}
    neighbors: dict[int, set[int]] = defaultdict(set)
    for _, group in train.sort_values(["order_id", "link_seq"]).groupby("order_id", sort=False):
        indices = [link_map.get(str(link_id), 0) for link_id in group.link_id.tolist()]
        for left, right in zip(indices[:-1], indices[1:]):
            if left and right and left != right:
                neighbors[left].add(right)
                neighbors[right].add(left)
    rng = np.random.default_rng(seed)
    matrix = np.zeros((len(link_map) + 1, max_neighbors), dtype="int64")
    for idx, values in neighbors.items():
        values = list(values)
        if len(values) > max_neighbors:
            values = rng.choice(values, size=max_neighbors, replace=False).tolist()
        matrix[idx, : len(values)] = values
    return link_map, torch.from_numpy(matrix)


class GNNRouteDataset(torch.utils.data.Dataset):
    def __init__(self, frame, numeric_columns, categorical_columns, cat_maps, link_map, numeric_mean, numeric_std):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.cat_maps = cat_maps
        self.link_map = link_map
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std.replace(0, 1).fillna(1)
        self.groups = [group.copy() for _, group in frame.groupby("order_id", sort=False)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        numeric = group[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = ((numeric - self.numeric_mean) / self.numeric_std).fillna(0).to_numpy(dtype="float32")
        cats = []
        for column in self.categorical_columns:
            cats.append(group[column].astype(str).map(self.cat_maps[column]).fillna(0).to_numpy(dtype="int64"))
        categorical = np.stack(cats, axis=1) if cats else np.zeros((len(group), 0), dtype="int64")
        link_idx = group.link_id.astype(str).map(self.link_map).fillna(0).to_numpy(dtype="int64")
        y = np.stack([pd.to_numeric(group[TARGETS[target][0]], errors="coerce").to_numpy(dtype="float32") for target in TARGET_ORDER], axis=1)
        masks = np.stack([group[TARGETS[target][1]].fillna(False).to_numpy(dtype=bool) for target in TARGET_ORDER], axis=1)
        high = np.stack([group[TARGETS[target][2]].fillna(False).to_numpy(dtype=bool) for target in TARGET_ORDER], axis=1)
        return {
            "numeric": torch.from_numpy(numeric),
            "categorical": torch.from_numpy(categorical),
            "link_idx": torch.from_numpy(link_idx),
            "target": torch.from_numpy(np.nan_to_num(y, nan=0.0)),
            "mask": torch.from_numpy(masks.astype("float32")),
            "high": torch.from_numpy(high.astype("float32")),
            "ids": group[ID_COLUMNS].to_dict(orient="records"),
        }


def collate(batch):
    max_len = max(item["numeric"].shape[0] for item in batch)
    size = len(batch)
    n_num = batch[0]["numeric"].shape[1]
    n_cat = batch[0]["categorical"].shape[1]
    numeric = torch.zeros(size, max_len, n_num)
    categorical = torch.zeros(size, max_len, n_cat, dtype=torch.long)
    link_idx = torch.zeros(size, max_len, dtype=torch.long)
    target = torch.zeros(size, max_len, len(TARGET_ORDER))
    mask = torch.zeros(size, max_len, len(TARGET_ORDER))
    high = torch.zeros(size, max_len, len(TARGET_ORDER))
    pad_mask = torch.ones(size, max_len, dtype=torch.bool)
    ids = []
    for i, item in enumerate(batch):
        length = item["numeric"].shape[0]
        numeric[i, :length] = item["numeric"]
        categorical[i, :length] = item["categorical"]
        link_idx[i, :length] = item["link_idx"]
        target[i, :length] = item["target"]
        mask[i, :length] = item["mask"]
        high[i, :length] = item["high"]
        pad_mask[i, :length] = False
        ids.append(item["ids"])
    return {"numeric": numeric, "categorical": categorical, "link_idx": link_idx, "target": target, "mask": mask, "high": high, "pad_mask": pad_mask, "ids": ids}


class GraphSequenceModel(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_sizes: list[int],
        neighbor_index: torch.Tensor,
        hidden_dim: int,
        cat_emb_dim: int,
        link_emb_dim: int,
        model_type: str,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, cat_emb_dim, padding_idx=0) for size in cat_sizes])
        self.link_embedding = nn.Embedding(neighbor_index.shape[0], link_emb_dim, padding_idx=0)
        self.register_buffer("neighbor_index", neighbor_index)
        self.graph_projection = nn.Sequential(nn.Linear(link_emb_dim * 2, link_emb_dim), nn.ReLU(), nn.LayerNorm(link_emb_dim))
        input_dim = num_numeric + len(cat_sizes) * cat_emb_dim + link_emb_dim
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.model_type = model_type
        if model_type == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 3, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            encoder_dim = hidden_dim
        else:
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            encoder_dim = hidden_dim * 2
        self.head = nn.Sequential(nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(TARGET_ORDER)))

    def graph_embed(self, link_idx: torch.Tensor) -> torch.Tensor:
        self_emb = self.link_embedding(link_idx)
        neighbor_ids = self.neighbor_index[link_idx]
        neighbor_emb = self.link_embedding(neighbor_ids)
        neighbor_mask = neighbor_ids.ne(0).unsqueeze(-1)
        neighbor_sum = (neighbor_emb * neighbor_mask).sum(dim=2)
        neighbor_count = neighbor_mask.sum(dim=2).clamp_min(1)
        neighbor_mean = neighbor_sum / neighbor_count
        return self.graph_projection(torch.cat([self_emb, neighbor_mean], dim=-1))

    def forward(self, numeric, categorical, link_idx, pad_mask):
        embeddings = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        graph = self.graph_embed(link_idx)
        x = torch.cat([numeric] + embeddings + [graph], dim=-1) if embeddings else torch.cat([numeric, graph], dim=-1)
        x = self.input(x)
        if self.model_type == "transformer":
            x = self.encoder(x, src_key_padding_mask=pad_mask)
        else:
            x, _ = self.encoder(x)
        return torch.sigmoid(self.head(x))


def predict(model, loader, device):
    rows = []
    ys = {target: [] for target in TARGET_ORDER}
    preds = {target: [] for target in TARGET_ORDER}
    highs = {target: [] for target in TARGET_ORDER}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["numeric"].to(device),
                batch["categorical"].to(device),
                batch["link_idx"].to(device),
                batch["pad_mask"].to(device),
            ).cpu().numpy()
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profile_path)
    train = collect_orders(args.dataset_root / "train.parquet", profiles, args.max_train_orders, args.max_seq_len, args.seed)
    profile_cols = [column for column in train.columns if column.startswith("profile_") and not column.endswith("_count")]
    numeric_columns = [column for column in NUMERIC_COLUMNS + profile_cols if column in train.columns]
    cat_maps, numeric_mean, numeric_std = build_metadata(train, numeric_columns)
    link_map, neighbor_index = build_link_graph(train, args.max_neighbors, args.seed)

    validation = collect_orders(args.dataset_root / "validation.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    test = collect_orders(args.dataset_root / "test.parquet", profiles, max_eval_orders(args.max_eval_orders), args.max_seq_len, args.seed)
    train_dataset = GNNRouteDataset(train, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    val_dataset = GNNRouteDataset(validation, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    test_dataset = GNNRouteDataset(test, numeric_columns, CATEGORICAL_COLUMNS, cat_maps, link_map, numeric_mean, numeric_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = GraphSequenceModel(
        len(numeric_columns),
        [len(cat_maps[column]) + 1 for column in CATEGORICAL_COLUMNS],
        neighbor_index.to(device),
        args.hidden_dim,
        args.cat_emb_dim,
        args.link_emb_dim,
        args.model_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["link_idx"].to(device), batch["pad_mask"].to(device))
            loss = masked_loss(pred, batch["target"].to(device), batch["high"].to(device), batch["mask"].to(device), args.tail_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.5f}", flush=True)

    val_pred, val_metrics = predict(model, val_loader, device)
    test_pred, test_metrics = predict(model, test_loader, device)
    val_pred.to_parquet(args.prediction_root / "gnn_sequence_validation.parquet", index=False, compression="zstd")
    test_pred.to_parquet(args.prediction_root / "gnn_sequence_test.parquet", index=False, compression="zstd")
    rows = []
    for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        for target, row in metrics.items():
            rows.append({"split": split, "target": target, **row})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_root / "gnn_metrics_by_target.csv", index=False)
    manifest = {
        "model_type": args.model_type,
        "device": str(device),
        "max_train_orders": args.max_train_orders,
        "max_eval_orders": args.max_eval_orders,
        "max_seq_len": args.max_seq_len,
        "max_neighbors": args.max_neighbors,
        "train_orders": len(train_dataset),
        "validation_orders": len(val_dataset),
        "test_orders": len(test_dataset),
        "link_nodes": len(link_map),
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "history": history,
        "metrics": rows,
    }
    (args.output_root / "gnn_metrics.json").write_text(json.dumps(safe_json_float(manifest), indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "manifest": manifest, "link_map": link_map}, args.output_root / "gnn_model.pt")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
