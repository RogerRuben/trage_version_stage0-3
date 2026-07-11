"""Train route-sequence neural upper-bound models for Stage2.

Default model: BiGRU sequence encoder with masked multi-head link predictions.
IIS missing labels are masked, never filled as zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_model_utils import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_COLUMNS,
    TARGETS,
    TARGET_ORDER,
    evaluate_predictions,
    safe_json_float,
    unique_existing_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_baselines/sequence_model"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_predictions"))
    parser.add_argument("--profile-path", type=Path, default=Path("stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet"))
    parser.add_argument("--model-type", choices=["bigru", "transformer", "tcn"], default="bigru")
    parser.add_argument("--max-train-orders", type=int, default=80_000)
    parser.add_argument("--max-eval-orders", default="all")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--cat-emb-dim", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tail-loss-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def max_eval_orders(value: str) -> int | None:
    return None if value.lower() == "all" else int(value)


def target_columns() -> list[str]:
    columns = []
    for target, mask, high in TARGETS.values():
        columns.extend([target, mask, high])
    return columns


def load_profiles(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def add_profiles(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return frame
    return frame.merge(profiles, on=["link_id", "time_bin"], how="left")


def collect_orders(
    path: Path,
    profiles: pd.DataFrame,
    max_orders: int | None,
    max_seq_len: int,
    seed: int,
) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    columns = unique_existing_columns(path, ID_COLUMNS + FEATURE_COLUMNS + target_columns())
    frames = []
    orders_seen = 0
    rng = np.random.default_rng(seed)
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
        frame = add_profiles(frame, profiles)
        order_ids = frame.order_id.drop_duplicates().to_numpy()
        if max_orders is not None:
            remaining = max_orders - orders_seen
            if remaining <= 0:
                break
            if len(order_ids) > remaining:
                order_ids = rng.choice(order_ids, size=remaining, replace=False)
                frame = frame[frame.order_id.isin(order_ids)]
        frames.append(frame)
        orders_seen += len(order_ids)
        if max_orders is not None and orders_seen >= max_orders:
            break
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    data = data.sort_values(["order_id", "link_seq"], kind="mergesort")
    if max_seq_len:
        data = data.groupby("order_id", group_keys=False).head(max_seq_len)
    return data.reset_index(drop=True)


def profile_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("profile_") and not column.endswith("_count")]


class RouteDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
        cat_maps: dict[str, dict],
        numeric_mean: pd.Series,
        numeric_std: pd.Series,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.cat_maps = cat_maps
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std.replace(0, 1).fillna(1)
        self.groups = [group.copy() for _, group in frame.groupby("order_id", sort=False)]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list]:
        group = self.groups[idx]
        numeric = group[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = ((numeric - self.numeric_mean) / self.numeric_std).fillna(0).to_numpy(dtype="float32")
        cats = []
        for column in self.categorical_columns:
            mapping = self.cat_maps[column]
            cats.append(group[column].astype(str).map(mapping).fillna(0).to_numpy(dtype="int64"))
        cat_array = np.stack(cats, axis=1) if cats else np.zeros((len(group), 0), dtype="int64")
        y = np.stack([pd.to_numeric(group[TARGETS[target][0]], errors="coerce").to_numpy(dtype="float32") for target in TARGET_ORDER], axis=1)
        masks = np.stack([group[TARGETS[target][1]].fillna(False).to_numpy(dtype=bool) for target in TARGET_ORDER], axis=1)
        high = np.stack([group[TARGETS[target][2]].fillna(False).to_numpy(dtype=bool) for target in TARGET_ORDER], axis=1)
        return {
            "numeric": torch.from_numpy(numeric),
            "categorical": torch.from_numpy(cat_array),
            "target": torch.from_numpy(np.nan_to_num(y, nan=0.0)),
            "mask": torch.from_numpy(masks.astype("float32")),
            "high": torch.from_numpy(high.astype("float32")),
            "ids": group[ID_COLUMNS].to_dict(orient="records"),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    max_len = max(item["numeric"].shape[0] for item in batch)
    n_num = batch[0]["numeric"].shape[1]
    n_cat = batch[0]["categorical"].shape[1]
    size = len(batch)
    numeric = torch.zeros(size, max_len, n_num)
    categorical = torch.zeros(size, max_len, n_cat, dtype=torch.long)
    target = torch.zeros(size, max_len, len(TARGET_ORDER))
    mask = torch.zeros(size, max_len, len(TARGET_ORDER))
    high = torch.zeros(size, max_len, len(TARGET_ORDER))
    pad_mask = torch.ones(size, max_len, dtype=torch.bool)
    ids = []
    for i, item in enumerate(batch):
        length = item["numeric"].shape[0]
        numeric[i, :length] = item["numeric"]
        categorical[i, :length] = item["categorical"]
        target[i, :length] = item["target"]
        mask[i, :length] = item["mask"]
        high[i, :length] = item["high"]
        pad_mask[i, :length] = False
        ids.append(item["ids"])
    return {"numeric": numeric, "categorical": categorical, "target": target, "mask": mask, "high": high, "pad_mask": pad_mask, "ids": ids}


class SequenceModel(nn.Module):
    def __init__(self, num_numeric: int, cat_sizes: list[int], hidden_dim: int, cat_emb_dim: int, model_type: str):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, cat_emb_dim, padding_idx=0) for size in cat_sizes])
        input_dim = num_numeric + len(cat_sizes) * cat_emb_dim
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.model_type = model_type
        if model_type == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 3, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            encoder_dim = hidden_dim
        elif model_type == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            encoder_dim = hidden_dim
        else:
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            encoder_dim = hidden_dim * 2
        self.head = nn.Sequential(nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(TARGET_ORDER)))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        embeddings = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([numeric] + embeddings, dim=-1) if embeddings else numeric
        x = self.input(x)
        if self.model_type == "transformer":
            x = self.encoder(x, src_key_padding_mask=pad_mask)
        elif self.model_type == "tcn":
            x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        else:
            x, _ = self.encoder(x)
        return torch.sigmoid(self.head(x))


def build_metadata(train: pd.DataFrame, numeric_columns: list[str]) -> tuple[dict[str, dict], pd.Series, pd.Series]:
    cat_maps = {}
    for column in CATEGORICAL_COLUMNS:
        values = train[column].astype(str).fillna("__MISSING__").unique().tolist()
        cat_maps[column] = {value: i + 1 for i, value in enumerate(values)}
    numeric = train[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return cat_maps, numeric.mean(), numeric.std().replace(0, 1).fillna(1)


def masked_loss(pred: torch.Tensor, target: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, tail_weight: float) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask).sum() / denominator
    bce = nn.functional.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), high, reduction="none")
    bce = (bce * mask).sum() / denominator
    return mse + tail_weight * bce


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    ys = {target: [] for target in TARGET_ORDER}
    preds = {target: [] for target in TARGET_ORDER}
    highs = {target: [] for target in TARGET_ORDER}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            output = model(numeric, categorical, pad_mask).cpu().numpy()
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
    metrics = {
        target: evaluate_predictions(np.array(ys[target]), np.array(preds[target]), np.array(highs[target], dtype=bool))
        for target in TARGET_ORDER
    }
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
    profile_cols = [column for column in train.columns if column.startswith("profile_") and not column.endswith("_count")]
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

    model = SequenceModel(
        num_numeric=len(numeric_columns),
        cat_sizes=[len(cat_maps[column]) + 1 for column in CATEGORICAL_COLUMNS],
        hidden_dim=args.hidden_dim,
        cat_emb_dim=args.cat_emb_dim,
        model_type=args.model_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["numeric"].to(device), batch["categorical"].to(device), batch["pad_mask"].to(device))
            loss = masked_loss(pred, batch["target"].to(device), batch["high"].to(device), batch["mask"].to(device), args.tail_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.5f}", flush=True)

    val_pred, val_metrics = predict(model, val_loader, device)
    test_pred, test_metrics = predict(model, test_loader, device)
    val_pred.to_parquet(args.prediction_root / "sequence_validation.parquet", index=False, compression="zstd")
    test_pred.to_parquet(args.prediction_root / "sequence_test.parquet", index=False, compression="zstd")
    metrics_rows = []
    for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        for target, row in metrics.items():
            metrics_rows.append({"split": split, "target": target, **row})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_root / "sequence_metrics_by_target.csv", index=False)
    manifest = {
        "model_type": args.model_type,
        "device": str(device),
        "max_train_orders": args.max_train_orders,
        "max_eval_orders": args.max_eval_orders,
        "max_seq_len": args.max_seq_len,
        "train_orders": len(train_dataset),
        "validation_orders": len(val_dataset),
        "test_orders": len(test_dataset),
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "history": history,
        "metrics": metrics_rows,
    }
    (args.output_root / "sequence_metrics.json").write_text(json.dumps(safe_json_float(manifest), indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "manifest": manifest}, args.output_root / "sequence_model.pt")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
