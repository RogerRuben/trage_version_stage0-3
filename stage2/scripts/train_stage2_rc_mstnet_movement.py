"""Train a movement-level IIS head for Deep v3.

This script keeps IIS separate from ordinary link heads.  It predicts
applicability and severity/tail only on movement rows, without filling missing
severity as zero.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import Timer, load_fold_config, metric_dict, safe_float  # noqa: E402


CATEGORICAL = ["planned_link_id", "from_link_id", "to_link_id", "node_id", "road_class", "area_grid", "turn_type", "estimated_time_bin"]
NUMERIC_BASE = [
    "planned_link_seq",
    "planned_route_link_count",
    "position_ratio",
    "distance_to_destination_ratio",
    "planned_link_length_m",
    "endpoint_degree",
    "node_degree",
    "junction_complexity",
    "turn_angle",
    "activity_intensity_index",
    "rolling_iis_raw_mean",
    "rolling_iis_raw_std",
    "rolling_iis_history_count",
]
ID_COLUMNS = ["order_id", "date", "planned_link_id", "planned_link_seq", "from_link_id", "node_id", "to_link_id", "movement_key"]
TARGET_COLUMNS = [
    "iis_applicable", "iis_observed", "target_iis_valid", "target_iis_raw", "target_iis_tail90_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/iis_movement_causal_dataset"))
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/feasibility_100k/rc_mstnet_movement"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_v3/rolling_predictions/rc_mstnet_movement"))
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--severity-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def existing_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def movement_read_columns(path: Path) -> list[str]:
    available = set(existing_columns(path))
    columns = [column for column in ID_COLUMNS + TARGET_COLUMNS + CATEGORICAL + NUMERIC_BASE if column in available]
    columns += [column for column in available if column.startswith("poi_density_100m_")]
    columns += [
        column for column in available
        if any(column.startswith(prefix) for prefix in ["link_recent_", "area_recent_", "network_recent_", "upstream_recent_", "downstream_recent_"])
        and "timestamp" not in column
    ]
    return list(dict.fromkeys(columns))


def read_dates(root: Path, dates: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    parts = []
    per_day_budget = None
    if max_rows:
        per_day_budget = max(1, int(np.ceil(max_rows / max(1, len(dates)))))
    for index, date in enumerate(dates):
        path = root / f"day={date}.parquet"
        frame = pd.read_parquet(path, columns=movement_read_columns(path))
        if per_day_budget and len(frame) > per_day_budget:
            frame = frame.sample(n=per_day_budget, random_state=seed + index)
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    return frame.reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    cols = [column for column in NUMERIC_BASE if column in frame.columns]
    cols += [column for column in frame.columns if column.startswith("poi_density_100m_")]
    cols += [column for column in frame.columns if any(column.startswith(prefix) for prefix in ["link_recent_", "area_recent_", "network_recent_", "upstream_recent_", "downstream_recent_"]) and "timestamp" not in column]
    return list(dict.fromkeys(cols))


def build_metadata(train: pd.DataFrame) -> dict:
    cats = [column for column in CATEGORICAL if column in train.columns]
    cat_maps = {}
    for column in cats:
        values = train[column].astype("string").fillna("__MISSING__")
        cat_maps[column] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
    nums = numeric_columns(train)
    numeric = train[nums].apply(pd.to_numeric, errors="coerce")
    return {
        "categorical": cats,
        "numeric": nums,
        "cat_maps": cat_maps,
        "mean": numeric.mean().fillna(0).to_dict(),
        "std": numeric.std().replace(0, 1).fillna(1).to_dict(),
    }


class MovementDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, metadata: dict):
        self.frame = frame.reset_index(drop=True)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        nums = []
        for column in self.metadata["numeric"]:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                value = self.metadata["mean"].get(column, 0.0)
            nums.append((float(value) - self.metadata["mean"].get(column, 0.0)) / self.metadata["std"].get(column, 1.0))
        cats = []
        for column in self.metadata["categorical"]:
            cats.append(self.metadata["cat_maps"][column].get(str(row.get(column, "__MISSING__")), 0))
        applicable = bool(row.get("iis_applicable", False))
        observed = bool(row.get("iis_observed", False)) and bool(row.get("target_iis_valid", False))
        severity = float(row.get("target_iis_raw", 0.0)) if observed and pd.notna(row.get("target_iis_raw")) else 0.0
        tail = float(row.get("target_iis_tail90_raw", 0.0)) if observed and pd.notna(row.get("target_iis_tail90_raw")) else 0.0
        ids = {key: row.get(key) for key in ID_COLUMNS if key in self.frame.columns}
        return {
            "numeric": torch.tensor(nums, dtype=torch.float32),
            "categorical": torch.tensor(cats, dtype=torch.long),
            "applicable": torch.tensor(float(applicable), dtype=torch.float32),
            "severity": torch.tensor(severity, dtype=torch.float32),
            "tail": torch.tensor(tail, dtype=torch.float32),
            "severity_mask": torch.tensor(float(observed), dtype=torch.float32),
            "id": ids,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "numeric": torch.stack([item["numeric"] for item in batch]),
        "categorical": torch.stack([item["categorical"] for item in batch]),
        "applicable": torch.stack([item["applicable"] for item in batch]),
        "severity": torch.stack([item["severity"] for item in batch]),
        "tail": torch.stack([item["tail"] for item in batch]),
        "severity_mask": torch.stack([item["severity_mask"] for item in batch]),
        "ids": [item["id"] for item in batch],
    }


class MovementNet(nn.Module):
    def __init__(self, n_numeric: int, cat_sizes: list[int], emb_dim: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(size, emb_dim, padding_idx=0) for size in cat_sizes])
        input_dim = n_numeric + len(cat_sizes) * emb_dim
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.15), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.app_head = nn.Linear(hidden_dim, 1)
        self.sev_head = nn.Linear(hidden_dim, 1)
        self.tail_head = nn.Linear(hidden_dim, 1)

    def forward(self, numeric, categorical):
        embeddings = [emb(categorical[:, i]) for i, emb in enumerate(self.emb)]
        x = torch.cat([numeric] + embeddings, dim=1) if embeddings else numeric
        h = self.backbone(x)
        return torch.sigmoid(self.app_head(h)).squeeze(1), torch.sigmoid(self.sev_head(h)).squeeze(1), torch.sigmoid(self.tail_head(h)).squeeze(1)


def run_fold(args: argparse.Namespace, fold: dict, device: torch.device) -> dict:
    fold_id = int(fold["fold"])
    seed = args.seed + fold_id
    train = read_dates(args.dataset_root, fold["train_dates"], args.max_train_rows, seed)
    validation = read_dates(args.dataset_root, [fold["validation_date"]], None, seed)
    test = read_dates(args.dataset_root, [fold["test_date"]], None, seed)
    metadata = build_metadata(train)
    datasets = {name: MovementDataset(frame, metadata) for name, frame in [("train", train), ("validation", validation), ("test", test)]}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collate),
    }
    model = MovementNet(len(metadata["numeric"]), [len(metadata["cat_maps"][column]) + 1 for column in metadata["categorical"]], args.emb_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        with Timer() as timer:
            for batch in loaders["train"]:
                optimizer.zero_grad()
                app, sev, tail = model(batch["numeric"].to(device), batch["categorical"].to(device))
                app_loss = nn.functional.binary_cross_entropy(app.clamp(1e-5, 1 - 1e-5), batch["applicable"].to(device))
                mask = batch["severity_mask"].to(device)
                sev_loss = (nn.functional.huber_loss(sev, batch["severity"].to(device), reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
                tail_loss = (nn.functional.binary_cross_entropy(tail.clamp(1e-5, 1 - 1e-5), batch["tail"].to(device), reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
                loss = app_loss + args.severity_weight * sev_loss + args.tail_weight * tail_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "epoch_seconds": timer.seconds})
        print(f"fold={fold_id} movement epoch={epoch} loss={history[-1]['train_loss']:.5f}", flush=True)

    def predict(split: str) -> tuple[pd.DataFrame, dict]:
        rows, app_y, app_p, sev_y, sev_p, tail_y, tail_p = [], [], [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loaders[split]:
                app, sev, tail = model(batch["numeric"].to(device), batch["categorical"].to(device))
                app = app.cpu().numpy()
                sev = sev.cpu().numpy()
                tail_pred = tail.cpu().numpy()
                for i, id_row in enumerate(batch["ids"]):
                    row = dict(id_row)
                    row["pred_iis_applicability"] = float(app[i])
                    row["pred_iis_severity"] = float(sev[i])
                    row["pred_iis_tail_prob"] = float(tail_pred[i])
                    row["iis_applicable"] = bool(batch["applicable"][i])
                    row["iis_observed"] = bool(batch["severity_mask"][i])
                    row["iis_prediction_available"] = True
                    row["iis_severity_prediction_available"] = True
                    row["target_iis_raw"] = float(batch["severity"][i]) if row["iis_observed"] else np.nan
                    row["target_iis_tail"] = bool(batch["tail"][i]) if row["iis_observed"] else False
                    rows.append(row)
                    app_y.append(row["iis_applicable"])
                    app_p.append(row["pred_iis_applicability"])
                    if row["iis_observed"]:
                        sev_y.append(row["target_iis_raw"])
                        sev_p.append(row["pred_iis_severity"])
                        tail_y.append(row["target_iis_tail"])
                        tail_p.append(row["pred_iis_tail_prob"])
        metrics = {
            "applicability_rows": len(app_y),
            "applicability_rate": float(np.mean(app_y)) if app_y else np.nan,
            "applicability_auc": float(roc_auc_score(app_y, app_p)) if len(set(app_y)) == 2 else np.nan,
            "applicability_ap": float(average_precision_score(app_y, app_p)) if len(set(app_y)) == 2 else np.nan,
            "severity": metric_dict(np.array(sev_y), np.array(sev_p), np.array(tail_p), np.array(tail_y, dtype=bool)) if sev_y else {},
        }
        return pd.DataFrame(rows), metrics

    pred_root = args.prediction_root / f"fold={fold_id}"
    pred_root.mkdir(parents=True, exist_ok=True)
    fold_root = args.output_root / f"fold={fold_id}"
    fold_root.mkdir(parents=True, exist_ok=True)
    val_pred, val_metrics = predict("validation")
    test_pred, test_metrics = predict("test")
    val_pred.to_parquet(pred_root / "validation_movement_predictions.parquet", index=False, compression="zstd")
    test_pred.to_parquet(pred_root / "test_movement_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "fold": fold_id,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }
    (fold_root / "manifest.json").write_text(json.dumps(safe_float(manifest), indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, fold_root / "rc_mstnet_movement.pt")
    return manifest


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = {int(part.strip()) for part in args.folds.split(",") if part.strip()}
    manifests = [run_fold(args, fold, device) for fold in load_fold_config(args.fold_config) if int(fold["fold"]) in selected]
    rows = []
    for manifest in manifests:
        for split, metrics in [("validation", manifest["validation_metrics"]), ("test", manifest["test_metrics"])]:
            sev = metrics.get("severity", {})
            rows.append({"fold": manifest["fold"], "split": split, "target": "IIS_APP", "auc": metrics.get("applicability_auc"), "ap": metrics.get("applicability_ap"), "rows": metrics.get("applicability_rows")})
            rows.append({"fold": manifest["fold"], "split": split, "target": "IIS_SEVERITY", **sev})
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "rc_mstnet_movement_metrics_by_fold.csv", index=False)
    (args.output_root / "rc_mstnet_movement_manifest.json").write_text(json.dumps(safe_float({"folds": manifests}), indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
