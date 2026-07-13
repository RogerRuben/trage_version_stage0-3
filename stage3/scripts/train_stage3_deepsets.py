"""Lightweight route-attention/DeepSets Stage3 order predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

STAGE2_SCRIPTS = Path(__file__).resolve().parents[2] / "stage2" / "scripts"
if str(STAGE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE2_SCRIPTS))
from stage2_deep_v3_utils import metric_dict  # noqa: E402


TARGETS = ["lcs", "pmis", "rts"]
LINK_FEATURES = sum(([f"{target}_raw_pred", f"{target}_tail_prob_calibrated", f"{target}_uncertainty"] for target in TARGETS), []) + ["position_ratio", "route_link_length_m"]
IIS_ORDER_FEATURES = [
    "iis_prediction_available",
    "iis_availability",
    "iis_applicability_mean",
    "iis_applicability_q90",
    "iis_predicted_applicable_share",
    "iis_severity_mean",
    "iis_severity_q90",
    "iis_severity_max",
    "iis_tail_prob_mean",
    "iis_tail_prob_q90",
    "iis_tail_share_50",
    "modality_coverage_score",
    "route_prediction_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--order-feature-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/deepsets"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-iis", action="store_true")
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, default=None, help="Optional Stage3 rolling fold id.")
    return parser.parse_args()


class OrderSequenceDataset(Dataset):
    def __init__(self, link: pd.DataFrame, target: pd.DataFrame, mean: pd.Series, std: pd.Series, order_features: pd.DataFrame | None, order_mean: pd.Series | None, order_std: pd.Series | None):
        target = target.set_index("order_id")
        features = order_features.set_index("order_id") if order_features is not None else None
        self.items = []
        for order_id, group in link.sort_values(["order_id", "route_link_seq"]).groupby("order_id", sort=False):
            if order_id not in target.index:
                continue
            x = group[LINK_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(mean)
            x = ((x - mean) / std).fillna(0).to_numpy("float32")
            if features is not None and order_id in features.index:
                order_x = features.loc[order_id, IIS_ORDER_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(order_mean)
                order_x = ((order_x - order_mean) / order_std).fillna(0).to_numpy("float32")
                availability = float(bool(features.loc[order_id].get("iis_prediction_available", False)))
            else:
                order_x = np.zeros(len(IIS_ORDER_FEATURES), dtype="float32")
                availability = 0.0
            row = target.loc[order_id]
            raw = np.array([row[f"order_{name}_raw"] for name in TARGETS], dtype="float32")
            tail = np.array([row[f"order_{name}_tail"] for name in TARGETS], dtype="float32")
            self.items.append((order_id, x, order_x, availability, np.nan_to_num(raw), np.isfinite(raw).astype("float32"), tail, float(row["order_overall_high_stress"])))

    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        order, x, order_x, availability, raw, mask, tail, overall = self.items[index]
        return {"order_id": order, "x": torch.from_numpy(x), "order_x": torch.from_numpy(order_x), "availability": torch.tensor(availability, dtype=torch.float32), "raw": torch.from_numpy(raw), "mask": torch.from_numpy(mask), "tail": torch.from_numpy(tail), "overall": torch.tensor(overall, dtype=torch.float32)}


def collate(batch):
    length = max(len(item["x"]) for item in batch); features = batch[0]["x"].shape[1]
    x = torch.zeros(len(batch), length, features); pad = torch.ones(len(batch), length, dtype=torch.bool)
    for index, item in enumerate(batch): x[index, :len(item["x"])] = item["x"]; pad[index, :len(item["x"])] = False
    return {"order_id": [item["order_id"] for item in batch], "x": x, "pad": pad, "order_x": torch.stack([item["order_x"] for item in batch]), "availability": torch.stack([item["availability"] for item in batch]), "raw": torch.stack([item["raw"] for item in batch]), "mask": torch.stack([item["mask"] for item in batch]), "tail": torch.stack([item["tail"] for item in batch]), "overall": torch.stack([item["overall"] for item in batch])}


class RouteAttention(nn.Module):
    def __init__(self, inputs: int, hidden: int, order_inputs: int = 0):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.attention = nn.Linear(hidden, 1)
        self.order_encoder = nn.Sequential(nn.Linear(order_inputs, hidden), nn.GELU(), nn.LayerNorm(hidden)) if order_inputs else None
        self.rho = nn.Sequential(nn.Linear(hidden * (3 if order_inputs else 2), hidden), nn.GELU(), nn.Dropout(.15))
        self.raw = nn.Linear(hidden, 3); self.tail = nn.Linear(hidden, 3); self.overall = nn.Linear(hidden, 1)

    def forward(self, x, pad, order_x=None):
        h = self.phi(x); logits = self.attention(h).squeeze(-1).masked_fill(pad, -1e4)
        pooled_attention = (h * torch.softmax(logits, dim=1).unsqueeze(-1)).sum(1)
        pooled_max = h.masked_fill(pad.unsqueeze(-1), -1e4).amax(1)
        parts = [pooled_attention, pooled_max]
        if self.order_encoder is not None:
            parts.append(self.order_encoder(order_x))
        z = self.rho(torch.cat(parts, dim=1))
        return torch.sigmoid(self.raw(z)), self.tail(z), self.overall(z).squeeze(1)


def read_many(base: Path) -> pd.DataFrame:
    paths = sorted(base.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {base}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_split(warehouse: Path, targets: Path, split: str, order_feature_root: Path | None, fold: int | None = None):
    if fold is None:
        link_base = warehouse / "link_predictions" / f"split={split}"
    else:
        link_base = warehouse / "link_predictions" / f"fold={fold}" / f"split={split}"
    link = read_many(link_base)
    target = pd.read_parquet(targets / f"split={split}" / "order_targets.parquet")
    features = pd.read_parquet(order_feature_root / f"split={split}" / "order_features.parquet") if order_feature_root else None
    return link, target, features


def evaluate(model, loader, device):
    rows = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            raw, tail_logits, overall_logits = model(batch["x"].to(device), batch["pad"].to(device), batch["order_x"].to(device))
            raw = raw.cpu().numpy(); prob = torch.sigmoid(tail_logits).cpu().numpy(); overall = torch.sigmoid(overall_logits).cpu().numpy()
            for i, order in enumerate(batch["order_id"]):
                for k, target in enumerate(TARGETS): rows.append({"order_id": order, "target": target.upper(), "true_raw": float(batch["raw"][i,k]), "true_tail": bool(batch["tail"][i,k]), "pred_raw": float(raw[i,k]), "pred_probability": float(prob[i,k])})
                rows.append({"order_id": order, "target": "OVERALL", "true_raw": float(batch["raw"][i].max()), "true_tail": bool(batch["overall"][i]), "pred_raw": float(overall[i]), "pred_probability": float(overall[i])})
    frame = pd.DataFrame(rows); metrics=[]
    for target, group in frame.groupby("target"):
        metrics.append({"target": target, **metric_dict(group.true_raw.to_numpy(), group.pred_raw.to_numpy(), group.pred_probability.to_numpy(), group.true_tail.to_numpy(bool))})
    return frame, pd.DataFrame(metrics)


def main():
    args=parse_args(); args.output_root.mkdir(parents=True, exist_ok=True); np.random.seed(args.seed); torch.manual_seed(args.seed)
    order_feature_root = args.order_feature_root if args.use_iis else None
    data={split:load_split(args.warehouse_root,args.target_root,split,order_feature_root,args.fold) for split in ["train","validation","test"]}
    numeric=data["train"][0][LINK_FEATURES].apply(pd.to_numeric,errors="coerce"); mean=numeric.mean().fillna(0); std=numeric.std().replace(0,1).fillna(1)
    if args.use_iis:
        order_numeric=data["train"][2][IIS_ORDER_FEATURES].apply(pd.to_numeric,errors="coerce"); order_mean=order_numeric.mean().fillna(0); order_std=order_numeric.std().replace(0,1).fillna(1)
    else:
        order_mean=None; order_std=None
    datasets={split:OrderSequenceDataset(link,target,mean,std,features if args.use_iis else None,order_mean,order_std) for split,(link,target,features) in data.items()}
    loaders={split:DataLoader(dataset,batch_size=args.batch_size,shuffle=split=="train",collate_fn=collate) for split,dataset in datasets.items()}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=RouteAttention(len(LINK_FEATURES),args.hidden_dim,len(IIS_ORDER_FEATURES) if args.use_iis else 0).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4)
    best=-np.inf; state=None
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loaders["train"]:
            optimizer.zero_grad()
            order_x = batch["order_x"].to(device)
            if args.use_iis and args.modality_dropout > 0:
                keep = (torch.rand(order_x.shape[0], 1, device=device) >= args.modality_dropout).float()
                order_x = order_x * keep
            raw,tail,overall=model(batch["x"].to(device),batch["pad"].to(device),order_x); mask=batch["mask"].to(device)
            raw_loss=(nn.functional.huber_loss(raw,batch["raw"].to(device),reduction="none")*mask).sum()/mask.sum().clamp_min(1)
            tail_loss=(nn.functional.binary_cross_entropy_with_logits(tail,batch["tail"].to(device),reduction="none")*mask).sum()/mask.sum().clamp_min(1)
            overall_loss=nn.functional.binary_cross_entropy_with_logits(overall,batch["overall"].to(device)); loss=raw_loss+.5*tail_loss+.5*overall_loss
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        _, val_metrics=evaluate(model,loaders["validation"],device); score=val_metrics.ap.mean(); print(f"epoch={epoch} loss={np.mean(losses):.5f} val_ap={score:.4f}",flush=True)
        if score>best: best=score; state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(state); metric_parts=[]; prediction_parts=[]
    for split in ["validation","test"]:
        prediction,metrics=evaluate(model,loaders[split],device); prediction["split"]=split; metrics["split"]=split; metrics["model"]="deepsets_route_attention"; prediction_parts.append(prediction); metric_parts.append(metrics)
    metrics=pd.concat(metric_parts,ignore_index=True); metrics.to_csv(args.output_root/"metrics.csv",index=False); pd.concat(prediction_parts,ignore_index=True).to_parquet(args.output_root/"predictions.parquet",index=False,compression="zstd")
    torch.save({"state":model.state_dict(),"features":LINK_FEATURES,"order_features":IIS_ORDER_FEATURES if args.use_iis else [],"mean":mean.to_dict(),"std":std.to_dict(),"order_mean":order_mean.to_dict() if order_mean is not None else {},"order_std":order_std.to_dict() if order_std is not None else {}},args.output_root/"model.pt")
    (args.output_root/"report.md").write_text("# Stage3 DeepSets / route attention\n\n"+metrics[metrics.split.eq("test")].to_markdown(index=False,floatfmt=".4f"),encoding="utf-8"); print(metrics[metrics.split.eq("test")].to_string(index=False))


if __name__=="__main__": main()
