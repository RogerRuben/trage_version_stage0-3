"""Run frozen Stage3 models on a full-day Stage2 prediction product.

The script consumes real Stage2 RC-MSTNet link predictions plus
validation-only calibration/uncertainty outputs and exports a Stage4
condition-vector table.  It does not create stress fallbacks.  If full-day IIS
movement predictions are unavailable, IIS fields are exported as unavailable
and the Core+IIS model is evaluated with its missing-modality path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_stage3_deepsets import IIS_ORDER_FEATURES, LINK_FEATURES, RouteAttention  # noqa: E402


TARGETS = ["lcs", "pmis", "rts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20161023")
    parser.add_argument("--link-predictions", type=Path, required=True)
    parser.add_argument("--calibrated-predictions", type=Path, required=True)
    parser.add_argument("--uncertainty-predictions", type=Path, required=True)
    parser.add_argument("--route-conditioned", type=Path, required=True)
    parser.add_argument("--od-path", type=Path, required=True)
    parser.add_argument("--core-checkpoint", type=Path, required=True)
    parser.add_argument("--extended-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iis-mode", choices=["unavailable"], default="unavailable")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_stage3_model(path: Path, device: torch.device) -> tuple[RouteAttention, dict]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["state"]
    first_weight = state["phi.0.weight"]
    hidden = int(first_weight.shape[0])
    inputs = int(first_weight.shape[1])
    order_inputs = len(checkpoint.get("order_features", []))
    model = RouteAttention(inputs, hidden, order_inputs).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


class Stage3InferenceDataset(Dataset):
    def __init__(self, link: pd.DataFrame, checkpoint: dict, use_order_features: bool):
        self.features = list(checkpoint["features"])
        self.mean = pd.Series(checkpoint["mean"], dtype=float)
        self.std = pd.Series(checkpoint["std"], dtype=float).replace(0, 1).fillna(1)
        self.use_order_features = use_order_features
        self.order_features = list(checkpoint.get("order_features", []))
        self.items = []
        for order_id, group in link.sort_values(["order_id", "route_link_seq"], kind="mergesort").groupby("order_id", sort=False):
            x = group[self.features].apply(pd.to_numeric, errors="coerce").fillna(self.mean)
            x = ((x - self.mean) / self.std).fillna(0).to_numpy("float32")
            if use_order_features and self.order_features:
                # Missing full-day IIS movement predictions are represented by
                # the same zero order-feature vector used by the training
                # dataset when an order has no IIS feature row.
                order_x = np.zeros(len(self.order_features), dtype="float32")
            else:
                order_x = np.zeros(len(self.order_features), dtype="float32")
            self.items.append((str(order_id), x, order_x))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        order_id, x, order_x = self.items[index]
        return {"order_id": order_id, "x": torch.from_numpy(x), "order_x": torch.from_numpy(order_x)}


def _collate(batch: list[dict]) -> dict:
    length = max(item["x"].shape[0] for item in batch)
    features = batch[0]["x"].shape[1]
    x = torch.zeros(len(batch), length, features)
    pad = torch.ones(len(batch), length, dtype=torch.bool)
    for index, item in enumerate(batch):
        n = item["x"].shape[0]
        x[index, :n] = item["x"]
        pad[index, :n] = False
    return {
        "order_id": [item["order_id"] for item in batch],
        "x": x,
        "pad": pad,
        "order_x": torch.stack([item["order_x"] for item in batch]),
    }


def _predict(model: RouteAttention, dataset: Stage3InferenceDataset, device: torch.device, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    rows = []
    with torch.no_grad():
        for batch in loader:
            raw, tail_logits, overall_logits = model(batch["x"].to(device), batch["pad"].to(device), batch["order_x"].to(device))
            raw = raw.cpu().numpy()
            prob = torch.sigmoid(tail_logits).cpu().numpy()
            overall = torch.sigmoid(overall_logits).cpu().numpy()
            for i, order_id in enumerate(batch["order_id"]):
                row = {"order_id": order_id, "overall_probability": float(overall[i])}
                for k, target in enumerate(TARGETS):
                    row[f"{target}_expected"] = float(raw[i, k])
                    row[f"{target}_tail_probability"] = float(prob[i, k])
                rows.append(row)
    return pd.DataFrame(rows)


def _prepare_link_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_cols = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"] + sum(
        ([f"pred_{target}_raw", f"pred_{target}_tail_prob"] for target in TARGETS), []
    )
    pred = pd.read_parquet(args.link_predictions, columns=pred_cols)
    cal = pd.read_parquet(args.calibrated_predictions)
    unc = pd.read_parquet(args.uncertainty_predictions)
    route_cols = ["order_id", "route_link_id", "route_link_seq", "estimated_link_entry_time", "route_link_length_m", "position_ratio", "route_link_count"]
    route = pd.read_parquet(args.route_conditioned, columns=route_cols)
    for frame in [pred, cal, unc, route]:
        frame["order_id"] = frame["order_id"].astype(str)
        frame["route_link_id"] = frame["route_link_id"].astype(str)
        frame["route_link_seq"] = pd.to_numeric(frame["route_link_seq"], errors="coerce").astype("int32")
    keys = ["order_id", "route_link_id", "route_link_seq"]
    link = pred.merge(cal, on=["order_id", "driver_id", "date", "route_link_id", "route_link_seq"], how="left", validate="one_to_one")
    link = link.merge(unc, on=["order_id", "driver_id", "date", "route_link_id", "route_link_seq"], how="left", validate="one_to_one")
    link = link.merge(route, on=keys, how="left", validate="one_to_one")
    for target in TARGETS:
        link[f"{target}_raw_pred"] = link[f"pred_{target}_raw"]
        link[f"{target}_tail_prob_calibrated"] = link[f"{target}_tail_prob_calibrated"].fillna(link[f"pred_{target}_tail_prob"])
    link["route_link_length_m"] = pd.to_numeric(link["route_link_length_m"], errors="coerce").fillna(0)
    link["position_ratio"] = pd.to_numeric(link["position_ratio"], errors="coerce").fillna(0)
    order_summary = link.groupby("order_id", as_index=False).agg(
        driver_id=("driver_id", "first"),
        date=("date", "first"),
        route_length_m=("route_link_length_m", "sum"),
        link_count=("route_link_seq", "count"),
        decision_time=("estimated_link_entry_time", "first"),
        rc_lcs_uncertainty_q90=("lcs_uncertainty", lambda value: float(np.nanquantile(value, 0.90))),
        rc_pmis_uncertainty_q90=("pmis_uncertainty", lambda value: float(np.nanquantile(value, 0.90))),
        rc_rts_uncertainty_q90=("rts_uncertainty", lambda value: float(np.nanquantile(value, 0.90))),
        route_prediction_confidence=("route_link_seq", lambda value: 1.0),
    )
    order_summary["movement_count"] = 0
    order_summary["iis_prediction_available"] = False
    order_summary["iis_availability"] = False
    order_summary["modality_coverage_score"] = 0.75
    return link, order_summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage4_dir = args.output_root / "stage4_inputs"
    stage4_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage4_dir / "stage4_inputs.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")

    link, order_summary = _prepare_link_frame(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_model, core_ckpt = _load_stage3_model(args.core_checkpoint, device)
    extended_model, ext_ckpt = _load_stage3_model(args.extended_checkpoint, device)
    core_pred = _predict(core_model, Stage3InferenceDataset(link, core_ckpt, use_order_features=False), device, args.batch_size)
    ext_pred = _predict(extended_model, Stage3InferenceDataset(link, ext_ckpt, use_order_features=True), device, args.batch_size)

    output = order_summary.merge(core_pred, on="order_id", how="inner", validate="one_to_one")
    output = output.rename(columns={"overall_probability": "core_overall_high_stress_probability"})
    ext_pred = ext_pred[["order_id", "overall_probability"]].rename(columns={"overall_probability": "extended_overall_high_stress_probability"})
    output = output.merge(ext_pred, on="order_id", how="inner", validate="one_to_one")
    od = pd.read_parquet(args.od_path, columns=["order_id", "origin_lon", "origin_lat", "destination_lon", "destination_lat", "origin_timestamp"])
    output = output.merge(od, on="order_id", how="left", validate="one_to_one")
    output["decision_time"] = pd.to_datetime(output["decision_time"], errors="coerce")
    output["origin_timestamp"] = pd.to_datetime(output["origin_timestamp"], errors="coerce")
    output["decision_time_source"] = "estimated_first_route_link_entry"
    fallback = output["decision_time"].isna() & output["origin_timestamp"].notna()
    output.loc[fallback, "decision_time"] = output.loc[fallback, "origin_timestamp"]
    output.loc[fallback, "decision_time_source"] = "origin_timestamp_fallback"
    output["route_id"] = output["order_id"].astype(str).map(lambda value: f"observed_matched_service_route_proxy:{value}")
    output["route_proxy_type"] = "observed_matched_service_route_proxy"
    output["intersection_applicability"] = np.nan
    output["intersection_severity"] = np.nan
    output["intersection_tail_probability"] = np.nan
    output["iis_availability"] = False
    output["iis_applicability"] = np.nan
    output["iis_severity"] = np.nan
    output["iis_tail_probability"] = np.nan
    output["iis_coverage_quality"] = 0.0
    output["composite_expected"] = output[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1)
    output["pred_stop_go_stress"] = output["lcs_tail_probability"]
    output["pred_poi_mediated_stress"] = output["pmis_tail_probability"]
    output["pred_reliability_stress"] = output["rts_tail_probability"]
    output["pred_intersection_stress"] = np.nan
    output["pred_composite_operational_stress"] = output["composite_expected"]
    output["overall_uncertainty"] = output[["rc_lcs_uncertainty_q90", "rc_pmis_uncertainty_q90", "rc_rts_uncertainty_q90"]].mean(axis=1)
    output["model_version"] = "Stage2=RC-MSTNet-fold7-full-day;Stage3-core=fold3-core_deepsets;extended=fold3-core_iis_dropout;IIS=unavailable_full_day"
    output["prediction_cutoff_time"] = output["decision_time"]
    for column in ["decision_time", "origin_timestamp", "prediction_cutoff_time"]:
        output[column] = output[column].astype(str)
    output.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "date": args.date,
        "orders": int(output["order_id"].nunique()),
        "rows": int(len(output)),
        "link_rows": int(len(link)),
        "iis_mode": args.iis_mode,
        "iis_available_orders": int(output["iis_availability"].sum()),
        "stage4_inputs": str(output_path),
        "status": "PASS",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
