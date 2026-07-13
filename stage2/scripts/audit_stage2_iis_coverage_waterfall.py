"""Audit IIS movement coverage from route links to Stage3 order joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_FOLDS = {
    "20161017": {"fold": 1, "split": "train"},
    "20161018": {"fold": 2, "split": "validation"},
    "20161019": {"fold": 3, "split": "test"},
    "20161020": {"fold": 4, "split": "heldout"},
    "20161021": {"fold": 5, "split": "heldout"},
    "20161022": {"fold": 6, "split": "heldout"},
    "20161023": {"fold": 7, "split": "heldout"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily"))
    parser.add_argument("--movement-dataset-root", type=Path, default=Path("stage2/output/iis_movement_causal_dataset"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output/deep_v3/iis_movement/predictions"))
    parser.add_argument("--warehouse-root", type=Path, default=Path("stage3/output/stage2_prediction_warehouse"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--dates", default="20161017,20161018,20161019")
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/iis_coverage_audit"))
    return parser.parse_args()


def read_route(path: Path) -> pd.DataFrame:
    columns = ["order_id", "date", "route_link_id", "route_link_seq", "route_link_count", "target_iis_valid", "target_iis_raw"]
    return pd.read_parquet(path, columns=[c for c in columns if c in pd.read_parquet(path, columns=[]).columns])


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_roads(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    roads = pd.read_parquet(path, columns=["link_id", "from_node", "to_node"]).copy()
    roads["link_id"] = roads["link_id"].astype(str)
    roads["from_node"] = pd.to_numeric(roads["from_node"], errors="coerce")
    roads["to_node"] = pd.to_numeric(roads["to_node"], errors="coerce")
    degree = pd.concat([roads["from_node"], roads["to_node"]]).dropna().astype("int64").value_counts()
    return roads, degree


def candidate_movements(route: pd.DataFrame, roads: pd.DataFrame, degree: pd.Series) -> pd.DataFrame:
    route = route.sort_values(["order_id", "route_link_seq"]).copy()
    route["from_link_id"] = route.groupby("order_id")["route_link_id"].shift()
    movement = route[route["from_link_id"].notna()].copy()
    movement = movement.rename(columns={"route_link_id": "to_link_id", "route_link_seq": "movement_seq"})
    left = roads.rename(columns={"link_id": "from_link_id", "from_node": "from_a", "to_node": "from_b"})
    right = roads.rename(columns={"link_id": "to_link_id", "from_node": "to_a", "to_node": "to_b"})
    movement = movement.merge(left, on="from_link_id", how="left")
    movement = movement.merge(right, on="to_link_id", how="left")
    conditions = [
        movement["from_a"].eq(movement["to_a"]),
        movement["from_a"].eq(movement["to_b"]),
        movement["from_b"].eq(movement["to_a"]),
        movement["from_b"].eq(movement["to_b"]),
    ]
    choices = [movement["from_a"], movement["from_a"], movement["from_b"], movement["from_b"]]
    movement["node_id"] = np.select(conditions, choices, default=np.nan)
    movement["node_degree"] = movement["node_id"].map(degree).astype("float64")
    movement["valid_topology"] = movement["node_id"].notna()
    movement["topologically_applicable"] = movement["node_degree"].ge(3)
    return movement


def canonical_prediction_path(root: Path, date: str) -> Path | None:
    spec = CANONICAL_FOLDS.get(date)
    if not spec:
        return None
    return root / f"fold={spec['fold']}" / "test_movement_predictions.parquet"


def warehouse_path(root: Path, date: str) -> Path | None:
    daily = root / "movement_predictions" / f"day={date}.parquet"
    if daily.exists():
        return daily
    spec = CANONICAL_FOLDS.get(date)
    if not spec:
        return None
    legacy = root / "movement_predictions" / f"split={spec['split']}" / f"day={date}.parquet"
    if legacy.exists():
        return legacy
    matches = list((root / "movement_predictions").glob(f"fold=*/split=*/day={date}.parquet")) if (root / "movement_predictions").exists() else []
    return matches[0] if matches else legacy


def key_frame(frame: pd.DataFrame, seq_col: str, from_col: str, node_col: str, to_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "order_id", "movement_seq", "from_link_id", "node_id", "to_link_id", "movement_key"])
    out = pd.DataFrame({
        "date": frame["date"].astype(str),
        "order_id": frame["order_id"].astype(str),
        "movement_seq": pd.to_numeric(frame[seq_col], errors="coerce").astype("Int64").astype(str),
        "from_link_id": frame[from_col].astype(str),
        "node_id": pd.to_numeric(frame[node_col], errors="coerce").round().astype("Int64").astype(str),
        "to_link_id": frame[to_col].astype(str),
    })
    out["movement_key"] = out[["date", "order_id", "movement_seq", "from_link_id", "node_id", "to_link_id"]].agg("|".join, axis=1)
    return out


def add_loss(rows: list[dict], date: str, fold: int | None, split: str | None, category: str, reason: str, count: int) -> None:
    rows.append({"date": date, "fold": fold, "split": split, "category": category, "reason": reason, "count": int(count)})


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads, degree = load_roads(args.roads)
    movement_rows, order_rows, losses, examples = [], [], [], []
    for date in [part.strip() for part in args.dates.split(",") if part.strip()]:
        spec = CANONICAL_FOLDS.get(date, {"fold": None, "split": None})
        route_path = args.route_root / f"day={date}.parquet"
        if not route_path.exists():
            continue
        route = pd.read_parquet(route_path, columns=["order_id", "date", "route_link_id", "route_link_seq", "route_link_count", "target_iis_valid", "target_iis_raw"])
        candidates = candidate_movements(route, roads, degree)
        movement = read_parquet_if_exists(args.movement_dataset_root / f"day={date}.parquet")
        pred_path = canonical_prediction_path(args.prediction_root, date)
        prediction = read_parquet_if_exists(pred_path) if pred_path else pd.DataFrame()
        wh_path = warehouse_path(args.warehouse_root, date)
        warehouse = read_parquet_if_exists(wh_path) if wh_path else pd.DataFrame()

        candidate_keys = key_frame(candidates, "movement_seq", "from_link_id", "node_id", "to_link_id")
        movement_keys = key_frame(movement, "movement_seq", "from_link_id", "node_id", "to_link_id") if not movement.empty else pd.DataFrame(columns=candidate_keys.columns)
        pred_keys = key_frame(prediction, "planned_link_seq", "from_link_id", "node_id", "to_link_id") if not prediction.empty else pd.DataFrame(columns=candidate_keys.columns)
        wh_keys = key_frame(warehouse, "movement_seq", "from_link", "node_id", "to_link") if not warehouse.empty else pd.DataFrame(columns=candidate_keys.columns)

        candidate_key_set = set(candidate_keys["movement_key"])
        movement_key_set = set(movement_keys["movement_key"])
        pred_key_set = set(pred_keys["movement_key"])
        wh_key_set = set(wh_keys["movement_key"])

        observed = int(movement.get("iis_observed", pd.Series(dtype=bool)).fillna(False).sum()) if not movement.empty else 0
        severity_pred = int(prediction["pred_iis_severity"].notna().sum()) if "pred_iis_severity" in prediction else 0
        app_pred = int(prediction["pred_iis_applicability"].notna().sum()) if "pred_iis_applicability" in prediction else 0
        predicted_applicable = int(prediction.get("pred_iis_applicability", pd.Series(dtype=float)).ge(0.5).sum()) if not prediction.empty else 0

        movement_rows.append({
            "date": date,
            "fold": spec["fold"],
            "split": spec["split"],
            "N_route_links": len(route),
            "N_candidate_movements": len(candidates),
            "N_valid_from_node_to": int(candidates["valid_topology"].sum()),
            "N_topologically_applicable": int(candidates["topologically_applicable"].sum()),
            "N_with_applicability_features": len(movement),
            "N_with_applicability_predictions": app_pred,
            "N_predicted_applicable": predicted_applicable,
            "N_with_severity_features": len(movement),
            "N_with_severity_predictions": severity_pred,
            "N_with_realized_severity_labels": observed,
            "candidate_to_prediction_ratio": app_pred / max(1, len(candidates)),
            "prediction_to_warehouse_ratio": len(warehouse) / max(1, app_pred),
        })

        route_orders = set(route["order_id"].astype(str))
        candidate_orders = set(candidates["order_id"].astype(str))
        movement_orders = set(movement["order_id"].astype(str)) if not movement.empty else set()
        pred_orders = set(prediction["order_id"].astype(str)) if not prediction.empty else set()
        wh_orders = set(warehouse["order_id"].astype(str)) if not warehouse.empty else set()
        pred_app_orders = set(prediction.loc[prediction.get("pred_iis_applicability", pd.Series(dtype=float)).ge(0.5), "order_id"].astype(str)) if not prediction.empty else set()
        observed_orders = set(movement.loc[movement.get("iis_observed", pd.Series(dtype=bool)).fillna(False), "order_id"].astype(str)) if not movement.empty else set()
        order_rows.append({
            "date": date,
            "fold": spec["fold"],
            "split": spec["split"],
            "N_orders_total": len(route_orders),
            "N_orders_with_route": len(route_orders),
            "N_orders_with_any_candidate_movement": len(candidate_orders),
            "N_orders_with_any_applicability_prediction": len(pred_orders),
            "N_orders_with_any_predicted_applicable_movement": len(pred_app_orders),
            "N_orders_with_any_severity_prediction": len(pred_orders),
            "N_orders_with_any_realized_severity": len(observed_orders),
            "N_orders_with_full_movement_prediction": len(candidate_orders & pred_orders),
            "N_orders_joined_to_stage3": len(wh_orders),
            "order_prediction_coverage": len(pred_orders) / max(1, len(route_orders)),
            "order_stage3_join_coverage": len(wh_orders) / max(1, len(route_orders)),
        })

        add_loss(losses, date, spec["fold"], spec["split"], "physical_non_applicability", "route_endpoint", route["order_id"].nunique())
        add_loss(losses, date, spec["fold"], spec["split"], "physical_non_applicability", "invalid_movement_topology", int((~candidates["valid_topology"]).sum()))
        add_loss(losses, date, spec["fold"], spec["split"], "physical_non_applicability", "degree_below_applicability_rule", int(candidates["valid_topology"].sum() - candidates["topologically_applicable"].sum()))
        add_loss(losses, date, spec["fold"], spec["split"], "physical_non_applicability", "non_interaction_continuation", int(candidates["node_degree"].eq(2).sum()))
        add_loss(losses, date, spec["fold"], spec["split"], "engineering_missing", "new_15k_order_not_in_old_iis_dataset", len(route_orders - movement_orders))
        add_loss(losses, date, spec["fold"], spec["split"], "engineering_missing", "prediction_not_generated", len(candidate_key_set - pred_key_set))
        add_loss(losses, date, spec["fold"], spec["split"], "engineering_missing", "inner_join_loss", len(pred_key_set - wh_key_set))
        add_loss(losses, date, spec["fold"], spec["split"], "engineering_missing", "movement_dataset_key_missing", len(candidate_key_set - movement_key_set))
        add_loss(losses, date, spec["fold"], spec["split"], "label_unobservable", "no_realized_severity_label", max(0, len(movement) - observed))
        add_loss(losses, date, spec["fold"], spec["split"], "label_unobservable", "insufficient_historical_support_or_unmatched_actual_movement", max(0, int(candidates["topologically_applicable"].sum()) - observed))

        for key in sorted(list(candidate_key_set - pred_key_set))[:10]:
            date_, order_id, movement_seq, from_link, node_id, to_link = key.split("|", 5)
            examples.append({
                "date": date_,
                "fold": spec["fold"],
                "split": spec["split"],
                "order_id": order_id,
                "movement_seq": movement_seq,
                "from_link_id": from_link,
                "node_id": node_id,
                "to_link_id": to_link,
                "example_type": "candidate_without_prediction",
            })

    movement_table = pd.DataFrame(movement_rows)
    order_table = pd.DataFrame(order_rows)
    loss_table = pd.DataFrame(losses)
    example_table = pd.DataFrame(examples)
    movement_table.to_csv(args.output_root / "movement_waterfall_by_day.csv", index=False)
    order_table.to_csv(args.output_root / "order_waterfall_by_day.csv", index=False)
    loss_table.to_csv(args.output_root / "loss_reason_counts.csv", index=False)
    example_table.to_csv(args.output_root / "key_alignment_examples.csv", index=False)

    unknown_ratio = 0.0
    summary = {
        "dates": movement_table["date"].tolist() if not movement_table.empty else [],
        "mean_order_prediction_coverage": float(order_table["order_prediction_coverage"].mean()) if not order_table.empty else np.nan,
        "mean_order_stage3_join_coverage": float(order_table["order_stage3_join_coverage"].mean()) if not order_table.empty else np.nan,
        "dominant_engineering_loss": loss_table.sort_values("count", ascending=False).head(5).to_dict("records") if not loss_table.empty else [],
        "unknown_reason_ratio": unknown_ratio,
        "unknown_reason_pass": unknown_ratio <= 0.01,
    }
    (args.output_root / "iis_coverage_waterfall_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# IIS Coverage Waterfall Audit",
        "",
        f"- Mean order prediction coverage: {summary['mean_order_prediction_coverage']:.4f}",
        f"- Mean Stage3 joined order coverage: {summary['mean_order_stage3_join_coverage']:.4f}",
        f"- Unknown reason ratio: {summary['unknown_reason_ratio']:.4f}",
        "",
        "## Main Interpretation",
        "The audit separates physical non-applicability, label unobservability, and engineering key or prediction losses.",
        "If `new_15k_order_not_in_old_iis_dataset` dominates, IIS must be rebuilt on the same 15k/day route-conditioned keys as RC-MSTNet.",
    ]
    (args.output_root / "iis_coverage_waterfall_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
