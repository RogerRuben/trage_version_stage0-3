"""Shared single-order processing for fixed validation and Stage 1 production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .canonical_mapper import CanonicalEdgeMapper
from .config import Stage0V6Config
from .eligibility import evaluate_modeling_eligibility
from .final_quality import CanonicalGeometryStore, FinalQualityResult, build_final_quality
from .parser import parse_trace_attributes
from .preprocess import PreprocessResult, preprocess_order
from .products import build_order_products
from .quality import evaluate_dynamic_measurement_quality, evaluate_route_quality
from .valhalla_client import ValhallaMatcher


@dataclass(frozen=True)
class ProcessedOrder:
    preprocess: PreprocessResult
    eligibility: dict[str, Any]
    matched_points: pd.DataFrame
    products: dict[str, pd.DataFrame]
    route_quality: dict[str, Any]
    dynamic_quality: dict[str, Any]
    final_quality: FinalQualityResult
    match_metrics: dict[str, Any]
    stage1_core_eligible: bool
    stage1_rejection_reason: str


def stage1_core_decision(
    final_quality: dict[str, Any],
    accounting: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[bool, str]:
    """Apply the frozen Stage 1 core contract without relaxing thresholds."""

    reasons: list[str] = []
    if final_quality["route_status"] != "route_pass":
        reasons.append("ROUTE_NOT_PASS")
    gps_status = str(final_quality["gps_status"])
    if gps_status == "local_outlier":
        if (
            float(final_quality["outlier_time_share"])
            > float(settings["maximum_local_outlier_time_share"])
            or float(final_quality["outlier_distance_share"])
            > float(settings["maximum_local_outlier_distance_share"])
        ):
            reasons.append("LOCAL_OUTLIER_AFFECTS_CORRIDOR")
    elif gps_status != "clean":
        reasons.append("GPS_NOT_CORE_ELIGIBLE")
    if final_quality["canonical_status"] not in {"unique", "chain_resolved"}:
        reasons.append("CANONICAL_NOT_RESOLVED")
    if final_quality["dynamic_status"] not in {
        "dynamic_strict",
        "dynamic_partial",
    }:
        reasons.append("DYNAMIC_UNUSABLE")
    record = accounting.iloc[0].to_dict() if len(accounting) else {}
    if not bool(record.get("time_conservation_valid", False)):
        reasons.append("TIME_CONSERVATION_FAILURE")
    if not bool(record.get("distance_conservation_valid", False)):
        reasons.append("DISTANCE_CONSERVATION_FAILURE")
    if int(record.get("duplicate_interval_allocation_count", 0)) != 0:
        reasons.append("DUPLICATE_INTERVAL_ALLOCATION")
    if int(record.get("non_direct_observed_time_violation_count", 0)) != 0:
        reasons.append("NON_DIRECT_OBSERVED_TIME")
    if int(record.get("valid_direct_interval_count", 0)) < int(
        settings["minimum_valid_direct_interval_count"]
    ):
        reasons.append("INSUFFICIENT_DIRECT_INTERVALS")
    if int(record.get("unique_timed_edge_count", 0)) < int(
        settings["minimum_unique_timed_edge_count"]
    ):
        reasons.append("INSUFFICIENT_TIMED_EDGES")
    return not reasons, "|".join(reasons)


class Stage0OrderProcessor:
    """Keep expensive Valhalla, canonical, and geometry state warm."""

    def __init__(
        self,
        config: Stage0V6Config,
        *,
        matcher: ValhallaMatcher | None = None,
    ) -> None:
        self.config = config
        self.matcher = matcher or ValhallaMatcher(
            config.section("valhalla"),
            valhalla_config_path=config.path("valhalla_config"),
        )
        self.mapper = CanonicalEdgeMapper.from_parquet(
            config.path("canonical_edges")
        )
        self.geometry_store = CanonicalGeometryStore(
            config.path("canonical_edges")
        )

    def process(self, raw_order: pd.DataFrame) -> ProcessedOrder:
        prep = preprocess_order(
            raw_order, **self.config.section("preprocess")
        )
        clean = prep.points
        eligibility = evaluate_modeling_eligibility(
            clean,
            prep.metrics,
            **self.config.section("modeling_eligibility"),
        )
        matched_frames: list[pd.DataFrame] = []
        route_frames: list[pd.DataFrame] = []
        metric_totals: dict[str, Any] = {
            "backend": self.matcher.backend,
            "request_point_count": 0,
            "matched_edge_count": 0,
            "matched_point_count": 0,
            "unmatched_point_count": 0,
            "interpolated_point_count": 0,
            "discontinuity_count": 0,
            "retry_count": 0,
            "response_ms": 0.0,
            "oneway_candidate_compared_count": 0,
            "selected_ignore_oneways_count": 0,
        }
        for subtrace_id, subtrace in clean.groupby(
            "subtrace_id", sort=False
        ):
            usable = bool(
                eligibility["modeling_eligible"]
                and subtrace.usable_subtrace.iloc[0]
            )
            result = (
                self.matcher.match_order(subtrace.reset_index(drop=True))
                if usable
                else {"raw_response": {}, "status": "ineligible"}
            )
            if result.get("status") == "error":
                raise RuntimeError(str(result.get("error_message", "Valhalla error")))
            for key in (
                "request_point_count",
                "matched_edge_count",
                "matched_point_count",
                "unmatched_point_count",
                "interpolated_point_count",
                "discontinuity_count",
                "retry_count",
                "response_ms",
            ):
                metric_totals[key] += result.get(key, 0) or 0
            metric_totals["oneway_candidate_compared_count"] += int(
                bool(result.get("oneway_candidate_compared", False))
            )
            metric_totals["selected_ignore_oneways_count"] += int(
                bool(result.get("selected_ignore_oneways", False))
            )
            matched, routes = parse_trace_attributes(
                result.get("raw_response") or {},
                subtrace.reset_index(drop=True),
                order_id=str(clean.order_id.iloc[0]),
                subtrace_id=str(subtrace_id),
            )
            matched_frames.append(matched)
            route_frames.append(routes)
        matched_points = (
            pd.concat(matched_frames, ignore_index=True)
            if matched_frames
            else pd.DataFrame()
        )
        valhalla_routes = (
            pd.concat(route_frames, ignore_index=True)
            if route_frames
            else pd.DataFrame()
        )
        mapped_routes, _ = self.mapper.map_route_parts(valhalla_routes)
        products = build_order_products(
            clean,
            matched_points,
            mapped_routes,
            preprocess_breaks=prep.preprocess_breaks,
            **self.config.section("products"),
        )
        route_quality = evaluate_route_quality(
            clean,
            matched_points,
            products["route_parts"],
            products["interval_measurements"],
            self.config.section("quality"),
        )
        dynamic_quality = evaluate_dynamic_measurement_quality(
            products["route_parts"],
            products["link_traversals"],
            products["interval_measurements"],
            products["interval_accounting"],
            self.config.section("quality"),
        )
        final_quality = build_final_quality(
            clean,
            matched_points,
            products["route_parts"],
            products["interval_measurements"],
            self.geometry_store,
            eligibility,
            self.config.section("final_quality"),
        )
        accepted, reason = stage1_core_decision(
            final_quality.order_quality,
            products["interval_accounting"],
            self.config.section("stage1_core"),
        )
        return ProcessedOrder(
            preprocess=prep,
            eligibility=eligibility,
            matched_points=matched_points,
            products=products,
            route_quality=route_quality,
            dynamic_quality=dynamic_quality,
            final_quality=final_quality,
            match_metrics=metric_totals,
            stage1_core_eligible=accepted,
            stage1_rejection_reason=reason,
        )
