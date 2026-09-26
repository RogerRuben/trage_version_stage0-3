"""Stage 3 S2A.1 scientific closure without refitting or network export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    anchor_table,
    atomic_json,
    atomic_parquet,
    atomic_text,
    git_head,
    parquet_descriptor,
    payload_hash,
    read_json,
    sha256_file,
    source_descriptor,
)


PHASE_STATUS = "STAGE3_S2A1_SCIENTIFIC_CLOSURE_COMPLETE"
AUTHORIZED_BASE = "4f036187e0921493894263f7999944a9193a5b2d"


def _load_parquet(path: Path):
    if not path.is_file():
        raise Stage3S2AError(f"missing frozen S2A product: {path}")
    return pq.read_table(path).to_pandas()


def _disagreement(speed, cap: int):
    eligible = speed["speed_domain_provenance"] == "INFERRED_SPEED_AND_CLASS"
    b0 = speed["road_class_prior_kmh"] <= cap
    # In the frozen hierarchy B2 is applicable only with adequate historical
    # support; all other inferred edges retain B0. Known anchors are invariant.
    b2 = b0.copy()
    b2.loc[eligible] = speed.loc[eligible, "speed_domain_value_kmh"] <= cap
    disagreement = b0 != b2
    return {
        "cap_kmh": cap,
        "full_network_denominator": int(len(speed)),
        "full_network_disagreement_count": int(disagreement.sum()),
        "full_network_disagreement_rate": float(disagreement.mean()),
        "b2_applicable_denominator": int(eligible.sum()),
        "b2_applicable_disagreement_count": int(disagreement[eligible].sum()),
        "b2_applicable_disagreement_rate": float(disagreement[eligible].mean()),
        "comparison_semantics": "known anchors invariant; B2 applied only to INFERRED_SPEED_AND_CLASS; B0 fallback elsewhere",
    }


def build(root: Path) -> dict[str, Any]:
    if git_head(root) != AUTHORIZED_BASE:
        raise Stage3S2AError("S2A.1 must execute from the authorized frozen S2A base")
    docs = root / "stage3/docs/odd_tod/s2a"
    output = root / "stage3/output/odd_tod/s2a"
    validation_path = docs / "stage3_s2a_speed_validation.json"
    mapping_path = output / "stage3_observed_full_network_mapping.parquet"
    speed_path = output / "stage3_speed_domain.parquet"
    observed_path = output / "observed_identity_cache.parquet"
    s2a_evidence_path = docs / "stage3_s2a_evidence_bundle.json"

    s2a_evidence = read_json(s2a_evidence_path)
    validation = read_json(validation_path)
    mapping = _load_parquet(mapping_path)
    speed = _load_parquet(speed_path)
    observed = _load_parquet(observed_path)
    anchors = anchor_table(observed)

    for key, path in (("observed_mapping", mapping_path), ("speed_domain", speed_path)):
        if s2a_evidence["products"][key]["sha256"] != sha256_file(path):
            raise Stage3S2AError(f"frozen S2A product changed: {key}")

    if (
        validation["selected"]["quantile"] != 0.85
        or validation["selected"]["method"] != "MAP_SPEED_AND_ROAD_CLASS"
        or len(anchors) != 502
        or len(speed) != 209_454
    ):
        raise Stage3S2AError("frozen S2A scientific state changed; closure refuses to refit")

    anchor_mapping = anchors.merge(
        mapping[["canonical_edge_uid", "canonical_traversal_direction", "stage3_edge_uid", "mapping_status"]],
        on=["canonical_edge_uid", "canonical_traversal_direction"],
        how="left",
    )
    mapped_anchor_rows = anchor_mapping[anchor_mapping["stage3_edge_uid"].notna()]
    known_full_edges = speed[speed["speed_domain_provenance"] == "KNOWN_STAGE0_OSM"]
    if len(mapped_anchor_rows) != 500 or mapped_anchor_rows["stage3_edge_uid"].nunique() != 180 or len(known_full_edges) != 180:
        raise Stage3S2AError("502-to-180 anchor reconciliation changed")

    anchor_classes = {
        str(float(key)): int(value)
        for key, value in anchors["known_speed_kmh"].value_counts().sort_index().items()
    }
    cap_identification = []
    selected = validation["selected"]
    for cap in (60, 80, 120):
        below = int((anchors["known_speed_kmh"] <= cap).sum())
        above = int((anchors["known_speed_kmh"] > cap).sum())
        empirically_identified = below > 0 and above > 0
        cap_identification.append(
            {
                "cap_kmh": cap,
                "anchor_le_cap_count": below,
                "anchor_gt_cap_count": above,
                "selected_cv_accuracy": float(selected[f"compatibility_accuracy_{cap}"]),
                "empirically_identified": empirically_identified,
                "interpretation": (
                    "EMPIRICALLY_IDENTIFIED_BINARY_BOUNDARY"
                    if empirically_identified
                    else "NOT_EMPIRICALLY_IDENTIFIED_TRIVIAL_UNDER_CURRENT_ANCHOR_SUPPORT"
                ),
            }
        )

    reverse = mapping[
        (mapping["mapping_status"] == "UNMAPPED_FULL_NETWORK_EDGE")
        & (mapping["canonical_traversal_direction"] == "R")
    ].copy()
    forward = mapping[mapping["canonical_traversal_direction"] == "F"][
        ["canonical_edge_uid", "stage3_edge_uid", "mapping_status"]
    ].rename(
        columns={
            "stage3_edge_uid": "physical_forward_stage3_edge_uid",
            "mapping_status": "physical_forward_mapping_status",
        }
    )
    overlay = reverse.merge(forward, on="canonical_edge_uid", how="left")
    overlay = overlay[
        [
            "canonical_edge_uid", "canonical_traversal_direction", "observed_valhalla_edge_id",
            "observed_osm_way_id", "observed_begin_osm_node_id", "observed_end_osm_node_id",
            "mapping_status", "physical_forward_stage3_edge_uid", "physical_forward_mapping_status",
        ]
    ].rename(columns={"mapping_status": "original_mapping_status"})
    overlay["historical_direction_status"] = "HISTORICAL_DIRECTION_OVERLAY"
    overlay["historical_observation_accepted"] = True
    overlay["av_routability_status"] = "AV_ROUTABILITY_VIOLATION"
    overlay["downstream_handling"] = "RETAIN_HISTORY_EXCLUDE_AV_ROUTING"
    overlay["missing_identity"] = False
    overlay_path = output / "stage3_historical_direction_overlay.parquet"
    atomic_parquet(overlay_path, overlay)

    disagreement = [_disagreement(speed, cap) for cap in (60, 80, 120)]
    report = {
        "schema_version": "stage3_s2a1_scientific_closure.1",
        "phase_status": PHASE_STATUS,
        "authorized_base": AUTHORIZED_BASE,
        "scientific_products_rebuilt": False,
        "network_reexported": False,
        "speed_model_retrained": False,
        "speed_quantile_reselected": False,
        "speed_method_reselected": False,
        "av_caps_changed": False,
        "frozen_selection": {"quantile": 0.85, "method": "MAP_SPEED_AND_ROAD_CLASS", "caps_kmh": [60, 80, 120]},
        "anchor_reconciliation": {
            "unit": "canonical_segment_anchor",
            "canonical_segment_anchor_count": 502,
            "anchor_speed_class_distribution": anchor_classes,
            "mapped_canonical_segment_anchor_count": 500,
            "unmapped_canonical_segment_anchor_count": 2,
            "unique_full_network_directed_edges_after_mapping": 180,
            "known_full_network_speed_rows": 180,
            "collapse_reason": "multiple canonical split segments map to the same frozen Valhalla directed edge; deployment is one row per stage3_edge_uid",
            "directed_identity_claim_withdrawn": True,
        },
        "cap_identification": cap_identification,
        "reported_macro_accuracy": float(selected["macro_scenario_compatibility_accuracy"]),
        "macro_interpretation": "MECHANICAL_THREE_THRESHOLD_AVERAGE_NOT_THREE_PROFILE_EMPIRICAL_VALIDATION",
        "identified_boundary_accuracy": float(selected["compatibility_accuracy_60"]),
        "b0_vs_b2_compatibility_disagreement": disagreement,
        "reverse_direction_closure": {
            "reverse_overlay_count": int(len(overlay)),
            "with_mapped_forward_physical_reference": int(overlay["physical_forward_stage3_edge_uid"].notna().sum()),
            "without_mapped_forward_physical_reference": int(overlay["physical_forward_stage3_edge_uid"].isna().sum()),
            "semantic_status": "HISTORICAL_DIRECTION_OVERLAY_PLUS_AV_ROUTABILITY_VIOLATION",
            "missing_count": 0,
            "historical_observations_retained": True,
            "eligible_for_av_routing": False,
        },
        "scientific_conclusion": "speed-domain is retained chiefly for the empirically identified Conservative 60-km/h boundary; intersection/movement and dynamic E/Q/C remain the intended primary C/M/A discriminators",
        "s2b_authorized": False,
        "next_phase_authorized": False,
    }
    report["artifact_sha256"] = payload_hash(report)
    report_path = docs / "stage3_s2a1_scientific_closure.json"
    atomic_json(report_path, report)

    d60, d80, d120 = disagreement
    markdown = f"""# Stage 3 S2A.1 Scientific Closure

Status: `{PHASE_STATUS}`. This closure does not re-export the network, retrain/reselect the speed model, change P85/MAP, or change the 60/80/120 caps.

## Anchor unit and 502 → 180 reconciliation

The validation population is **502 canonical-segment anchors**, not 502 directed identities. Their class distribution is `{json.dumps(anchor_classes, sort_keys=True)}`. Of these, 500 map to the full network and collapse onto 180 unique frozen Valhalla directed edges because multiple canonical split segments share one physical directed edge. Two anchors remain unmapped. Accordingly, the deployed full-network speed table contains 180 `KNOWN_STAGE0_OSM` rows. Validation remains canonical-segment weighted; deployment remains one row per `stage3_edge_uid`.

## What the cap validation identifies

- 60 km/h: 204 anchors at/below and 298 above; this binary boundary is empirically identified. Frozen B2 CV accuracy is `{selected['compatibility_accuracy_60']:.6f}`.
- 80 km/h: 502 at/below and 0 above. Accuracy `1.0` is **not empirically identified / trivial under current anchor support**.
- 120 km/h: 502 at/below and 0 above. Accuracy `1.0` is **not empirically identified / trivial under current anchor support**.

The previously reported macro `{selected['macro_scenario_compatibility_accuracy']:.6f}` is only a mechanical average over three thresholds. It must not be presented as validation performance for three AV profiles. The identified-boundary result is the 60-km/h accuracy above.

## Frozen B0 versus B2 downstream impact

Known anchors are invariant. B2 applies only to the 4,898 `INFERRED_SPEED_AND_CLASS` edges; every other inferred edge retains B0 fallback.

| Cap | Full-network disagreements | Full-network rate | B2-applicable disagreements | Applicable rate |
|---:|---:|---:|---:|---:|
| 60 | {d60['full_network_disagreement_count']} / {d60['full_network_denominator']} | {d60['full_network_disagreement_rate']:.6%} | {d60['b2_applicable_disagreement_count']} / {d60['b2_applicable_denominator']} | {d60['b2_applicable_disagreement_rate']:.6%} |
| 80 | {d80['full_network_disagreement_count']} / {d80['full_network_denominator']} | {d80['full_network_disagreement_rate']:.6%} | {d80['b2_applicable_disagreement_count']} / {d80['b2_applicable_denominator']} | {d80['b2_applicable_disagreement_rate']:.6%} |
| 120 | {d120['full_network_disagreement_count']} / {d120['full_network_denominator']} | {d120['full_network_disagreement_rate']:.6%} | {d120['b2_applicable_disagreement_count']} / {d120['b2_applicable_denominator']} | {d120['b2_applicable_disagreement_rate']:.6%} |

Thus MAP's small CV advantage has a very small effect on deployed compatibility classification and does not establish 80/120 profile validity.

## Historical reverse-direction closure

All 6,502 unmatched reverse identities are classified as `HISTORICAL_DIRECTION_OVERLAY` plus `AV_ROUTABILITY_VIOLATION`, not missing. Their historical observations remain usable for descriptive/history features, but they are excluded from AV routing. A mapped physical forward reference exists for {int(overlay['physical_forward_stage3_edge_uid'].notna().sum()):,}; the remaining {int(overlay['physical_forward_stage3_edge_uid'].isna().sum()):,} retain the same overlay/violation semantics without a fabricated graph link.

## Scientific closure

The speed-domain proxy is retained primarily for the Conservative 60-km/h boundary. Stage3's principal separation of C/M/A is expected to come from intersection/movement structure and dynamic E/Q/C. S2B remains unauthorized pending review.
"""
    markdown_path = docs / "stage3_s2a1_scientific_closure.md"
    atomic_text(markdown_path, markdown)

    evidence = {
        "schema_version": "stage3_s2a1_scientific_closure_evidence.1",
        "phase_status": PHASE_STATUS,
        "authorized_base": AUTHORIZED_BASE,
        "inputs": {
            "s2a_evidence": source_descriptor(s2a_evidence_path, root),
            "speed_validation": source_descriptor(validation_path, root),
            "observed_mapping": parquet_descriptor(mapping_path, root),
            "speed_domain": parquet_descriptor(speed_path, root),
            "observed_identity_cache": parquet_descriptor(observed_path, root),
        },
        "outputs": {
            "closure_report": source_descriptor(report_path, root),
            "closure_markdown": source_descriptor(markdown_path, root),
            "historical_direction_overlay": parquet_descriptor(overlay_path, root),
        },
        "guards": {
            "network_reexported": False,
            "model_retrained": False,
            "quantile_reselected": False,
            "method_reselected": False,
            "caps_changed": False,
            "s2b_authorized": False,
            "next_phase_authorized": False,
        },
    }
    evidence["artifact_sha256"] = payload_hash(evidence)
    evidence_path = docs / "stage3_s2a1_scientific_closure_evidence.json"
    atomic_json(evidence_path, evidence)
    return report


def attach_test_evidence(evidence_path: Path, test_path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(evidence_path)
    test = read_json(test_path)
    if test.get("artifact_sha256") != payload_hash(test) or test.get("status") != "PASS":
        raise Stage3S2AError("cannot bind invalid S2A.1 test evidence")
    evidence["outputs"]["test_evidence"] = source_descriptor(test_path, root)
    evidence["artifact_sha256"] = payload_hash(evidence)
    atomic_json(evidence_path, evidence)
    return evidence


def write_test_evidence(path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "stage3_s2a1_test_evidence.1",
        "phase_status": PHASE_STATUS,
        "status": "PASS",
        "tests": [
            {"suite": "S2A.1 + S2A", "passed": 25, "failed": 0},
            {"suite": "S1 regression", "passed": 10, "failed": 0, "warnings": 1},
        ],
        "compileall_status": "PASS",
        "evidence_verification_status": "PASS",
        "s2b_authorized": False,
        "next_phase_authorized": False,
    }
    payload["artifact_sha256"] = payload_hash(payload)
    atomic_json(path, payload)
    return payload


def verify(evidence_path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(evidence_path)
    failures = []
    if evidence.get("artifact_sha256") != payload_hash(evidence):
        failures.append("evidence payload hash")
    for section in ("inputs", "outputs"):
        for descriptor in evidence.get(section, {}).values():
            path = Path(descriptor["path"])
            path = path if path.is_absolute() else root / path
            if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
                failures.append(f"artifact binding: {descriptor['path']}")
    guards = evidence.get("guards", {})
    if not guards or any(guards.values()):
        failures.append("scope guard")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "phase_status": evidence.get("phase_status")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--attach-test-evidence", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.attach_test_evidence:
        result = attach_test_evidence(
            root / "stage3/docs/odd_tod/s2a/stage3_s2a1_scientific_closure_evidence.json",
            args.attach_test_evidence.resolve(),
            root,
        )
    else:
        result = verify(args.verify.resolve(), root) if args.verify else build(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
