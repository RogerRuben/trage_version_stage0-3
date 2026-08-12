"""Freeze the adjudicated 10m S2B products and publish the S2B-to-S3 interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
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


AUTHORIZED_BASE = "800c9e9d0747b6fe875c00e5c1d8932626b29fc0"
PHASE_STATUS = "STAGE3_S2B_INTERSECTION_COMPLEX_FROZEN"
SELECTED_BUFFER_RADIUS_M = 10
ALLOWED_LABELS = {"10_CORRECT", "5_CORRECT", "BOTH_ACCEPTABLE", "NEITHER", "UNCERTAIN"}
EXPECTED_LABEL_COUNTS = {
    "10_CORRECT": 7,
    "5_CORRECT": 0,
    "BOTH_ACCEPTABLE": 61,
    "NEITHER": 2,
    "UNCERTAIN": 0,
}
EXPECTED_STRATA = {
    "signalized": {"10_CORRECT": 3, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 16, "NEITHER": 1, "UNCERTAIN": 0},
    "multi_node_divided_road": {"10_CORRECT": 1, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 19, "NEITHER": 0, "UNCERTAIN": 0},
    "high_degree": {"10_CORRECT": 0, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 10, "NEITHER": 0, "UNCERTAIN": 0},
    "grade_separated": {"10_CORRECT": 2, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 7, "NEITHER": 1, "UNCERTAIN": 0},
    "random_changed": {"10_CORRECT": 1, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 9, "NEITHER": 0, "UNCERTAIN": 0},
}
INTERFACE_COLUMNS = [
    "route_position", "intersection_complex_uid", "incoming_stage3_edge_uid",
    "outgoing_stage3_edge_uid", "route_turn_type", "signed_turn_angle_deg",
    "signal_state", "roundabout", "topological_path_exists",
    "restriction_evidence_present", "restriction_enforcement_certified",
    "movement_legality_state", "interface_status",
]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise Stage3S2AError(f"missing frozen product: {path}")
    return pq.read_table(path).to_pandas()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _descriptor_matches(descriptor: Mapping[str, Any], root: Path) -> bool:
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
        return False
    if path.suffix.lower() == ".parquet" and "row_count" in descriptor and pq.ParquetFile(path).metadata.num_rows != descriptor["row_count"]:
        return False
    return True


def _verify_evidence(path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(path)
    if evidence.get("artifact_sha256") != payload_hash(evidence):
        raise Stage3S2AError(f"evidence payload hash mismatch: {path}")
    for section in ("inputs", "outputs", "products", "documents"):
        for descriptor in evidence.get(section, {}).values():
            if isinstance(descriptor, Mapping) and "path" in descriptor and not _descriptor_matches(descriptor, root):
                raise Stage3S2AError(f"upstream binding mismatch: {descriptor['path']}")
    return evidence


def validate_adjudication(path: Path, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise Stage3S2AError(f"missing completed adjudication: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"adjudication_case_id", "complex_r10", "selection_stratum", "adjudication_label", "reviewer_note"}
    if len(frame) != 70 or not required.issubset(frame.columns) or frame["adjudication_case_id"].nunique() != 70:
        raise Stage3S2AError("completed adjudication row count/schema is not exact")
    invalid = sorted(set(frame["adjudication_label"]) - ALLOWED_LABELS)
    counts = {label: int((frame["adjudication_label"] == label).sum()) for label in EXPECTED_LABEL_COUNTS}
    if invalid or counts != EXPECTED_LABEL_COUNTS:
        raise Stage3S2AError(f"completed adjudication label mismatch: invalid={invalid}, counts={counts}")
    by_stratum: dict[str, dict[str, int]] = {}
    for stratum, expected in EXPECTED_STRATA.items():
        subset = frame[frame["selection_stratum"] == stratum]
        actual = {label: int((subset["adjudication_label"] == label).sum()) for label in EXPECTED_LABEL_COUNTS}
        if actual != expected:
            raise Stage3S2AError(f"adjudication stratum mismatch for {stratum}: {actual}")
        by_stratum[stratum] = actual
    neither = frame[frame["adjudication_label"] == "NEITHER"]
    if len(neither) != 2 or (neither["reviewer_note"].str.len() == 0).any():
        raise Stage3S2AError("two NEITHER reviewer notes are not preserved")
    descriptor = source_descriptor(path, root)
    descriptor.update({
        "row_count": len(frame),
        "header": list(frame.columns),
        "header_sha256": __import__("hashlib").sha256(",".join(frame.columns).encode()).hexdigest(),
        "label_counts": counts,
        "per_stratum_counts": by_stratum,
        "neither_cases": neither[["adjudication_case_id", "complex_r10", "selection_stratum", "reviewer_note"]].to_dict("records"),
    })
    return frame, descriptor


def _promote_immutable(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise Stage3S2AError(f"immutable promotion hash mismatch: {source}")
    os.replace(temporary, destination)
    source_table, final_table = pq.ParquetFile(source), pq.ParquetFile(destination)
    if source_table.metadata.num_rows != final_table.metadata.num_rows or source_table.schema_arrow != final_table.schema_arrow:
        raise Stage3S2AError(f"immutable promotion row/schema mismatch: {source}")
    return {
        "method": "BYTE_IDENTICAL_IMMUTABLE_PROMOTION",
        "source_sha256": sha256_file(source),
        "final_sha256": sha256_file(destination),
        "content_identity": True,
    }


def _promote_complexes_with_radius_alias(source: Path, destination: Path) -> dict[str, Any]:
    """Retain frozen r10 fields and add the contract-required semantic alias."""
    frame = _read_parquet(source)
    if set(frame["tolerance_m"]) != {10}:
        raise Stage3S2AError("complex source is not the frozen r10 product")
    frame.insert(1, "buffer_radius_m", frame["tolerance_m"].astype("int64"))
    atomic_parquet(destination, frame)
    final = _read_parquet(destination)
    if list(final.drop(columns=["buffer_radius_m"]).columns) != list(frame.drop(columns=["buffer_radius_m"]).columns):
        raise Stage3S2AError("complex normalization changed frozen source columns")
    source_frame = _read_parquet(source)
    if not source_frame.equals(final.drop(columns=["buffer_radius_m"])):
        raise Stage3S2AError("complex normalization changed frozen source values")
    return {
        "method": "FROZEN_R10_PLUS_BUFFER_RADIUS_ALIAS",
        "source_sha256": sha256_file(source),
        "final_sha256": sha256_file(destination),
        "source_columns_value_identity": True,
        "derived_field": "buffer_radius_m = tolerance_m = 10",
    }


def _promote_hardlink_alias(source: Path, destination: Path) -> dict[str, Any]:
    """Create the required lookup path without duplicating the movement bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        method = "HARDLINK_ALIAS_NO_DUPLICATE_PAYLOAD"
    except OSError:
        shutil.copyfile(source, temporary)
        method = "BYTE_IDENTICAL_ALIAS_COPY_HARDLINK_UNAVAILABLE"
    os.replace(temporary, destination)
    if sha256_file(source) != sha256_file(destination):
        raise Stage3S2AError("movement lookup alias hash mismatch")
    return {
        "method": method,
        "source_sha256": sha256_file(source),
        "final_sha256": sha256_file(destination),
        "content_identity": True,
        "same_file_identity": bool(os.path.samefile(source, destination)),
    }


def build_boundary_index(edges: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    node_complex = membership.set_index("stage3_node_uid")["intersection_complex_uid"]
    if node_complex.index.duplicated().any():
        raise Stage3S2AError("10m membership assigns a node to multiple complexes")
    frame = edges[["stage3_edge_uid", "from_stage3_node_uid", "to_stage3_node_uid"]].copy()
    frame["from_complex"] = frame["from_stage3_node_uid"].map(node_complex)
    frame["to_complex"] = frame["to_stage3_node_uid"].map(node_complex)
    rows = []
    internal = frame[frame["from_complex"].notna() & frame["from_complex"].eq(frame["to_complex"])]
    rows.extend((row.stage3_edge_uid, row.from_complex, "INTERNAL") for row in internal.itertuples(index=False))
    incoming = frame[frame["to_complex"].notna() & frame["to_complex"].ne(frame["from_complex"])]
    rows.extend((row.stage3_edge_uid, row.to_complex, "INCOMING") for row in incoming.itertuples(index=False))
    outgoing = frame[frame["from_complex"].notna() & frame["from_complex"].ne(frame["to_complex"])]
    rows.extend((row.stage3_edge_uid, row.from_complex, "OUTGOING") for row in outgoing.itertuples(index=False))
    result = pd.DataFrame(rows, columns=["stage3_edge_uid", "intersection_complex_uid", "boundary_role"])
    return result.drop_duplicates().sort_values(["stage3_edge_uid", "intersection_complex_uid", "boundary_role"]).reset_index(drop=True)


def adapt_route_edge_sequence(
    route_edge_sequence: Sequence[str],
    movement_lookup: pd.DataFrame,
    complexes: pd.DataFrame,
    boundary_index: pd.DataFrame,
    reverse_overlay: pd.DataFrame,
    full_network_edge_uids: set[str],
) -> pd.DataFrame:
    """Return deterministic static descriptors without producing route F/U/I."""
    movement_by_pair = movement_lookup.set_index(["incoming_stage3_edge_uid", "outgoing_stage3_edge_uid"], verify_integrity=True)
    complex_by_id = complexes.set_index("intersection_complex_uid")
    network_edges = full_network_edge_uids
    historical = reverse_overlay.set_index("canonical_edge_uid", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    if not route_edge_sequence:
        return pd.DataFrame(columns=INTERFACE_COLUMNS)
    for position in range(max(1, len(route_edge_sequence) - 1)):
        incoming = str(route_edge_sequence[position]) if route_edge_sequence else ""
        outgoing = str(route_edge_sequence[position + 1]) if position + 1 < len(route_edge_sequence) else ""
        base = {column: None for column in INTERFACE_COLUMNS}
        base.update({"route_position": position, "incoming_stage3_edge_uid": incoming or None, "outgoing_stage3_edge_uid": outgoing or None})
        reverse_identity = incoming if incoming in historical.index else (outgoing if outgoing in historical.index else None)
        if reverse_identity is not None:
            overlay = historical.loc[reverse_identity]
            if overlay["av_routability_status"] != "AV_ROUTABILITY_VIOLATION" or bool(overlay["missing_identity"]):
                raise Stage3S2AError("historical reverse overlay semantics changed")
            base["interface_status"] = "AV_ROUTABILITY_VIOLATION"
        elif incoming not in network_edges or (outgoing and outgoing not in network_edges):
            base["interface_status"] = "UNRESOLVED_NETWORK_IDENTITY"
        elif outgoing and (incoming, outgoing) in movement_by_pair.index:
            movement = movement_by_pair.loc[(incoming, outgoing)]
            complex_row = complex_by_id.loc[movement["intersection_complex_uid"]]
            base.update({
                "intersection_complex_uid": movement["intersection_complex_uid"],
                "route_turn_type": movement["route_turn_type"],
                "signed_turn_angle_deg": movement["signed_turn_angle_deg"],
                "signal_state": complex_row["signal_state"],
                "roundabout": bool(complex_row["roundabout_evidence_present"]),
                "topological_path_exists": bool(movement["topological_path_exists"]),
                "restriction_evidence_present": bool(movement["restriction_evidence_present"]),
                "restriction_enforcement_certified": bool(movement["restriction_enforcement_certified"]),
                "movement_legality_state": movement["movement_legality_state"],
                "interface_status": "MATCHED_TOPOLOGICAL_MOVEMENT",
            })
        else:
            base["interface_status"] = "NO_COMPLEX_TRANSITION"
        rows.append(base)
    return pd.DataFrame(rows, columns=INTERFACE_COLUMNS)


def _write_human_summary(path_json: Path, path_md: Path, descriptor: Mapping[str, Any]) -> None:
    summary = {
        "schema_version": "stage3_s2b2_human_adjudication.1",
        "phase_status": PHASE_STATUS,
        "selected_buffer_radius_m": 10,
        "completed_adjudication": dict(descriptor),
        "overall_label_counts": descriptor["label_counts"],
        "per_stratum_counts": descriptor["per_stratum_counts"],
        "neither_reviewer_notes_verbatim": descriptor["neither_cases"],
        "strict_5m_preference_count": 0,
        "sampling_warning": "Targeted stratified diagnostic review; not a population-random accuracy estimate.",
        "decision_statement": "10m selected because it reduces observed under-consolidation without observed strict 5m preference in the targeted review set, while 15m and 20m had already shown increasing structural over-consolidation.",
        "known_limitation": "A single 10m radius can still under-consolidate very large interchange systems; no case-specific patch or manual topology override is introduced.",
    }
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(path_json, summary)
    notes = "\n".join(f"- `{row['adjudication_case_id']}` ({row['selection_stratum']}): {row['reviewer_note']}" for row in descriptor["neither_cases"])
    strata = "\n".join(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`" for key, value in descriptor["per_stratum_counts"].items())
    atomic_text(path_md, f"""# Stage 3 S2B Human Adjudication Summary

Selected buffer radius: **10 m**.

Overall labels: `{json.dumps(descriptor['label_counts'], sort_keys=True)}`.

## Stratified diagnostic results

{strata}

## NEITHER cases — reviewer notes preserved verbatim

{notes}

These two cases indicate that a single 10m radius can still under-consolidate very large interchange systems. They do not prefer 5m, and no manual topology override is added.

This was targeted stratified diagnostic review, not population-random sampling. It supports 10m over 5m but does not estimate population accuracy.
""")


def run(root: Path) -> dict[str, Any]:
    if git_head(root) != AUTHORIZED_BASE:
        raise Stage3S2AError(f"S2B-2 requires authorized base {AUTHORIZED_BASE}")
    base = root / "stage3/output/odd_tod"
    s2a, calibration, final = base / "s2a", base / "s2b/calibration", base / "s2b/final"
    docs = root / "stage3/docs/odd_tod/s2b"
    adjudication_path = docs / "stage3_s2b_completed_adjudication.csv"

    upstream_evidence_paths = {
        "s2a": root / "stage3/docs/odd_tod/s2a/stage3_s2a_evidence_bundle.json",
        "s2a1": root / "stage3/docs/odd_tod/s2a/stage3_s2a1_scientific_closure_evidence.json",
        "s2b1": docs / "stage3_s2b1_evidence_bundle.json",
        "s2b11": docs / "stage3_s2b11_evidence_bundle.json",
    }
    upstream_evidence = {key: _verify_evidence(path, root) for key, path in upstream_evidence_paths.items()}
    _, adjudication = validate_adjudication(adjudication_path, root)

    inputs = {
        "full_network_edges": s2a / "stage3_full_network_edges.parquet",
        "full_network_nodes": s2a / "stage3_full_network_nodes.parquet",
        "control_evidence": s2a / "stage3_control_evidence.parquet",
        "turn_restrictions": s2a / "stage3_turn_restrictions.parquet",
        "speed_domain": s2a / "stage3_speed_domain.parquet",
        "historical_direction_overlay": s2a / "stage3_historical_direction_overlay.parquet",
        "junction_candidates": calibration / "junction_candidates.parquet",
        "complexes_r10": calibration / "complexes_r10.parquet",
        "movements_r10": calibration / "movements_r10.parquet",
        "membership_r10": calibration / "node_membership_r10.parquet",
        "tolerance_comparison": docs / "stage3_s2b_tolerance_comparison.json",
        "s2b11_closure": docs / "stage3_s2b11_5v10_closure.json",
        "s2b11_evidence": docs / "stage3_s2b11_evidence_bundle.json",
        "completed_adjudication": adjudication_path,
    }
    for path in inputs.values():
        if not path.is_file(): raise Stage3S2AError(f"missing required frozen input: {path}")
    # Cross-check the three r10 source hashes against S2B-1's own evidence.
    for source_key, evidence_key in (("complexes_r10", "complexes_r10"), ("movements_r10", "movements_r10"), ("membership_r10", "node_membership_r10")):
        if sha256_file(inputs[source_key]) != upstream_evidence["s2b1"]["products"][evidence_key]["sha256"]:
            raise Stage3S2AError(f"r10 source no longer matches S2B-1 evidence: {source_key}")

    final_paths = {
        "complexes": final / "stage3_intersection_complexes.parquet",
        "movements": final / "stage3_intersection_movements.parquet",
        "membership": final / "stage3_intersection_node_membership.parquet",
        "boundary_index": final / "stage3_edge_complex_boundary_index.parquet",
        "movement_lookup": final / "stage3_route_movement_lookup.parquet",
    }
    promotions = {
        "complexes": _promote_complexes_with_radius_alias(inputs["complexes_r10"], final_paths["complexes"]),
        "movements": _promote_immutable(inputs["movements_r10"], final_paths["movements"]),
        "membership": _promote_immutable(inputs["membership_r10"], final_paths["membership"]),
    }
    promotions["movement_lookup"] = _promote_hardlink_alias(final_paths["movements"], final_paths["movement_lookup"])
    edges = _read_parquet(inputs["full_network_edges"])
    complexes = _read_parquet(final_paths["complexes"])
    movements = _read_parquet(final_paths["movements"])
    membership = _read_parquet(final_paths["membership"])
    overlay = _read_parquet(inputs["historical_direction_overlay"])
    if set(complexes["tolerance_m"]) != {10} or set(complexes["buffer_radius_m"]) != {10} or set(movements["tolerance_m"]) != {10} or set(membership["tolerance_m"]) != {10}:
        raise Stage3S2AError("a non-10m product reached final promotion")
    if "UNSIGNALIZED" in set(complexes["signal_state"]):
        raise Stage3S2AError("missing control was converted to UNSIGNALIZED")
    if set(overlay["historical_direction_status"]) != {"HISTORICAL_DIRECTION_OVERLAY"} or set(overlay["av_routability_status"]) != {"AV_ROUTABILITY_VIOLATION"} or overlay["missing_identity"].any():
        raise Stage3S2AError("reverse overlay semantics changed")
    if set(overlay["canonical_edge_uid"]) & set(edges["stage3_edge_uid"]):
        raise Stage3S2AError("historical reverse identities were inserted into AV topology")

    boundary = build_boundary_index(edges, membership)
    atomic_parquet(final_paths["boundary_index"], boundary)
    docs.mkdir(parents=True, exist_ok=True)
    human_json, human_md = docs / "stage3_s2b_human_adjudication_summary.json", docs / "stage3_s2b_human_adjudication_summary.md"
    _write_human_summary(human_json, human_md, adjudication)

    contract_path = docs / "stage3_s2b_to_s3_contract.md"
    atomic_text(contract_path, """# Stage 3 S2B to S3 Contract

Status: `STAGE3_S2B_INTERSECTION_COMPLEX_FROZEN`. S3 remains unauthorized.

The selected **buffer radius is 10m**. Because candidate-node buffers overlap, the spatial candidate-overlap condition is center distance <= 20m; 10m is not a direct pairwise node-distance threshold. POI evidence remains excluded.

S3 may join static network identity (`stage3_edge_uid`, `intersection_complex_uid`), intersection descriptors (`external_physical_connection_count`, `topological_movement_count`, `internal_length_m`, `road_class_diversity`, `signal_state`, roundabout and grade-separation descriptors), and route-specific movement descriptors (incoming/outgoing edges, geometric turn type/angle, control state, and restriction uncertainty). Frozen speed-domain data joins by `stage3_edge_uid`.

The movement table describes **topological movements, not legally certified movements**. Missing control evidence is `UNKNOWN_CONTROL`, not unsignalized. Grade-separation fields are network-structure and anti-merge evidence, not automatic AV infeasibility. QA flags remain `QA_ONLY` and are not AV capability thresholds.

Historical reverse identities remain historical observations. They are not missing, are not inserted into the AV topology, and return `AV_ROUTABILITY_VIOLATION`. Of 6,502 overlays, 6,388 retain a mapped forward physical reference; no reference is fabricated for the remaining 114. S2B-2 does not propagate this into route F/U/I. The deterministic adapter accepts an ordered edge sequence plus the frozen movement, complex, boundary, reverse-overlay, and full-network identity sets; full-network edges with no complex transition return `NO_COMPLEX_TRANSITION`.

S3 must not interpret topological movement as legal movement, operational stress as AV safety probability, or missing control evidence as unsignalized.
""")
    method_path = docs / "stage3_s2b_final_methodology_note.md"
    atomic_text(method_path, """# S2B Final Methodology Note

The frozen intersection representation promotes the already calibrated r10 products byte-for-byte. Its `buffer_radius_m = 10`; candidate buffers overlap when center distance is <= 20m. No clustering, tolerance comparison, manual topology override, or large-interchange rule was executed in S2B-2.

Targeted human adjudication supports 10m over 5m. It is diagnostic evidence, not a population accuracy estimate. The two NEITHER cases document possible under-consolidation of very large interchange systems under a single global radius.
""")

    product_descriptors = {key: parquet_descriptor(path, root) for key, path in final_paths.items()}
    input_descriptors = {key: (parquet_descriptor(path, root) if path.suffix == ".parquet" else source_descriptor(path, root)) for key, path in inputs.items()}
    input_descriptors["completed_adjudication"].update({key: adjudication[key] for key in ("row_count", "header", "header_sha256", "label_counts", "per_stratum_counts")})
    overlay_summary = {
        "row_count": len(overlay),
        "historical_direction_status": "HISTORICAL_DIRECTION_OVERLAY",
        "av_routability_status": "AV_ROUTABILITY_VIOLATION",
        "mapped_forward_physical_reference_count": int(overlay["physical_forward_stage3_edge_uid"].notna().sum()),
        "without_forward_physical_reference_count": int(overlay["physical_forward_stage3_edge_uid"].isna().sum()),
        "missing_identity_count": int(overlay["missing_identity"].sum()),
        "inserted_into_av_topology": False,
    }
    guards = {
        "tolerance_recalibration_performed": False, "candidate_products_recomputed": False,
        "new_tolerance_tested": False, "full_network_reexported": False,
        "speed_model_changed": False, "stage2_inference_performed": False,
        "dynamic_eqc_calibrated": False, "av_profile_calibrated": False,
        "test31_route_assessment_performed": False, "fallback_routing_performed": False,
        "stage4_performed": False, "manual_topology_override_created": False,
        "large_interchange_rule_created": False, "s3_authorized": False,
        "next_phase_authorized": False,
    }
    report = {
        "schema_version": "stage3_s2b2_closure.1", "phase_status": PHASE_STATUS,
        "authorized_base": AUTHORIZED_BASE, "selected_buffer_radius_m": 10,
        "spatial_candidate_overlap_condition": "center_distance_m <= 20",
        "buffer_radius_not_pairwise_threshold": True,
        "human_adjudication_counts": EXPECTED_LABEL_COUNTS,
        "final_product_rows": {key: value["row_count"] for key, value in product_descriptors.items()},
        "promotions": promotions, "reverse_overlay": overlay_summary,
        "restriction_claim_boundary": "TOPOLOGICAL_MOVEMENT_NOT_LEGALLY_CERTIFIED_MOVEMENT",
        "control_claim_boundary": "MISSING_CONTROL_EVIDENCE_IS_UNKNOWN_NOT_UNSIGNALIZED",
        "s2b_complete": True, "s3_authorized": False, "next_phase_authorized": False,
        "known_limitation": "10m can still under-consolidate very large interchange systems.",
    }
    report["artifact_sha256"] = payload_hash(report)
    report_json, report_md = docs / "stage3_s2b_final_closure_report.json", docs / "stage3_s2b_final_closure_report.md"
    atomic_json(report_json, report)
    atomic_text(report_md, f"""# Stage 3 S2B Final Closure

Status: `{PHASE_STATUS}`. Selected buffer radius: **10m**; candidate-overlap condition: center distance <= 20m.

Human adjudication: `{json.dumps(EXPECTED_LABEL_COUNTS, sort_keys=True)}`. This targeted review supports 10m over 5m and is not a population accuracy estimate.

Final rows: complexes `{len(complexes):,}`, movements `{len(movements):,}`, membership `{len(membership):,}`, edge-complex boundary index `{len(boundary):,}`, route-movement lookup `{len(movements):,}`. Movement, membership, and lookup tables are byte-identical promotions of frozen r10 sources. The complex table retains every frozen r10 source column/value and adds only `buffer_radius_m = tolerance_m = 10` as the contract-required semantic alias.

Historical reverse overlays remain `HISTORICAL_DIRECTION_OVERLAY + AV_ROUTABILITY_VIOLATION`; they are neither missing nor inserted into AV topology. Movement legality and control-state claim boundaries remain conservative.

`S3_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
""")

    manifest_path = docs / "stage3_s2b_final_release_manifest.json"
    manifest = {
        "schema_version": "stage3_s2b2_release_manifest.1", "phase_status": PHASE_STATUS,
        "base_commit": AUTHORIZED_BASE,
        "final_commit": "RECORDED_BY_GIT_COMMIT_AND_REMOTE_HEAD_OUTSIDE_SELF_HASHED_MANIFEST",
        "final_commit_recording_note": "A Git commit cannot embed its own SHA without changing that SHA; the authoritative final commit is the commit containing this manifest and is reported after push.",
        "selected_buffer_radius_m": 10,
        "spatial_candidate_overlap_condition": "center_distance_m <= 20",
        "buffer_radius_not_pairwise_threshold": True,
        "selection_evidence": {key: source_descriptor(path, root) for key, path in {
            "s2b1_calibration": upstream_evidence_paths["s2b1"],
            "s2b11_closure": inputs["s2b11_closure"],
            "completed_human_adjudication": adjudication_path,
        }.items()},
        "human_adjudication_counts": EXPECTED_LABEL_COUNTS,
        "rejected_baseline_radii_m": [5, 15, 20],
        "frozen_inputs": input_descriptors,
        "final_outputs": product_descriptors,
        "source_to_final_provenance": promotions,
        "reverse_overlay_semantics": overlay_summary,
        "restriction_claim_boundary": report["restriction_claim_boundary"],
        "control_claim_boundary": report["control_claim_boundary"],
        "qa_flags_semantics": "QA_ONLY_NOT_AV_CAPABILITY_THRESHOLDS",
        "scope_guards": guards,
    }
    manifest["artifact_sha256"] = payload_hash(manifest)
    atomic_json(manifest_path, manifest)

    evidence_path = docs / "stage3_s2b2_evidence_bundle.json"
    document_paths = {
        "human_summary_json": human_json, "human_summary_markdown": human_md,
        "s2b_to_s3_contract": contract_path, "methodology_note": method_path,
        "closure_json": report_json, "closure_markdown": report_md,
        "release_manifest": manifest_path,
    }
    test_evidence_path = docs / "stage3_s2b2_test_evidence.json"
    if test_evidence_path.is_file(): document_paths["test_evidence"] = test_evidence_path
    evidence = {
        "schema_version": "stage3_s2b2_evidence.1", "phase_status": PHASE_STATUS,
        "authorized_base": AUTHORIZED_BASE,
        "upstream_evidence": {key: source_descriptor(path, root) for key, path in upstream_evidence_paths.items()},
        "inputs": input_descriptors, "outputs": product_descriptors,
        "documents": {key: source_descriptor(path, root) for key, path in document_paths.items()},
        "scope_guards": guards,
    }
    evidence["artifact_sha256"] = payload_hash(evidence)
    atomic_json(evidence_path, evidence)
    return report


def verify(path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(path); failures = []
    if evidence.get("artifact_sha256") != payload_hash(evidence): failures.append("payload_hash")
    for section in ("upstream_evidence", "inputs", "outputs", "documents"):
        for descriptor in evidence.get(section, {}).values():
            if not _descriptor_matches(descriptor, root): failures.append(f"binding:{descriptor.get('path')}")
    guards = evidence.get("scope_guards", {})
    if not guards or any(guards.values()): failures.append("scope_guards")
    if evidence.get("phase_status") != PHASE_STATUS: failures.append("phase_status")
    return {"status": "PASS" if not failures else "FAIL", "phase_status": evidence.get("phase_status"), "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv); root = args.root.resolve()
    result = verify(args.verify.resolve(), root) if args.verify else run(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
