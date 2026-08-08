from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from stage2.v5_2.contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from stage2.v5_2.protocols import get_protocol, protocol_role_dates
from stage2.v5_2.support_transfer import (
    TAU_CANDIDATE_LABELS, _payload_hash, fit_train_support, freeze_tau_selection,
    select_tau_once,
)
from stage2.v5_2.verification import (
    FINAL_GATE_SPECS, FINAL_REQUIRED_GATES, sha256_file, verify_final_gate_bundle,
)


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _duplicate_tau_inputs() -> tuple[dict[str, object], dict[str, object]]:
    support = fit_train_support(["a", "b", "c", "d"], fit_dates=["20161009"]).to_payload()
    support["protocol_id"] = "transfer_tuning"
    support["artifact_sha256"] = _payload_hash(support)
    baseline = {target: 1.0 for target in CORE_TRANSFER_TARGETS}
    metrics: dict[str, object] = {
        "schema_version": "stage2_v5_2_tau_evaluation.2", "status": "PASS",
        "protocol_id": "transfer_tuning", "protocol_hash": "p" * 64,
        "train_dates": [f"201610{day:02d}" for day in range(9, 19)],
        "validation_dates": ["20161019", "20161020"],
        "support_artifact_embedded_sha256": support["artifact_sha256"],
        "support_artifact_sha256": "s" * 64, "feature_artifact_sha256": "f" * 64,
        "m1_source_checkpoint_sha256": "a" * 64, "m1_checkpoint_sha256": "b" * 64,
        "m1_evaluation_manifest_sha256": "c" * 64, "evaluation_code_sha256": "d" * 64,
        "evaluation_schema": "fixture", "m1_core_mae": baseline,
        "m4_candidates": {
            label: {
                "support_tau_candidate": label, "support_tau_value": 1.0,
                "core_mae": {target: 1.0 + index * 0.1 for target in CORE_TRANSFER_TARGETS},
            }
            for index, label in enumerate(TAU_CANDIDATE_LABELS)
        },
    }
    metrics["artifact_sha256"] = _payload_hash(metrics)
    return support, metrics


def test_tau_labels_survive_duplicate_numeric_quantiles() -> None:
    support, metrics = _duplicate_tau_inputs()
    selection = select_tau_once(metrics, support)
    assert selection["candidate_labels"] == ["p25", "p50", "p75"]
    assert [selection["candidate_table"][label]["support_tau_value"] for label in TAU_CANDIDATE_LABELS] == [1.0, 1.0, 1.0]
    assert selection["selected_candidate"] == "p25"


def test_non_tuning_tau_requires_exact_config_bound_freeze(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from stage2.v5_2.cli import _resolve_training_tau
    support, metrics = _duplicate_tau_inputs()
    support_path, metrics_path = tmp_path / "support.json", tmp_path / "metrics.json"
    support_sha, metrics_sha = _write(support_path, support), _write(metrics_path, metrics)
    selection = select_tau_once(
        metrics, support, metrics_manifest_sha256=metrics_sha, support_artifact_sha256=support_sha,
    )
    selection_path = tmp_path / "selection.json"
    selection_sha = _write(selection_path, selection)
    freeze = freeze_tau_selection(
        selection, metrics, support, selection_artifact_sha256=selection_sha,
        metrics_manifest_sha256=metrics_sha, support_artifact_sha256=support_sha,
    )
    freeze_path = tmp_path / "stage2_v5_2_tau_freeze.json"
    freeze_sha = _write(freeze_path, freeze)
    args = argparse.Namespace(
        model="M4", protocol="fold_1", tau_candidate=None, tau_artifact=None,
        tau_freeze_artifact=str(freeze_path), support_artifact=str(support_path),
    )
    tau, provenance = _resolve_training_tau(
        args, {"tau_freeze": {"expected_file_sha256": freeze_sha}}
    )
    assert tau == 1.0 and provenance["tau_freeze_artifact_sha256"] == freeze_sha
    with pytest.raises(Stage2V52ContractError):
        _resolve_training_tau(args, {"tau_freeze": {"expected_file_sha256": "0" * 64}})


def _final_bundle(tmp_path: Path) -> dict[str, object]:
    protocol_id = "fold_1"
    release = {
        "schema_version": "stage2_v5_2_release_manifest.3", "git_commit": "g" * 40,
        "protocol_id": protocol_id, "protocol_sha256": get_protocol(protocol_id).digest,
        "stage1_release_manifest_sha256": "1" * 64,
        "artifact_hashes": {
            "tau_freeze": "t" * 64, "transfer_manifest": "x" * 64,
            "selected_checkpoint": "c" * 64,
        },
    }
    release_path = tmp_path / "release.json"
    release_sha = _write(release_path, release)
    context = {
        "git_commit": release["git_commit"], "protocol_id": protocol_id,
        "protocol_sha256": release["protocol_sha256"],
        "stage1_release_manifest_sha256": release["stage1_release_manifest_sha256"],
        "tau_freeze_sha256": "t" * 64, "transfer_manifest_sha256": "x" * 64,
        "selected_checkpoint_sha256": "c" * 64,
    }
    rolling_dates = sorted({date for index in range(1, 4) for date in get_protocol(f"fold_{index}").evaluation_dates})
    protocol_dates = sorted({date for values in protocol_role_dates(protocol_id).values() for date in values})
    gates: dict[str, dict[str, object]] = {}
    for name in FINAL_REQUIRED_GATES:
        spec = FINAL_GATE_SPECS[name]
        protocol = {
            "release": protocol_id, "rolling": "rolling_origin_fold_1_2_3",
            "legacy": "legacy_31", "global": None,
        }[spec["protocol"]]
        dates = {
            "none": [], "protocol_scope": protocol_dates,
            "rolling_evaluation": rolling_dates, "legacy": ["20161031"],
        }.get(spec["dates"], list(get_protocol(protocol_id).evaluation_dates))
        model = next((value for value in spec["models"] if value is not None), None)
        report: dict[str, object] = {
            "schema_version": spec["schemas"][0], "status": "PASS",
            "protocol_id": protocol, "model_id": model, "evaluation_dates": dates,
        }
        if spec["dates"] == "canonical_role":
            report["role"] = "evaluation" if protocol_id != "legacy_31" else "legacy"
        if name == "spatial_adoption":
            report["adopt"] = True
        path = tmp_path / f"{name}.json"
        gates[name] = {"report_path": str(path), "report_sha256": _write(path, report)}
    return {
        "release_manifest_path": str(release_path), "release_manifest_sha256": release_sha,
        "release_context": context, "required_gates": gates,
    }


def test_final_gate_specs_reject_arbitrary_fixture_schema(tmp_path: Path) -> None:
    bundle = _final_bundle(tmp_path)
    gate = bundle["required_gates"]["performance"]
    path = Path(gate["report_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = "caller_defined_fixture.1"
    gate["report_sha256"] = _write(path, report)
    result = verify_final_gate_bundle(bundle)
    assert result["status"] == "FAIL"
    assert "schema_version" in result["gate_results"]["performance"]["policy_failures"]


def test_final_gate_specs_accept_one_release_bound_protocol(tmp_path: Path) -> None:
    result = verify_final_gate_bundle(_final_bundle(tmp_path))
    assert result["status"] == "PASS"


def test_final_gate_specs_reject_cross_protocol_mixture(tmp_path: Path) -> None:
    bundle = _final_bundle(tmp_path)
    gate = bundle["required_gates"]["m1_baseline_complete"]
    path = Path(gate["report_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    report["protocol_id"] = "fold_2"
    gate["report_sha256"] = _write(path, report)
    result = verify_final_gate_bundle(bundle)
    assert result["status"] == "FAIL"
    assert "protocol_id" in result["gate_results"]["m1_baseline_complete"]["policy_failures"]
