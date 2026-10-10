"""Narrow opt-in calibration reference; frozen mixed-fleet runs are unchanged."""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path("stage4/config/replay_calibration_v1.json")
FIXED = {
    "version": "replay_calibration_v1", "test_date": "20161031",
    "base_config": "stage4/config/symmetric_flexibility_full_day_v1.json",
    "acceleration_config": "stage4/config/symmetric_flexibility_acceleration_v4.json",
    "reference_policy": "SERVICE_PRESERVING_LOOKAHEAD",
    "requested_q_a": 0.0, "passenger_acceptance_rate": 1.0,
    "chain_sample_size": 1200, "chain_sample_seed": 20261006,
    "chain_max_gap_s": 5400,
    "all_hv_output": "stage4/output/replay_calibration_v1/all_hv",
    "diagnostic_output": "stage4/output/replay_calibration_v1/diagnostic",
    "doc_output": "stage4/docs/replay_calibration_v1",
    "parameter_search": False, "change_request_times": False,
    "change_pickup_eta_calibration": False, "repositioning_enabled": False,
}


def load_calibration(root, path):
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("calibration configuration must stay in the workspace")
    spec = json.loads(resolved.read_text(encoding="utf-8"))
    if spec != FIXED:
        raise ValueError("only the declared one-condition all-HV calibration is authorized")
    return spec, resolved
