from __future__ import annotations

import pandas as pd
import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.micro_products import DIMENSIONS, fit_train_cdf_thresholds
from stage2.v5_2.support_transfer import fit_train_support_frame


def test_support_rejects_non_train_rows_and_wrong_dates() -> None:
    frame = pd.DataFrame({
        "split": ["train", "evaluation"], "date": ["20161009", "20161022"],
        "order_id": ["a", "b"], "traversal_id": [1, 2],
        "observed_directed_edge_uid": ["e1", "e2"],
    })
    with pytest.raises(Stage2V52ContractError, match="non-Train"):
        fit_train_support_frame(
            frame, protocol_id="transfer_tuning", protocol_train_dates=("20161009",),
            input_sha256="fixture",
        )


def test_support_deduplicates_physical_traversal_without_nan_edge() -> None:
    frame = pd.DataFrame({
        "split": ["train"] * 3, "date": ["20161009"] * 3,
        "order_id": ["a", "a", "b"], "traversal_id": [1, 1, 2],
        "observed_directed_edge_uid": ["e1", "e1", None],
    })
    artifact = fit_train_support_frame(
        frame, protocol_id="transfer_tuning", protocol_train_dates=("20161009",),
        input_sha256="fixture",
    ).to_payload()
    assert artifact["counts"] == {"e1": 1}
    assert artifact["duplicate_removed_count"] == 1
    assert artifact["missing_edge_count"] == 1
    assert "artifact_sha256" in artifact


def test_cdf_rejects_evaluation_rows() -> None:
    frame = pd.DataFrame({
        "split": ["evaluation"], "date": ["20161022"],
        "protocol_id": ["fold_1"], "model_id": ["M4"], "prediction_source": ["model"],
    })
    for column in DIMENSIONS.values():
        frame[column] = 0.5
    with pytest.raises(Stage2V52ContractError, match="non-Train"):
        fit_train_cdf_thresholds(
            frame, protocol_id="fold_1", protocol_train_dates=("20161009",),
            input_sha256="fixture",
        )
