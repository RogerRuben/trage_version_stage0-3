"""Train-fitted, decision-time-only static road structure features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Stage2V52ContractError, require_columns, validate_model_inputs


CATEGORICAL_STATIC_FIELDS = ("canonical_highway", "road_class", "observed_direction")
BOOLEAN_STATIC_FIELDS = ("bridge", "tunnel", "synthetic_reverse_edge", "osm_direction_disagreement")
STRUCTURE_REQUIRED_FIELDS = (
    "split", "date", "order_id", "route_sequence", "row_id",
    *CATEGORICAL_STATIC_FIELDS, *BOOLEAN_STATIC_FIELDS,
)


@dataclass(frozen=True)
class StaticStructureArtifact:
    vocabularies: dict[str, tuple[str, ...]]
    fit_dates: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "stage2_v5_2_static_structure.1",
            "fit_scope": "train_only",
            "fit_dates": list(self.fit_dates),
            "vocabularies": {name: list(values) for name, values in self.vocabularies.items()},
            "label_fields_used": [],
            "evaluation_rows_used": 0,
        }


def fit_static_structure_artifact(
    train_frames: Iterable[pd.DataFrame],
    *,
    fit_dates: Sequence[str],
) -> StaticStructureArtifact:
    values = {name: set() for name in CATEGORICAL_STATIC_FIELDS}
    row_count = 0
    for frame in train_frames:  # Bounded Train-partition streaming; no frame accumulation.
        require_columns(frame.columns, STRUCTURE_REQUIRED_FIELDS, product="Train static structure")
        validate_model_inputs(column for column in frame.columns if column in CATEGORICAL_STATIC_FIELDS + BOOLEAN_STATIC_FIELDS)
        row_count += len(frame)
        for name in CATEGORICAL_STATIC_FIELDS:  # Fixed three-field loop.
            values[name].update(frame[name].astype("string").dropna().astype(str).unique())
    if row_count == 0:
        raise Stage2V52ContractError("cannot fit static structure on empty Train data")
    return StaticStructureArtifact(
        vocabularies={name: tuple(sorted(entries)) for name, entries in values.items()},
        fit_dates=tuple(str(value) for value in fit_dates),
    )


def _one_hot(values: pd.Series, vocabulary: Sequence[str]) -> np.ndarray:
    mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
    code = values.astype("string").map(mapping).fillna(0).to_numpy(np.int64)
    return np.eye(len(vocabulary) + 1, dtype=np.float32)[code]


def build_static_structure_features(
    frame: pd.DataFrame,
    artifact: StaticStructureArtifact | Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Encode structure, then scatter features back to the caller's row order."""
    require_columns(frame.columns, STRUCTURE_REQUIRED_FIELDS, product="static structure input")
    payload = artifact.to_payload() if isinstance(artifact, StaticStructureArtifact) else artifact
    if payload.get("fit_scope") != "train_only" or payload.get("evaluation_rows_used") != 0:
        raise Stage2V52ContractError("static structure artifact must be Train-only")
    vocabularies = payload.get("vocabularies", {})
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise Stage2V52ContractError("static structure row_id must be complete and unique")
    working = frame.sort_values(
        ["split", "date", "order_id", "route_sequence", "row_id"], kind="stable"
    ).reset_index(drop=True)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for field in CATEGORICAL_STATIC_FIELDS:  # Fixed schema loop, never an edge loop.
        vocabulary = tuple(str(value) for value in vocabularies.get(field, ()))
        encoded = _one_hot(working[field], vocabulary)
        arrays.append(encoded)
        names.extend(f"{field}={value}" for value in ("__UNSEEN_OR_MISSING__", *vocabulary))
    road_vocabulary = tuple(str(value) for value in vocabularies.get("road_class", ()))
    order = working["order_id"].astype(str)
    upstream = working["road_class"].shift(1).where(order.eq(order.shift(1)))
    downstream = working["road_class"].shift(-1).where(order.eq(order.shift(-1)))
    for field, values in (("upstream_road_class", upstream), ("downstream_road_class", downstream)):
        arrays.append(_one_hot(values, road_vocabulary))
        names.extend(f"{field}={value}" for value in ("__UNSEEN_OR_MISSING__", *road_vocabulary))
    for field in BOOLEAN_STATIC_FIELDS:
        values = working[field].astype("boolean")
        arrays.append(values.fillna(False).to_numpy(np.float32)[:, None])
        arrays.append(values.notna().to_numpy(np.float32)[:, None])
        names.extend((field, f"{field}_available"))
    sorted_features = np.column_stack(arrays).astype(np.float32, copy=False)
    original_row_id = frame["row_id"].to_numpy(copy=True)
    original_index = pd.Index(original_row_id)
    destination = original_index.get_indexer(working["row_id"])
    if np.any(destination < 0):
        raise Stage2V52ContractError("cannot scatter static features back to input row_id")
    features = np.empty_like(sorted_features)
    features[destination] = sorted_features
    return features, tuple(names), original_row_id


def validate_feature_alignment(expected_row_id: np.ndarray, feature_row_id: np.ndarray) -> None:
    if not np.array_equal(np.asarray(expected_row_id), np.asarray(feature_row_id)):
        raise Stage2V52ContractError("static structure features are not aligned to the model batch row_id")
