"""Train-only support artifacts and support-aware edge representations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Stage2V52ContractError
from .contracts import CORE_TRANSFER_TARGETS

try:  # The data-contract helpers remain importable without torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in lightweight readers.
    torch = None
    nn = None


SUPPORT_GROUPS = ("unseen", "low", "medium", "high")


def support_gate(support: np.ndarray | Iterable[float], tau: float) -> np.ndarray:
    """Return n/(n+tau); unseen edges are exactly structure-only."""
    values = np.asarray(support, dtype=np.float64)
    if not np.isfinite(tau) or tau <= 0:
        raise Stage2V52ContractError("tau must be a finite positive Train-only support statistic")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise Stage2V52ContractError("edge support must be finite and non-negative")
    return values / (values + float(tau))


@dataclass(frozen=True)
class TrainSupportArtifact:
    counts: dict[str, int]
    positive_quantiles: dict[str, float]
    tau_candidates: tuple[float, ...]
    group_boundaries: tuple[float, float]
    fit_dates: tuple[str, ...]
    protocol_id: str = "unspecified"
    source_row_count: int = 0
    unique_traversal_count: int = 0
    duplicate_removed_count: int = 0
    missing_edge_count: int = 0
    input_sha256: str = ""
    source: str = "train_only_observed_directed_edge_uid"

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": "stage2_v5_2_train_support.1",
            "counts": self.counts,
            "positive_quantiles": self.positive_quantiles,
            "tau_candidates": list(self.tau_candidates),
            "group_boundaries": list(self.group_boundaries),
            "fit_dates": list(self.fit_dates),
            "fit_dates_observed": list(self.fit_dates),
            "protocol_id": self.protocol_id,
            "source_row_count": self.source_row_count,
            "unique_traversal_count": self.unique_traversal_count,
            "duplicate_removed_count": self.duplicate_removed_count,
            "missing_edge_count": self.missing_edge_count,
            "input_sha256": self.input_sha256,
            "source": self.source,
            "evaluation_support_used": False,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload


def fit_train_support(edge_ids: Iterable[object], *, fit_dates: Iterable[str]) -> TrainSupportArtifact:
    values = pd.Series(edge_ids, copy=False).astype(str).to_numpy()
    if values.size == 0:
        raise Stage2V52ContractError("cannot fit support from an empty Train partition")
    unique, counts = np.unique(values, return_counts=True)
    positive = counts[counts > 0].astype(np.float64)
    quantile_values = np.quantile(positive, (0.25, 0.50, 0.75, 0.90), method="higher")
    quantiles = {name: float(value) for name, value in zip(("p25", "p50", "p75", "p90"), quantile_values)}
    return TrainSupportArtifact(
        counts={str(edge): int(count) for edge, count in zip(unique, counts)},
        positive_quantiles=quantiles,
        tau_candidates=(quantiles["p25"], quantiles["p50"], quantiles["p75"]),
        group_boundaries=(quantiles["p25"], quantiles["p75"]),
        fit_dates=tuple(str(value) for value in fit_dates),
    )


def fit_train_support_frame(
    frame: pd.DataFrame,
    *,
    protocol_id: str,
    protocol_train_dates: Iterable[str],
    input_sha256: str,
    allowed_date_subset: Iterable[str] | None = None,
) -> TrainSupportArtifact:
    """Fit support from unique physical Train traversals with verified dates."""
    required = ("split", "date", "order_id", "traversal_id", "observed_directed_edge_uid")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise Stage2V52ContractError(f"support input is missing: {missing}")
    split = frame["split"].astype(str)
    if not split.eq("train").all():
        raise Stage2V52ContractError("support input contains non-Train rows")
    observed_dates = tuple(sorted(frame["date"].astype(str).unique()))
    expected = tuple(str(value) for value in protocol_train_dates)
    allowed = tuple(str(value) for value in allowed_date_subset) if allowed_date_subset is not None else expected
    if not set(allowed) <= set(expected) or observed_dates != tuple(sorted(allowed)):
        raise Stage2V52ContractError(
            f"support dates {observed_dates} do not match the allowed protocol Train dates {tuple(sorted(allowed))}"
        )
    identity = ["date", "order_id", "traversal_id"]
    edge = frame["observed_directed_edge_uid"].astype("string")
    valid = edge.notna() & edge.str.len().fillna(0).gt(0)
    working = frame.loc[valid, [*identity, "observed_directed_edge_uid"]].copy()
    inconsistent = working.groupby(identity, sort=False, observed=True)["observed_directed_edge_uid"].nunique(dropna=True)
    if (inconsistent > 1).any():
        raise Stage2V52ContractError("one physical traversal maps to multiple directed edges in support input")
    unique = working.drop_duplicates(identity, keep="first")
    base = fit_train_support(unique["observed_directed_edge_uid"], fit_dates=observed_dates)
    return TrainSupportArtifact(
        counts=base.counts,
        positive_quantiles=base.positive_quantiles,
        tau_candidates=base.tau_candidates,
        group_boundaries=base.group_boundaries,
        fit_dates=observed_dates,
        protocol_id=str(protocol_id),
        source_row_count=int(len(frame)),
        unique_traversal_count=int(len(unique)),
        duplicate_removed_count=int(len(working) - len(unique)),
        missing_edge_count=int((~valid).sum()),
        input_sha256=str(input_sha256),
    )


def lookup_train_support(edge_ids: Iterable[object], artifact: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if artifact.get("evaluation_support_used") is not False:
        raise Stage2V52ContractError("support artifact does not prove Train-only fitting")
    counts = artifact.get("counts")
    if not isinstance(counts, Mapping):
        raise Stage2V52ContractError("support artifact has no count mapping")
    edges = pd.Series(edge_ids, copy=False).astype(str)
    support = edges.map(counts).fillna(0).to_numpy(dtype=np.int64)
    boundaries = artifact.get("group_boundaries", ())
    if len(boundaries) != 2:
        raise Stage2V52ContractError("support artifact requires frozen low/high boundaries")
    low, high = map(float, boundaries)
    group = np.full(len(edges), "medium", dtype=object)
    group[support == 0] = "unseen"
    group[(support > 0) & (support <= low)] = "low"
    group[support > high] = "high"
    return support, group.astype(str)


def select_tau_once(
    candidate_mae_by_target: Mapping[float, Mapping[str, float]],
    v5_1_mae_by_target: Mapping[str, float],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Select tau by frozen four-target macro normalized validation MAE."""
    allowed = tuple(float(value) for value in artifact.get("tau_candidates", ()))
    if not allowed or set(map(float, candidate_mae_by_target)) != set(allowed):
        raise Stage2V52ContractError("tau scores must cover exactly Train P25/P50/P75 candidates")
    if set(v5_1_mae_by_target) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("tau baseline must contain exactly four core micro targets")
    scores: dict[float, float] = {}
    for tau in allowed:
        target_mae = candidate_mae_by_target[tau]
        if set(target_mae) != set(CORE_TRANSFER_TARGETS):
            raise Stage2V52ContractError("tau candidate includes a non-core target or misses a core target")
        normalized = []
        for target in CORE_TRANSFER_TARGETS:
            baseline = float(v5_1_mae_by_target[target])
            candidate = float(target_mae[target])
            if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(candidate):
                raise Stage2V52ContractError("tau selection MAE values must be finite with positive M1 baseline")
            normalized.append(candidate / baseline)
        scores[tau] = float(np.mean(normalized))
    selected = min(allowed, key=lambda value: (scores[value], value))
    return {
        "schema_version": "stage2_v5_2_tau_selection.1",
        "selection_protocol": "transfer_tuning",
        "train_dates": [f"201610{day:02d}" for day in range(9, 19)],
        "validation_dates": ["20161019", "20161020"],
        "rolling_reselection_allowed": False,
        "selection_metric": "macro_normalized_mae_over_4_core_micro_targets",
        "core_targets": list(CORE_TRANSFER_TARGETS),
        "rts_used": False,
        "pace_used": False,
        "candidates": list(allowed),
        "scores": {str(value): scores[value] for value in allowed},
        "selected_tau": float(selected),
        "tie_break": "smaller_tau",
    }


if nn is not None:
    class StructureEncoder(nn.Module):
        """Encode decision-time static road semantics for seen or unseen edges."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
            super().__init__()
            hidden = int(hidden_dim or output_dim)
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, static_features: "torch.Tensor") -> "torch.Tensor":
            return self.network(static_features)


    class SupportAwareEdgeRepresentation(nn.Module):
        """S0-S3 compatible edge representation with a Train-only gate."""

        MODES = {"identity", "structure_only", "concat", "support_aware"}

        def __init__(
            self,
            *,
            edge_vocabulary_size: int,
            static_feature_count: int,
            embedding_dim: int,
            tau: float,
            mode: str = "support_aware",
            padding_idx: int = 0,
        ):
            super().__init__()
            if mode not in self.MODES:
                raise Stage2V52ContractError(f"unknown spatial transfer mode: {mode}")
            if tau <= 0:
                raise Stage2V52ContractError("tau must be positive")
            self.mode = mode
            self.tau = float(tau)
            self.id_embedding = nn.Embedding(edge_vocabulary_size, embedding_dim, padding_idx=padding_idx)
            self.structure_encoder = StructureEncoder(static_feature_count, embedding_dim)
            self.concat_projection = nn.Linear(embedding_dim * 2, embedding_dim)

        def forward(
            self,
            edge_index: "torch.Tensor",
            static_features: "torch.Tensor",
            train_support: "torch.Tensor",
        ) -> "torch.Tensor":
            if torch.any(train_support < 0):
                raise Stage2V52ContractError("train_support cannot be negative")
            identity = self.id_embedding(edge_index)
            structure = self.structure_encoder(static_features)
            if self.mode == "identity":
                return identity
            if self.mode == "structure_only":
                return structure
            if self.mode == "concat":
                return self.concat_projection(torch.cat((identity, structure), dim=-1))
            gate = train_support.to(structure.dtype) / (train_support.to(structure.dtype) + self.tau)
            return gate.unsqueeze(-1) * identity + (1.0 - gate.unsqueeze(-1)) * structure
else:  # pragma: no cover
    StructureEncoder = None
    SupportAwareEdgeRepresentation = None
