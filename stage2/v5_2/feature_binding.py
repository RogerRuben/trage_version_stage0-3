"""Fail-closed binding to the frozen v5.1 preprocessing and checkpoint schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from stage2.v5.shards import CATEGORY_NAMES, RESERVED_TOKENS

from .contracts import Stage2V52ContractError


@dataclass(frozen=True)
class V51FeatureSchemaBinding:
    categorical_names: tuple[str, ...]
    edge_category_name: str
    edge_source_field: str
    edge_column_index: int
    pad_index: int
    unseen_index: int
    categorical_sizes: tuple[int, ...]
    feature_artifact_sha256: str
    edge_vocabulary_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_v51_feature_schema(
    feature_artifact_path: str | Path,
    *,
    checkpoint_state: Mapping[str, Any],
    expected_edge_source_field: str = "observed_directed_edge_uid",
) -> V51FeatureSchemaBinding:
    """Bind categorical order and indices from artifact bytes and checkpoint tables."""
    path = Path(feature_artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocabularies = payload.get("vocabularies")
    if not isinstance(vocabularies, dict) or not vocabularies:
        raise Stage2V52ContractError("v5.1 feature artifact has no ordered vocabularies")
    categorical_names = tuple(payload.get("categorical_names", tuple(vocabularies.keys())))
    if categorical_names != tuple(vocabularies.keys()) or categorical_names != tuple(CATEGORY_NAMES):
        raise Stage2V52ContractError("categorical order differs from the frozen v5.1 preprocessing schema")
    category_sources = payload.get("categorical_source_fields", {})
    edge_category = next(
        (name for name in categorical_names if category_sources.get(name, name) == expected_edge_source_field),
        None,
    )
    if edge_category is None:
        # Frozen v5.1 artifacts use the explicit category name `edge`; the raw
        # source binding is fixed in v5.1 shard code and is recorded here.
        if "edge" not in categorical_names:
            raise Stage2V52ContractError("cannot bind observed_directed_edge_uid to a v5.1 category")
        edge_category = "edge"
    edge_index = categorical_names.index(edge_category)
    edge_vocabulary = vocabularies[edge_category]
    token_to_index = edge_vocabulary.get("token_to_index", {})
    if "__PAD__" not in token_to_index or "__UNSEEN__" not in token_to_index:
        raise Stage2V52ContractError("v5.1 edge vocabulary lacks PAD/UNSEEN tokens")
    reserved_indices = tuple(token_to_index.get(token) for token in RESERVED_TOKENS)
    if reserved_indices != tuple(range(len(RESERVED_TOKENS))):
        raise Stage2V52ContractError("edge reserved-token indices differ from frozen v5.1 preprocessing")
    sizes = tuple(len(vocabularies[name].get("token_to_index", {})) for name in categorical_names)
    for index, size in enumerate(sizes):
        key = f"embeddings.{index}.weight"
        if key not in checkpoint_state:
            raise Stage2V52ContractError(f"v5.1 checkpoint is missing {key}")
        if int(checkpoint_state[key].shape[0]) != size:
            raise Stage2V52ContractError(
                f"categorical size mismatch at {categorical_names[index]}: artifact={size}, checkpoint={checkpoint_state[key].shape[0]}"
            )
    return V51FeatureSchemaBinding(
        categorical_names=categorical_names,
        edge_category_name=edge_category,
        edge_source_field=expected_edge_source_field,
        edge_column_index=edge_index,
        pad_index=int(token_to_index["__PAD__"]),
        unseen_index=int(token_to_index["__UNSEEN__"]),
        categorical_sizes=sizes,
        feature_artifact_sha256=_sha256(path),
        edge_vocabulary_sha256=_canonical_hash(edge_vocabulary),
    )


def validate_binding_against_model(binding: V51FeatureSchemaBinding, model: Any) -> None:
    if binding.edge_source_field != "observed_directed_edge_uid":
        raise Stage2V52ContractError("edge source field is not the frozen v5.1 directed-edge identity")
    if len(model.embeddings) != len(binding.categorical_names):
        raise Stage2V52ContractError("model categorical table count differs from feature binding")
    for index, (embedding, expected_size) in enumerate(zip(model.embeddings, binding.categorical_sizes)):
        if embedding.num_embeddings != expected_size:
            raise Stage2V52ContractError(
                f"model embedding size mismatch at {binding.categorical_names[index]}"
            )
    edge_embedding = model.embeddings[binding.edge_column_index]
    if edge_embedding.padding_idx != binding.pad_index:
        raise Stage2V52ContractError("checkpoint/model PAD index differs from feature artifact")
