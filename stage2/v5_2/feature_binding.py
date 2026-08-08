"""Fail-closed bindings to frozen v5.1 feature and source-model artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from stage2.v5.shards import CATEGORY_NAMES, RESERVED_TOKENS

from .contracts import Stage2V52ContractError
from .protocols import get_protocol


SOURCE_BACKBONE_KEYS = (
    "hidden_dim",
    "categorical_embedding_dim",
    "transformer_layers",
    "attention_heads",
    "dropout",
    "minimum_log_scale",
    "maximum_log_scale",
    "distribution_family",
    "maximum_log_p50",
    "maximum_log_p90_p50_ratio",
    "maximum_log_p95_p90_ratio",
    "history_mode",
)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return payload


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_resolved_source_config(path: str | Path) -> dict[str, Any]:
    """Resolve the frozen v5 layered JSON config without accepting CLI overrides."""
    source = Path(path).resolve()
    payload = _load_json(source)
    base_value = payload.get("base_config")
    if not base_value:
        return payload
    base_path = Path(str(base_value))
    if not base_path.is_absolute():
        candidates = (source.parent / base_path, Path.cwd() / base_path)
        base_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[-1].resolve())
    if not base_path.is_file():
        raise Stage2V52ContractError(f"source config base_config does not exist: {base_path}")
    return _deep_merge(load_resolved_source_config(base_path), payload)


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
    categorical_vocabulary_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V51SourceModelBinding:
    protocol_id: str
    source_protocol_id: str
    fit_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    feature_artifact_path: str
    feature_artifact_sha256: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_model_manifest_path: str
    source_model_manifest_sha256: str
    source_model_id: str
    source_config_path: str
    source_config_sha256: str
    resolved_source_config_sha256: str
    distribution_family: str
    history_mode: str
    numeric_features: tuple[str, ...]
    categorical_vocabulary_sha256: str
    model_config: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def bind_v51_feature_schema(
    feature_artifact_path: str | Path,
    *,
    checkpoint_state: Mapping[str, Any],
    expected_edge_source_field: str = "observed_directed_edge_uid",
) -> V51FeatureSchemaBinding:
    """Bind the frozen code order, never JSON object iteration order."""
    path = Path(feature_artifact_path)
    payload = _load_json(path)
    vocabularies = payload.get("vocabularies")
    if not isinstance(vocabularies, dict) or not vocabularies:
        raise Stage2V52ContractError("v5.1 feature artifact has no vocabularies")
    categorical_names = tuple(CATEGORY_NAMES)
    explicit_names = payload.get("categorical_names")
    if explicit_names is not None and tuple(explicit_names) != categorical_names:
        raise Stage2V52ContractError("explicit categorical_names differ from frozen v5.1 code")
    missing_names = sorted(set(categorical_names) - set(vocabularies))
    if missing_names:
        raise Stage2V52ContractError(f"v5.1 feature artifact is missing categories: {missing_names}")
    category_sources = payload.get("categorical_source_fields", {})
    edge_category = next(
        (name for name in categorical_names if category_sources.get(name, name) == expected_edge_source_field),
        "edge" if "edge" in categorical_names else None,
    )
    if edge_category is None:
        raise Stage2V52ContractError("cannot bind observed_directed_edge_uid to v5.1 edge category")
    edge_index = categorical_names.index(edge_category)
    edge_vocabulary = vocabularies[edge_category]
    token_to_index = edge_vocabulary.get("token_to_index", {})
    for name in categorical_names:
        reserved_indices = tuple(
            vocabularies[name].get("token_to_index", {}).get(token) for token in RESERVED_TOKENS
        )
        if reserved_indices != tuple(range(len(RESERVED_TOKENS))):
            raise Stage2V52ContractError(
                f"{name} reserved-token indices differ from frozen v5.1 preprocessing"
            )
    sizes = tuple(len(vocabularies[name].get("token_to_index", {})) for name in categorical_names)
    for index, (name, size) in enumerate(zip(categorical_names, sizes)):
        key = f"embeddings.{index}.weight"
        if key not in checkpoint_state:
            raise Stage2V52ContractError(f"v5.1 checkpoint is missing {key}")
        if int(checkpoint_state[key].shape[0]) != size:
            raise Stage2V52ContractError(
                f"categorical size mismatch at {name}: artifact={size}, checkpoint={checkpoint_state[key].shape[0]}"
            )
    return V51FeatureSchemaBinding(
        categorical_names=categorical_names,
        edge_category_name=edge_category,
        edge_source_field=expected_edge_source_field,
        edge_column_index=edge_index,
        pad_index=int(token_to_index["__PAD__"]),
        unseen_index=int(token_to_index["__UNSEEN__"]),
        categorical_sizes=sizes,
        feature_artifact_sha256=sha256_path(path),
        edge_vocabulary_sha256=_canonical_hash(edge_vocabulary),
        categorical_vocabulary_sha256=_canonical_hash(
            {name: vocabularies[name] for name in categorical_names}
        ),
    )


def bind_v51_source_model(
    *,
    protocol_id: str,
    feature_artifact_path: str | Path,
    source_checkpoint_path: str | Path,
    source_model_manifest_path: str | Path,
    source_config_path: str | Path,
    backbone_kwargs: Mapping[str, Any],
) -> tuple[V51SourceModelBinding, V51FeatureSchemaBinding]:
    """Bind a v5.1 source to the exact Train window and model configuration."""
    import torch

    protocol = get_protocol(protocol_id)
    expected_fit_dates = tuple(protocol.train_dates)
    source_protocol_id = {
        "transfer_tuning": "fold_1", "development": "development",
        "fold_1": "fold_1", "fold_2": "fold_2", "fold_3": "fold_3",
        "legacy_31": "legacy",
    }[protocol_id]
    feature_path = Path(feature_artifact_path)
    checkpoint_path = Path(source_checkpoint_path)
    manifest_path = Path(source_model_manifest_path)
    config_path = Path(source_config_path)
    feature = _load_json(feature_path)
    manifest = _load_json(manifest_path)
    resolved_config = load_resolved_source_config(config_path)
    artifact_fit_dates = tuple(str(value) for value in feature.get("fit_dates", ()))
    model_fit_dates = tuple(str(value) for value in manifest.get("fit_dates", ()))
    if artifact_fit_dates != expected_fit_dates or model_fit_dates != expected_fit_dates:
        raise Stage2V52ContractError(
            f"v5.1 source fit dates do not match protocol {protocol_id}: "
            f"artifact={artifact_fit_dates}, model={model_fit_dates}, expected={expected_fit_dates}"
        )
    expected_validation = tuple(protocol.validation_dates)
    manifest_validation = tuple(str(value) for value in manifest.get("validation_dates", ()))
    if manifest_validation != expected_validation:
        raise Stage2V52ContractError("v5.1 source validation dates differ from frozen protocol")
    checkpoint_sha = sha256_path(checkpoint_path)
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise Stage2V52ContractError("v5.1 model manifest checkpoint hash mismatch")
    saved = torch.load(checkpoint_path, map_location="cpu")
    state = saved.get("model_state_dict", saved)
    checkpoint_config = saved.get("model_config")
    if not isinstance(checkpoint_config, Mapping):
        raise Stage2V52ContractError("v5.1 checkpoint has no model_config")
    feature_binding = bind_v51_feature_schema(feature_path, checkpoint_state=state)
    numeric_features = tuple(str(value) for value in feature.get("numeric_features", ()))
    if not numeric_features or int(checkpoint_config.get("numeric_feature_count", -1)) != len(numeric_features):
        raise Stage2V52ContractError("v5.1 numeric feature list/order is missing or incompatible")
    for key in SOURCE_BACKBONE_KEYS:
        if key not in checkpoint_config or key not in backbone_kwargs:
            raise Stage2V52ContractError(f"source/backbone configuration is missing {key}")
        observed = checkpoint_config[key]
        requested = backbone_kwargs[key]
        if isinstance(observed, float) or isinstance(requested, float):
            if abs(float(observed) - float(requested)) > 1.0e-12:
                raise Stage2V52ContractError(f"backbone {key} differs from v5.1 checkpoint")
        elif observed != requested:
            raise Stage2V52ContractError(f"backbone {key} differs from v5.1 checkpoint")
    split = resolved_config.get("split", {})
    if tuple(str(value) for value in split.get("train_dates", ())) != expected_fit_dates:
        raise Stage2V52ContractError("resolved source config Train dates differ from protocol")
    expected_config_protocol = {
        "transfer_tuning": "fold_1", "development": "development_temporal_evaluation",
        "fold_1": "fold_1", "fold_2": "fold_2", "fold_3": "fold_3",
        "legacy_31": "legacy_frozen_benchmark",
    }[protocol_id]
    if str(split.get("protocol_name", "")) != expected_config_protocol:
        raise Stage2V52ContractError("resolved source config protocol identity mismatch")
    distribution = resolved_config.get("distribution", {})
    deep = resolved_config.get("deep", {})
    source_history_mode = str(manifest.get("history_mode", checkpoint_config["history_mode"]))
    source_distribution = str(manifest.get("distribution_family", checkpoint_config["distribution_family"]))
    if source_history_mode != str(checkpoint_config["history_mode"]):
        raise Stage2V52ContractError("source model manifest history_mode mismatch")
    if source_distribution != str(checkpoint_config["distribution_family"]):
        raise Stage2V52ContractError("source model manifest distribution_family mismatch")
    if distribution and str(distribution.get("family", source_distribution)) != source_distribution:
        raise Stage2V52ContractError("resolved source config distribution family mismatch")
    if deep and "dropout" in deep and float(deep["dropout"]) != float(checkpoint_config["dropout"]):
        raise Stage2V52ContractError("resolved source config dropout mismatch")
    model_id = str(manifest.get("model_id", ""))
    if not model_id:
        raise Stage2V52ContractError("v5.1 source model manifest has no model_id")
    manifest_protocol = str(
        manifest.get("protocol_id", manifest.get("fold_id", manifest.get("source_protocol_id", "")))
    )
    if manifest_protocol != source_protocol_id:
        raise Stage2V52ContractError("v5.1 source model manifest protocol identity mismatch")
    return V51SourceModelBinding(
        protocol_id=protocol_id,
        source_protocol_id=source_protocol_id,
        fit_dates=expected_fit_dates,
        validation_dates=expected_validation,
        feature_artifact_path=feature_path.as_posix(),
        feature_artifact_sha256=sha256_path(feature_path),
        source_checkpoint_path=checkpoint_path.as_posix(),
        source_checkpoint_sha256=checkpoint_sha,
        source_model_manifest_path=manifest_path.as_posix(),
        source_model_manifest_sha256=sha256_path(manifest_path),
        source_model_id=model_id,
        source_config_path=config_path.as_posix(),
        source_config_sha256=sha256_path(config_path),
        resolved_source_config_sha256=_canonical_hash(resolved_config),
        distribution_family=source_distribution,
        history_mode=source_history_mode,
        numeric_features=numeric_features,
        categorical_vocabulary_sha256=feature_binding.categorical_vocabulary_sha256,
        model_config={key: checkpoint_config[key] for key in SOURCE_BACKBONE_KEYS},
    ), feature_binding


def validate_binding_against_model(binding: V51FeatureSchemaBinding, model: Any) -> None:
    if binding.edge_source_field != "observed_directed_edge_uid":
        raise Stage2V52ContractError("edge source field is not frozen v5.1 directed-edge identity")
    if len(model.embeddings) != len(binding.categorical_names):
        raise Stage2V52ContractError("model categorical table count differs from feature binding")
    for index, (embedding, expected_size) in enumerate(zip(model.embeddings, binding.categorical_sizes)):
        if embedding.num_embeddings != expected_size:
            raise Stage2V52ContractError(f"model embedding size mismatch at {binding.categorical_names[index]}")
    if model.embeddings[binding.edge_column_index].padding_idx != binding.pad_index:
        raise Stage2V52ContractError("checkpoint/model PAD index differs from feature artifact")
