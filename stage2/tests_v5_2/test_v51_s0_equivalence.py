from __future__ import annotations

import json

import pytest


def test_s0_is_numerically_identical_to_v51_for_same_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from stage2.v5.models.rc_mstnet_v5 import RCMSTNetV5
    from stage2.v5_2.feature_binding import bind_v51_feature_schema
    from stage2.v5_2.models.rc_mstnet_transfer import RCMSTNetTransfer

    names = ("edge", "highway", "time_bin", "position_bucket", "route_length_bucket")
    sizes = (7, 6, 6, 6, 6)
    vocabularies = {}
    for name, size in zip(names, sizes):
        tokens = {"__PAD__": 0, "__UNSEEN__": 1, "__RARE__": 2, "__MISSING__": 3}
        tokens.update({f"{name}_{index}": index for index in range(4, size)})
        vocabularies[name] = {"token_to_index": tokens, "seen_tokens": []}
    artifact_path = tmp_path / "feature_artifacts.json"
    artifact_path.write_text(json.dumps({"vocabularies": vocabularies}), encoding="utf-8")
    options = {
        "hidden_dim": 16, "categorical_embedding_dim": 4,
        "transformer_layers": 1, "attention_heads": 4, "dropout": 0.0,
    }
    source = RCMSTNetV5(numeric_feature_count=3, categorical_sizes=sizes, **options).eval()
    checkpoint_path = tmp_path / "v51.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)
    binding = bind_v51_feature_schema(artifact_path, checkpoint_state=source.state_dict())
    transfer = RCMSTNetTransfer(
        numeric_feature_count=3, binding=binding, static_feature_count=5,
        support_tau=2.0, spatial_mode="identity", temporal_mode="none",
        backbone_kwargs=options,
    ).eval()
    transfer.initialize_from_v51(
        checkpoint_path, source_model_id="fixture-v5.1",
        source_feature_artifact_path=artifact_path,
    )
    torch.manual_seed(17)
    numeric = torch.randn(2, 4, 3)
    missing = torch.zeros_like(numeric, dtype=torch.bool)
    categorical = torch.stack(
        [torch.randint(0, size, (2, 4)) for size in sizes], dim=-1
    )
    route_sequence = torch.arange(4).repeat(2, 1)
    pad_mask = torch.zeros(2, 4, dtype=torch.bool)
    with torch.no_grad():
        expected = source(numeric, missing, categorical, route_sequence, pad_mask)
        actual = transfer(numeric, missing, categorical, route_sequence, pad_mask)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.allclose(actual[name], expected[name], rtol=0, atol=0), name
