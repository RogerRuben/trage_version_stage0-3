from __future__ import annotations

from pathlib import Path

import yaml

from stage2.v5.final_upstream import audit_stage0_config_delta


REPO = Path(__file__).resolve().parents[2]


def test_final_upstream_changes_only_dates_and_isolated_paths() -> None:
    result = audit_stage0_config_delta(
        REPO / "stage0/config/stage0_v6_final.yaml",
        REPO / "stage2/config/stage2_v5_final_stage0.yaml",
        v5_config_path=REPO / "stage2/config/stage2_v5.json",
    )
    assert result["status"] == "PASS"
    assert result["materialized_dates"] == ["20161028", "20161029", "20161030"]
    assert result["unexpected_changed_fields"] == []


def test_final_upstream_rejects_quality_threshold_change(tmp_path: Path) -> None:
    source = REPO / "stage2/config/stage2_v5_final_stage0.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["final_quality"]["order_pass_minimum_buffer20_share"] = 0.5
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = audit_stage0_config_delta(
        REPO / "stage0/config/stage0_v6_final.yaml",
        changed,
        v5_config_path=REPO / "stage2/config/stage2_v5.json",
    )
    assert result["status"] == "FAIL"
    assert "final_quality.order_pass_minimum_buffer20_share" in result["unexpected_changed_fields"]
