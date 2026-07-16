from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_paper_pipeline_blocks_formal_stage4_and_uses_stage35_manifest():
    config = yaml.safe_load((ROOT / "config/pipeline_research_v3.yaml").read_text(encoding="utf-8"))
    assert config["formal_stage4_enabled"] is False
    assert config["stage4"]["formal_full_day_gate"] == "HOLD"
    assert config["stage4"]["input_manifest"] == config["stage35"]["output_manifest"]
    assert 500 <= config["stage4"]["maximum_technical_validation_orders"] <= 2000


def test_paper_pipeline_test_dates_are_held_out():
    config = yaml.safe_load((ROOT / "config/pipeline_research_v3.yaml").read_text(encoding="utf-8"))
    test = set(config["dates"]["test"])
    prior = set(config["dates"]["upstream_fit"] + config["dates"]["train"] + config["dates"]["validation"])
    assert not test & prior
