from pathlib import Path


def test_full_day_runner_declares_parquet_streaming_support():
    source = Path("stage0/scripts/run_full_day_2017.py").read_text(encoding="utf-8")
    assert "ParquetFile(source)" in source
    assert "iter_batches" in source
