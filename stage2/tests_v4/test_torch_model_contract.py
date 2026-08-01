from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTORCH_PYTHON = Path("D:/anaconda/envs/pytorch/python.exe")


def test_padding_invariance_and_masked_losses() -> None:
    script = r"""
import torch
from stage2.v4.models.losses import masked_huber, stop_two_part_loss
from stage2.v4.models.rc_mstnet_v4 import RCMSTNetV4

torch.manual_seed(7)
model = RCMSTNetV4(
    numeric_feature_count=3,
    categorical_sizes=(8, 5),
    hidden_dim=16,
    categorical_embedding_dim=4,
    transformer_layers=1,
    attention_heads=4,
    dropout=0.0,
).eval()
numeric = torch.randn(1, 3, 3)
missing = torch.zeros_like(numeric, dtype=torch.bool)
categorical = torch.ones(1, 3, 2, dtype=torch.long)
sequence = torch.arange(3).view(1, 3)
mask = torch.zeros(1, 3, dtype=torch.bool)
short = model(numeric, missing, categorical, sequence, mask)["rts_raw"]

numeric_pad = torch.cat((numeric, torch.randn(1, 4, 3)), dim=1)
missing_pad = torch.cat((missing, torch.ones(1, 4, 3, dtype=torch.bool)), dim=1)
categorical_pad = torch.cat(
    (categorical, torch.zeros(1, 4, 2, dtype=torch.long)),
    dim=1,
)
sequence_pad = torch.cat((sequence, torch.zeros(1, 4, dtype=torch.long)), dim=1)
mask_pad = torch.tensor([[False, False, False, True, True, True, True]])
long = model(
    numeric_pad,
    missing_pad,
    categorical_pad,
    sequence_pad,
    mask_pad,
)["rts_raw"][:, :3]
assert torch.allclose(short, long, atol=1e-6), (short, long)

prediction = torch.tensor([0.0, 1.0])
target = torch.tensor([1.0, 0.0])
assert masked_huber(prediction, target, torch.tensor([True, False])) > 0
assert masked_huber(prediction, target, torch.tensor([False, False])) == 0
loss = stop_two_part_loss(
    torch.tensor([0.0, 0.0]),
    torch.tensor([0.4, 0.5]),
    torch.tensor([0.0, 0.5]),
    torch.tensor([True, True]),
)
assert torch.isfinite(loss)
"""
    result = subprocess.run(
        [str(PYTORCH_PYTHON), "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
