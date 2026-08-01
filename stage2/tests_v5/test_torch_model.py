from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTORCH_PYTHON = Path("D:/anaconda/envs/pytorch/python.exe")


def test_physical_outputs_horizon_ablation_and_trained_scale() -> None:
    script = r'''
import torch
from stage2.v5.models.losses import weighted_lognormal_nll
from stage2.v5.models.rc_mstnet_v5 import RCMSTNetV5

torch.manual_seed(7)
model = RCMSTNetV5(numeric_feature_count=3, categorical_sizes=(8, 5), hidden_dim=16, categorical_embedding_dim=4, transformer_layers=1, attention_heads=4, dropout=0.0).eval()
numeric = torch.randn(2, 5, 3)
missing = torch.zeros_like(numeric, dtype=torch.bool)
categorical = torch.ones(2, 5, 2, dtype=torch.long)
sequence = torch.arange(5).repeat(2, 1)
pad = torch.zeros(2, 5, dtype=torch.bool)
recent = torch.randn(2, 5, 4)
profile = torch.randn(2, 5, 3)
controls = dict(forecast_horizon_s=torch.ones(2,5)*300, history_age_s=torch.ones(2,5)*60, history_support=torch.ones(2,5)*10)
out = model(numeric, missing, categorical, sequence, pad, recent_history=recent, profile_history=profile, **controls)
assert torch.all(out['crawl_share'] >= 0) and torch.all(out['stop_share'] >= 0)
assert torch.all(out['crawl_share'] + out['stop_share'] <= 1.0 + 1e-7)
assert torch.all(out['pace_pred_mean'] > 0)
assert torch.all(out['pace_pred_p50'] <= out['pace_pred_p90'])
assert torch.all(out['pace_pred_p90'] <= out['pace_pred_p95'])
assert torch.all((out['history_recent_gate'] >= 0) & (out['history_recent_gate'] <= 1))
near = model(numeric, missing, categorical, sequence, pad, recent_history=recent, profile_history=profile, forecast_horizon_s=torch.ones(2,5)*60, history_age_s=torch.ones(2,5)*30, history_support=torch.ones(2,5)*10)
far = model(numeric, missing, categorical, sequence, pad, recent_history=recent, profile_history=profile, forecast_horizon_s=torch.ones(2,5)*3600, history_age_s=torch.ones(2,5)*30, history_support=torch.ones(2,5)*10)
assert near['history_recent_gate'].mean() > far['history_recent_gate'].mean()
without_recent = model(numeric, missing, categorical, sequence, pad, recent_history=recent, profile_history=profile, use_recent=False, **controls)
without_profile = model(numeric, missing, categorical, sequence, pad, recent_history=recent, profile_history=profile, use_profile=False, **controls)
assert torch.all(without_recent['history_recent_gate'] == 0)
assert torch.all(without_profile['history_recent_gate'] == 1)
model.train()
out = model(numeric, missing, categorical, sequence, pad)
out['pace_log_scale'].retain_grad()
loss = weighted_lognormal_nll(out['pace_log_mu'], out['pace_log_scale'], torch.ones(2,5)*0.3, torch.ones(2,5,dtype=torch.bool), torch.ones(2,5))
loss.backward()
assert out['pace_log_scale'].grad is not None
assert torch.isfinite(out['pace_log_scale'].grad).all()
'''
    result = subprocess.run([str(PYTORCH_PYTHON), "-c", script], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
