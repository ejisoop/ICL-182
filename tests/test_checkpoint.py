"""Resume path: checkpoints must restore weights, optimizer and RNG state, also when loaded
with map_location="cuda" (as train.py does). Regression test for a resume crash
("RNG state must be a torch.ByteTensor") found when resuming Experiment 2 runs."""
import random

import numpy as np
import pytest
import torch

from src.training.checkpoint import load_checkpoint, save_checkpoint

LOCS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("loc", LOCS)
def test_checkpoint_roundtrip_restores_rng(tmp_path, loc):
    model = torch.nn.Linear(4, 3).to(loc)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4, device=loc)).sum().backward(); opt.step()
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    save_checkpoint(tmp_path / "c.pt", model, opt, step=7)
    expect = (random.random(), np.random.rand(), torch.rand(1).item(),
              torch.rand(1, device="cuda").item() if torch.cuda.is_available() else None)
    w = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    st = load_checkpoint(tmp_path / "c.pt", model, opt, map_location=loc)
    got = (random.random(), np.random.rand(), torch.rand(1).item(),
           torch.rand(1, device="cuda").item() if torch.cuda.is_available() else None)
    assert st["step"] == 7 and got == expect and torch.equal(model.weight, w)
