"""Atomic checkpoint save / load with full RNG state for exact resumption."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: str | Path, model, optimizer, step: int, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, model, optimizer=None, restore_rng: bool = True,
                    map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        r = state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        # RNG states must be CPU ByteTensors; map_location="cuda" would have moved them
        torch.set_rng_state(r["torch"].cpu())
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in r["cuda"]])
    return state
