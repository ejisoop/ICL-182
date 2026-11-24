"""Model B: hybrid Gated DeltaNet with exactly one causal attention layer.

Convention: the hybrid has the same depth as the pure model; the token mixer of
block ``attn_layer`` is replaced by causal softmax attention (the MLP, norms and
residual structure of that block are kept).  Placement names map to depth
fractions of the n_layers-deep stack:

    early  -> round(0.25 * n_layers)   (layer 2 of 0..7 for n_layers=8)
    middle -> round(0.50 * n_layers)   (layer 4)
    late   -> round(0.75 * n_layers)   (layer 6)
"""
from __future__ import annotations

import dataclasses

from .common import LayerStackLM, ModelConfig

PLACEMENT_FRACTIONS = {"early": 0.25, "middle": 0.5, "late": 0.75}


def placement_to_layer(placement: str | int, n_layers: int) -> int:
    if isinstance(placement, int):
        idx = placement
    else:
        idx = int(round(PLACEMENT_FRACTIONS[placement] * n_layers))
    if not 0 <= idx < n_layers:
        raise ValueError(f"attention layer index {idx} out of range for {n_layers} layers")
    return idx


def build_hybrid(cfg: ModelConfig, placement: str | int) -> LayerStackLM:
    idx = placement_to_layer(placement, cfg.n_layers)
    cfg = dataclasses.replace(cfg, attn_layers=[idx])
    model = LayerStackLM(cfg)
    assert model.attention_layer_indices() == [idx]
    return model
