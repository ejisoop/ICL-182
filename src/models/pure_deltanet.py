"""Model A: pure Gated DeltaNet (no conventional attention)."""
from __future__ import annotations

from .common import LayerStackLM, ModelConfig


def build_pure(cfg: ModelConfig) -> LayerStackLM:
    assert not cfg.attn_layers, "pure Gated DeltaNet must not contain attention layers"
    return LayerStackLM(cfg)
