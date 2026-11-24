from dataclasses import replace

from .common import ModelConfig, LayerStackLM
from .pure_deltanet import build_pure
from .hybrid import build_hybrid, placement_to_layer


def build_model(arch: str, cfg: ModelConfig):
    if arch == "transformer":
        return LayerStackLM(replace(cfg, attn_layers=list(range(cfg.n_layers))))
    if arch == "pure":
        return build_pure(cfg)
    if arch.startswith("hybrid_"):
        return build_hybrid(cfg, arch.split("_", 1)[1])
    raise ValueError(arch)
