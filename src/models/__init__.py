from .common import ModelConfig, LayerStackLM
from .pure_deltanet import build_pure
from .hybrid import build_hybrid, placement_to_layer


def build_model(arch: str, cfg: ModelConfig):
    """arch in {'pure', 'hybrid_early', 'hybrid_middle', 'hybrid_late'}."""
    if arch == "pure":
        return build_pure(cfg)
    if arch.startswith("hybrid_"):
        return build_hybrid(cfg, arch.split("_", 1)[1])
    raise ValueError(arch)
