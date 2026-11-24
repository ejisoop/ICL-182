"""Shared building blocks for the pure and hybrid Gated DeltaNet language models.

Both model families are instances of :class:`LayerStackLM`, which is a stack of
pre-norm residual blocks ``x + mixer(norm(x))`` followed by ``x + mlp(norm(x))``.
The *only* difference between families is the token-mixer used in each block:

* ``"gdn"``  -- Gated DeltaNet (fla.layers.GatedDeltaNet; Yang et al., 2025)
* ``"attn"`` -- conventional multi-head causal softmax self-attention with RoPE

Embedding, MLPs, norms, output head, depth and hidden size are identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    head_dim: int = 64
    mlp_ratio: float = 4.0
    gdn_expand_v: float = 1.0
    gdn_conv_size: int = 4
    attn_layers: List[int] = field(default_factory=list)  # indices of attention blocks
    rope_theta: float = 10000.0
    init_std: float = 0.02
    norm_eps: float = 1e-5
    gdn_head_dim: int | None = None  # defaults to legacy head_dim (64)

    def __post_init__(self):
        if self.gdn_head_dim is None:
            self.gdn_head_dim = self.head_dim
        if self.gdn_head_dim <= 0:
            raise ValueError("gdn_head_dim must be positive")

    def layer_types(self) -> List[str]:
        return ["attn" if i in self.attn_layers else "gdn" for i in range(self.n_layers)]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, ratio: float):
        super().__init__()
        hidden = int(2 * ratio * d / 3)
        hidden = 64 * math.ceil(hidden / 64)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def rope_cache(seq_len: int, dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); rotate pairs (first half, second half)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Conventional multi-head causal softmax attention (the single 'attention layer')."""

    is_conventional_attention = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.head_dim
        inner = self.h * self.dh
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.theta = cfg.rope_theta

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = rope_cache(T, self.dh, self.theta, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.h * self.dh))


class GDNMixer(nn.Module):
    """Wrapper around Flash Linear Attention's GatedDeltaNet layer.

    fla.layers.GatedDeltaNet implements the gated delta rule
        S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T,   o_t = S_t q_t
    with data-dependent decay alpha_t and learning rate beta_t, short causal
    convolutions on q/k/v, L2-normalised q/k and an output gate, following
    Yang et al. (2025).  It contains no softmax attention.
    """

    is_conventional_attention = False

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        from fla.layers import GatedDeltaNet
        # FLA expresses value width relative to key width. Hold value width fixed.
        expand_v = cfg.gdn_expand_v * (cfg.head_dim / cfg.gdn_head_dim)
        self.inner = GatedDeltaNet(hidden_size=cfg.d_model, expand_v=expand_v,
                                   head_dim=cfg.gdn_head_dim, num_heads=cfg.n_heads,
                                   mode="chunk", use_gate=True, use_short_conv=True,
                                   conv_size=cfg.gdn_conv_size, layer_idx=layer_idx,
                                   norm_eps=cfg.norm_eps)

    def forward(self, x):
        return self.inner(x)[0]


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, kind: str):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = CausalSelfAttention(cfg) if kind == "attn" else GDNMixer(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LayerStackLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i, k) for i, k in enumerate(cfg.layer_types()))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # GPT-2 style scaled init for residual output projections
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w_down.weight")):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm_f(x))

    # -- introspection helpers used by tests / reports ----------------------
    def attention_layer_indices(self) -> List[int]:
        return [i for i, b in enumerate(self.blocks)
                if getattr(b.mixer, "is_conventional_attention", False)]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        out = {"embed": self.embed.weight.numel(), "lm_head": self.lm_head.weight.numel(),
               "norm_f": self.norm_f.weight.numel()}
        for i, b in enumerate(self.blocks):
            out[f"block{i}_{b.kind}_mixer"] = sum(p.numel() for p in b.mixer.parameters())
            out[f"block{i}_mlp+norms"] = sum(p.numel() for p in b.mlp.parameters()) + \
                b.norm1.weight.numel() + b.norm2.weight.numel()
        return out
