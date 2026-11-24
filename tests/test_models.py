import pytest
import torch
from src.data.tokenizer import VOCAB_SIZE
from src.models import ModelConfig, build_model
from src.models.common import CausalSelfAttention, GDNMixer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels need CUDA")
ARCHS = ["pure", "hybrid_early", "hybrid_middle", "hybrid_late"]


def cfg():
    return ModelConfig(vocab_size=VOCAB_SIZE)


def count_attn(m):
    return sum(isinstance(x, CausalSelfAttention) for x in m.modules()), \
        sum(isinstance(x, torch.nn.MultiheadAttention) for x in m.modules())


@cuda
def test_pure_has_no_attention():
    m = build_model("pure", cfg())
    assert count_attn(m) == (0, 0)
    assert m.attention_layer_indices() == []
    assert sum(isinstance(x, GDNMixer) for x in m.modules()) == 8


@cuda
@pytest.mark.parametrize("arch,idx", [("hybrid_early", 2), ("hybrid_middle", 4), ("hybrid_late", 6)])
def test_hybrid_exactly_one_attention_at_placement(arch, idx):
    m = build_model(arch, cfg())
    assert count_attn(m) == (1, 0)
    assert m.attention_layer_indices() == [idx]
    assert sum(isinstance(x, GDNMixer) for x in m.modules()) == 7
    assert len(m.blocks) == 8


@cuda
def test_param_counts_close():
    n = {a: build_model(a, cfg()).num_params() for a in ARCHS}
    assert abs(n["hybrid_middle"] - n["pure"]) / n["pure"] < 0.02
    assert n["hybrid_early"] == n["hybrid_middle"] == n["hybrid_late"]


def test_attention_layer_is_causal_cpu():
    torch.manual_seed(0)
    att = CausalSelfAttention(cfg())
    x = torch.randn(2, 20, 256)
    y = att(x)
    x2 = x.clone(); x2[:, 12:] = torch.randn(2, 8, 256)
    y2 = att(x2)
    assert torch.allclose(y[:, :12], y2[:, :12], atol=1e-5)
    assert not torch.allclose(y[:, 12:], y2[:, 12:], atol=1e-3)


@cuda
@pytest.mark.parametrize("arch", ARCHS)
def test_full_model_causal_and_output_shape(arch):
    torch.manual_seed(0)
    m = build_model(arch, cfg()).cuda().float()
    x = torch.randint(3, VOCAB_SIZE, (2, 64), device="cuda")
    y = m(x)
    assert y.shape == (2, 64, VOCAB_SIZE)
    assert torch.isfinite(y).all()
    x2 = x.clone(); x2[:, 40:] = torch.randint(3, VOCAB_SIZE, (2, 24), device="cuda")
    y2 = m(x2)
    assert torch.allclose(y[:, :40], y2[:, :40], atol=1e-3)


@cuda
def test_right_padding_does_not_change_prefix_outputs():
    torch.manual_seed(0)
    m = build_model("hybrid_middle", cfg()).cuda().float()
    x = torch.randint(3, VOCAB_SIZE, (1, 30), device="cuda")
    xp = torch.cat([x, torch.zeros(1, 50, dtype=torch.long, device="cuda")], 1)
    assert torch.allclose(m(x), m(xp)[:, :30], atol=1e-3)


@cuda
def test_early_init_transplant(tmp_path, monkeypatch):
    """--early-init-seed copies exactly the layer-0/1 GDN A_log/dt_bias/conv tensors from the
    donor seed and leaves every other tensor equal to the plain `seed` initialisation."""
    import src.training.train as T
    from src.utils.reproducibility import set_seed
    from src.utils.logging import load_config
    mc = T.model_config_from(load_config("configs/base.yaml"))
    set_seed(10); base = {k: v.clone() for k, v in build_model("pure", mc).state_dict().items()}
    set_seed(11); donor = {k: v.clone() for k, v in build_model("pure", mc).state_dict().items()}
    captured = {}

    def fake_opt(model, tc):          # stop right after model construction
        captured.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        raise RuntimeError("stop")
    monkeypatch.setattr(T, "make_optimizer", fake_opt)
    with pytest.raises(RuntimeError, match="stop"):
        T.train("configs/base.yaml", "pure", 10, str(tmp_path), early_init_seed=11)
    keys = [k for k in base if T.EARLY_KEYS.match(k)]
    assert len(keys) == 10
    for k in base:
        ref = donor[k] if k in keys else base[k]
        assert torch.equal(captured[k], ref.cpu()), k
    assert any(not torch.equal(base[k], donor[k]) for k in keys)
