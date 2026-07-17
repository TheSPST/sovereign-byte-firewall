"""
Tests for the Mamba backbone (experiment/mamba-backbone).
Run on a machine with torch:  pytest tests/test_model_mamba.py -v
Uses force_torch_scan=True so it runs anywhere (no mamba_ssm/CUDA needed).
"""
import torch
from src.model_mamba import MambaBytePatcher, build_backbone


def _tiny(**kw):
    return MambaBytePatcher(d_model=32, num_layers=2, max_sequence_length=64,
                            d_state=8, force_torch_scan=True, **kw)


def test_forward_shape_and_logits():
    m = _tiny().eval()
    x = torch.randint(0, 256, (3, 40))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (3, 40, 256), out.shape
    assert torch.isfinite(out).all()


def test_padding_sentinel_is_safe():
    """-1 padding must not crash the embedding (clamped internally)."""
    m = _tiny().eval()
    x = torch.randint(0, 256, (2, 20))
    x[:, -5:] = -1
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 20, 256)


def test_causality():
    """Output at position t must not depend on inputs at positions > t."""
    m = _tiny().eval()
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        out1 = m(x)
        x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 123) % 256
        out2 = m(x2)
    past = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    last = (out1[:, -1] - out2[:, -1]).abs().max().item()
    assert past < 1e-5, f"causality violated: past changed by {past}"
    assert last > 1e-6, f"last-step output should respond, got {last}"


def test_no_positional_embedding():
    """SSMs encode order via recurrence -> no pos embedding (unlike the transformer)."""
    m = _tiny()
    assert not any("pos" in n.lower() for n, _ in m.named_parameters())


def test_backend_flag_and_factory():
    m = _tiny()
    assert m.backend == "torch_scan"
    fac = build_backbone("mamba", d_model=32, num_layers=1, max_sequence_length=64,
                         force_torch_scan=True)
    assert isinstance(fac, MambaBytePatcher)


def test_param_count_reasonable():
    """Sanity: a 2-layer d128 Mamba is in the same lightweight ballpark as the
    transformer (~1-2M params), not orders of magnitude larger."""
    m = MambaBytePatcher(d_model=128, num_layers=2, max_sequence_length=512,
                         force_torch_scan=True)
    n = sum(p.numel() for p in m.parameters())
    assert 0.3e6 < n < 5e6, f"unexpected param count {n}"
