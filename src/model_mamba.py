"""
src/model_mamba.py
===================
Mamba (Selective SSM) backbone for raw-byte network sequence modeling.
Drop-in replacement for NetworkBytePatcher (src/model.py): accepts a (B, T)
tensor of bytes (0..255, with -1 padding clamped internally) and returns
(B, T, 256) next-byte logits — so the existing training loop and
`evaluate_zero_day.py` work by swapping the model class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _Mamba1
    _HAS_MAMBA1 = True
except Exception:
    _HAS_MAMBA1 = False
try:
    from mamba_ssm import Mamba2 as _Mamba2
    _HAS_MAMBA2 = True
except Exception:
    _HAS_MAMBA2 = False

_HAS_MAMBA_SSM = _HAS_MAMBA1 or _HAS_MAMBA2


class MambaBlockTorch(nn.Module):
    """Self-contained selective-SSM (Mamba/S6) block, pure PyTorch, causal."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xx, z = self.in_proj(x).chunk(2, dim=-1)
        xx = self.conv1d(xx.transpose(1, 2))[..., :L].transpose(1, 2)
        xx = F.silu(xx)
        dt, Bm, Cm = torch.split(self.x_proj(xx),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=xx.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBx = (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xx[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + xx * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBytePatcher(nn.Module):
    """Byte-level next-byte model with a Mamba backbone. Drop-in for
    NetworkBytePatcher (same forward contract), no positional embedding."""

    def __init__(self, d_model=128, num_layers=2, max_sequence_length=512,
                 d_state=None, d_conv=4, expand=2, variant="mamba2",
                 headdim=64, force_torch_scan=False):
        super().__init__()
        self.vocab_size = 256
        self.max_sequence_length = max_sequence_length
        self.byte_embedding = nn.Embedding(self.vocab_size, d_model)

        want2 = (variant == "mamba2")
        if force_torch_scan:
            backend = "torch_scan"
        elif want2 and _HAS_MAMBA2:
            backend = "mamba2"
        elif (not want2) and _HAS_MAMBA1:
            backend = "mamba1"
        elif _HAS_MAMBA2:
            backend = "mamba2"
        elif _HAS_MAMBA1:
            backend = "mamba1"
        else:
            backend = "torch_scan"

        ds = d_state if d_state is not None else (64 if backend == "mamba2" else 16)

        def make_block():
            if backend == "mamba2":
                return _Mamba2(d_model=d_model, d_state=ds, d_conv=d_conv,
                               expand=expand, headdim=headdim)
            if backend == "mamba1":
                return _Mamba1(d_model=d_model, d_state=ds, d_conv=d_conv, expand=expand)
            return MambaBlockTorch(d_model, d_state=ds, d_conv=d_conv, expand=expand)

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.blocks = nn.ModuleList([make_block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, self.vocab_size)
        self.backend = backend

    def forward(self, x):
        B, T = x.size()
        assert T <= self.max_sequence_length, \
            f"seq len {T} exceeds max_sequence_length {self.max_sequence_length}"
        x_clamped = torch.clamp(x, min=0)
        h = self.byte_embedding(x_clamped)
        for norm, blk in zip(self.norms, self.blocks):
            h = h + blk(norm(h))
        h = self.ln_f(h)
        return self.predictor(h)


# Alias Mamba2NetworkBytePatcher to MambaBytePatcher for backwards compatibility
Mamba2NetworkBytePatcher = MambaBytePatcher
