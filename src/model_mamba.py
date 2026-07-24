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


class Mamba2BlockTorch(nn.Module):
    """Pure PyTorch fallback implementation matching mamba_ssm.Mamba2 state_dict schema."""

    def __init__(self, d_model=128, d_state=64, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_ssm = self.nheads * self.headdim

        # in_proj outputs: [z (d_inner), x (d_inner), B (d_state), C (d_state), dt (nheads)] -> total 2*d_inner + 2*d_state + nheads (644)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # conv1d processes [x, B, C] -> total d_inner + 2*d_state (384)
        d_conv_in = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(d_conv_in, d_conv_in, d_conv, groups=d_conv_in, padding=d_conv - 1)

        self.dt_bias = nn.Parameter(torch.ones(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = nn.LayerNorm(self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        zxbcdt = self.in_proj(x)
        z, xbc, dt_raw = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1
        )
        xbc = self.conv1d(xbc.transpose(1, 2))[..., :L].transpose(1, 2)
        xbc = F.silu(xbc)

        x_ssm, B_ssm, C_ssm = torch.split(
            xbc, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(dt_raw + self.dt_bias)
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.nheads, self.headdim, self.d_state, device=x.device, dtype=x.dtype)
        x_reshaped = x_ssm.view(B, L, self.nheads, self.headdim)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            A_t = A.view(1, self.nheads, 1, 1)
            dA = torch.exp(dt_t * A_t)

            B_t = B_ssm[:, t].unsqueeze(1).unsqueeze(1)
            x_t = x_reshaped[:, t].unsqueeze(-1)

            dBx = dt_t * (x_t @ B_t)
            h = dA * h + dBx

            C_t = C_ssm[:, t].unsqueeze(1).unsqueeze(-1)
            y_t = (h @ C_t).squeeze(-1)
            ys.append(y_t.view(B, self.d_inner))

        y = torch.stack(ys, dim=1)
        y = y + x_ssm * self.D.repeat_interleave(self.headdim).view(1, 1, self.d_inner)
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)


class MambaBlockTorch(nn.Module):
    """Self-contained selective-SSM (Mamba-1 / S6) block, pure PyTorch, causal."""

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
            backend = "mamba2_torch" if want2 else "torch_scan"
        ds = d_state if d_state is not None else (64 if "mamba2" in backend else 16)
        self.is_native_mamba = backend in ("mamba1", "mamba2")

        def make_block():
            if backend == "mamba2":
                return _Mamba2(d_model=d_model, d_state=ds, d_conv=d_conv,
                               expand=expand, headdim=headdim)
            if backend == "mamba1":
                return _Mamba1(d_model=d_model, d_state=ds, d_conv=d_conv, expand=expand)
            if backend == "mamba2_torch":
                return Mamba2BlockTorch(d_model=d_model, d_state=ds, d_conv=d_conv,
                                        expand=expand, headdim=headdim)
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
