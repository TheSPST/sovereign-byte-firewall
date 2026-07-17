"""
model_mamba.py
==============
EXPERIMENT (branch: experiment/mamba-backbone).

A Mamba / state-space-model backbone as a drop-in alternative to the transformer
`NetworkBytePatcher`. Same forward contract — `forward(x)` takes a (B, T) int
tensor of bytes (0..255, with -1 padding clamped internally) and returns
(B, T, 256) next-byte logits — so the existing training loop and
`evaluate_zero_day.py` work by swapping the model class.

Why: Mamba processes a sequence in linear time O(N) vs the transformer's O(N²),
and the literature (NetMamba 1–60× faster inference, MambaNetBurst byte-level
with no pretraining) suggests it can match accuracy at materially better speed —
which is what the edge / pre-filter throughput story needs. This file exists to
run the A/B: same data, same held-out protocol, compare AUC AND throughput.

Two backends, auto-selected:
  - `mamba_ssm.Mamba` (fused CUDA kernel) when installed — use this for REAL
    training/benchmarking on GPU (`pip install mamba-ssm causal-conv1d`).
  - a self-contained pure-PyTorch selective-SSM (below) otherwise — correct and
    linear-time but the sequential scan is Python-loop slow; for shape/causality
    tests and CPU/MPS smoke runs only.

Note: SSMs encode order through their recurrence, so there is NO positional
embedding (a genuine simplification vs the transformer). Checkpoints are NOT
cross-compatible with the transformer — Mamba must be trained from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mamba-1 (classic selective scan) and Mamba-2 (SSD / matmul-optimized) both ship
# in mamba_ssm. Mamba-2 is faster on GPU (tensor-core matmuls, larger d_state) and
# its scalar-A structure may regularize — tested in the accuracy A/B, not assumed.
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

_HAS_MAMBA_SSM = _HAS_MAMBA1 or _HAS_MAMBA2  # any fused kernel available


class MambaBlockTorch(nn.Module):
    """Self-contained selective-SSM (Mamba/S6) block, pure PyTorch, causal.

    x -> in_proj -> [xx, gate z]; xx through a causal depthwise conv + SiLU;
    data-dependent (delta, B, C) drive a diagonal selective state-space scan;
    output is gated by SiLU(z) and projected back. Linear-time; the scan loop is
    slow in Python but numerically faithful for testing."""

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
        xx, z = self.in_proj(x).chunk(2, dim=-1)               # (B, L, d_inner) each
        # causal depthwise conv (left-pad, drop the right overhang)
        xx = self.conv1d(xx.transpose(1, 2))[..., :L].transpose(1, 2)
        xx = F.silu(xx)
        dt, Bm, Cm = torch.split(self.x_proj(xx),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B, L, d_inner)
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=xx.dtype)
        ys = []
        for t in range(L):                                     # O(L) causal scan
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                       # (B, d_inner, d_state)
            dBx = (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xx[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))                   # (B, d_inner)
        y = torch.stack(ys, dim=1) + xx * self.D              # (B, L, d_inner)
        y = y * F.silu(z)                                      # gate
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

        # Resolve backend: requested variant if its kernel is present, else the
        # other Mamba kernel, else the pure-torch scan (Mamba-1 style).
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

        # Mamba-2 runs a larger state dim well (matmul path); Mamba-1 favors 16.
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
        x_clamped = torch.clamp(x, min=0)                      # -1 padding is safe (loss ignores it)
        h = self.byte_embedding(x_clamped)
        for norm, blk in zip(self.norms, self.blocks):
            h = h + blk(norm(h))                               # pre-norm residual
        h = self.ln_f(h)
        return self.predictor(h)                               # (B, T, 256)


def build_backbone(name="transformer", **kwargs):
    """Factory so training/eval can A/B by name. 'transformer' imports the
    existing NetworkBytePatcher; 'mamba' returns MambaBytePatcher.
    NOTE: Mamba ignores nhead; transformer ignores d_state/d_conv/expand."""
    if name in ("mamba", "mamba1", "mamba2"):
        variant = "mamba1" if name == "mamba1" else "mamba2" if name == "mamba2" \
            else kwargs.get("variant", "mamba2")
        return MambaBytePatcher(
            d_model=kwargs.get("d_model", 128),
            num_layers=kwargs.get("num_layers", 2),
            max_sequence_length=kwargs.get("max_sequence_length", 512),
            d_state=kwargs.get("d_state", None),
            d_conv=kwargs.get("d_conv", 4),
            expand=kwargs.get("expand", 2),
            variant=variant,
            headdim=kwargs.get("headdim", 64),
            force_torch_scan=kwargs.get("force_torch_scan", False),
        )
    from src.model import NetworkBytePatcher
    return NetworkBytePatcher(
        d_model=kwargs.get("d_model", 128),
        nhead=kwargs.get("nhead", 4),
        num_layers=kwargs.get("num_layers", 2),
        max_sequence_length=kwargs.get("max_sequence_length", 512),
    )
