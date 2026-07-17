#!/usr/bin/env python3
"""
bench_backbones.py  (branch: experiment/mamba-backbone)
=======================================================
The Mamba A/B, part 1: THROUGHPUT + a learning sanity check. No dataset needed.

Throughput is the whole reason to consider Mamba, and it needs no training, so
run this first. It times forward passes for the transformer and the Mamba
backbone at several sequence lengths (so the O(N) vs O(N^2) scaling shows), and
runs a quick learning check on a periodic pattern to prove the Mamba
implementation actually trains end-to-end (loss drops), not just runs forward.

KAGGLE (GPU on):
    !git clone -b experiment/mamba-backbone https://github.com/TheSPST/sovereign-byte-firewall.git
    %cd sovereign-byte-firewall
    !pip -q install mamba-ssm causal-conv1d   # fused CUDA kernel; else slow fallback
    !python bench_backbones.py

Reads: which Mamba backend loaded (mamba_ssm fused vs torch fallback), params,
windows/sec + ms/batch per backbone per seq_len, and the learning-check losses.
"""

import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import NetworkBytePatcher
from src.model_mamba import MambaBytePatcher, _HAS_MAMBA1, _HAS_MAMBA2


def params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def bench_forward(model, device, seq_len, batch, steps):
    model.eval().to(device)
    x = torch.randint(0, 256, (batch, seq_len), device=device)
    for _ in range(3):  # warmup
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return (steps * batch) / dt, (dt / steps) * 1000.0  # windows/sec, ms/batch


def learn_check(model, device, seq_len, steps, lr=1e-3):
    """Periodic byte pattern (byte[i] = i % 37) -> predictable. Loss should drop
    sharply if backprop through the backbone works. Proves the impl trains."""
    model.train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = (torch.arange(seq_len, device=device) % 37).long().unsqueeze(0).repeat(16, 1)
    first = last = None
    for s in range(steps):
        logits = model(base[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), base[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def build(name, d_model, layers, seq_len, d_state=None):
    if name == "transformer":
        return NetworkBytePatcher(d_model=d_model, nhead=4, num_layers=layers,
                                  max_sequence_length=seq_len)
    variant = "mamba2" if name == "mamba2" else "mamba1"
    return MambaBytePatcher(d_model=d_model, num_layers=layers,
                            max_sequence_length=seq_len, variant=variant, d_state=d_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--learn_steps", type=int, default=150)
    ap.add_argument("--d_state", type=int, default=None,
                    help="SSM state dim (Mamba). Larger favors Mamba-2's SSD matmul path — "
                         "raise it (e.g. 128/256) to test where Mamba-2 pulls ahead.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device} | Mamba-1 kernel: {_HAS_MAMBA1} | Mamba-2 (SSD) kernel: {_HAS_MAMBA2}")
    if not (_HAS_MAMBA1 or _HAS_MAMBA2):
        print("  WARNING: install mamba-ssm + causal-conv1d for a fair speed test on GPU.")

    # Backbones to compare: transformer + whichever Mamba kernels are present.
    names = ["transformer"]
    if _HAS_MAMBA1:
        names.append("mamba1")
    if _HAS_MAMBA2:
        names.append("mamba2")
    if len(names) == 1:
        names.append("mamba1")  # torch_scan fallback, so at least one Mamba shows

    max_seq = max(args.seq_lens)
    models = {n: build(n, args.d_model, args.layers, max_seq, d_state=args.d_state) for n in names}
    pstr = " | ".join(f"{n} {params(m):,} ({params(m)/params(models['transformer']):.2f}x)"
                      for n, m in models.items())
    print(f"Params: {pstr}\n")

    print(f"{'seq_len':>8} | {'backbone':>11} | {'windows/sec':>12} | {'ms/batch':>9} | {'vs tfm':>7}")
    print("-" * 62)
    for L in args.seq_lens:
        base = None
        for n in names:
            wps, mspb = bench_forward(models[n], device, L, args.batch, args.steps)
            if n == "transformer":
                base = wps
            sp = f"{wps/base:.2f}x" if base else "-"
            print(f"{L:>8} | {n:>11} | {wps:>12,.0f} | {mspb:>9.2f} | {sp:>7}")
        print("-" * 62)
    print()

    print("Learning sanity check (periodic pattern, loss should drop):")
    for n in names:
        f_, l_ = learn_check(build(n, args.d_model, args.layers, max_seq, d_state=args.d_state),
                             device, 256, args.learn_steps)
        ok = "OK" if l_ < f_ * 0.5 else "?? (loss did not halve)"
        print(f"  {n:>11}: loss {f_:.3f} -> {l_:.3f}   {ok}")

    print("\nRead: if mamba matches accuracy (separate eval) at higher windows/sec,")
    print("especially as seq_len grows, the edge/pre-filter economics favor it.")


if __name__ == "__main__":
    main()
