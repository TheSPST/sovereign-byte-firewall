#!/usr/bin/env python3
"""
train_ab.py  (branch: experiment/mamba-backbone)
================================================
Self-contained A/B trainer: trains EITHER backbone (transformer or mamba) on a
benign pcap with the SAME loop / loss / optimizer, so the accuracy comparison is
fair. Saves an eval-compatible checkpoint (records backbone + config) that
`evaluate_zero_day.py --backbone ...` can load directly.

Deliberately minimal (not the SLURM/HF orchestration) to keep the experiment
easy to reason about and reproduce. Same next-byte cross-entropy objective and
masking as the real training, via get_pcap_dataloader.

FAIR A/B on Kaggle GPU (after `pip install mamba-ssm causal-conv1d --no-build-isolation`):
    # train both to the SAME number of steps on the SAME benign corpus
    python train_ab.py --backbone transformer --train_pcap <Monday.pcap> --steps 20000 --out ckpt_tfm.pt
    python train_ab.py --backbone mamba       --train_pcap <Monday.pcap> --steps 20000 --out ckpt_mmb.pt
    # then evaluate both on the SAME held-out zero-day split
    python evaluate_zero_day.py --backbone transformer --checkpoint_path ckpt_tfm.pt --score_agg topk --topk_frac 0.1 ...
    python evaluate_zero_day.py --backbone mamba       --checkpoint_path ckpt_mmb.pt --score_agg topk --topk_frac 0.1 ...

Compare calibration AUC + held-out detection @ ~1% FPR. Fair-capacity note: at
d_model=128/2L the Mamba is ~2.4x smaller; raise --d_model / --num_layers on the
Mamba side to match params if you want an iso-capacity comparison.

SMOKE TEST FIRST (cheap, proves the wiring end-to-end before you spend hours):
    python train_ab.py --backbone mamba --train_pcap <any.pcap> --steps 20 --seq_len 128 --out /tmp/smoke.pt
    python evaluate_zero_day.py --backbone mamba --checkpoint_path /tmp/smoke.pt \
        --benign_calibration_pcap <a> --benign_holdout_pcap <b> --attack_dir <d> \
        --holdout_attack_pcap <c> --score_agg topk --topk_frac 0.1
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import get_pcap_dataloader
from src.model_mamba import build_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["transformer", "mamba1", "mamba2"], required=True,
                   help="mamba2 = SSD/matmul-optimized (faster on GPU, recommended)")
    p.add_argument("--train_pcap", required=True, help="Benign training corpus (e.g. Monday)")
    p.add_argument("--out", required=True, help="Output checkpoint path")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_backbone(
        args.backbone, d_model=args.d_model, num_layers=args.num_layers,
        max_sequence_length=args.seq_len, nhead=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    backend = getattr(model, "backend", "transformer")
    print(f"Backbone: {args.backbone} ({backend}) | params {n_params:,} | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    def save(step):
        torch.save({
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "max_sequence_length": args.seq_len,
            "step": step,
        }, args.out)
        print(f"  saved {args.out} @ step {step}")

    step, t0, running = 0, time.time(), 0.0
    while step < args.steps:
        dl = get_pcap_dataloader(pcap_path=args.train_pcap, batch_size=args.batch_size,
                                 num_workers=0, max_sequence_length=args.seq_len,
                                 label_anomalies=False)
        for batch in dl:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                                   ignore_index=-1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item(); step += 1
            if step % args.log_every == 0:
                rate = step / (time.time() - t0)
                print(f"step {step:>7} | loss {running/args.log_every:.4f} | {rate:.1f} steps/s")
                running = 0.0
            if step % args.ckpt_every == 0:
                save(step)
            if step >= args.steps:
                break
    save(step)
    print("done.")


if __name__ == "__main__":
    main()
