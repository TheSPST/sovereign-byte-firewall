#!/usr/bin/env python3
"""
evaluate_sequential.py
======================
Trace-level sequential detection via CUSUM (Page 1954), on top of the
per-window surprise scores from the byte-level model.

WHY (the math):
  The per-window metric answers "is THIS 512-byte window anomalous?" — a
  single weak test repeated thousands of times, which is exactly the regime
  where the base-rate fallacy (Axelsson 2000) kills you: at any threshold
  giving useful per-window recall, benign volume swamps you with false
  positives; at <1% FPR the recall collapses (our best: 32%).

  But a real attack is not one window — it is a sustained shift in the score
  DISTRIBUTION across consecutive windows. Detecting a persistent mean shift
  in a stream is a solved problem: Page's CUSUM statistic

      C_0 = 0,   C_t = max(0, C_{t-1} + (s_t - mu_b) / sigma_b - k)

  accumulates standardized excess surprise above a slack k (in benign-sigma
  units) and alarms when C_t > h. It is the SPRT-optimal (minimal expected
  detection delay for a given false-alarm rate, Lorden 1971 / Moustakides
  1986) test for a mean shift of 2k sigmas. Windows that are individually
  sub-threshold — say each only 0.5 sigma above benign mean — become
  detectable in aggregate within ~h/(0.5-k)/1 windows, while benign noise is
  mean-reverting and keeps C_t pinned near 0.

  Operationally this reframes the goal: per-ATTACK detection rate at a
  budgeted false-alarm rate per benign traffic volume, which is both the
  operational quantity SOCs care about and a strictly easier statistical
  target than per-window classification.

IMPORTANT: window ORDER matters here, so scoring runs with the shuffle
buffer disabled (shuffle_buffer_windows=1), unlike evaluate_zero_day.py
where order is irrelevant.

Usage (defaults match the project pcap layout):
  python evaluate_sequential.py --checkpoint_path checkpoints/latest_patcher.pt \
      --score_agg topk --topk_frac 0.10 [--complexity_correction zlib]
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch

# Reuse the model loader and per-window scorer from the main harness so the
# two evaluations can never silently diverge in scoring methodology.
from evaluate_zero_day import load_model, _window_complexity_bits_per_byte  # noqa: F401
import evaluate_zero_day as ezd
from src.dataloader import get_pcap_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="CUSUM trace-level sequential evaluation")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/latest_patcher.pt")
    p.add_argument("--benign_calibration_pcap", type=str, default="scratch/archive_upload/normal.pcap")
    p.add_argument("--benign_holdout_pcap", type=str, default="scratch/archive_upload/normal2.pcap")
    p.add_argument("--attack_dir", type=str, default="scratch/archive_upload")
    p.add_argument("--holdout_attack_pcap", type=str, default="scratch/archive_upload/0day.pcap")
    p.add_argument("--max_sequence_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--score_agg", type=str, default="topk", choices=["mean", "max", "topk"])
    p.add_argument("--topk_frac", type=float, default=0.10)
    p.add_argument("--complexity_correction", type=str, default="none", choices=["none", "zlib", "bz2"])
    p.add_argument("--cusum_k", type=float, default=0.5,
                   help="CUSUM slack in benign-sigma units; the test is optimal for "
                        "detecting a mean shift of 2k sigmas (default 0.5 => 1-sigma shifts)")
    p.add_argument("--target_alarms_per_10k", type=float, default=0.3,
                   help="False-alarm budget as alarms per 10k benign windows (default 0.3, "
                        "i.e. ~1 alarm per 33k windows). h is the smallest threshold whose "
                        "alarm count on the CALIBRATION benign trace stays within this rate. "
                        "The held-out benign trace is NEVER used for h selection — its alarm "
                        "count is reported as the genuine held-out false-alarm measurement.")
    p.add_argument("--output_dir", type=str, default="results/sequential_eval")
    return p.parse_args()


def ordered_scores(model, pcap_path, device, args, max_seq_len):
    """Per-window scores in FILE ORDER (shuffle buffer disabled)."""
    if not os.path.exists(pcap_path):
        print(f"  WARNING: '{pcap_path}' not found, skipping.")
        return []
    dataloader = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=args.batch_size,
        num_workers=0,
        max_sequence_length=max_seq_len,
        shuffle_buffer_windows=1,   # ORDER MATTERS for sequential statistics
        label_anomalies=False,
    )
    # Delegate the per-batch math to the shared scorer by monkey-free reuse:
    # ezd.score_pcap builds its own dataloader (shuffled), so inline the loop here.
    import math as _math
    import torch.nn.functional as F
    scores = []
    with torch.no_grad():
        for batch in dataloader:
            comp_bits = None
            if args.complexity_correction != "none":
                comp_bits = _window_complexity_bits_per_byte(batch[:, 1:].numpy(), args.complexity_correction)
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            valid_mask = targets != -1
            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
            gather_idx = torch.clamp(targets, min=0).unsqueeze(-1)
            token_logprob = log_probs.gather(-1, gather_idx).squeeze(-1)
            surprise_bits = -token_logprob / _math.log(2)
            valid_counts = valid_mask.sum(dim=1)

            if args.score_agg == "mean":
                sm = surprise_bits.masked_fill(~valid_mask, float("nan"))
                per_window = torch.nanmean(sm, dim=1)
            elif args.score_agg == "max":
                filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
                per_window = filled.max(dim=1).values
            else:  # topk
                filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
                k = max(1, min(filled.shape[1], int(round(args.topk_frac * filled.shape[1]))))
                topk_vals, _ = torch.topk(filled, k=k, dim=1)
                tv = torch.isfinite(topk_vals)
                topk_vals = torch.where(tv, topk_vals, torch.zeros_like(topk_vals))
                per_window = topk_vals.sum(dim=1) / tv.sum(dim=1).clamp(min=1)
            per_window = torch.where(valid_counts > 0, per_window,
                                     torch.full_like(per_window, float("nan")))
            pw = per_window.cpu().numpy()
            if comp_bits is not None:
                pw = pw - comp_bits
            scores.extend([s for s in pw.tolist() if not np.isnan(s)])
    return scores


def cusum_path(scores, mu, sigma, k):
    """Standardized one-sided CUSUM statistic over an ordered score stream."""
    c, path = 0.0, np.empty(len(scores))
    inv_sigma = 1.0 / max(sigma, 1e-9)
    for i, s in enumerate(scores):
        c = max(0.0, c + (s - mu) * inv_sigma - k)
        path[i] = c
    return path


def simulate_alarms(scores, mu, sigma, k, h):
    """CUSUM with reset after each alarm: (num_alarms, first_alarm_index)."""
    c, count, first = 0.0, 0, None
    inv_sigma = 1.0 / max(sigma, 1e-9)
    for i, s in enumerate(scores):
        c = max(0.0, c + (s - mu) * inv_sigma - k)
        if c > h:
            count += 1
            if first is None:
                first = i
            c = 0.0  # reset and keep monitoring
    return count, first


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("==================================================")
    print("   SEQUENTIAL (CUSUM) TRACE-LEVEL EVALUATION")
    print("==================================================")
    print(f"Device: {device} | agg={args.score_agg} | k={args.cusum_k} | "
          f"complexity_correction={args.complexity_correction}")

    model, max_seq_len = load_model(args.checkpoint_path, device, args.max_sequence_length)

    # --- Benign reference distribution (calibration file, ordered) ---
    print(f"\nScoring benign calibration (ordered): {args.benign_calibration_pcap}")
    benign_cal = ordered_scores(model, args.benign_calibration_pcap, device, args, max_seq_len)
    if len(benign_cal) < 100:
        print("ERROR: not enough benign calibration windows.", file=sys.stderr)
        sys.exit(1)
    mu_b, sigma_b = float(np.mean(benign_cal)), float(np.std(benign_cal))
    print(f"  mu_b={mu_b:.3f} bits, sigma_b={sigma_b:.3f} bits over {len(benign_cal)} windows")

    # --- Choose h on the CALIBRATION benign trace (no held-out leakage) ---
    # FIX: an earlier version selected h on the held-out benign trace, which
    # made the reported held-out false-alarm count partly in-sample. h is now
    # fit entirely on calibration data; the held-out benign alarm count below
    # is a genuine out-of-sample measurement.
    cal_budget = args.target_alarms_per_10k * len(benign_cal) / 10000.0
    h_grid = np.linspace(1.0, 200.0, 400)
    chosen_h = None
    for h in h_grid:
        n_alarms, _ = simulate_alarms(benign_cal, mu_b, sigma_b, args.cusum_k, h)
        if n_alarms <= cal_budget:
            chosen_h = float(h)
            break
    if chosen_h is None:
        chosen_h = float(h_grid[-1])
        print("WARNING: even the largest h in the grid exceeds the benign alarm budget.")
    cal_alarms, _ = simulate_alarms(benign_cal, mu_b, sigma_b, args.cusum_k, chosen_h)
    print(f"  chosen h={chosen_h:.2f} on calibration ({cal_alarms} alarm(s) / "
          f"{len(benign_cal)} windows, budget {cal_budget:.2f})")

    # --- Held-out benign: pure out-of-sample false-alarm measurement ---
    print(f"Scoring held-out benign (ordered, UNTOUCHED by fitting): {args.benign_holdout_pcap}")
    benign_hold = ordered_scores(model, args.benign_holdout_pcap, device, args, max_seq_len)
    benign_alarms, _ = simulate_alarms(benign_hold, mu_b, sigma_b, args.cusum_k, chosen_h)
    print(f"  -> {benign_alarms} held-out benign alarm(s) over {len(benign_hold)} windows "
          f"({10000.0 * benign_alarms / max(1, len(benign_hold)):.2f} per 10k windows)")

    # --- Attack traces ---
    exclude = {os.path.basename(args.benign_calibration_pcap),
               os.path.basename(args.benign_holdout_pcap)}
    attack_files = sorted(glob.glob(os.path.join(args.attack_dir, "*.pcap")))
    attack_files = [f for f in attack_files if os.path.basename(f) not in exclude]

    per_file = {}
    detected = 0
    evaluated = 0
    for f in attack_files:
        name = os.path.basename(f)
        s = ordered_scores(model, f, device, args, max_seq_len)
        if not s:
            continue
        evaluated += 1
        n_alarms, first = simulate_alarms(s, mu_b, sigma_b, args.cusum_k, chosen_h)
        hit = n_alarms > 0
        detected += int(hit)
        per_file[name] = {
            "windows": len(s),
            "alarms": n_alarms,
            "first_alarm_window": first,
            "detected": hit,
        }
        holdout_tag = " [HELD-OUT]" if name == os.path.basename(args.holdout_attack_pcap) else ""
        print(f"  {name}{holdout_tag}: {'DETECTED' if hit else 'missed'} "
              f"({n_alarms} alarms, first at window {first}, {len(s)} windows)")

    print("\n================ TRACE-LEVEL RESULTS ================")
    print(f"Attack traces detected: {detected}/{evaluated} "
          f"({detected / max(1, evaluated):.1%}) at {benign_alarms} benign alarm(s) "
          f"per {len(benign_hold)} held-out benign windows")
    print("======================================================")

    metrics = {
        "checkpoint_path": args.checkpoint_path,
        "score_agg": args.score_agg,
        "topk_frac": args.topk_frac,
        "complexity_correction": args.complexity_correction,
        "cusum": {"k": args.cusum_k, "h": chosen_h, "mu_benign": mu_b, "sigma_benign": sigma_b,
                  "h_fit_on": "calibration", "calibration_alarms": cal_alarms,
                  "target_alarms_per_10k": args.target_alarms_per_10k},
        "benign_holdout": {"windows": len(benign_hold), "alarms": benign_alarms,
                           "alarms_per_10k": 10000.0 * benign_alarms / max(1, len(benign_hold))},
        "trace_detection_rate": detected / max(1, evaluated),
        "per_file": per_file,
    }
    out = os.path.join(args.output_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics: {out}")


if __name__ == "__main__":
    main()
