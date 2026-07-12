#!/usr/bin/env python3
"""
evaluate_zero_day.py
=====================
Proof-of-Work evaluation harness for the Sovereign Byte-Level Anomaly
Detection Engine.

This is the missing piece between "we built an entropy-based anomaly
scorer" and "we proved it detects zero-days": it scores a trained Stage 1
checkpoint against KNOWN attack traffic and a genuinely held-out file
(default: 0day.pcap), then reports ROC-AUC, a calibrated detection
threshold, and the resulting detection rate / false-positive rate.

Scoring methodology (identical to src/sniffer.py's live inference path,
so the numbers reported here are a faithful proxy for live deployment):
  For each byte window, run the causal model and compute the mean
  "surprise" (bits) the model assigns to the ACTUAL observed next byte:
      surprise_t = -log2 P(byte_t+1 | byte_<=t)
  This is next-byte predictive cross-entropy, NOT the distribution's own
  entropy (which is what generate_patch_lengths() uses for patch cuts —
  a different, unconditional quantity). High surprise = the model has
  learned "normal" protocol grammar poorly predicts this byte sequence.

Dataset split (avoids circular validation):
  - benign_calibration_pcap + attack_dir files  -> used ONLY to fit the
    ROC curve and pick a threshold (Youden's J and a target-FPR point).
  - benign_holdout_pcap + holdout_attack_pcap    -> NEVER used for
    threshold selection. Scored once, at the end, to report a genuine
    generalization number (this is where 0day.pcap belongs).

Usage (matches project pcap layout under scratch/archive_upload/):
  python evaluate_zero_day.py \
    --checkpoint_path checkpoints/latest_patcher.pt \
    --benign_calibration_pcap scratch/archive_upload/normal.pcap \
    --benign_holdout_pcap scratch/archive_upload/normal2.pcap \
    --attack_dir scratch/archive_upload \
    --holdout_attack_pcap scratch/archive_upload/0day.pcap \
    --output_dir results/zero_day_eval
"""

import os
import sys
import json
import glob
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

from src.model import NetworkBytePatcher
from src.dataloader import get_pcap_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-day proof-of-work evaluation harness")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/latest_patcher.pt")
    parser.add_argument("--benign_calibration_pcap", type=str, default="scratch/archive_upload/normal.pcap",
                         help="Benign traffic used to fit the threshold (label=0)")
    parser.add_argument("--benign_holdout_pcap", type=str, default="scratch/archive_upload/normal2.pcap",
                         help="Benign traffic NEVER used for threshold fitting (final FPR check)")
    parser.add_argument("--attack_dir", type=str, default="scratch/archive_upload",
                         help="Directory of labeled attack pcaps used to fit the threshold (label=1)")
    parser.add_argument("--holdout_attack_pcap", type=str, default="scratch/archive_upload/0day.pcap",
                         help="Attack file NEVER used for threshold fitting (final recall check)")
    parser.add_argument("--max_sequence_length", type=int, default=None,
                         help="Override sequence length; auto-detected from checkpoint by default")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_fpr", type=float, default=0.01,
                         help="Additionally report the threshold nearest this false-positive rate")
    parser.add_argument("--output_dir", type=str, default="results/zero_day_eval")
    parser.add_argument("--score_agg", type=str, default="mean", choices=["mean", "max", "topk"],
                         help="How to collapse per-byte surprise into one per-window score. "
                              "'mean' is the original methodology (can dilute a small anomalous "
                              "trigger inside a mostly-ordinary window). 'max' uses the single most "
                              "surprising byte in the window. 'topk' averages the top --topk_frac "
                              "fraction of bytes by surprise (a smoothed version of 'max').")
    parser.add_argument("--topk_frac", type=float, default=0.1,
                         help="Fraction of bytes (by surprise, highest first) to average when "
                              "--score_agg=topk (default: 0.1 = top 10%% of the window)")
    return parser.parse_args()


def load_model(checkpoint_path, device, override_seq_len=None):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at '{checkpoint_path}'", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]

    has_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if has_prefix:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    pos_weight = state_dict.get("pos_embedding.weight")
    checkpoint_max_seq_len = pos_weight.shape[0] if pos_weight is not None else 8192

    # IMPORTANT: the model architecture (and its pos_embedding table) must always be
    # built at the checkpoint's native size, or load_state_dict will fail on a shape
    # mismatch. A shorter --max_sequence_length only limits how long a window we feed
    # through the model during *scoring* (see main()) — the model just uses the first
    # N rows of the same embedding table, which is always valid since N <= native size.
    model = NetworkBytePatcher(max_sequence_length=checkpoint_max_seq_len).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_window_len = override_seq_len or checkpoint_max_seq_len
    if eval_window_len > checkpoint_max_seq_len:
        print(f"WARNING: requested --max_sequence_length {eval_window_len} exceeds the "
              f"checkpoint's native {checkpoint_max_seq_len}; capping to native size.")
        eval_window_len = checkpoint_max_seq_len

    print(f"Loaded checkpoint '{checkpoint_path}' | native max_sequence_length={checkpoint_max_seq_len} "
          f"| scoring window length={eval_window_len}")
    return model, eval_window_len


@torch.no_grad()
def score_pcap(model, pcap_path, device, batch_size, max_sequence_length, agg="mean", topk_frac=0.1):
    """
    Returns a list of per-window "surprise" scores (bits) — negative log2
    probability the model assigned to the true next byte in each window,
    collapsed to one score per window via `agg`. NaN-safe against -1
    padding sentinels.

    agg="mean"  : average surprise across the whole window (original method;
                  a small anomalous trigger inside a mostly-ordinary window
                  gets diluted by all the ordinary bytes around it).
    agg="max"   : the single most surprising byte in the window. Sensitive
                  to a lone outlier, but can be noisy (one weird-but-benign
                  byte can trigger it).
    agg="topk"  : average of the top `topk_frac` fraction of bytes by
                  surprise — a smoothed middle ground between mean and max.
    """
    if not os.path.exists(pcap_path):
        print(f"  WARNING: '{pcap_path}' not found, skipping.")
        return []

    dataloader = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=batch_size,
        num_workers=0,
        max_sequence_length=max_sequence_length,
    )

    scores = []
    for batch in dataloader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        valid_mask = targets != -1

        logits = model(inputs)  # model handles -1 clamping internally
        log_probs = F.log_softmax(logits, dim=-1)

        gather_idx = torch.clamp(targets, min=0).unsqueeze(-1)
        token_logprob = log_probs.gather(-1, gather_idx).squeeze(-1)
        surprise_bits = -token_logprob / math.log(2)

        valid_counts = valid_mask.sum(dim=1)

        if agg == "mean":
            surprise_masked = surprise_bits.masked_fill(~valid_mask, float("nan"))
            per_window = torch.nanmean(surprise_masked, dim=1)
        elif agg == "max":
            filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
            per_window = filled.max(dim=1).values
            per_window = torch.where(valid_counts > 0, per_window, torch.full_like(per_window, float("nan")))
        elif agg == "topk":
            filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
            k = max(1, min(filled.shape[1], int(round(topk_frac * filled.shape[1]))))
            topk_vals, _ = torch.topk(filled, k=k, dim=1)
            topk_valid = torch.isfinite(topk_vals)
            topk_vals = torch.where(topk_valid, topk_vals, torch.zeros_like(topk_vals))
            denom = topk_valid.sum(dim=1).clamp(min=1)
            per_window = topk_vals.sum(dim=1) / denom
            per_window = torch.where(valid_counts > 0, per_window, torch.full_like(per_window, float("nan")))
        else:
            raise ValueError(f"Unknown --score_agg: {agg}")

        per_window = per_window.cpu().numpy()
        scores.extend([s for s in per_window.tolist() if not math.isnan(s)])

    return scores


def discover_attack_files(attack_dir, exclude_basenames):
    files = sorted(glob.glob(os.path.join(attack_dir, "*.pcap")))
    return [f for f in files if os.path.basename(f) not in exclude_basenames]


def pick_thresholds(y_true, y_scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    # Youden's J statistic: maximize (tpr - fpr)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    youden_threshold = float(thresholds[best_idx])
    youden_tpr, youden_fpr = float(tpr[best_idx]), float(fpr[best_idx])

    # Nearest point to the requested target FPR
    fpr_idx = int(np.argmin(np.abs(fpr - target_fpr)))
    target_threshold = float(thresholds[fpr_idx])
    target_tpr, target_fpr_actual = float(tpr[fpr_idx]), float(fpr[fpr_idx])

    return {
        "auc": float(auc),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
        "youden": {"threshold": youden_threshold, "tpr": youden_tpr, "fpr": youden_fpr},
        "target_fpr": {"requested": target_fpr, "threshold": target_threshold,
                        "tpr": target_tpr, "fpr": target_fpr_actual},
    }


def apply_threshold(scores, threshold):
    if not scores:
        return None
    arr = np.array(scores)
    return float(np.mean(arr > threshold))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("==================================================")
    print("   ZERO-DAY PROOF-OF-WORK EVALUATION HARNESS")
    print("==================================================")
    print(f"Device: {device}")
    agg_label = f"{args.score_agg}" + (f" (top {args.topk_frac:.0%})" if args.score_agg == "topk" else "")
    print(f"Score aggregation: {agg_label}")

    model, max_seq_len = load_model(args.checkpoint_path, device, args.max_sequence_length)

    exclude = {
        os.path.basename(args.benign_calibration_pcap),
        os.path.basename(args.benign_holdout_pcap),
        os.path.basename(args.holdout_attack_pcap),
    }
    attack_files = discover_attack_files(args.attack_dir, exclude)
    print(f"\nCalibration attack files ({len(attack_files)}): "
          f"{[os.path.basename(f) for f in attack_files]}")

    # --- Calibration set ---
    print(f"\nScoring benign calibration file: {args.benign_calibration_pcap}")
    benign_scores = score_pcap(model, args.benign_calibration_pcap, device, args.batch_size, max_seq_len,
                                agg=args.score_agg, topk_frac=args.topk_frac)
    print(f"  -> {len(benign_scores)} windows scored")

    per_file_attack_scores = {}
    attack_scores = []
    for f in attack_files:
        print(f"Scoring attack file: {f}")
        s = score_pcap(model, f, device, args.batch_size, max_seq_len,
                        agg=args.score_agg, topk_frac=args.topk_frac)
        print(f"  -> {len(s)} windows scored")
        per_file_attack_scores[os.path.basename(f)] = s
        attack_scores.extend(s)

    if not benign_scores or not attack_scores:
        print("ERROR: Insufficient calibration data (need both benign and attack windows).", file=sys.stderr)
        sys.exit(1)

    y_true = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
    y_scores = np.array(benign_scores + attack_scores)
    calib_metrics = pick_thresholds(y_true, y_scores, args.target_fpr)

    print(f"\nCalibration AUC: {calib_metrics['auc']:.4f}")
    print(f"Youden threshold: {calib_metrics['youden']['threshold']:.3f} bits "
          f"(TPR={calib_metrics['youden']['tpr']:.3f}, FPR={calib_metrics['youden']['fpr']:.3f})")
    print(f"Target-FPR threshold (~{args.target_fpr:.1%}): "
          f"{calib_metrics['target_fpr']['threshold']:.3f} bits "
          f"(TPR={calib_metrics['target_fpr']['tpr']:.3f}, FPR={calib_metrics['target_fpr']['fpr']:.3f})")

    # --- True held-out generalization check ---
    print(f"\nScoring HELD-OUT benign file (never used for calibration): {args.benign_holdout_pcap}")
    holdout_benign_scores = score_pcap(model, args.benign_holdout_pcap, device, args.batch_size, max_seq_len,
                                        agg=args.score_agg, topk_frac=args.topk_frac)
    print(f"  -> {len(holdout_benign_scores)} windows scored")

    print(f"Scoring HELD-OUT attack file (never used for calibration): {args.holdout_attack_pcap}")
    holdout_attack_scores = score_pcap(model, args.holdout_attack_pcap, device, args.batch_size, max_seq_len,
                                        agg=args.score_agg, topk_frac=args.topk_frac)
    print(f"  -> {len(holdout_attack_scores)} windows scored")

    holdout_results = {}
    for label, thresh in [("youden", calib_metrics["youden"]["threshold"]),
                           ("target_fpr", calib_metrics["target_fpr"]["threshold"])]:
        holdout_results[label] = {
            "threshold": thresh,
            "holdout_benign_false_positive_rate": apply_threshold(holdout_benign_scores, thresh),
            "holdout_attack_detection_rate": apply_threshold(holdout_attack_scores, thresh),
        }

    print("\n================ HELD-OUT GENERALIZATION RESULTS ================")
    for label, r in holdout_results.items():
        fpr_str = f"{r['holdout_benign_false_positive_rate']:.3f}" if r["holdout_benign_false_positive_rate"] is not None else "N/A"
        det_str = f"{r['holdout_attack_detection_rate']:.3f}" if r["holdout_attack_detection_rate"] is not None else "N/A"
        print(f"  [{label}] threshold={r['threshold']:.3f} bits | "
              f"held-out benign FPR={fpr_str} | held-out 0day detection rate={det_str}")
    print("===================================================================\n")

    # --- Save artifacts ---
    metrics = {
        "checkpoint_path": args.checkpoint_path,
        "max_sequence_length": max_seq_len,
        "score_agg": args.score_agg,
        "topk_frac": args.topk_frac if args.score_agg == "topk" else None,
        "calibration": {
            "benign_file": args.benign_calibration_pcap,
            "attack_files": list(per_file_attack_scores.keys()),
            "num_benign_windows": len(benign_scores),
            "num_attack_windows": len(attack_scores),
            "auc": calib_metrics["auc"],
            "youden": calib_metrics["youden"],
            "target_fpr": calib_metrics["target_fpr"],
            "per_file_mean_score": {
                "benign_calibration": float(np.mean(benign_scores)),
                **{k: float(np.mean(v)) for k, v in per_file_attack_scores.items() if v},
            },
        },
        "holdout": {
            "benign_file": args.benign_holdout_pcap,
            "attack_file": args.holdout_attack_pcap,
            "num_benign_windows": len(holdout_benign_scores),
            "num_attack_windows": len(holdout_attack_scores),
            "results": holdout_results,
        },
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # ROC curve plot
    plt.figure(figsize=(7, 6))
    plt.plot(calib_metrics["roc_curve"]["fpr"], calib_metrics["roc_curve"]["tpr"],
              label=f"ROC (AUC={calib_metrics['auc']:.3f})", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    plt.scatter([calib_metrics["youden"]["fpr"]], [calib_metrics["youden"]["tpr"]],
                color="#d62728", zorder=5, label="Youden's J threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Zero-Day Proof-of-Work: Calibration ROC Curve (agg={agg_label})")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.5)
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {roc_path}")

    # Score distribution plot (calibration + holdout overlay)
    plt.figure(figsize=(10, 6))
    plt.hist(benign_scores, bins=50, alpha=0.5, label="Benign (calibration)", color="#2ca02c", density=True)
    plt.hist(attack_scores, bins=50, alpha=0.5, label="Attacks (calibration)", color="#d62728", density=True)
    if holdout_benign_scores:
        plt.hist(holdout_benign_scores, bins=50, alpha=0.5, label="Benign (held-out)", color="#98df8a",
                  density=True, histtype="step", linewidth=2)
    if holdout_attack_scores:
        plt.hist(holdout_attack_scores, bins=50, alpha=0.5, label="0day (held-out)", color="#ff9896",
                  density=True, histtype="step", linewidth=2)
    plt.axvline(calib_metrics["youden"]["threshold"], color="black", linestyle="--",
                label=f"Youden threshold ({calib_metrics['youden']['threshold']:.2f} bits)")
    plt.xlabel(f"Next-Byte Surprise (bits), agg={agg_label}")
    plt.ylabel("Density")
    plt.title(f"Zero-Day Proof-of-Work: Score Distributions (agg={agg_label})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    dist_path = os.path.join(args.output_dir, "score_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {dist_path}")

    print("\nProof-of-work evaluation complete.")


if __name__ == "__main__":
    main()
