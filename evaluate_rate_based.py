#!/usr/bin/env python3
"""
evaluate_rate_based.py
=======================
Companion rate/flow-based detector for attacks that are structurally
invisible to the byte-level content model (src/model.py + evaluate_zero_day.py).

Why this exists: hydra_ssh / hydra_ftp brute-force traffic is made of
individually well-formed, protocol-valid packets. Each single packet looks
completely normal at the byte level — the only thing that marks it as an
attack is the RATE of connection attempts from one source over time, which
a per-window byte model has no way to see (it only ever looks at one
window's raw bytes, with no cross-window/temporal state). No amount of
tuning the byte-level model's aggregation, threshold, or training will ever
close this gap; it needs an orthogonal signal.

This script buckets a pcap into fixed-width time windows (default 1s) and,
for each window, computes the highest new-connection rate (SYN packets)
seen from any single source IP in that window. That's the "surprise" score
for this detector — analogous in spirit to evaluate_zero_day.py's per-byte
surprise score, but operating on connection-rate statistics instead of raw
content. Same calibration/holdout split and threshold machinery (Youden's J
+ EVT/POT) as evaluate_zero_day.py, so results are directly comparable and
this can eventually be fused (OR-combined) with the byte-level detector's
alerts.

Usage:
  python evaluate_rate_based.py \
    --benign_calibration_pcap scratch/archive_upload/normal.pcap \
    --benign_holdout_pcap scratch/archive_upload/normal2.pcap \
    --attack_dir scratch/archive_upload \
    --holdout_attack_pcap scratch/archive_upload/0day.pcap \
    --output_dir results/rate_based_eval
"""

import os
import sys
import json
import glob
import math
import argparse
from collections import defaultdict, deque

import numpy as np
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import genpareto


def parse_args():
    parser = argparse.ArgumentParser(description="Rate/flow-based companion detector evaluation harness")
    parser.add_argument("--benign_calibration_pcap", type=str, default="scratch/archive_upload/normal.pcap")
    parser.add_argument("--benign_holdout_pcap", type=str, default="scratch/archive_upload/normal2.pcap")
    parser.add_argument("--attack_dir", type=str, default="scratch/archive_upload")
    parser.add_argument("--holdout_attack_pcap", type=str, default="scratch/archive_upload/0day.pcap")
    parser.add_argument("--window_sec", type=float, default=1.0,
                         help="Time-bucket width in seconds; one score is produced per bucket (default: 1.0)")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--evt_tail_quantile", type=float, default=0.98)
    parser.add_argument("--output_dir", type=str, default="results/rate_based_eval")
    return parser.parse_args()


def score_pcap_connection_rate(pcap_path, window_sec):
    """
    Returns a list of per-time-window scores: for each fixed-width time
    bucket, the highest number of new-connection attempts (bare SYN
    packets: SYN=1, ACK=0) seen from any single source IP within that
    bucket. Buckets are keyed by wall-clock packet timestamp (from the
    pcap's own capture metadata), not by byte position, since rate is
    inherently a time-domain property.
    """
    if not os.path.exists(pcap_path):
        print(f"  WARNING: '{pcap_path}' not found, skipping.")
        return []

    # {bucket_index: {src_ip: syn_count}}
    buckets = defaultdict(lambda: defaultdict(int))
    first_ts = None

    with RawPcapReader(pcap_path) as pcap_reader:
        for packet_data, metadata in pcap_reader:
            ts = getattr(metadata, "sec", None)
            if ts is None:
                # Older scapy metadata tuple form: (sec, usec, ...)
                try:
                    ts = metadata[0]
                except Exception:
                    continue
            if first_ts is None:
                first_ts = ts

            try:
                pkt = Ether(packet_data)
                if not pkt.haslayer(TCP) or not pkt.haslayer(IP):
                    continue
                flags = int(pkt[TCP].flags)
                is_bare_syn = bool(flags & 0x02) and not bool(flags & 0x10)
                if not is_bare_syn:
                    continue
                src_ip = pkt[IP].src
            except Exception:
                continue

            bucket_idx = int((ts - first_ts) // window_sec)
            buckets[bucket_idx][src_ip] += 1

    scores = []
    for bucket_idx, src_counts in buckets.items():
        if src_counts:
            scores.append(max(src_counts.values()))
        else:
            scores.append(0)
    return scores


def discover_attack_files(attack_dir, exclude_basenames):
    files = sorted(glob.glob(os.path.join(attack_dir, "*.pcap")))
    return [f for f in files if os.path.basename(f) not in exclude_basenames]


def fit_evt_threshold(benign_scores, target_fpr, tail_quantile=0.98):
    """Same POT/GPD method as evaluate_zero_day.py — see that file for the derivation."""
    arr = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(arr)
    if n < 30:
        return None
    t0 = float(np.quantile(arr, tail_quantile))
    excesses = arr[arr > t0] - t0
    Nt = len(excesses)
    if Nt < 10:
        return None
    shape, _, scale = genpareto.fit(excesses, floc=0)
    q = target_fpr
    if abs(shape) < 1e-6:
        threshold = t0 - scale * math.log(q * n / Nt)
    else:
        threshold = t0 + (scale / shape) * (((n / Nt) * q) ** (-shape) - 1)
    return {"threshold": float(threshold), "tail_quantile": tail_quantile, "t0": t0,
            "gpd_shape": float(shape), "gpd_scale": float(scale),
            "num_tail_points": Nt, "target_fpr": target_fpr}


def pick_thresholds(y_true, y_scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    youden_threshold = float(thresholds[best_idx])
    youden_tpr, youden_fpr = float(tpr[best_idx]), float(fpr[best_idx])
    fpr_idx = int(np.argmin(np.abs(fpr - target_fpr)))
    target_threshold = float(thresholds[fpr_idx])
    target_tpr, target_fpr_actual = float(tpr[fpr_idx]), float(fpr[fpr_idx])
    return {
        "auc": float(auc),
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

    print("==================================================")
    print("   RATE-BASED COMPANION DETECTOR EVALUATION")
    print("==================================================")
    print(f"Window width: {args.window_sec}s | scoring = max SYNs/window from any single source IP\n")

    exclude = {
        os.path.basename(args.benign_calibration_pcap),
        os.path.basename(args.benign_holdout_pcap),
        os.path.basename(args.holdout_attack_pcap),
    }
    attack_files = discover_attack_files(args.attack_dir, exclude)
    print(f"Calibration attack files ({len(attack_files)}): {[os.path.basename(f) for f in attack_files]}")

    print(f"\nScoring benign calibration file: {args.benign_calibration_pcap}")
    benign_scores = score_pcap_connection_rate(args.benign_calibration_pcap, args.window_sec)
    print(f"  -> {len(benign_scores)} time-windows scored")

    per_file_attack_scores = {}
    attack_scores = []
    for f in attack_files:
        print(f"Scoring attack file: {f}")
        s = score_pcap_connection_rate(f, args.window_sec)
        print(f"  -> {len(s)} time-windows scored, max SYN-rate seen: {max(s) if s else 'N/A'}")
        per_file_attack_scores[os.path.basename(f)] = s
        attack_scores.extend(s)

    if not benign_scores or not attack_scores:
        print("ERROR: Insufficient calibration data.", file=sys.stderr)
        sys.exit(1)

    y_true = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
    y_scores = np.array(benign_scores + attack_scores, dtype=float)
    calib_metrics = pick_thresholds(y_true, y_scores, args.target_fpr)
    evt_result = fit_evt_threshold(benign_scores, args.target_fpr, args.evt_tail_quantile)

    print(f"\nCalibration AUC: {calib_metrics['auc']:.4f}")
    print(f"Youden threshold: {calib_metrics['youden']['threshold']:.1f} SYNs/window "
          f"(TPR={calib_metrics['youden']['tpr']:.3f}, FPR={calib_metrics['youden']['fpr']:.3f})")
    print(f"Target-FPR threshold (~{args.target_fpr:.1%}): {calib_metrics['target_fpr']['threshold']:.1f} SYNs/window "
          f"(TPR={calib_metrics['target_fpr']['tpr']:.3f}, FPR={calib_metrics['target_fpr']['fpr']:.3f})")
    if evt_result is not None:
        print(f"EVT/POT threshold: {evt_result['threshold']:.1f} SYNs/window")

    print(f"\nScoring HELD-OUT benign file: {args.benign_holdout_pcap}")
    holdout_benign_scores = score_pcap_connection_rate(args.benign_holdout_pcap, args.window_sec)
    print(f"  -> {len(holdout_benign_scores)} time-windows scored")

    print(f"Scoring HELD-OUT attack file: {args.holdout_attack_pcap}")
    holdout_attack_scores = score_pcap_connection_rate(args.holdout_attack_pcap, args.window_sec)
    print(f"  -> {len(holdout_attack_scores)} time-windows scored")

    threshold_candidates = [("youden", calib_metrics["youden"]["threshold"]),
                            ("target_fpr", calib_metrics["target_fpr"]["threshold"])]
    if evt_result is not None:
        threshold_candidates.append(("evt", evt_result["threshold"]))

    holdout_results = {}
    for label, thresh in threshold_candidates:
        holdout_results[label] = {
            "threshold": thresh,
            "holdout_benign_false_positive_rate": apply_threshold(holdout_benign_scores, thresh),
            "holdout_attack_detection_rate": apply_threshold(holdout_attack_scores, thresh),
        }

    print("\n================ HELD-OUT GENERALIZATION RESULTS ================")
    for label, r in holdout_results.items():
        fpr_str = f"{r['holdout_benign_false_positive_rate']:.3f}" if r["holdout_benign_false_positive_rate"] is not None else "N/A"
        det_str = f"{r['holdout_attack_detection_rate']:.3f}" if r["holdout_attack_detection_rate"] is not None else "N/A"
        print(f"  [{label}] threshold={r['threshold']:.1f} SYNs/window | "
              f"held-out benign FPR={fpr_str} | held-out 0day detection rate={det_str}")
    print("===================================================================\n")

    # Per-attack-file mean score breakdown — the key sanity check: hydra_ssh /
    # hydra_ftp (rate-based attacks the byte-level model structurally can't
    # see) should show a clearly elevated rate score here, validating this
    # detector is catching a genuinely different signal.
    print("Per-attack-file mean max-SYN-rate (benign calibration mean shown for reference):")
    print(f"  benign_calibration: {np.mean(benign_scores):.2f}")
    for name, s in per_file_attack_scores.items():
        if s:
            print(f"  {name}: {np.mean(s):.2f}")

    metrics = {
        "window_sec": args.window_sec,
        "calibration": {
            "benign_file": args.benign_calibration_pcap,
            "attack_files": list(per_file_attack_scores.keys()),
            "num_benign_windows": len(benign_scores),
            "num_attack_windows": len(attack_scores),
            "auc": calib_metrics["auc"],
            "youden": calib_metrics["youden"],
            "target_fpr": calib_metrics["target_fpr"],
            "evt": evt_result,
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
    print(f"\nSaved metrics: {metrics_path}")
    print("Rate-based companion detector evaluation complete.")


if __name__ == "__main__":
    main()
