#!/usr/bin/env python3
"""
fuse_detectors.py
==================
Combines the byte-level content detector (evaluate_zero_day.py) and the
rate-based companion detector (evaluate_rate_based.py) via OR-fusion: a
file/flow is flagged as anomalous if EITHER detector's score crosses ITS
OWN threshold.

Why OR, not AND: the two detectors cover DISJOINT attack categories by
design. The byte-level model catches content anomalies (a malicious
payload embedded in otherwise-normal-looking traffic) but is structurally
blind to rate-based attacks (hydra_ssh/hydra_ftp brute-force — every
individual packet looks valid, only the connection RATE is anomalous).
The rate detector is the mirror image: it catches bursty connection
patterns but has zero signal for a single well-formed malicious payload
sent over a normal-rate connection (confirmed empirically — it scored
0day.pcap as having zero SYN packets at all). Since neither one can
detect what the other is built for, requiring BOTH to fire (AND) would
only ever reduce coverage; OR is the correct fusion for orthogonal
detectors covering different attack classes.

This script reads the already-saved metrics.json files from both
evaluation harnesses and reports what combined detection coverage looks
like across the calibration attack files, using each file's mean score
against its own detector's Youden threshold as a per-file proxy (a
coarser, file-level approximation — real per-window fusion would need
time-aligned window scores from both detectors, which isn't captured in
the saved summary stats, but this is enough to demonstrate the value and
correct shape of the fusion).

Usage:
  python fuse_detectors.py \
    --byte_level_metrics results/zero_day_eval_gs865000_topk10/metrics.json \
    --rate_based_metrics results/rate_based_eval/metrics.json
"""

import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse byte-level + rate-based detector results")
    parser.add_argument("--byte_level_metrics", type=str,
                         default="results/zero_day_eval_gs865000_topk10/metrics.json")
    parser.add_argument("--rate_based_metrics", type=str,
                         default="results/rate_based_eval/metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.byte_level_metrics, "r", encoding="utf-8") as f:
        byte_metrics = json.load(f)
    with open(args.rate_based_metrics, "r", encoding="utf-8") as f:
        rate_metrics = json.load(f)

    byte_thresh = byte_metrics["calibration"]["youden"]["threshold"]
    byte_scores = byte_metrics["calibration"]["per_file_mean_score"]

    rate_thresh = rate_metrics["calibration"]["youden"]["threshold"]
    rate_scores = rate_metrics["calibration"]["per_file_mean_score"]

    print("==================================================")
    print("   OR-FUSION: byte-level + rate-based detectors")
    print("==================================================")
    print(f"Byte-level Youden threshold: {byte_thresh:.3f} bits")
    print(f"Rate-based Youden threshold: {rate_thresh:.1f} SYNs/window\n")

    all_files = sorted((set(byte_scores.keys()) | set(rate_scores.keys())) - {"benign_calibration"})

    print(f"{'file':28s} {'byte_score':>12s} {'byte_hit':>9s} {'rate_score':>12s} {'rate_hit':>9s} {'FUSED':>7s}")
    fused_hits = 0
    total_files = 0
    byte_only_misses_caught_by_rate = 0
    rate_only_misses_caught_by_byte = 0
    for name in all_files:
        b_score = byte_scores.get(name)
        r_score = rate_scores.get(name)
        b_hit = (b_score is not None) and (b_score > byte_thresh)
        r_hit = (r_score is not None) and (r_score > rate_thresh)
        fused = b_hit or r_hit
        total_files += 1
        fused_hits += int(fused)
        if r_hit and not b_hit:
            byte_only_misses_caught_by_rate += 1
        if b_hit and not r_hit:
            rate_only_misses_caught_by_byte += 1

        b_str = f"{b_score:.3f}" if b_score is not None else "N/A"
        r_str = f"{r_score:.1f}" if r_score is not None else "N/A"
        print(f"{name:28s} {b_str:>12s} {'YES' if b_hit else 'no':>9s} "
              f"{r_str:>12s} {'YES' if r_hit else 'no':>9s} {'YES' if fused else 'no':>7s}")

    print(f"\nFused detection: {fused_hits}/{total_files} calibration attack files flagged by at least one detector")
    print(f"Files the byte-level detector alone would have MISSED but the rate detector caught: "
          f"{byte_only_misses_caught_by_rate}")
    print(f"Files the rate detector alone would have MISSED but the byte-level detector caught: "
          f"{rate_only_misses_caught_by_byte}")

    # Combined FPR: report both individually since we don't have time-aligned
    # per-window scores to compute a true joint FPR. Under an independence
    # assumption, combined_FPR ~= 1 - (1-FPR_a)(1-FPR_b); report as an
    # upper-bound estimate, clearly labeled as approximate.
    byte_holdout_fpr = byte_metrics["holdout"]["results"]["youden"]["holdout_benign_false_positive_rate"]
    rate_holdout_fpr = rate_metrics["holdout"]["results"]["youden"]["holdout_benign_false_positive_rate"]
    approx_combined_fpr = 1 - (1 - byte_holdout_fpr) * (1 - rate_holdout_fpr)
    print(f"\nHeld-out benign FPR — byte-level: {byte_holdout_fpr:.4f}, rate-based: {rate_holdout_fpr:.4f}")
    print(f"Approx. combined FPR under an independence assumption (1-(1-FPR_a)(1-FPR_b)): {approx_combined_fpr:.4f}")
    print("(This is an approximation, not a measured joint FPR — a true measurement needs "
          "time-aligned per-window scores from both detectors on the same holdout traffic.)")


if __name__ == "__main__":
    main()
