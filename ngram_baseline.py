#!/usr/bin/env python3
"""
ngram_baseline.py
=================
DECISION EXPERIMENT: does the transformer actually beat a cheap byte n-gram?

The whole engine rests on the claim that a causal transformer's next-byte
surprise is a better anomaly signal than trivial statistics. A byte n-gram /
Markov model ALSO produces a next-byte probability -> surprise score, at a tiny
fraction of the cost. If the n-gram gets within a point or two of the
transformer on the same held-out zero-day protocol, the transformer is not
earning its complexity and we should simplify. If the transformer clearly wins,
the complexity is justified -- and now we can prove it to a technical reviewer.

This runs the SAME protocol as evaluate_zero_day.py -- same pcaps, same masking
(via get_pcap_dataloader), same topk-10% aggregation, same benign-calibration /
held-out-attack split -- but swaps the transformer for an order-K add-alpha
smoothed n-gram with uniform backoff on unseen contexts.

USAGE (mirror your transformer eval, same files):
    python ngram_baseline.py \
        --benign_calibration_pcap <normal.pcap> \
        --benign_holdout_pcap <normal2.pcap> \
        --attack_dir <dir of calibration attack pcaps> \
        --holdout_attack_pcap <held-out attack.pcap> \
        --order 3 --topk_frac 0.1

Then compare the printed AUC / held-out detection @ ~1% FPR to the transformer's
numbers for the SAME split (e.g. UNSW Shellcode: transformer AUC 0.752, 8.5%).

DECISION RULE (write the verdict in the repo after running):
  - n-gram within ~2 pts of transformer detection @ 1% FPR  -> transformer NOT
    justified at current scale; simplify or shrink the model.
  - transformer beats n-gram by a clear margin (>~5 pts)     -> complexity
    justified; keep, and cite this ablation in due diligence.
  - middle ground -> run a larger n-gram order and a Mamba backbone before
    deciding.
"""

import os
import sys
import glob
import math
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pure n-gram model (no torch/scapy dependency -> unit-testable standalone)
# ---------------------------------------------------------------------------
class NGramModel:
    """Order-K byte model. P(b|ctx) = (count(ctx,b)+a) / (total(ctx)+a*256)
    for a seen context; uniform 1/256 when the K-gram context was never seen.
    Contexts are the K bytes preceding the predicted byte."""

    def __init__(self, order=3, alpha=0.1):
        self.order = order
        self.alpha = alpha
        self.ctx_counts = defaultdict(lambda: defaultdict(int))  # ctx bytes -> {byte: n}
        self.ctx_total = defaultdict(int)                        # ctx bytes -> n
        self._log2_256 = math.log2(256)

    def train_on_sequence(self, seq):
        """seq: iterable of ints 0..255 (a masked byte stream, -1 padding removed)."""
        K = self.order
        buf = bytes(seq)
        for i in range(K, len(buf)):
            ctx = buf[i - K:i]
            b = buf[i]
            self.ctx_counts[ctx][b] += 1
            self.ctx_total[ctx] += 1

    def surprise_bits(self, seq):
        """Per-byte surprise (-log2 P) for each predicted position (i>=order).
        Returns a list aligned to seq[order:]."""
        K, a = self.order, self.alpha
        denom_a = a * 256.0
        buf = bytes(seq)
        out = []
        for i in range(K, len(buf)):
            ctx = buf[i - K:i]
            b = buf[i]
            tot = self.ctx_total.get(ctx, 0)
            if tot == 0:
                out.append(self._log2_256)  # unseen context -> uniform
            else:
                cnt = self.ctx_counts[ctx].get(b, 0)
                p = (cnt + a) / (tot + denom_a)
                out.append(-math.log2(p))
        return out

    def window_score(self, window, topk_frac=0.1):
        """topk-10% mean surprise for one window (list of ints, no -1)."""
        sb = self.surprise_bits(window)
        if not sb:
            return None
        sb.sort(reverse=True)
        k = max(1, min(len(sb), int(round(topk_frac * len(sb)))))
        return sum(sb[:k]) / k


# ---------------------------------------------------------------------------
# Metrics (no sklearn dependency)
# ---------------------------------------------------------------------------
def roc_auc(benign, attack):
    """Rank-based AUC (Mann-Whitney). higher score = more anomalous."""
    if not benign or not attack:
        return float("nan")
    combined = sorted(((s, 0) for s in benign), key=lambda x: x[0]) + \
               sorted(((s, 1) for s in attack), key=lambda x: x[0])
    combined.sort(key=lambda x: x[0])
    # average ranks
    rank_sum_attack = 0.0
    i = 0
    n = len(combined)
    r = 1
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if combined[k][1] == 1:
                rank_sum_attack += avg_rank
        r += (j - i)
        i = j
    n_a, n_b = len(attack), len(benign)
    u = rank_sum_attack - n_a * (n_a + 1) / 2.0
    return u / (n_a * n_b)


def threshold_at_fpr(benign, target_fpr=0.01):
    """Smallest threshold whose benign exceedance rate <= target_fpr."""
    s = sorted(benign)
    idx = int(math.ceil((1 - target_fpr) * len(s))) - 1
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


def detection_rate(scores, thr):
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s > thr) / len(scores)


# ---------------------------------------------------------------------------
# pcap -> windows (reuses the exact eval dataloader + masking)
# ---------------------------------------------------------------------------
def iter_windows(pcap_path, seq_len, max_windows=None):
    """Yield windows (lists of ints, -1 padding stripped) using the same
    masked dataloader as evaluate_zero_day.py, so this is apples-to-apples."""
    from src.dataloader import get_pcap_dataloader
    dl = get_pcap_dataloader(pcap_path=pcap_path, batch_size=64, num_workers=0,
                             max_sequence_length=seq_len, label_anomalies=False)
    count = 0
    for batch in dl:
        for row in batch.tolist():
            w = [b for b in row if b >= 0]
            if len(w) > 1:
                yield w
                count += 1
                if max_windows and count >= max_windows:
                    return


def build_model(pcap_path, order, alpha, seq_len, max_train_windows):
    m = NGramModel(order=order, alpha=alpha)
    n = 0
    for w in iter_windows(pcap_path, seq_len, max_windows=max_train_windows):
        m.train_on_sequence(w)
        n += 1
    print(f"  trained order-{order} n-gram on {n} windows, {len(m.ctx_total)} distinct contexts")
    return m


def score_file(m, pcap_path, topk_frac, seq_len, max_windows):
    out = []
    for w in iter_windows(pcap_path, seq_len, max_windows=max_windows):
        s = m.window_score(w, topk_frac=topk_frac)
        if s is not None:
            out.append(s)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Byte n-gram baseline vs the transformer")
    p.add_argument("--benign_calibration_pcap", required=True)
    p.add_argument("--benign_holdout_pcap", required=True)
    p.add_argument("--attack_dir", required=True)
    p.add_argument("--holdout_attack_pcap", required=True)
    p.add_argument("--train_pcap", default=None,
                   help="Benign corpus to train the n-gram on (default: the calibration pcap). "
                        "Point at the full Monday training file for a same-corpus fair fight vs the transformer.")
    p.add_argument("--order", type=int, default=3, help="n-gram context length (bytes)")
    p.add_argument("--alpha", type=float, default=0.1, help="add-alpha smoothing")
    p.add_argument("--topk_frac", type=float, default=0.1)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--target_fpr", type=float, default=0.01)
    p.add_argument("--max_train_windows", type=int, default=200000)
    p.add_argument("--max_score_windows", type=int, default=200000)
    p.add_argument("--max_pcap_size_mb", type=float, default=None,
                   help="Skip attack pcaps larger than this (match the transformer eval's set)")
    return p.parse_args()


def main():
    args = parse_args()
    train_pcap = args.train_pcap or args.benign_calibration_pcap
    print(f"Building order-{args.order} n-gram from {train_pcap} ...")
    m = build_model(train_pcap, args.order, args.alpha,
                    args.seq_len, args.max_train_windows)

    print("Scoring benign calibration ...")
    benign_cal = score_file(m, args.benign_calibration_pcap, args.topk_frac, args.seq_len, args.max_score_windows)

    # Attack calibration files = every pcap in attack_dir EXCEPT the benign and
    # held-out files (which often live in the same directory).
    exclude = {os.path.abspath(p) for p in
               (args.benign_calibration_pcap, args.benign_holdout_pcap, args.holdout_attack_pcap)}
    attack_files = [f for f in sorted(glob.glob(os.path.join(args.attack_dir, "*.pcap")))
                    if os.path.abspath(f) not in exclude
                    and (args.max_pcap_size_mb is None
                         or os.path.getsize(f) / 1e6 <= args.max_pcap_size_mb)]
    attack_cal = []
    for f in attack_files:
        s = score_file(m, f, args.topk_frac, args.seq_len, args.max_score_windows)
        attack_cal += s
        print(f"  {os.path.basename(f)}: {len(s)} windows")

    auc = roc_auc(benign_cal, attack_cal)
    thr = threshold_at_fpr(benign_cal, args.target_fpr)

    print("Scoring held-out benign + held-out attack ...")
    benign_ho = score_file(m, args.benign_holdout_pcap, args.topk_frac, args.seq_len, args.max_score_windows)
    attack_ho = score_file(m, args.holdout_attack_pcap, args.topk_frac, args.seq_len, args.max_score_windows)

    ho_fpr = detection_rate(benign_ho, thr)
    ho_det = detection_rate(attack_ho, thr)

    print("\n================ N-GRAM BASELINE RESULTS ================")
    print(f"  order={args.order} alpha={args.alpha} topk={args.topk_frac}")
    print(f"  Calibration AUC:            {auc:.4f}")
    print(f"  Threshold @ ~{args.target_fpr:.0%} FPR:     {thr:.3f} bits")
    print(f"  Held-out benign FPR:        {ho_fpr:.4f}")
    print(f"  Held-out 0day detection:    {ho_det:.4f}")
    print("=========================================================")
    print("Compare to the transformer's AUC / held-out detection on the SAME")
    print("split. See the decision rule at the top of this file.")


if __name__ == "__main__":
    main()
