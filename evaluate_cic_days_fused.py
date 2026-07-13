#!/usr/bin/env python3
"""
evaluate_cic_days_fused.py
==========================
Same-environment CIC-IDS2017 day evaluation with BYTE + RATE detector fusion.

Motivation (measured on Wednesday, gs75000, 2026-07-13): the byte-level model
detects payload exploits (Heartbleed) and volumetric floods (Hulk, GoldenEye)
robustly, but the slow low-rate connection attacks (slowloris, Slowhttptest)
produce few byte alarms and drop out under a strict false-alarm budget. Those
are exactly the attacks a RATE/connection detector sees clearly (it counts
new-connection SYNs per source). This harness runs BOTH detectors over one day
and OR-combines their alarms, so an attack counts as detected if EITHER fires —
closing the slow-attack gap the byte model structurally can't.

Both detectors:
  * are calibrated ONLY on the leading benign period of the SAME day
    (capture start -> just before the first scheduled attack),
  * emit alarms as capture timestamps,
  * are OR-fused per attack interval, with a combined false-alarm rate.

Byte side  : per-window next-byte surprise -> CUSUM (Page 1954) mean-shift test.
Rate side  : per-1s bucket, max bare-SYN count from any single source IP ->
             threshold on the benign lead-in.

Usage (GPU/Kaggle; Wednesday from the attached CYBERA dataset):
  python evaluate_cic_days_fused.py \
    --checkpoint_path ckpt_best/checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt \
    --pcap /kaggle/input/.../Wednesday-workingHours.pcap \
    --day wednesday --max_sequence_length 512 \
    --window_subsample 20 \
    --target_alarms_per_10k 0.03 \
    --output_dir ./results/wed_gs75000_fused
"""
import os
import sys
import json
import argparse
import datetime

import numpy as np
import torch
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP

# Reuse the pcapng-correct timestamp decode, the byte-window scorer, the model
# loader and the built-in attack schedules from the byte-only harness.
from evaluate_cic_days import (_packet_epoch, stream_window_scores,
                               CIC_SCHEDULES, load_schedule)
from evaluate_zero_day import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Byte + rate fusion, same-environment CIC day eval")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--pcap", type=str, required=True)
    p.add_argument("--day", type=str, choices=list(CIC_SCHEDULES.keys()), default=None)
    p.add_argument("--schedule_json", type=str, default=None)
    p.add_argument("--utc_offset_hours", type=float, default=-3.0)
    p.add_argument("--max_sequence_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--score_agg", type=str, default="topk", choices=["mean", "max", "topk"])
    p.add_argument("--topk_frac", type=float, default=0.10)
    p.add_argument("--window_subsample", type=int, default=1,
                   help="Byte side: score 1 of every N windows (fast on dense captures).")
    p.add_argument("--stop_after_minutes", type=float, default=None,
                   help="Stop both passes this many minutes after the first packet.")
    # Byte CUSUM
    p.add_argument("--cusum_k", type=float, default=0.5)
    p.add_argument("--target_alarms_per_10k", type=float, default=0.3,
                   help="Per-detector benign false-alarm budget on the fit period "
                        "(alarms per 10k benign samples). Applied to BOTH detectors.")
    # Rate detector
    p.add_argument("--rate_window_sec", type=float, default=1.0,
                   help="Rate side: time-bucket width; score = max bare-SYNs/bucket/source-IP.")
    p.add_argument("--output_dir", type=str, default="results/cic_day_fused")
    return p.parse_args()


def _local(ts, off):
    return datetime.datetime.utcfromtimestamp(ts + off * 3600)


# ---------------------------------------------------------------------------
# Rate detector: one cheap pass counting bare-SYNs per time bucket per source.
# ---------------------------------------------------------------------------
def rate_scores_over_time(pcap_path, window_sec, stop_epoch=None):
    import gzip
    import dpkt
    from collections import defaultdict
    
    fobj = gzip.open(pcap_path, "rb") if pcap_path.endswith(".gz") else open(pcap_path, "rb")
    buckets = {}
    first_ts = None
    
    try:
        reader = dpkt.pcap.Reader(fobj)
    except ValueError:
        fobj.seek(0)
        reader = dpkt.pcapng.Reader(fobj)
        
    for ts, packet_data in reader:
        if first_ts is None:
            first_ts = ts
        if stop_epoch is not None and ts > stop_epoch:
            break
            
        try:
            eth = dpkt.ethernet.Ethernet(packet_data)
            if not isinstance(eth.data, dpkt.ip.IP): continue
            ip = eth.data
            if not isinstance(ip.data, dpkt.tcp.TCP): continue
            tcp = ip.data
            
            if not (bool(tcp.flags & dpkt.tcp.TH_SYN) and not bool(tcp.flags & dpkt.tcp.TH_ACK)):
                continue
            src = ip.src
        except Exception:
            continue
            
        b = int((ts - first_ts) // window_sec)
        buckets.setdefault(b, defaultdict(int))[src] += 1
        
    fobj.close()
    
    if not buckets:
        return np.array([]), np.array([])
        
    idxs = sorted(buckets)
    epochs = np.array([first_ts + b * window_sec for b in idxs], dtype=float)
    scores = np.array([max(buckets[b].values()) for b in idxs], dtype=float)
    return epochs, scores


# ---------------------------------------------------------------------------
# CUSUM helper (byte side): standardized one-sided mean-shift, reset-on-alarm.
# ---------------------------------------------------------------------------
def cusum_alarm_epochs(scores, tss, mu, sigma, k, h):
    c = 0.0
    inv = 1.0 / max(sigma, 1e-9)
    out = []
    for s, t in zip(scores, tss):
        c = max(0.0, c + (s - mu) * inv - k)
        if c > h:
            out.append(t)
            c = 0.0
    return np.array(out, dtype=float)


def cusum_count_on(scores, mu, sigma, k, h):
    c, n = 0.0, 0
    inv = 1.0 / max(sigma, 1e-9)
    for s in scores:
        c = max(0.0, c + (s - mu) * inv - k)
        if c > h:
            n += 1
            c = 0.0
    return n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    off = args.utc_offset_hours
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    schedule = load_schedule(args)
    print("==================================================")
    print("  CIC-IDS2017 FUSED (BYTE + RATE) DAY EVALUATION")
    print("==================================================")
    print(f"Device: {device} | pcap: {args.pcap}")
    print(f"Attacks: {[a['name'] for a in schedule]}")

    model, seq_len = load_model(args.checkpoint_path, device, args.max_sequence_length)

    # ---- BYTE pass: per-window surprise + timestamps ----
    print("\n[byte] streaming + scoring windows ...")
    b_scores, b_tss = [], []
    t_start = None
    stop_epoch = None
    for s, t in stream_window_scores(model, device, args.pcap, seq_len,
                                     args.batch_size, args.score_agg, args.topk_frac,
                                     subsample=args.window_subsample):
        if t_start is None:
            t_start = t
            print(f"[byte] first packet: {_local(t, off):%Y-%m-%d %H:%M:%S} local")
            if args.stop_after_minutes is not None:
                stop_epoch = t_start + args.stop_after_minutes * 60
        if stop_epoch is not None and t > stop_epoch:
            break
        b_scores.append(s)
        b_tss.append(t)
    b_scores = np.asarray(b_scores)
    b_tss = np.asarray(b_tss)
    print(f"[byte] {len(b_scores)} windows "
          f"({_local(b_tss[0], off):%H:%M} -> {_local(b_tss[-1], off):%H:%M} local)")

    # ---- RATE pass: per-bucket max SYN rate + timestamps ----
    print("[rate] streaming + counting SYNs ...")
    r_tss, r_scores = rate_scores_over_time(args.pcap, args.rate_window_sec, stop_epoch)
    print(f"[rate] {len(r_scores)} time-buckets")

    # ---- Attack intervals + shared benign fit window ----
    t0 = float(b_tss[0])
    lt0 = _local(t0, off)
    midnight = t0 - (lt0.hour * 3600 + lt0.minute * 60 + lt0.second)

    def hhmm_epoch(hhmm):
        h, m = map(int, hhmm.split(":"))
        return midnight + h * 3600 + m * 60

    MARGIN = 120.0  # seconds of slack around each attack interval
    intervals = [(a["name"], hhmm_epoch(a["start"]) - MARGIN, hhmm_epoch(a["end"]) + MARGIN)
                 for a in schedule]
    first_attack = min(s for _, s, _ in intervals)
    fit_end = first_attack - 5 * 60      # benign lead-in ends 5 min before first attack
    span_end = float(b_tss[-1])
    # only evaluate attacks that actually fall within the captured span
    eval_intervals = [(n, s, e) for (n, s, e) in intervals if s <= span_end]
    outside = [n for (n, s, e) in intervals if s > span_end]

    # ---- Fit BYTE CUSUM on benign lead-in ----
    b_fit = b_scores[b_tss < fit_end]
    mu_b, sig_b = float(b_fit.mean()), float(b_fit.std())
    b_budget = args.target_alarms_per_10k * len(b_fit) / 10000.0
    b_h = None
    for h in np.geomspace(1.0, 5000.0, 500):
        if cusum_count_on(b_fit, mu_b, sig_b, args.cusum_k, h) <= b_budget:
            b_h = float(h); break
    b_h = b_h or 5000.0
    byte_alarms = cusum_alarm_epochs(b_scores, b_tss, mu_b, sig_b, args.cusum_k, b_h)
    print(f"[byte] fit {len(b_fit)} windows mu={mu_b:.2f} sigma={sig_b:.2f} h={b_h:.1f} "
          f"(budget {b_budget:.2f}); {len(byte_alarms)} total alarms")

    # ---- Fit RATE threshold on benign lead-in ----
    rate_alarms = np.array([])
    r_thresh = None
    if len(r_scores):
        r_fit = r_scores[r_tss < fit_end]
        r_budget = args.target_alarms_per_10k * len(r_fit) / 10000.0
        # smallest integer threshold whose benign-lead-in exceedances <= budget
        for cand in range(1, int(r_scores.max()) + 2):
            if int((r_fit > cand).sum()) <= r_budget:
                r_thresh = float(cand); break
        if r_thresh is None:
            r_thresh = float(int(r_scores.max()) + 1)
        rate_alarms = r_tss[r_scores > r_thresh]
        print(f"[rate] fit {len(r_fit)} buckets threshold={r_thresh:.0f} SYNs/bucket "
              f"(budget {r_budget:.2f}); {len(rate_alarms)} total alarms")

    # ---- Per-attack fusion ----
    def any_in(alarms, s, e):
        return bool(len(alarms)) and bool(((alarms >= s) & (alarms <= e)).any())

    per_attack = {}
    detected = 0
    print("\nPer-attack (byte | rate | fused):")
    for name, s, e in eval_intervals:
        by = any_in(byte_alarms, s, e)
        rt = any_in(rate_alarms, s, e)
        fused = by or rt
        detected += int(fused)
        who = ",".join([d for d, f in [("byte", by), ("rate", rt)] if f]) or "none"
        per_attack[name] = {"byte": by, "rate": rt, "detected": fused, "by": who}
        print(f"  {name}: {'DETECTED' if fused else 'missed'}  [{who}]")
    for name in outside:
        per_attack[name] = {"detected": None, "outside_captured_range": True}
        print(f"  {name}: outside captured range")

    # ---- Combined false alarms (union of byte+rate alarms outside all intervals) ----
    def outside_all(alarms):
        if not len(alarms):
            return np.array([])
        keep = np.ones(len(alarms), dtype=bool)
        for _, s, e in eval_intervals:
            keep &= ~((alarms >= s) & (alarms <= e))
        # also drop anything in the fit period (not part of the eval benign span)
        keep &= (alarms >= fit_end)
        return alarms[keep]

    fa_byte = outside_all(byte_alarms)
    fa_rate = outside_all(rate_alarms)
    fa_union = np.unique(np.concatenate([fa_byte, fa_rate])) if (len(fa_byte) or len(fa_rate)) else np.array([])
    attack_secs = sum(max(0.0, min(e, span_end) - max(s, fit_end)) for _, s, e in eval_intervals)
    benign_hours = max(1e-9, ((span_end - fit_end) - attack_secs) / 3600.0)

    print("\n================ FUSED SAME-ENVIRONMENT RESULTS ================")
    print(f"Attacks detected (fused): {detected}/{len(eval_intervals)} evaluated"
          + (f" ({len(outside)} outside captured range)" if outside else ""))
    print(f"False alarms/hour  byte={len(fa_byte)/benign_hours:.2f}  "
          f"rate={len(fa_rate)/benign_hours:.2f}  fused={len(fa_union)/benign_hours:.2f}")
    print("=================================================================")

    metrics = {
        "checkpoint_path": args.checkpoint_path, "pcap": args.pcap,
        "byte": {"mu": mu_b, "sigma": sig_b, "h": b_h, "cusum_k": args.cusum_k,
                 "total_alarms": int(len(byte_alarms))},
        "rate": {"window_sec": args.rate_window_sec, "threshold": r_thresh,
                 "total_alarms": int(len(rate_alarms))},
        "target_alarms_per_10k": args.target_alarms_per_10k,
        "per_attack": per_attack,
        "detected_fused": detected, "n_evaluated": len(eval_intervals),
        "false_alarms_per_hour": {
            "byte": len(fa_byte) / benign_hours,
            "rate": len(fa_rate) / benign_hours,
            "fused": len(fa_union) / benign_hours,
        },
    }
    out = os.path.join(args.output_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out}")


if __name__ == "__main__":
    main()
