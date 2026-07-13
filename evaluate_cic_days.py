#!/usr/bin/env python3
"""
evaluate_cic_days.py
====================
Same-environment, time-aligned attack detection on CIC-IDS2017 attack-day
pcaps (Tuesday/Wednesday/Friday), with CUSUM over ordered window scores.

WHY THIS EXISTS (2026-07-13 finding):
  Trace-level evaluation on the scratch/archive_upload collection is
  confounded: the attack pcaps come from different capture environments than
  the benign traces, so ANY checkpoint — including a known-bad one — separates
  them perfectly at trace level (gs800000 scored 25/25, same as gs865000).
  That protocol measures capture-source differences, not attack detection.

  CIC-IDS2017 fixes this by construction: each attack day records benign AND
  attack traffic on the SAME testbed, same hosts, same link, in one continuous
  capture, with published attack time-windows (Sharafaldin et al., ICISSP
  2018). Detection = CUSUM alarms falling inside documented attack intervals;
  false alarms = alarms in the benign gaps of the SAME capture. No domain
  shift shortcut exists.

METHOD
  1. Stream the day's pcap IN ORDER through the same masking dataloader
     logic used in training, tracking the capture timestamp of every
     512-byte window.
  2. Fit mu/sigma and the CUSUM threshold h on the leading benign period of
     the SAME day (capture start until shortly before the first scheduled
     attack) at a false-alarm budget.
  3. Run CUSUM over the rest of the day. Report, per scheduled attack:
     detected yes/no + detection delay; plus false alarms per hour over the
     benign gaps.

  Timestamps: CIC-IDS2017 was captured in ADT (UTC-3); attack times below
  are local clock times from the dataset documentation. Override the UTC
  offset or supply --schedule_json for custom intervals.

USAGE
  python evaluate_cic_days.py \
    --checkpoint_path checkpoints/latest_patcher.pt \
    --pcap data/cic-ids2017/Wednesday-workingHours.pcap \
    --day wednesday [--max_sequence_length 512]
"""

import os
import sys
import json
import math
import bisect
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_zero_day import load_model
from src.dataloader import RawPcapIterableDataset
from scapy.utils import RawPcapReader

# Published CIC-IDS2017 attack schedules (local ADT clock, from the dataset
# documentation). Minor +/- minutes of drift exist in the literature; the
# comparison below pads each interval with --margin_minutes.
CIC_SCHEDULES = {
    "tuesday": [
        {"name": "FTP-Patator", "start": "09:20", "end": "10:20"},
        {"name": "SSH-Patator", "start": "14:00", "end": "15:00"},
    ],
    "wednesday": [
        {"name": "DoS-slowloris", "start": "09:47", "end": "10:10"},
        {"name": "DoS-Slowhttptest", "start": "10:14", "end": "10:35"},
        {"name": "DoS-Hulk", "start": "10:43", "end": "11:00"},
        {"name": "DoS-GoldenEye", "start": "11:10", "end": "11:23"},
        {"name": "Heartbleed", "start": "15:12", "end": "15:32"},
    ],
    "thursday": [
        {"name": "Web-BruteForce", "start": "09:20", "end": "10:00"},
        {"name": "Web-XSS", "start": "10:15", "end": "10:35"},
        {"name": "Web-SQLi", "start": "10:40", "end": "10:42"},
        {"name": "Infiltration", "start": "14:19", "end": "15:45"},
    ],
    "friday": [
        {"name": "Botnet-ARES", "start": "10:02", "end": "11:02"},
        {"name": "PortScan", "start": "13:55", "end": "15:29"},
        {"name": "DDoS-LOIT", "start": "15:56", "end": "16:16"},
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Same-environment CIC-IDS2017 day evaluation")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/latest_patcher.pt")
    p.add_argument("--pcap", type=str, required=True, help="Attack-day pcap (or .gz)")
    p.add_argument("--day", type=str, choices=list(CIC_SCHEDULES.keys()), default=None,
                   help="Which built-in CIC-IDS2017 schedule to use")
    p.add_argument("--schedule_json", type=str, default=None,
                   help="Custom schedule: JSON list of {name, start: 'HH:MM', end: 'HH:MM'} "
                        "(local capture clock). Overrides --day.")
    p.add_argument("--utc_offset_hours", type=float, default=-3.0,
                   help="Capture local-time offset from UTC (CIC-IDS2017 = ADT = -3)")
    p.add_argument("--max_sequence_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--score_agg", type=str, default="topk", choices=["mean", "max", "topk"])
    p.add_argument("--topk_frac", type=float, default=0.10)
    p.add_argument("--cusum_k", type=float, default=0.5)
    p.add_argument("--target_alarms_per_10k", type=float, default=0.3,
                   help="False-alarm budget on the leading benign fit period")
    p.add_argument("--fit_until_min_before_first_attack", type=float, default=5.0,
                   help="Benign fit period = capture start until this many minutes "
                        "before the first scheduled attack")
    p.add_argument("--margin_minutes", type=float, default=2.0,
                   help="Slack around each attack interval (schedule times are approximate); "
                        "alarms inside interval+margin count as detections, and these regions "
                        "are excluded from false-alarm counting")
    p.add_argument("--output_dir", type=str, default="results/cic_day_eval")
    p.add_argument("--stop_after", type=str, default=None,
                   help="Local HH:MM at which to stop streaming (fast partial run). "
                        "e.g. '11:30' on Wednesday covers the benign lead-in + all 4 DoS "
                        "attacks (done by 11:23) and skips the afternoon Heartbleed — "
                        "~half the runtime. Attacks after the cutoff are marked "
                        "'outside captured range', not counted as missed.")
    return p.parse_args()


def load_schedule(args):
    if args.schedule_json:
        with open(args.schedule_json, encoding="utf-8") as f:
            return json.load(f)
    if args.day is None:
        print("ERROR: pass --day or --schedule_json", file=sys.stderr)
        sys.exit(1)
    return CIC_SCHEDULES[args.day]


def stream_window_scores(model, device, pcap_path, seq_len, batch_size, agg, topk_frac):
    """
    Stream the pcap in order; yield (score, capture_epoch_ts) per 512-byte
    window. Uses the SAME masking as training (RawPcapIterableDataset's
    parser, with cross-packet TLS state) but tracks byte-offset->timestamp
    marks so every window gets the capture time of its first byte.
    """
    # Borrow the masking method without paying dataset __init__ (hashing etc.)
    masker = RawPcapIterableDataset.__new__(RawPcapIterableDataset)
    tls_state = {}

    import gzip
    fobj = gzip.open(pcap_path, "rb") if pcap_path.endswith(".gz") else open(pcap_path, "rb")

    buffer = bytearray()
    marks_off, marks_ts = [], []   # byte offset of each packet start (in buffer coords), its ts
    consumed = 0                   # bytes already cut into windows (global coords)
    pending = []                   # [(window_bytes, ts)]

    @torch.no_grad()
    def flush(force=False):
        while len(pending) >= batch_size or (force and pending):
            chunk, ts_chunk = zip(*pending[:batch_size])
            del pending[:batch_size]
            batch = torch.stack([torch.frombuffer(bytearray(w), dtype=torch.uint8).long()
                                 for w in chunk]).to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            surprise = -token_logprob / math.log(2)
            if agg == "mean":
                pw = surprise.mean(dim=1)
            elif agg == "max":
                pw = surprise.max(dim=1).values
            else:
                k = max(1, int(round(topk_frac * surprise.shape[1])))
                pw = torch.topk(surprise, k=k, dim=1).values.mean(dim=1)
            for s, t in zip(pw.cpu().tolist(), ts_chunk):
                yield s, t

    with RawPcapReader(fobj) as reader:
        global_off = 0
        for packet_data, meta in reader:
            ts = getattr(meta, "sec", None)
            if ts is None and isinstance(meta, tuple):
                ts = meta[0]
            usec = getattr(meta, "usec", 0) or 0
            ts = float(ts) + float(usec) / 1e6 if ts is not None else 0.0

            masked = masker._mask_packet_addresses(packet_data, stream_tls_state=tls_state)
            marks_off.append(global_off)
            marks_ts.append(ts)
            buffer.extend(masked)
            global_off += len(masked)

            while len(buffer) >= seq_len:
                w = bytes(buffer[:seq_len])
                # timestamp of the packet containing this window's first byte
                i = bisect.bisect_right(marks_off, consumed) - 1
                pending.append((w, marks_ts[max(0, i)]))
                del buffer[:seq_len]
                consumed += seq_len
            # prune old marks
            if len(marks_off) > 4096:
                cut = bisect.bisect_right(marks_off, consumed) - 1
                if cut > 0:
                    del marks_off[:cut]
                    del marks_ts[:cut]
            yield from flush()
        yield from flush(force=True)
    fobj.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    schedule = load_schedule(args)
    print("==================================================")
    print("  CIC-IDS2017 SAME-ENVIRONMENT DAY EVALUATION")
    print("==================================================")
    print(f"Device: {device} | pcap: {args.pcap} | attacks: {[a['name'] for a in schedule]}")

    model, seq_len = load_model(args.checkpoint_path, device, args.max_sequence_length)

    # ---- Pass over the day (single streaming pass; scores + timestamps) ----
    # --stop_after HH:MM: stop streaming once capture time passes this local
    # clock. Lets you evaluate just the morning DoS block (all 4 DoS attacks
    # finish by ~11:23) in ~half the runtime, skipping the afternoon. Attacks
    # scheduled after the cutoff are reported as 'outside captured range'.
    stop_epoch = None            # resolved lazily once we know the first timestamp
    scores, tss = [], []
    n = 0
    for s, t in stream_window_scores(model, device, args.pcap, seq_len,
                                     args.batch_size, args.score_agg, args.topk_frac):
        if args.stop_after and stop_epoch is None:
            _lt0 = datetime.datetime.utcfromtimestamp(t + args.utc_offset_hours * 3600)
            _mid = t - (_lt0.hour * 3600 + _lt0.minute * 60 + _lt0.second)
            _h, _m = map(int, args.stop_after.split(":"))
            stop_epoch = _mid + _h * 3600 + _m * 60
        if stop_epoch is not None and t > stop_epoch:
            print(f"  [stop_after {args.stop_after}] reached — halting stream early.")
            break
        scores.append(s)
        tss.append(t)
        n += 1
        if n % 100000 == 0:
            print(f"  ... {n} windows scored "
                  f"(capture time {datetime.datetime.utcfromtimestamp(t + args.utc_offset_hours * 3600):%H:%M})")
    scores = np.asarray(scores)
    tss = np.asarray(tss)
    print(f"Total: {len(scores)} windows scored"
          + (f" (stopped at {args.stop_after} local)" if args.stop_after else " across the day"))
    if len(scores) < 1000:
        print("ERROR: too few windows.", file=sys.stderr)
        sys.exit(1)

    t0 = float(tss[0])
    # Map local 'HH:MM' to epoch: find the capture's local midnight in pure
    # UTC arithmetic (never the runner machine's timezone).
    local_t0 = datetime.datetime.utcfromtimestamp(t0 + args.utc_offset_hours * 3600)
    secs_since_local_midnight = local_t0.hour * 3600 + local_t0.minute * 60 + local_t0.second
    midnight_epoch = t0 - secs_since_local_midnight

    def hhmm_epoch(hhmm):
        h, m = map(int, hhmm.split(":"))
        return midnight_epoch + h * 3600 + m * 60

    intervals = [(a["name"], hhmm_epoch(a["start"]) - args.margin_minutes * 60,
                  hhmm_epoch(a["end"]) + args.margin_minutes * 60) for a in schedule]
    first_attack_start = min(s for _, s, _ in intervals)
    fit_end = first_attack_start - args.fit_until_min_before_first_attack * 60

    fit_mask = tss < fit_end
    benign_fit = scores[fit_mask]
    print(f"Benign fit period: capture start -> "
          f"{datetime.datetime.utcfromtimestamp(fit_end + args.utc_offset_hours * 3600):%H:%M} "
          f"local ({fit_mask.sum()} windows)")
    if fit_mask.sum() < 1000:
        print("ERROR: benign fit period too small — check --utc_offset_hours / schedule.",
              file=sys.stderr)
        sys.exit(1)
    mu_b, sigma_b = float(benign_fit.mean()), float(benign_fit.std())
    print(f"  mu_b={mu_b:.3f} bits, sigma_b={sigma_b:.3f} bits")

    def cusum_alarm_indices(vals, h):
        c, idx = 0.0, []
        inv = 1.0 / max(sigma_b, 1e-9)
        for i, s in enumerate(vals):
            c = max(0.0, c + (s - mu_b) * inv - args.cusum_k)
            if c > h:
                idx.append(i)
                c = 0.0
        return idx

    budget = args.target_alarms_per_10k * len(benign_fit) / 10000.0
    chosen_h = None
    for h in np.geomspace(1.0, 5000.0, 500):
        if len(cusum_alarm_indices(benign_fit, h)) <= budget:
            chosen_h = float(h)
            break
    chosen_h = chosen_h or 5000.0
    print(f"  chosen h={chosen_h:.2f} (budget {budget:.2f} alarms on fit period)")

    # ---- CUSUM over the whole day, map alarms to wall clock ----
    alarm_idx = cusum_alarm_indices(scores, chosen_h)
    alarm_ts = tss[alarm_idx] if alarm_idx else np.array([])

    span_end_ts = float(tss.max())
    per_attack = {}
    evaluated_intervals = []
    for name, s, e in intervals:
        # Attack window falls entirely after where the stream stopped -> not evaluated.
        if s > span_end_ts:
            per_attack[name] = {"detected": None, "alarms_in_window": 0,
                                "detection_delay_sec": None, "outside_captured_range": True}
            print(f"  {name}: outside captured range (stream stopped before it)")
            continue
        evaluated_intervals.append((name, s, e))
        hits = alarm_ts[(alarm_ts >= s) & (alarm_ts <= e)] if len(alarm_ts) else np.array([])
        delay = float(hits.min() - (s + args.margin_minutes * 60)) if len(hits) else None
        per_attack[name] = {"detected": bool(len(hits)), "alarms_in_window": int(len(hits)),
                            "detection_delay_sec": delay}
        status = "DETECTED" if len(hits) else "missed"
        print(f"  {name}: {status} ({len(hits)} alarms in interval"
              + (f", delay {delay:.0f}s" if delay is not None else "") + ")")

    in_any_attack = np.zeros(len(alarm_ts), dtype=bool)
    for _, s, e in evaluated_intervals:
        in_any_attack |= (alarm_ts >= s) & (alarm_ts <= e)
    # Benign evaluation time = post-fit captured span minus the attack intervals.
    span_end = span_end_ts
    attack_secs = sum(max(0.0, min(e, span_end) - max(s, fit_end)) for _, s, e in evaluated_intervals)
    benign_hours = max(1e-9, ((span_end - fit_end) - attack_secs) / 3600.0)
    false_alarms = int((~in_any_attack).sum())
    detected = sum(1 for v in per_attack.values() if v["detected"])
    n_evaluated = len(evaluated_intervals)

    print("\n================ SAME-ENVIRONMENT RESULTS ================")
    print(f"Attacks detected: {detected}/{n_evaluated} evaluated"
          + (f" ({len(per_attack) - n_evaluated} outside captured range)"
             if n_evaluated < len(per_attack) else ""))
    print(f"False alarms outside attack windows: {false_alarms} "
          f"(~{false_alarms / benign_hours:.2f}/hour of benign traffic)")
    print("===========================================================")

    metrics = {
        "checkpoint_path": args.checkpoint_path, "pcap": args.pcap,
        "score_agg": args.score_agg, "topk_frac": args.topk_frac,
        "cusum": {"k": args.cusum_k, "h": chosen_h, "mu": mu_b, "sigma": sigma_b},
        "fit_windows": int(fit_mask.sum()),
        "per_attack": per_attack,
        "detected": detected, "total_attacks": len(per_attack),
        "false_alarms_benign": false_alarms,
        "false_alarms_per_benign_hour": false_alarms / benign_hours,
        "alarm_epochs": alarm_ts.tolist(),
    }
    out = os.path.join(args.output_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out}")


if __name__ == "__main__":
    main()
