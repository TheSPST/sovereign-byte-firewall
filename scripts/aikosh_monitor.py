#!/usr/bin/env python3
"""
aikosh_monitor.py — active checkpoint watchdog for the one-shot AI Kosh A100 run.

WHY THIS EXISTS
  This project has direct evidence that (1) training past a sharp optimum causes
  real regression (gs865000: 32% held-out detection -> ~6-7% by gs1.76M+), and
  (2) calibration AUC does NOT predict held-out detection. So the run cannot be
  trusted unattended: checkpoints pushed to HF Hub during training must be pulled
  and scored with the real zero-day harness, and the job killed once detection
  stops improving.

WHAT IT DOES (each poll cycle)
  1. Lists the HF repo (default: spst01/sovereign-byte-firewall-aikosh) for new
     checkpoint files (*.pt) not yet evaluated.
  2. Downloads the newest one and runs evaluate_zero_day.py on it with the
     project-standard scoring config (topk 10% aggregation).
  3. Appends one summary line per checkpoint to results/aikosh_monitor_log.csv
     and prints a verdict against the benchmark to beat.

BENCHMARK TO BEAT (pre-TLS-fix best, from PROJECT_STATUS_2026-07-12):
  gs865000 + topk10% + Youden: 32.0% held-out detection @ 1.0% held-out FPR.

USAGE (run from repo root on the machine that has the eval pcaps):
  python scripts/aikosh_monitor.py                     # poll forever, every 15 min
  python scripts/aikosh_monitor.py --once              # single pass
  python scripts/aikosh_monitor.py --poll_minutes 30

KILL RULE OF THUMB
  Give the run headroom early (undertrained checkpoints score near-random), but
  once a checkpoint has beaten the benchmark, two consecutive later checkpoints
  scoring materially worse = the overfitting regression has started -> scancel
  the SLURM job and keep the peak checkpoint.
"""
import os
import sys
import csv
import json
import time
import argparse
import subprocess
import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK = {"label": "gs865000 topk10 youden", "detection": 0.32, "fpr": 0.01}


def parse_args():
    p = argparse.ArgumentParser(description="AI Kosh checkpoint watchdog")
    p.add_argument("--hf_repo", default=os.environ.get("HF_REPO_ID", "spst01/sovereign-byte-firewall-aikosh"))
    p.add_argument("--poll_minutes", type=float, default=15.0)
    p.add_argument("--once", action="store_true", help="Single pass instead of polling forever")
    p.add_argument("--download_dir", default="checkpoints/aikosh_pulled")
    p.add_argument("--log_csv", default="results/aikosh_monitor_log.csv")
    p.add_argument("--score_agg", default="topk")
    p.add_argument("--topk_frac", type=float, default=0.10)
    p.add_argument("--benign_calibration_pcap", default="scratch/archive_upload/normal.pcap")
    p.add_argument("--benign_holdout_pcap", default="scratch/archive_upload/normal2.pcap")
    p.add_argument("--attack_dir", default="scratch/archive_upload")
    p.add_argument("--holdout_attack_pcap", default="scratch/archive_upload/0day.pcap")
    return p.parse_args()


def list_remote_checkpoints(repo_id):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id=repo_id)
    return sorted(f for f in files if f.endswith(".pt"))


def already_evaluated(log_csv):
    if not os.path.exists(log_csv):
        return set()
    with open(log_csv, newline="", encoding="utf-8") as f:
        return {row["checkpoint"] for row in csv.DictReader(f)}


def evaluate_checkpoint(args, local_ckpt, remote_name):
    out_dir = os.path.join("results", "aikosh_monitor",
                           os.path.splitext(os.path.basename(remote_name))[0])
    cmd = [
        sys.executable, "evaluate_zero_day.py",
        "--checkpoint_path", local_ckpt,
        "--score_agg", args.score_agg,
        "--topk_frac", str(args.topk_frac),
        "--benign_calibration_pcap", args.benign_calibration_pcap,
        "--benign_holdout_pcap", args.benign_holdout_pcap,
        "--attack_dir", args.attack_dir,
        "--holdout_attack_pcap", args.holdout_attack_pcap,
        "--output_dir", out_dir,
    ]
    print(f"[monitor] evaluating {remote_name} ...")
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"[monitor] WARNING: eval failed for {remote_name} (rc={res.returncode})")
        return None
    with open(os.path.join(REPO_ROOT, out_dir, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def append_log(log_csv, remote_name, metrics):
    youden = metrics["holdout"]["results"].get("youden", {})
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": remote_name,
        "calib_auc": f"{metrics['calibration']['auc']:.4f}",
        "youden_threshold_bits": f"{youden.get('threshold', float('nan')):.3f}",
        "holdout_detection": youden.get("holdout_attack_detection_rate"),
        "holdout_fpr": youden.get("holdout_benign_false_positive_rate"),
    }
    exists = os.path.exists(log_csv)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    return row


def verdict(row):
    det = row["holdout_detection"]
    if det is None:
        return "NO HOLDOUT NUMBER — check eval output manually."
    det = float(det)
    if det > BENCHMARK["detection"]:
        return (f"*** BEATS benchmark ({det:.1%} > {BENCHMARK['detection']:.0%} from "
                f"{BENCHMARK['label']}) — new best, keep this checkpoint safe. ***")
    return (f"below benchmark ({det:.1%} vs {BENCHMARK['detection']:.0%}). Fine early in the "
            f"run; if this is the 2nd consecutive drop AFTER a peak, kill the job (scancel).")


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.download_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download

    print(f"[monitor] watching {args.hf_repo} | benchmark: {BENCHMARK['detection']:.0%} "
          f"detection @ {BENCHMARK['fpr']:.0%} FPR ({BENCHMARK['label']})")
    while True:
        try:
            done = already_evaluated(args.log_csv)
            remote = [f for f in list_remote_checkpoints(args.hf_repo) if f not in done]
            if remote:
                newest = remote[-1]  # newest unevaluated; skip backlog to stay current
                local = hf_hub_download(repo_id=args.hf_repo, filename=newest,
                                        local_dir=args.download_dir)
                metrics = evaluate_checkpoint(args, local, newest)
                if metrics:
                    row = append_log(args.log_csv, newest, metrics)
                    print(f"[monitor] {newest}: detection={row['holdout_detection']} "
                          f"fpr={row['holdout_fpr']} auc={row['calib_auc']}")
                    print(f"[monitor] VERDICT: {verdict(row)}")
            else:
                print(f"[monitor] no new checkpoints ({len(done)} already scored).")
        except Exception as e:
            print(f"[monitor] cycle error (will retry): {e}")
        if args.once:
            break
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
