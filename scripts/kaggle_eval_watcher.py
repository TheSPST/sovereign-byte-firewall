#!/usr/bin/env python3
"""
kaggle_eval_watcher.py — background per-checkpoint evaluation, in-process on Kaggle.

Runs ALONGSIDE training in the same notebook: launch it (detached) before the
training cell, and it polls the local checkpoint directory. Each time a NEW
mid-epoch checkpoint appears, it scores that checkpoint with the project's
proven signal — per-window surprise, topk-10% aggregation, Youden threshold on
held-out 0day.pcap (evaluate_zero_day.py) — and appends one row to a CSV plus a
human-readable verdict against the 32% benchmark.

DESIGN CHOICES (deliberate):
  * Runs on CPU (CUDA_VISIBLE_DEVICES=""). The model is tiny; per-window eval on
    the small scratch pcaps is a few minutes on CPU, and this GUARANTEES it never
    contends with training for the single GPU (no OOM, no throughput hit).
  * Evaluates the NEWEST unscored checkpoint each cycle and skips any backlog, so
    it stays current with a fast-moving run instead of falling behind.
  * Reads checkpoints from the LOCAL dir (no HF round-trip); HF backup is
    orthogonal and still happens from the training process.

EVAL DATA (gitignored — must be provided): pass --eval_data_dir pointing at a
directory that contains normal.pcap, normal2.pcap, 0day.pcap and the attack
pcaps (e.g. an attached Kaggle dataset). If the dir/files are missing the
watcher prints a clear message and exits rather than looping uselessly.

USAGE (from a Kaggle cell, detached):
  import subprocess, os
  env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # force CPU
  proc = subprocess.Popen(
      ["python", "scripts/kaggle_eval_watcher.py",
       "--checkpoints_dir", "/kaggle/working/checkpoints",
       "--eval_data_dir", "/kaggle/input/sovereign-eval-pcaps",
       "--max_sequence_length", "512"],
      stdout=open("/kaggle/working/eval_watcher.log", "w"),
      stderr=subprocess.STDOUT, env=env, start_new_session=True)
"""
import os
import sys
import csv
import glob
import json
import time
import argparse
import datetime
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_DETECTION = 0.32   # gs865000 topk10 youden, old masking — the number to beat


def parse_args():
    p = argparse.ArgumentParser(description="Background per-checkpoint eval watcher")
    p.add_argument("--checkpoints_dir", default="/kaggle/working/checkpoints")
    p.add_argument("--eval_data_dir", required=True,
                   help="Dir with normal.pcap, normal2.pcap, 0day.pcap + attack pcaps")
    p.add_argument("--benign_calibration", default="normal.pcap")
    p.add_argument("--benign_holdout", default="normal2.pcap")
    p.add_argument("--holdout_attack", default="0day.pcap")
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--score_agg", default="topk")
    p.add_argument("--topk_frac", type=float, default=0.10)
    p.add_argument("--poll_seconds", type=float, default=120.0)
    p.add_argument("--log_csv", default="/kaggle/working/eval_watcher_results.csv")
    p.add_argument("--ckpt_glob", default="*_mid_epoch.pt",
                   help="Local-source only: which checkpoints to evaluate")
    p.add_argument("--checkpoint_source", default="hf", choices=["hf", "local"],
                   help="Where to find checkpoints. 'hf' (default): pull gs-named "
                        "checkpoints from the HF repo — REQUIRED on Kaggle, because "
                        "training only writes a single overwritten 'latest_patcher.pt' "
                        "locally; the distinct gs-named files exist ONLY on HF. "
                        "'local': scan --checkpoints_dir with --ckpt_glob.")
    p.add_argument("--hf_repo", default=os.environ.get("HF_REPO_ID", ""),
                   help="HF repo to pull checkpoints from (default: $HF_REPO_ID)")
    p.add_argument("--hf_download_dir", default="/kaggle/working/hf_ckpt_eval",
                   help="Where to stage checkpoints downloaded from HF for scoring")
    return p.parse_args()


def hf_list_checkpoints(repo_id):
    """Return HF checkpoint files (checkpoints/*.pt) newest-first by gs step."""
    import re
    from huggingface_hub import HfApi
    token, _ = get_hf_credentials()
    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model")
    ckpts = [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")
             and "latest_patcher_" in f]

    def step_of(f):
        m = re.search(r"_gs(\d+)_", f)
        return int(m.group(1)) if m else -1
    return sorted(ckpts, key=step_of)


def preflight(args):
    # Always log what's actually in the eval dir — makes filename mismatches
    # self-diagnosing from the log instead of a silent exit.
    if not os.path.isdir(args.eval_data_dir):
        print(f"[watcher] eval_data_dir '{args.eval_data_dir}' does not exist. "
              f"Exiting (training is unaffected).", flush=True)
        sys.exit(0)
    present = sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.eval_data_dir, "*.pcap")))
    print(f"[watcher] eval_data_dir '{args.eval_data_dir}' contains {len(present)} pcap(s): "
          f"{present}", flush=True)

    # Hard requirements: a benign calibration file and a held-out attack file.
    # The benign HOLDOUT is optional — if absent we reuse the calibration file
    # (the held-out FPR becomes in-sample, noted in the log, but the detection
    # number the watcher tracks is unaffected).
    hard = {"benign_calibration": args.benign_calibration, "holdout_attack": args.holdout_attack}
    missing_hard = {k: v for k, v in hard.items() if not os.path.exists(os.path.join(args.eval_data_dir, v))}
    if missing_hard:
        print(f"[watcher] MISSING required eval files {missing_hard} in '{args.eval_data_dir}'. "
              f"Available: {present}. Re-launch with --benign_calibration / --holdout_attack "
              f"set to real filenames from the list above. Exiting (training is unaffected).",
              flush=True)
        sys.exit(0)

    if not os.path.exists(os.path.join(args.eval_data_dir, args.benign_holdout)):
        print(f"[watcher] NOTE: benign holdout '{args.benign_holdout}' absent — reusing "
              f"'{args.benign_calibration}' for the held-out FPR check (in-sample; detection "
              f"metric unaffected).", flush=True)
        args.benign_holdout = args.benign_calibration

    need = {args.benign_calibration, args.benign_holdout, args.holdout_attack}
    n_attack = len([f for f in present if f not in need])
    print(f"[watcher] eval data OK: {n_attack} calibration attack pcaps + held-out set", flush=True)


def _step_of(name):
    """Parse the gs training step out of a checkpoint filename (-1 if absent)."""
    import re
    m = re.search(r"_gs(\d+)_", name)
    return int(m.group(1)) if m else -1


def already_done(log_csv):
    if not os.path.exists(log_csv):
        return set()
    with open(log_csv, newline="") as f:
        return {r["checkpoint"] for r in csv.DictReader(f)}


def max_scored_step(done):
    """Highest gs step already scored — the 'start from the latest' high-water mark."""
    return max((_step_of(n) for n in done), default=-1)


def evaluate(args, ckpt_path):
    name = os.path.basename(ckpt_path)
    out_dir = os.path.join("/kaggle/working/eval_watcher", os.path.splitext(name)[0])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # CPU only — no GPU contention
    cmd = [
        sys.executable, "evaluate_zero_day.py",
        "--checkpoint_path", ckpt_path,
        "--max_sequence_length", str(args.max_sequence_length),
        "--score_agg", args.score_agg, "--topk_frac", str(args.topk_frac),
        "--benign_calibration_pcap", os.path.join(args.eval_data_dir, args.benign_calibration),
        "--benign_holdout_pcap", os.path.join(args.eval_data_dir, args.benign_holdout),
        "--attack_dir", args.eval_data_dir,
        "--holdout_attack_pcap", os.path.join(args.eval_data_dir, args.holdout_attack),
        "--output_dir", out_dir,
        "--batch_size", "256",            # Vectorization throughput on CPU
        "--num_workers", "2",             # Prefetch packets in worker threads
        "--max_pcap_size_mb", "5.0",      # Exclude huge files (like 93MB mirai.pcap) from calibration to save CPU time
    ]
    print(f"[watcher] evaluating {name} on CPU ...", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if r.returncode != 0:
        print(f"[watcher] eval FAILED for {name} (rc={r.returncode})", flush=True)
        return None
    with open(os.path.join(REPO_ROOT, out_dir, "metrics.json")) as f:
        return json.load(f)


def log_row(args, name, m):
    y = m["holdout"]["results"].get("youden", {})
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": name,
        "calib_auc": round(m["calibration"]["auc"], 4),
        "holdout_detection": y.get("holdout_attack_detection_rate"),
        "holdout_fpr": y.get("holdout_benign_false_positive_rate"),
        "benign_sigma_proxy": round(
            m["calibration"]["per_file_mean_score"].get("benign_calibration", float("nan")), 3),
    }
    exists = os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    det = row["holdout_detection"]
    verdict = "no holdout number" if det is None else (
        f"*** BEATS benchmark ({det:.1%} > {BENCHMARK_DETECTION:.0%}) — keep this checkpoint ***"
        if float(det) > BENCHMARK_DETECTION else
        f"below benchmark ({det:.1%} vs {BENCHMARK_DETECTION:.0%})")
    print(f"[watcher] {name}: detection={det} fpr={row['holdout_fpr']} auc={row['calib_auc']} "
          f"-> {verdict}", flush=True)


def get_hf_credentials():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token() or ""
        except Exception:
            pass
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    return token, repo_id


def upload_to_hf(local_path, path_in_repo, commit_message):
    token, repo_id = get_hf_credentials()
    if not token or not repo_id:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        print(f"[watcher] ✓ Uploaded '{local_path}' to HF Hub '{repo_id}/{path_in_repo}'", flush=True)
    except Exception as e:
        print(f"[watcher] HF upload failed: {e}", flush=True)


def upload_folder_to_hf(local_dir, path_in_repo, commit_message):
    token, repo_id = get_hf_credentials()
    if not token or not repo_id:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        print(f"[watcher] ✓ Uploaded folder '{local_dir}' to HF Hub '{repo_id}/{path_in_repo}'", flush=True)
    except Exception as e:
        print(f"[watcher] HF folder upload failed: {e}", flush=True)


def _next_checkpoint(args, done):
    """
    Return (local_path, score_name) for the newest not-yet-scored checkpoint,
    or (None, None). HF source is the default and correct choice on Kaggle
    (training only keeps one overwritten latest_patcher.pt locally; the distinct
    gs-named files live on HF).
    """
    hwm = max_scored_step(done)   # only ever move FORWARD from the latest scored step
    if args.checkpoint_source == "hf":
        if not args.hf_repo:
            print("[watcher] checkpoint_source=hf but no --hf_repo/$HF_REPO_ID set.", flush=True)
            return None, None
        # Candidates strictly newer than the high-water mark and not already scored.
        remote = [f for f in hf_list_checkpoints(args.hf_repo)
                  if _step_of(os.path.basename(f)) > hwm
                  and os.path.basename(f) not in done]
        if not remote:
            return None, None
        newest = remote[-1]                       # highest gs step available
        from huggingface_hub import hf_hub_download
        token, _ = get_hf_credentials()
        local = hf_hub_download(repo_id=args.hf_repo, filename=newest, repo_type="model",
                                local_dir=args.hf_download_dir, token=token)
        return local, os.path.basename(newest)
    # local source
    cands = sorted(glob.glob(os.path.join(args.checkpoints_dir, args.ckpt_glob)),
                   key=os.path.getmtime)
    todo = [c for c in cands if _step_of(os.path.basename(c)) > hwm
            and os.path.basename(c) not in done]
    if not todo:
        return None, None
    return todo[-1], os.path.basename(todo[-1])


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    preflight(args)
    src_desc = (f"HF repo '{args.hf_repo}'" if args.checkpoint_source == "hf"
                else f"local dir '{args.checkpoints_dir}'")
    print(f"[watcher] source: {src_desc} | benchmark {BENCHMARK_DETECTION:.0%} "
          f"held-out detection | results -> {args.log_csv}", flush=True)
    while True:
        try:
            done = already_done(args.log_csv)
            ckpt_path, score_name = _next_checkpoint(args, done)
            if ckpt_path:
                print(f"[watcher] scoring {score_name} ...", flush=True)
                m = evaluate(args, ckpt_path)
                if m:
                    log_row(args, score_name, m)
                    upload_to_hf(args.log_csv, "eval/eval_watcher_results.csv",
                                 f"update eval logs for {score_name}")
                    ckpt_id = os.path.splitext(score_name)[0]
                    local_eval_dir = os.path.join("/kaggle/working/eval_watcher", ckpt_id)
                    if os.path.isdir(local_eval_dir):
                        upload_folder_to_hf(local_eval_dir, f"eval/{ckpt_id}",
                                            f"upload eval metrics for {score_name}")
                # free disk: drop the downloaded checkpoint after scoring
                if args.checkpoint_source == "hf" and os.path.exists(ckpt_path):
                    try:
                        os.remove(ckpt_path)
                    except OSError:
                        pass
            else:
                print(f"[watcher] no new checkpoints ({len(done)} scored)", flush=True)
        except Exception as e:
            print(f"[watcher] cycle error (will retry): {e}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
