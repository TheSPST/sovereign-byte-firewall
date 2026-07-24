#!/usr/bin/env python3
"""
kaggle_eval_watcher.py — background per-checkpoint evaluation, in-process on Kaggle.

Runs ALONGSIDE training in the same notebook: launch it (detached) before the
training cell, and it polls for new checkpoints. Each time a NEW mid-epoch
checkpoint appears, it scores that checkpoint (per-window surprise, topk-10%
aggregation, evaluate_zero_day.py) and appends one row to a CSV.

SELECTION POLICY (revised 2026-07-25) — read this before changing it:
  * Checkpoints are ranked by held-out detection at the EVT/POT threshold,
    admissible only if held-out benign FPR stays within --fpr_budget.
  * Calibration AUC is recorded but NEVER ranked on. Repeatedly in this project
    AUC has moved opposite to deployed performance.
  * Degenerate checkpoints are detected and excluded. A collapsed model flags
    every window and therefore scores 100% detection; the previous version of
    this script ranked on Youden detection alone and consequently reported
    exactly those dead checkpoints as "BEATS benchmark — keep this checkpoint".
  * Mean benign surprise is tracked as an early collapse warning: it rose from
    8.06 bits at the healthy peak to 10.6 through collapse in the reference run,
    which is visible well before the detection numbers look absurd.

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
BENCHMARK_DETECTION = 0.15   # best verified EVT held-out detection (gs75000, 24-family calibration)

# --- Checkpoint selection policy (revised 2026-07-25) -----------------------
# Previously this watcher ranked checkpoints by held-out detection at the YOUDEN
# threshold and declared anything above 32% a winner. That is actively harmful:
# a COLLAPSED model flags every window, so it scores detection = 100% at ~98%
# false-positive rate and was reported as "*** BEATS benchmark — keep this
# checkpoint ***". eval_watcher_results.csv contains six such rows (gs105000,
# gs135000, gs165000, gs285000, gs320000, gs365000).
#
# Selection now uses the DEPLOYMENT metric: held-out detection at the EVT/POT
# threshold, admissible only if the checkpoint respects a false-alarm budget.
# Calibration AUC is logged but never used to rank — across this project it has
# repeatedly moved opposite to deployed performance (a 0.96-AUC Mamba-2 detects
# less at matched FPR than a 0.80-AUC Transformer; an lr1e4 run hit 0.942 AUC
# while detecting 5.5%).
COLLAPSE_FPR = 0.50      # held-out FPR above this = degenerate "flag everything"
COLLAPSE_AUC = 0.55      # calibration AUC at/below this = no better than chance


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
    p.add_argument("--output_root", default="/kaggle/working/eval_watcher",
                   help="Where per-checkpoint eval artifacts (metrics.json, plots) are "
                        "written. Set this to a local path when running OFF Kaggle "
                        "(e.g. on your Mac): --output_root ./eval_out")
    p.add_argument("--fpr_budget", type=float, default=0.01,
                   help="Maximum held-out benign false-positive rate for a checkpoint to "
                        "be eligible as 'best' (default 0.01 = 1%%). Checkpoints above "
                        "this are logged but never selected, however high their raw "
                        "detection rate — this is what stops a collapsed model that "
                        "flags everything from being crowned the winner.")
    p.add_argument("--collapse_sigma_drift", type=float, default=1.0,
                   help="Flag collapse when mean benign surprise rises this many bits "
                        "above the run's minimum (default 1.0). In the reference run it "
                        "went 8.06 at the healthy peak to 10.6 through collapse, making "
                        "this the earliest reliable warning signal.")
    p.add_argument("--abort_on_collapse", action="store_true",
                   help="Exit the watcher after --collapse_patience consecutive collapsed "
                        "checkpoints, so a dead run stops burning session time.")
    p.add_argument("--collapse_patience", type=int, default=3,
                   help="Consecutive collapsed checkpoints tolerated before aborting "
                        "(only meaningful with --abort_on_collapse; default 3)")
    p.add_argument("--best_json", default=None,
                   help="Path for the running best-checkpoint record "
                        "(default: <log_csv dir>/best_checkpoint.json)")
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
    out_dir = os.path.join(args.output_root, os.path.splitext(name)[0])
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


def _operating_points(m):
    """Extract all three threshold rules plus health signals from metrics.json."""
    res = m.get("holdout", {}).get("results", {}) or {}

    def pt(rule):
        d = res.get(rule) or {}
        return (d.get("holdout_attack_detection_rate"),
                d.get("holdout_benign_false_positive_rate"))

    evt_dr, evt_fpr = pt("evt")
    tf_dr, tf_fpr = pt("target_fpr")
    y_dr, y_fpr = pt("youden")
    calib = m.get("calibration", {}) or {}
    return {
        "auc": calib.get("auc"),
        "sigma": (calib.get("per_file_mean_score") or {}).get("benign_calibration"),
        "evt_dr": evt_dr, "evt_fpr": evt_fpr,
        "tf_dr": tf_dr, "tf_fpr": tf_fpr,
        "youden_dr": y_dr, "youden_fpr": y_fpr,
    }


def classify(x, sigma_min, drift_limit):
    """Return (status, reason). COLLAPSED means the checkpoint is degenerate and
    must never be selected regardless of its raw detection rate."""
    if x["evt_fpr"] is not None and x["evt_fpr"] > COLLAPSE_FPR:
        return "COLLAPSED", f"held-out FPR {x['evt_fpr']:.1%} — flagging nearly everything"
    if x["auc"] is not None and x["auc"] <= COLLAPSE_AUC:
        return "COLLAPSED", f"calibration AUC {x['auc']:.4f} — at or below chance"
    if (x["sigma"] is not None and sigma_min is not None
            and (x["sigma"] - sigma_min) > drift_limit):
        return "COLLAPSED", (f"benign mean surprise {x['sigma']:.3f} is "
                             f"+{x['sigma'] - sigma_min:.2f} bits above run minimum "
                             f"{sigma_min:.3f} — model is losing its baseline")
    return "OK", ""


def selection_metric(x, fpr_budget):
    """The deployment metric: held-out detection at the EVT threshold, admissible
    only within the false-alarm budget. Returns None if not eligible."""
    dr, fpr = x["evt_dr"], x["evt_fpr"]
    if dr is None or fpr is None:
        return None
    if fpr > fpr_budget:
        return None
    return float(dr)


def _read_prior(log_csv):
    """(min benign sigma seen, best eligible EVT detection so far, best name)."""
    sigma_min, best_dr, best_name = None, None, None
    if not os.path.exists(log_csv):
        return sigma_min, best_dr, best_name
    try:
        with open(log_csv, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    s = float(r.get("benign_sigma_proxy", ""))
                    sigma_min = s if sigma_min is None else min(sigma_min, s)
                except (TypeError, ValueError):
                    pass
                if r.get("status") == "OK" and r.get("eligible") == "yes":
                    try:
                        d = float(r.get("evt_detection", ""))
                    except (TypeError, ValueError):
                        continue
                    if best_dr is None or d > best_dr:
                        best_dr, best_name = d, r.get("checkpoint")
    except OSError:
        pass
    return sigma_min, best_dr, best_name


FIELDNAMES = ["timestamp", "checkpoint", "status", "eligible", "calib_auc",
              "evt_detection", "evt_fpr", "target_fpr_detection", "target_fpr_fpr",
              "youden_detection", "youden_fpr", "benign_sigma_proxy", "note"]


def log_row(args, name, m):
    """Append one row and report a verdict based on the DEPLOYMENT metric.

    Returns True if this checkpoint was collapsed (so the caller can count
    consecutive failures for --abort_on_collapse).
    """
    x = _operating_points(m)
    sigma_min, best_dr, best_name = _read_prior(args.log_csv)
    status, reason = classify(x, sigma_min, args.collapse_sigma_drift)
    sel = selection_metric(x, args.fpr_budget) if status == "OK" else None

    def r4(v):
        return None if v is None else round(float(v), 6)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": name,
        "status": status,
        "eligible": "yes" if sel is not None else "no",
        "calib_auc": r4(x["auc"]),
        "evt_detection": r4(x["evt_dr"]),
        "evt_fpr": r4(x["evt_fpr"]),
        "target_fpr_detection": r4(x["tf_dr"]),
        "target_fpr_fpr": r4(x["tf_fpr"]),
        "youden_detection": r4(x["youden_dr"]),
        "youden_fpr": r4(x["youden_fpr"]),
        "benign_sigma_proxy": None if x["sigma"] is None else round(float(x["sigma"]), 3),
        "note": reason,
    }

    # The CSV schema changed in 2026-07-25; retire an old-schema file rather than
    # appending mismatched columns to it.
    if os.path.exists(args.log_csv):
        try:
            with open(args.log_csv, newline="") as f:
                hdr = next(csv.reader(f), [])
            if hdr and hdr != FIELDNAMES:
                retired = args.log_csv + ".v1"
                os.replace(args.log_csv, retired)
                print(f"[watcher] old-schema CSV retired to {retired}", flush=True)
        except OSError:
            pass

    exists = os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)

    if status == "COLLAPSED":
        print(f"[watcher] {name}: *** COLLAPSED *** {reason}. "
              f"(raw youden detection {x['youden_dr']} is meaningless here.)", flush=True)
        return True

    if sel is None:
        print(f"[watcher] {name}: NOT ELIGIBLE — held-out FPR {x['evt_fpr']} exceeds budget "
              f"{args.fpr_budget:.2%}; detection {x['evt_dr']} disregarded.", flush=True)
        return False

    if best_dr is None or sel > best_dr:
        verdict = (f"*** NEW BEST *** EVT detection {sel:.2%} @ FPR {x['evt_fpr']:.3%}"
                   + (f" (previous best {best_dr:.2%}, {best_name})" if best_dr is not None else ""))
        best_path = args.best_json or os.path.join(
            os.path.dirname(os.path.abspath(args.log_csv)) or ".", "best_checkpoint.json")
        try:
            with open(best_path, "w") as f:
                json.dump({"checkpoint": name, "evt_detection": sel,
                           "evt_fpr": x["evt_fpr"], "calib_auc": x["auc"],
                           "benign_sigma_proxy": x["sigma"],
                           "selected_at": row["timestamp"],
                           "criterion": f"max held-out EVT detection s.t. held-out FPR <= "
                                        f"{args.fpr_budget}"}, f, indent=2)
        except OSError as e:
            print(f"[watcher] could not write best record: {e}", flush=True)
    else:
        verdict = (f"EVT detection {sel:.2%} @ FPR {x['evt_fpr']:.3%} "
                   f"(best remains {best_dr:.2%}, {best_name})")

    print(f"[watcher] {name}: {verdict} | auc={row['calib_auc']} (not used for ranking) "
          f"| benign_mean={row['benign_sigma_proxy']}", flush=True)
    return False


def get_hf_credentials():
    # Return (token_or_None, repo_id_or_None). Never return an empty STRING for
    # the token — passing token="" makes huggingface_hub send a broken
    # "Bearer " header ("Illegal header value"); None lets it auto-detect the
    # CLI-cached token instead.
    token = os.environ.get("HF_TOKEN", "").strip() or None
    if not token:
        try:
            from huggingface_hub import get_token   # modern, finds CLI-login cache
            token = get_token()
        except Exception:
            try:
                from huggingface_hub import HfFolder  # legacy fallback
                token = HfFolder.get_token()
            except Exception:
                token = None
    repo_id = os.environ.get("HF_REPO_ID", "").strip() or None
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
    print(f"[watcher] source: {src_desc}", flush=True)
    print(f"[watcher] selection = max held-out EVT detection subject to held-out FPR "
          f"<= {args.fpr_budget:.2%} (calibration AUC is logged, never ranked on); "
          f"reference {BENCHMARK_DETECTION:.0%} | results -> {args.log_csv}", flush=True)
    consecutive_collapsed = 0
    while True:
        try:
            done = already_done(args.log_csv)
            ckpt_path, score_name = _next_checkpoint(args, done)
            if ckpt_path:
                print(f"[watcher] scoring {score_name} ...", flush=True)
                m = evaluate(args, ckpt_path)
                if m:
                    collapsed = log_row(args, score_name, m)
                    consecutive_collapsed = consecutive_collapsed + 1 if collapsed else 0
                    if collapsed and args.abort_on_collapse and \
                            consecutive_collapsed >= args.collapse_patience:
                        print(f"[watcher] {consecutive_collapsed} consecutive collapsed "
                              f"checkpoints — aborting watcher. The run is not recovering; "
                              f"stop training and inspect gradient norms / LR schedule.",
                              flush=True)
                        upload_to_hf(args.log_csv, "eval/eval_watcher_results.csv",
                                     "final eval log (watcher aborted on collapse)")
                        return
                    upload_to_hf(args.log_csv, "eval/eval_watcher_results.csv",
                                 f"update eval logs for {score_name}")
                    ckpt_id = os.path.splitext(score_name)[0]
                    local_eval_dir = os.path.join(args.output_root, ckpt_id)
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
