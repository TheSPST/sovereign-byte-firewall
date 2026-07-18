#!/usr/bin/env python3
"""
kaggle_train_mamba2.py
=======================
Standalone Kaggle-notebook script: trains the Mamba-2 backbone (branch
experiment/mamba-backbone) on CIC-IDS2017 Monday (benign-only), downloading
the pcap remotely from Hugging Face if it isn't already attached as a Kaggle
Dataset. This is the accuracy-A/B run called for in MAMBA_EXPERIMENT.md —
throughput A/B is already done; this proves whether Mamba-2 matches the
transformer's detection quality.

Run this as a single script in a Kaggle notebook cell (or `!python
kaggle_train_mamba2.py` after uploading it), on a GPU T4 x2 instance.
P100 is NOT supported (sm_60 < required sm_70 for the installed torch).

What it does, in order:
  1. Clone (or pull) the repo, checkout experiment/mamba-backbone, merge
     latest main into it (so masking/labeling fixes on main are included).
  2. Install deps: scapy + causal-conv1d + mamba-ssm (fused CUDA kernel).
  3. Verify GPU is present and compute-capable.
  4. Resolve the Monday-WorkingHours.pcap:
       - check /kaggle/input for a manually-attached copy first (fast),
       - else check /kaggle/working for an already-downloaded copy,
       - else download it remotely from the public HF dataset
         bvsam/cic-ids-2017 (~10 GB, one-time).
     Refuses to proceed if the resolved file is Wednesday (attack/eval day).
  5. Smoke test train_ab.py --backbone mamba2 for 20 steps (proves wiring
     before committing to the long run).
  6. Real training run: --backbone mamba2, 20000 steps, seq_len 512 — same
     recipe used for the transformer A/B, for a fair comparison.
  7. Push the resulting checkpoint to HUGGING FACE HUB (same backup pattern
     as kaggle_train.ipynb uses for the transformer — train_ab.py itself has
     no push hook, so this uploads the .pt directly via huggingface_hub).
     Goes to a DEDICATED repo (HF_CKPT_REPO below) so mamba2 checkpoints
     don't mix with the transformer's gs-numbered ones. Requires a Kaggle
     secret named HF_TOKEN (write access). If missing, this step is skipped
     with a warning and the checkpoint is left in /kaggle/working for manual
     download.

Next step after this finishes (not run here): evaluate_zero_day.py
--backbone mamba2 --checkpoint_path <out> --score_agg topk --topk_frac 0.1
against the same held-out split used for the transformer, to compare AUC +
detection@1%FPR before deciding whether Mamba-2 merges to main.
"""

import os
import re
import sys
import subprocess

REPO_URL = "https://github.com/TheSPST/sovereign-byte-firewall.git"
REPO_DIR = "/kaggle/working/sovereign-byte-firewall"
BRANCH = "experiment/mamba-backbone"

MONDAY_HF_REPO = "bvsam/cic-ids-2017"
MONDAY_HF_FILENAME = "pcap/Monday-WorkingHours.pcap"
MONDAY_LOCAL_FALLBACK = "/kaggle/working/pcap/Monday-WorkingHours.pcap"
MONDAY_MIN_BYTES = 10 * 1e9  # sanity floor so a truncated download isn't reused

TRAIN_OUT = "/kaggle/working/checkpoints/ckpt_mamba2.pt"
SMOKE_OUT = "/kaggle/working/smoke_mamba2.pt"
HF_CKPT_REPO = "spst01/sovereign-byte-firewall-mamba2"  # dedicated repo, kept apart from the transformer's


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def step1_repo():
    print("\n=== [1/7] repo: clone/checkout/merge ===")
    if os.path.exists(REPO_DIR):
        run(["git", "-C", REPO_DIR, "fetch", "origin"])
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    # merge commits need an identity — Kaggle containers have none set by default
    run(["git", "config", "user.email", "kaggle-bot@users.noreply.github.com"])
    run(["git", "config", "user.name", "kaggle-training-bot"])
    run(["git", "checkout", BRANCH])
    run(["git", "pull", "origin", BRANCH])
    run(["git", "merge", "origin/main", "-m", "merge main into mamba branch"])
    log = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True).stdout
    print(log)


def step2_deps():
    print("\n=== [2/7] deps: scapy + mamba-ssm + causal-conv1d ===")
    run([sys.executable, "-m", "pip", "install", "scapy", "-q"])
    run([sys.executable, "-m", "pip", "install",
         "causal-conv1d>=1.2.0", "mamba-ssm", "--no-build-isolation", "-q"])


def step3_gpu_check():
    print("\n=== [3/7] GPU check ===")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected! Turn on GPU accelerator in Settings.")
    major, minor = torch.cuda.get_device_capability(0)
    if major < 7:
        raise RuntimeError(
            f"GPU compute capability {major}.{minor} < 7.0 — this torch build needs "
            f"sm_70+. Switch Kaggle accelerator to 'GPU T4 x2' (P100 is sm_60, unsupported)."
        )
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU[{i}]: {name} - {vram:.1f}GB VRAM")
    subprocess.run(["nvidia-smi"])


def step4_resolve_monday():
    print("\n=== [4/7] resolve Monday (benign-only) pcap ===")
    from huggingface_hub import hf_hub_download

    dataset_path = None

    # 1) Manually-attached Kaggle dataset (fastest — no remote download needed)
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if "onday" in f and f.endswith(".pcap"):
                dataset_path = os.path.join(root, f)
                print("Monday pcap found (Kaggle input, no download needed):", dataset_path)
                break
        if dataset_path:
            break

    # 2) Already downloaded to /kaggle/working from a prior run in this session
    if not dataset_path:
        if os.path.exists(MONDAY_LOCAL_FALLBACK) and os.path.getsize(MONDAY_LOCAL_FALLBACK) > MONDAY_MIN_BYTES:
            dataset_path = MONDAY_LOCAL_FALLBACK
            print("Monday pcap already on local disk (skipping download):", dataset_path)

    # 3) Remote download from the public, non-gated HF dataset
    if not dataset_path:
        print(f"Downloading {MONDAY_HF_FILENAME} from {MONDAY_HF_REPO} (~10 GB, one-time)...")
        dataset_path = hf_hub_download(
            repo_id=MONDAY_HF_REPO,
            filename=MONDAY_HF_FILENAME,
            repo_type="dataset",
            local_dir="/kaggle/working/",
        )
        print("Downloaded:", dataset_path)

    base = os.path.basename(dataset_path)
    assert "onday" in base, "Resolved file is not the Monday benign capture!"
    assert "ednesday" not in base, "Refusing to train on Wednesday (attack/eval day)!"
    print(f"Training corpus: {dataset_path} ({os.path.getsize(dataset_path)/1e9:.2f} GB)")
    return dataset_path


def step5_smoke_test(dataset_path):
    print("\n=== [5/7] smoke test (mamba2, 20 steps, ~2 min) ===")
    run([sys.executable, "train_ab.py",
         "--backbone", "mamba2", "--train_pcap", dataset_path,
         "--steps", "20", "--seq_len", "128",
         "--out", SMOKE_OUT])
    print("Smoke test passed — wiring confirmed end-to-end.")


def step6_real_run(dataset_path):
    print("\n=== [6/7] real Mamba-2 training run ===")
    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)
    run([sys.executable, "train_ab.py",
         "--backbone", "mamba2", "--train_pcap", dataset_path,
         "--steps", "20000", "--seq_len", "512", "--batch_size", "32",
         "--out", TRAIN_OUT])
    size_mb = os.path.getsize(TRAIN_OUT) / 1e6
    print(f"\nDone. Checkpoint: {TRAIN_OUT} ({size_mb:.1f} MB)")


def step7_push_checkpoint_to_hf(ckpt_path):
    print("\n=== [7/7] push checkpoint to Hugging Face Hub ===")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = ""
    if not token:
        print("WARNING: no HF_TOKEN secret found — skipping push.")
        print("Add a Kaggle secret named HF_TOKEN (a Hugging Face token with write "
              f"access) to push automatically. Checkpoint remains at: {ckpt_path}")
        return

    from huggingface_hub import HfApi, login
    login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id=HF_CKPT_REPO, repo_type="model", exist_ok=True, private=False)

    remote_name = f"checkpoints/{os.path.basename(ckpt_path)}"
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=remote_name,
        repo_id=HF_CKPT_REPO,
        repo_type="model",
    )
    print(f"Pushed to HF Hub: https://huggingface.co/{HF_CKPT_REPO}/blob/main/{remote_name}")
    print("Next: evaluate_zero_day.py --backbone mamba2 --checkpoint_path", ckpt_path,
          "--score_agg topk --topk_frac 0.1  (compare AUC + detection@1%FPR vs the transformer)")


def main():
    step1_repo()
    step2_deps()
    step3_gpu_check()
    dataset_path = step4_resolve_monday()
    step5_smoke_test(dataset_path)
    step6_real_run(dataset_path)
    step7_push_checkpoint_to_hf(TRAIN_OUT)


if __name__ == "__main__":
    main()
