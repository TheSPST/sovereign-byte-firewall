"""
src/cloud_backup.py
===================
Non-blocking checkpoint cloud backup for the AIKosh SLURM training environment.

Supports Hugging Face Hub (https://huggingface.co/docs/huggingface_hub).

Authentication (choose ONE method)
-----------------------------------
Method 1 — HF CLI login (recommended, persists across sessions):
    hf auth login
    # Paste your token from https://huggingface.co/settings/tokens
    # Once done, the token is cached at ~/.cache/huggingface/token
    # and picked up automatically by this module.

Method 2 — Environment variable (good for SLURM jobs):
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

Method 3 — HF_REPO_ID only (required regardless of auth method):
    export HF_REPO_ID="your-username/sovereign-byte-firewall"
    # Or set it inside sbatch_train.sh

Token resolution order (huggingface_hub builtin):
    1. HF_TOKEN env var
    2. ~/.cache/huggingface/token  (written by `hf auth login`)
    3. HUGGINGFACE_HUB_TOKEN env var  (legacy alias)

If no token is found OR HF_REPO_ID is not set, the upload is silently
skipped — training will never fail because of an unavailable cloud backend.

The upload runs in a background daemon thread so it never blocks the
training loop or the SLURM wall-time countdown.
"""

import os
import threading


def _resolve_token() -> str:
    """
    Resolve the HF token using the same priority order as `huggingface_hub`.
    Reads the CLI cache file (~/.cache/huggingface/token) as a fallback so
    tokens stored by `hf auth login` work without setting any env vars.
    """
    # 1. Explicit env var (highest priority — good for SLURM sbatch env)
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    # 2. Legacy env var alias
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    if token:
        return token

    # 3. Token cached by `hf auth login` / `huggingface-cli login`
    try:
        from huggingface_hub import HfFolder          # type: ignore
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    # 4. Read the CLI token file directly (handles older huggingface_hub versions)
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_file):
        try:
            with open(token_file, "r") as f:
                cached = f.read().strip()
            if cached:
                return cached
        except Exception:
            pass

    return ""


def _upload_worker(local_path: str, repo_id: str, token: str,
                   path_in_repo: str, commit_message: str) -> None:
    """Background thread target — uploads a single file to HF Hub."""
    try:
        from huggingface_hub import HfApi             # type: ignore
        # Pass token explicitly; HfApi also auto-reads the CLI cache
        api = HfApi(token=token if token else None)

        # Ensure the model repository exists (private by default)
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=True, exist_ok=True)

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"[CloudBackup] ✓ Uploaded '{path_in_repo}' → hf:{repo_id}")
    except ImportError:
        print("[CloudBackup] huggingface_hub not installed — skipping upload. "
              "Run: pip install 'huggingface_hub>=0.25'")
    except Exception as exc:
        # Never propagate — a backup failure must not crash the training job
        print(f"[CloudBackup] ✗ Upload failed (non-fatal): {exc}")


def push_checkpoint(
    local_path: str,
    epoch: int,
    global_step: int,
    checkpoint_type: str = "epoch",   # "epoch" | "mid_epoch" | "interrupt"
) -> None:
    """
    Asynchronously push a checkpoint file to Hugging Face Hub.

    Parameters
    ----------
    local_path      : Absolute or relative path to the .pt file on disk.
    epoch           : Epoch index (used in the remote filename).
    global_step     : Global training step (used in the remote filename).
    checkpoint_type : Label for the remote filename and commit message.
    """
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        # Silently skip — cloud backup is optional
        return

    token = _resolve_token()

    if not token:
        print("[CloudBackup] No HF token found. Run `hf auth login` or set "
              "HF_TOKEN env var. Skipping upload.")
        return

    if not os.path.isfile(local_path):
        print(f"[CloudBackup] File not found, skipping upload: {local_path}")
        return

    # Remote path inside the HF repo: checkpoints/latest_patcher_ep1_gs5000_epoch.pt
    basename = os.path.basename(local_path)
    root, ext = os.path.splitext(basename)
    remote_name = f"{root}_ep{epoch}_gs{global_step}_{checkpoint_type}{ext}"
    path_in_repo = f"checkpoints/{remote_name}"

    commit_msg = (
        f"[auto] {checkpoint_type} checkpoint | epoch={epoch} | step={global_step}"
    )

    thread = threading.Thread(
        target=_upload_worker,
        args=(local_path, repo_id, token, path_in_repo, commit_msg),
        daemon=True,   # killed automatically if the main process exits
        name=f"hf-upload-ep{epoch}-gs{global_step}",
    )
    thread.start()
    print(f"[CloudBackup] Background upload started → hf:{repo_id}/{path_in_repo}")
