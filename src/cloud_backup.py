"""
src/cloud_backup.py
===================
Non-blocking checkpoint cloud backup for the AIKosh SLURM training environment.

Supports Hugging Face Hub (https://huggingface.co/docs/huggingface_hub).

Usage
-----
The HF_TOKEN and HF_REPO_ID environment variables must be set before launching
the sbatch job.  Example in sbatch_train.sh or your cluster shell:

    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
    export HF_REPO_ID="your-username/sovereign-byte-firewall"

If either variable is missing, the upload is silently skipped — training will
never fail because of an unavailable cloud backend.

The upload runs in a background daemon thread so it never blocks the training
loop or the SLURM wall-time countdown.
"""

import os
import threading


def _upload_worker(local_path: str, repo_id: str, token: str,
                   path_in_repo: str, commit_message: str) -> None:
    """Background thread target — uploads a single file to HF Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

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
              "Run: pip install huggingface_hub")
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
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("HF_REPO_ID", "").strip()

    if not token or not repo_id:
        # Silently skip — cloud backup is optional
        return

    if not os.path.isfile(local_path):
        print(f"[CloudBackup] File not found, skipping upload: {local_path}")
        return

    # Remote path inside the HF repo: checkpoints/epoch3_step15000_epoch.pt
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
