#!/bin/bash
#SBATCH --job-name=sovereign_firewall
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --mem=32G

# Ensure log directory exists
mkdir -p logs

# Navigate to the job submission directory
cd "${SLURM_SUBMIT_DIR:-.}"
echo "Current working directory: $(pwd)"

# Trap SLURM termination signals (like pre-emption or time limits)
# SLURM sends SIGTERM (15) to notify the job it is about to be terminated.
trap 'handle_terminate' SIGTERM SIGINT

handle_terminate() {
    echo "=================================================="
    echo " WARNING: SLURM termination signal (SIGTERM) received!"
    echo " Triggering graceful checkpoint save in training script..."
    echo "=================================================="
    
    # Send SIGINT to the running python job to trigger KeyboardInterrupt & save state
    if [ ! -z "$PYTHON_PID" ]; then
        kill -s SIGINT "$PYTHON_PID"
        wait "$PYTHON_PID"
    fi
    exit 143 # Standard SLURM termination exit code
}

# Load cluster python module (e.g. CDAC custom environment setup if needed)
# Example: module load cuda/12.1 anaconda3
# Feel free to adjust these module loads for the specific AI Kosh node layout

# Read custom dataset path, epochs, and optional held-out validation path from arguments
# Usage: sbatch sbatch_train.sh <train_pcap> <epochs> [val_pcap]
#
# IMPORTANT: the default DATASET_PATH below is a placeholder / historically a
# 0-byte empty file in this project. ALWAYS pass the real training pcap
# explicitly as $1 (e.g. the Monday-clean split), do not rely on the default.
#
# IMPORTANT: EPOCHS defaults to 1, not more. We have direct evidence from this
# project that repeating epochs over the same training file causes real
# regression (zero-day detection rate dropped from 32% to ~6% once training
# started re-seeing the same Monday file a 2nd/3rd time) — more epochs is not
# automatically better here. Monitor pushed checkpoints via evaluate_zero_day.py
# during the run and kill the job early if detection stops improving, rather
# than trusting it to run unattended for the full epoch count.
DATASET_PATH=${1:-"./data/cic-ids2017/cic_ids.pcap"}
EPOCHS=${2:-1}
VAL_DATASET_PATH=${3:-""}

# ─── Hugging Face Cloud Backup ────────────────────────────────────────────────
# Checkpoints are automatically pushed to your HF Hub repo after every epoch.
#
# SETUP (run these ONCE on every machine / cluster node, not inside this script):
#
#   Step 1 — Install the HF CLI:
#     pip install "huggingface_hub>=0.25"
#
#   Step 2 — Login (saves token to ~/.cache/huggingface/token):
#     hf auth login
#     # Paste your token from: https://huggingface.co/settings/tokens
#     # Choose "write" permission.
#
#   Step 3 — Set your repo (edit the line below OR export before sbatch):
#     export HF_REPO_ID="TheSPST/sovereign-byte-firewall"
#
# Alternative — pass token directly via env var (for headless SLURM nodes):
#   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
#   export HF_REPO_ID="TheSPST/sovereign-byte-firewall"
#
# Leave HF_REPO_ID blank to disable cloud backup entirely (training still runs).
#
# IMPORTANT: this defaults to a NEW repo name, deliberately different from
# spst01/sovereign-byte-firewall and spst01/sovereign-byte-firewall-monday
# (both already in use from earlier Kaggle runs). A fresh run starting from
# global_step=0 will re-create filenames like gs5000/gs10000/etc, which would
# collide with and be easy to confuse with unrelated old checkpoints of the
# same name if pushed into a shared repo. Override with
# `export HF_REPO_ID=...` before calling sbatch if you want a different name.
export HF_REPO_ID="${HF_REPO_ID:-spst01/sovereign-byte-firewall-aikosh}"  # ← your HF repo
export HF_TOKEN="${HF_TOKEN:-}"             # optional — hf auth login is preferred

# Pre-flight: warn if no auth method is found
if [ -n "$HF_REPO_ID" ]; then
    if [ -z "$HF_TOKEN" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
        echo "[CloudBackup] WARNING: HF_REPO_ID is set but no token found."
        echo "[CloudBackup]   Run: hf auth login  OR  export HF_TOKEN=hf_..."
        echo "[CloudBackup]   Uploads will be skipped unless a token is available."
    else
        echo "[CloudBackup] HF cloud backup enabled → ${HF_REPO_ID}"
    fi
fi
# ──────────────────────────────────────────────────────────────────────────────

# Activate local virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Auto-extract if dataset is gzip compressed
if [[ "$DATASET_PATH" == *.gz ]]; then
    echo "Compressed dataset detected: $DATASET_PATH"
    EXTRACTED_PCAP="${DATASET_PATH%.gz}"
    if [ ! -f "$EXTRACTED_PCAP" ]; then
        echo "Extracting dataset to $EXTRACTED_PCAP..."
        gunzip -c "$DATASET_PATH" > "$EXTRACTED_PCAP"
    else
        echo "Extracted dataset already present at $EXTRACTED_PCAP. Skipping decompression."
    fi
    DATASET_PATH="$EXTRACTED_PCAP"
fi

# 1. Pre-flight diagnostics check (gatekeeper)
echo "Running system verification check on: $DATASET_PATH"
python setup_and_verify.py --dataset_path "$DATASET_PATH"
if [ $? -ne 0 ]; then
    echo "Verification check failed. Aborting training job."
    exit 1
fi

# 2. Start training run in background so bash can trap signals
#
# IMPORTANT: --use_focal_loss is False here, not True. FocalLoss was tested
# extensively earlier in this project and found to be both theoretically
# wrong for this near-balanced 256-byte-vocab next-token task (it suppresses
# learning from common/easy tokens, which here means suppressing the model's
# ability to learn genuinely "normal" byte patterns) and numerically fragile
# under fp16 (GradScaler silently skipping most optimizer steps). Plain
# CrossEntropy (with ignore_index=-1 for the -1 padding sentinel) is the
# proven-better config that produced every usable checkpoint in this project
# (gs590000 onward). Do not switch this back to True without re-deriving why.
#
# TOTAL_STEPS (optional env var): exact OneCycleLR step count. On a FIRST pass
# over a file the dataset length is only a file-size estimate, so the LR
# schedule is approximate (run_training pads it 1.5x to avoid a premature LR
# collapse). If you know the real windows-per-epoch count from a previous run
# (e.g. the Kaggle run over the same Monday file), export TOTAL_STEPS before
# sbatch for an exact schedule:  export TOTAL_STEPS=900000
TOTAL_STEPS="${TOTAL_STEPS:-}"
TOTAL_STEPS_FLAG=""
if [ -n "$TOTAL_STEPS" ]; then
    echo "Exact OneCycleLR schedule requested -> total_steps=$TOTAL_STEPS"
    TOTAL_STEPS_FLAG="--total_steps $TOTAL_STEPS"
fi

echo "Launching training orchestrator on $DATASET_PATH for $EPOCHS epochs..."
if [ -n "$VAL_DATASET_PATH" ]; then
    echo "Validation tracking enabled -> $VAL_DATASET_PATH"
    python run_training.py --dataset_path "$DATASET_PATH" --epochs "$EPOCHS" --use_focal_loss False --val_dataset_path "$VAL_DATASET_PATH" $TOTAL_STEPS_FLAG &
else
    python run_training.py --dataset_path "$DATASET_PATH" --epochs "$EPOCHS" --use_focal_loss False $TOTAL_STEPS_FLAG &
fi
PYTHON_PID=$!

# Wait for the python job to finish
wait "$PYTHON_PID"

echo "Job execution completed successfully."
