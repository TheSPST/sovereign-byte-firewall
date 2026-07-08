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

# Read custom dataset path and epochs from arguments (fall back to defaults)
DATASET_PATH=${1:-"./data/cic-ids2017/cic_ids.pcap"}
EPOCHS=${2:-10}

# Activate local virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# 1. Pre-flight diagnostics check (gatekeeper)
echo "Running system verification check on: $DATASET_PATH"
python setup_and_verify.py --dataset_path "$DATASET_PATH"
if [ $? -ne 0 ]; then
    echo "Verification check failed. Aborting training job."
    exit 1
fi

# 2. Start training run in background so bash can trap signals
echo "Launching training orchestrator on $DATASET_PATH for $EPOCHS epochs..."
python run_training.py --dataset_path "$DATASET_PATH" --epochs "$EPOCHS" --use_focal_loss True --focal_gamma 2.0 &
PYTHON_PID=$!

# Wait for the python job to finish
wait "$PYTHON_PID"

echo "Job execution completed successfully."
