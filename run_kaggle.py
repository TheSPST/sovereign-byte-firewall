#!/usr/bin/env python3
"""
Kaggle Notebook Training Orchestrator.
Adapts execution path structures, automatically downloads the training datasets
from Hugging Face, and scales sequence lengths/batch sizes to prevent VRAM OOMs
on P100/T4 instances.

Usage:
  # On Kaggle with GPU (downloads dataset automatically):
  python run_kaggle.py --dataset_url "https://huggingface.co/datasets/your-username/your-pcap-dataset/resolve/main/dataset.pcap"

  # Locally (MPS/CPU bypass):
  python run_kaggle.py --dataset_path local_test.pcap --epochs 1 --bypass_cuda_check
"""

import os
import sys
import argparse
import subprocess
import torch
from src.dataloader import get_pcap_dataloader
from src.model import NetworkBytePatcher
from src.training import train_patcher_on_kosh

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    # 1. Detect if running inside Kaggle environment
    is_kaggle = os.path.exists("/kaggle")
    
    default_dataset = "/kaggle/working/data/dataset.pcap" if is_kaggle else "local_test.pcap"
    default_checkpoints = "/kaggle/working/checkpoints" if is_kaggle else "./checkpoints"
    
    parser = argparse.ArgumentParser(description="Kaggle Notebook Training Orchestrator")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset,
        help=f"Path to local PCAP dataset file (default: {default_dataset})"
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        default=None,
        help="Hugging Face download URL for the PCAP dataset (if not locally present)"
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="Path to a SEPARATE held-out benign PCAP, never trained on, used to compute "
             "validation loss after every epoch (default: None — no validation tracking)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1 — project evidence shows repeating "
             "epochs over the same file causes real regression: held-out detection "
             "dropped 32%% -> ~6%% once training re-saw the same data)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Target batch size for training (default: 32)"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Target sequence length (default: 512)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=default_checkpoints,
        help=f"Directory to save pt checkpoints (default: {default_checkpoints})"
    )
    parser.add_argument(
        "--bypass_cuda_check",
        action="store_true",
        default=False,
        help="Bypass the CUDA GPU requirement (used for local CPU/MPS debugging)"
    )
    parser.add_argument(
        "--use_focal_loss",
        type=str2bool,
        default=False,
        help="Use Focal Loss instead of standard Cross Entropy (default: False — "
             "FocalLoss was tested and produced strictly worse convergence on this "
             "task; every usable checkpoint came from plain CrossEntropy)"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for Focal Loss (default: 2.0)"
    )
    parser.add_argument(
        "--label_anomalies",
        type=str2bool,
        default=False,
        help="Write the per-packet anomaly side-channel CSV during training (default: "
             "False — costs a full scapy parse per packet in the dataloader hot path "
             "and the loss never consumes it)"
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Explicit OneCycleLR total step count (streaming dataset length is an "
             "estimate on first pass; pass the known window count for an exact schedule)"
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-4,
        help="OneCycleLR PEAK learning rate (default 1e-4). The old hardcoded 5e-4 "
             "was diagnosed too high: held-out detection peaked during warmup then "
             "collapsed to random as sustained high LR wrecked the tiny model."
    )
    parser.add_argument(
        "--allow_resume",
        action="store_true",
        default=False,
        help="Allow auto-resume from an existing checkpoints/latest_patcher.pt. "
             "OFF by default: the masking scheme changed (TLS continuation, header "
             "stochastic fields, QUIC), so resuming a checkpoint trained under the OLD "
             "preprocessing silently mixes incompatible data distributions."
    )
    return parser.parse_args()

def handle_data_download(dataset_path, dataset_url):
    """
    Downloads dataset from Hugging Face / web if not already present.
    """
    if os.path.exists(dataset_path):
        print(f"Dataset Status: Found local dataset at '{dataset_path}' [OK]")
        return
        
    if not dataset_url:
        print(f"ERROR: Dataset not found at '{dataset_path}' and no --dataset_url was provided.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Dataset Status: Dataset missing. Downloading from: '{dataset_url}'...")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Execute streaming wget download with progress logging
    try:
        subprocess.run(["wget", "-O", dataset_path, dataset_url], check=True)
        print(f"Download Complete: Saved dataset to '{dataset_path}' [OK]")
    except Exception as e:
        print(f"ERROR: Failed to download dataset using wget: {e}", file=sys.stderr)
        sys.exit(1)

def configure_hardware_limits(args):
    print("==================================================")
    print("      KAGGLE HARDWARE & RESOURCE PRE-FLIGHT CHECK ")
    print("==================================================")
    
    # 1. Directory Permission checks
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if not os.access(args.checkpoints_dir, os.W_OK):
        print(f"ERROR: Checkpoints directory '{args.checkpoints_dir}' is NOT writable.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Checkpoints Directory: '{args.checkpoints_dir}' is WRITABLE. [OK]")
    
    # 2. CUDA Hardware Verification
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.bypass_cuda_check:
        print("ERROR: CUDA GPU is not available on this machine.", file=sys.stderr)
        print("Please verify your Kaggle Notebook has the 'GPU T4 x2' accelerator turned on.", file=sys.stderr)
        raise RuntimeError("Production training requires an NVIDIA GPU with CUDA.")
        
    if cuda_available:
        # Check GPU compute capability to avoid CUDA error: no kernel image is available
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"ERROR: Detected GPU '{gpu_name}' with CUDA capability {major}.{minor}.", file=sys.stderr)
            print("The pre-installed PyTorch version on Kaggle requires CUDA capability >= 7.0 (sm_70+).", file=sys.stderr)
            print("Tesla P100 (sm_60) is NOT compatible. Please switch your Kaggle accelerator to 'GPU T4 x2'.", file=sys.stderr)
            raise RuntimeError("Unsupported GPU architecture (compute capability < 7.0).")

    # Default parameters before scaling checks
    batch_size = args.batch_size
    max_sequence_length = args.max_sequence_length
    
    if cuda_available:
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        print(f"CUDA GPUs Detected:  {device_count}")
        
        primary_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        primary_vram_gb = primary_vram_bytes / (1024 ** 3)
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  -> GPU [{i}]: {name} ({vram:.2f} GB VRAM)")
            
        # 3. Hardware Scaling Auto-Tuner: Check if we are running on standard 16GB GPUs (T4 or P100)
        if primary_vram_gb <= 16.5:
            # P100/T4 OOM Prevention Fallback logic
            print("\nWARNING: Detected GPU with <= 16GB VRAM (T4 / P100 instance).")
            print("To prevent Out-Of-Memory (OOM) failures under quadratic attention scaling,")
            print("dynamic hyperparameters are auto-tuned:")
            
            if max_sequence_length > 2048:
                max_sequence_length = 2048
                print(f"  -> Capped max_sequence_length to: {max_sequence_length} bytes")
                
            # Allow up to 16 samples per GPU in batch
            max_batch_per_gpu = 16
            target_batch_size = max_batch_per_gpu * device_count
            if batch_size > target_batch_size:
                batch_size = target_batch_size
                print(f"  -> Capped batch_size to:           {batch_size} (max {max_batch_per_gpu} per GPU)")
    else:
        # Bypassed local fallback selection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Local Hardware: macOS Apple Silicon (MPS).")
        else:
            device = torch.device("cpu")
            print("Local Hardware: CPU fallback.")
            
        # Lower sequences locally to prevent memory pressure
        if max_sequence_length > 1024:
            max_sequence_length = 1024
            print(f"  -> Local debug mode: capping sequence length to: {max_sequence_length}")
        if batch_size > 4:
            batch_size = 4
            print(f"  -> Local debug mode: capping batch size to:       {batch_size}")

    print(f"Active Device:       {device}")
    print("==================================================\n")
    return device, batch_size, max_sequence_length

def main():
    args = parse_args()
    
    # 1. Pre-flight checks and scaling configuration
    device, batch_size, max_sequence_length = configure_hardware_limits(args)
    
    # 2. Automated data downloader
    handle_data_download(args.dataset_path, args.dataset_url)

    # 2b. Fresh-start guard: old-masking checkpoints must not be resumed silently.
    stale_ckpt = os.path.join(args.checkpoints_dir, "latest_patcher.pt")
    if os.path.exists(stale_ckpt) and not args.allow_resume:
        print(f"ERROR: found existing checkpoint '{stale_ckpt}'.\n"
              f"The masking scheme changed (TLS continuation, stochastic header fields, "
              f"QUIC) — resuming a checkpoint trained under the OLD preprocessing mixes "
              f"incompatible data distributions. Delete/move it for a fresh run, or pass "
              f"--allow_resume ONLY if it was trained with the current masking.", file=sys.stderr)
        sys.exit(1)

    # 3. Initialize model and dataloader
    print(f"\nInitializing DataLoader for PCAP at: '{args.dataset_path}'...")
    dataloader = get_pcap_dataloader(
        pcap_path=args.dataset_path,
        batch_size=batch_size,
        # 1 background worker: sequential stream (TLS mask state valid) but pcap
        # parse/mask/window runs off the training process so batches prefetch.
        num_workers=1,
        max_sequence_length=max_sequence_length,
        label_anomalies=args.label_anomalies
    )
    
    print("Initializing NetworkBytePatcher (ultra-lightweight configuration)...")
    model = NetworkBytePatcher(
        d_model=128,
        nhead=4,
        num_layers=2,
        max_sequence_length=max_sequence_length
    )

    model = model.to(device)

    # 3b. Optional held-out validation dataloader (never trained on)
    val_dataloader = None
    if args.val_dataset_path:
        if not os.path.exists(args.val_dataset_path):
            print(f"WARNING: --val_dataset_path '{args.val_dataset_path}' not found. "
                  f"Continuing without validation tracking.", file=sys.stderr)
        else:
            print(f"Initializing VALIDATION DataLoader for PCAP at: '{args.val_dataset_path}'...")
            val_dataloader = get_pcap_dataloader(
                pcap_path=args.val_dataset_path,
                batch_size=batch_size,
                num_workers=1,
                max_sequence_length=max_sequence_length,
                label_anomalies=False
            )

    # 4. Initiate training
    print("\nStarting Kaggle Training session...")
    train_patcher_on_kosh(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoints_dir,
        lr=args.lr,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        val_dataloader=val_dataloader,
        total_steps_override=args.total_steps,
        max_lr=args.max_lr
    )
    print("\nKaggle training script finished successfully!")

if __name__ == "__main__":
    main()
