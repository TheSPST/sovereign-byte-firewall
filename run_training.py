#!/usr/bin/env python3
"""
AI Kosh Cluster Training Orchestration Entrypoint.
Pre-flights directories, ensures CUDA compliance, initializes model/dataloader, 
and initiates the training & checkpointing loop.

Usage:
  # On Cluster (CUDA required):
  python run_training.py --dataset_path ./data/cic-ids2017/cic_ids.pcap --epochs 5 --batch_size 32

  # Locally (MPS/CPU bypass):
  python run_training.py --dataset_path local_test.pcap --epochs 2 --bypass_cuda_check
"""

import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="AI Kosh Training Orchestrator")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="local_test.pcap",
        help="Path to the PCAP dataset file (default: local_test.pcap)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for the PCAP streaming dataloader (default: 512)"
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
        default=True,
        help="Use Focal Loss instead of standard Cross Entropy (default: True)"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for Focal Loss (default: 2.0)"
    )
    return parser.parse_args()

def check_preflight_and_device(args):
    print("==================================================")
    print("      AI KOSH PRE-FLIGHT DIRECTORY & HARDWARE CHECK ")
    print("==================================================")
    
    # 1. Check current workspace directory permissions
    current_dir = os.getcwd()
    if not os.access(current_dir, os.W_OK):
        print(f"ERROR: Current working directory '{current_dir}' is NOT writable.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Check checkpoints folder write access (crucial for cluster ephemeral storage policy)
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not os.access(checkpoint_dir, os.W_OK):
        print(f"ERROR: Checkpoints directory '{checkpoint_dir}' is NOT writable.", file=sys.stderr)
        sys.exit(1)
    print("Directory Permissions: Working directory & checkpoints folder are WRITABLE. [OK]")
        
    # 3. Hardware Enforcement Check
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.bypass_cuda_check:
        print("ERROR: CUDA GPU is not available on this machine.", file=sys.stderr)
        print("To run local debug mode on macOS/CPU, please pass the '--bypass_cuda_check' flag.", file=sys.stderr)
        raise RuntimeError("Production training requires an NVIDIA GPU with CUDA.")
        
    # Set the target hardware device
    if cuda_available:
        device = torch.device("cuda")
        print(f"Hardware Status: CUDA GPU detected ({torch.cuda.get_device_name(0)}). [OK]")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Status: WARNING: Running in Local Debug Mode. MPS selected.")
    else:
        device = torch.device("cpu")
        print("Hardware Status: WARNING: Running in Local Debug Mode. CPU selected.")
        
    print(f"Target Device:   {device}")
    print("==================================================\n")
    return device, checkpoint_dir

def main():
    args = parse_args()
    device, checkpoint_dir = check_preflight_and_device(args)
    
    # Verify dataset existence
    if not os.path.exists(args.dataset_path):
        print(f"ERROR: Target PCAP dataset not found at: '{args.dataset_path}'", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading PCAP dataset: '{args.dataset_path}'...")
    dataloader = get_pcap_dataloader(
        pcap_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=0,  # Single-process sequential stream is robust
        max_sequence_length=args.max_sequence_length
    )
    
    print("Initializing NetworkBytePatcher model (ultra-lightweight configuration)...")
    # Instantiates the model using default ultra-lightweight configuration
    # (num_layers=2, d_model=128, nhead=4) for <1ms inference footprint per packet chunk
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2).to(device)
    
    print("\nStarting training orchestrator loop...")
    train_patcher_on_kosh(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        lr=args.lr,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    
    print("\nOrchestrated training job completed successfully!")

if __name__ == "__main__":
    main()
