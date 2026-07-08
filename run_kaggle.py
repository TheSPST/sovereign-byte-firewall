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
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
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
        default=8192,
        help="Target sequence length (default: 8192)"
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
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face User Access Token for private repositories (default: HF_TOKEN env var)"
    )
    return parser.parse_args()

def handle_data_download(dataset_path, dataset_url, hf_token=None):
    """
    Downloads dataset from Hugging Face / web if not already present.
    Supports authenticated private downloads via huggingface_hub.
    """
    if os.path.exists(dataset_path):
        print(f"Dataset Status: Found local dataset at '{dataset_path}' [OK]")
        return
        
    if not dataset_url:
        print(f"ERROR: Dataset not found at '{dataset_path}' and no --dataset_url was provided.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Dataset Status: Dataset missing. Initiating download...")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Check if this is a Hugging Face URL and we should use huggingface_hub
    if "huggingface.co" in dataset_url:
        try:
            # Parse repo_id, filename and repo_type from URL
            # e.g., https://huggingface.co/datasets/username/repo/resolve/main/folder/file.pcap
            parts = dataset_url.split("/")
            if "datasets" in parts:
                idx = parts.index("datasets")
                repo_id = f"{parts[idx+1]}/{parts[idx+2]}"
                resolve_idx = parts.index("resolve")
                # Parts after main/
                filename = "/".join(parts[resolve_idx+2:])
                repo_type = "dataset"
            else:
                # Model repo
                # e.g., https://huggingface.co/username/repo/resolve/main/file.pcap
                resolve_idx = parts.index("resolve")
                repo_id = f"{parts[resolve_idx-2]}/{parts[resolve_idx-1]}"
                filename = "/".join(parts[resolve_idx+2:])
                repo_type = "model"
                
            print(f"Hugging Face Hub Download:")
            print(f"  -> Repo ID:   {repo_id}")
            print(f"  -> Filename:  {filename}")
            print(f"  -> Type:      {repo_type}")
            
            from huggingface_hub import hf_hub_download
            
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                token=hf_token,
                local_dir=os.path.dirname(dataset_path),
                local_dir_use_symlinks=False
            )
            
            # Rename if the returned filename does not match dataset_path exactly
            if downloaded_path != dataset_path and os.path.exists(downloaded_path):
                os.replace(downloaded_path, dataset_path)
            
            print(f"Download Complete: Saved to '{dataset_path}' [OK]")
            return
        except Exception as e:
            print(f"Warning: Hugging Face SDK download failed: {e}. Falling back to wget...")
            
    # Fallback to standard wget
    try:
        cmd = ["wget", "-O", dataset_path, dataset_url]
        if hf_token:
            cmd.insert(1, f"--header=Authorization: Bearer {hf_token}")
        subprocess.run(cmd, check=True)
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
        print("Please verify your Kaggle Notebook has 'GPU T4 x2' or 'GPU P100' accelerator turned on.", file=sys.stderr)
        raise RuntimeError("Production training requires an NVIDIA GPU with CUDA.")
        
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
    handle_data_download(args.dataset_path, args.dataset_url, args.hf_token)
    
    # 3. Initialize model and dataloader
    print(f"\nInitializing DataLoader for PCAP at: '{args.dataset_path}'...")
    dataloader = get_pcap_dataloader(
        pcap_path=args.dataset_path,
        batch_size=batch_size,
        num_workers=0,
        max_sequence_length=max_sequence_length
    )
    
    print("Initializing NetworkBytePatcher (ultra-lightweight configuration)...")
    model = NetworkBytePatcher(
        d_model=128, 
        nhead=4, 
        num_layers=2, 
        max_sequence_length=max_sequence_length
    )
    
    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training via torch.nn.DataParallel!")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    # 4. Initiate training
    print("\nStarting Kaggle Training session...")
    train_patcher_on_kosh(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoints_dir,
        lr=args.lr
    )
    print("\nKaggle training script finished successfully!")

if __name__ == "__main__":
    main()
