#!/usr/bin/env python3
"""
AI Kosh Cluster Diagnostic and Verification Script.
Checks CUDA status, local directory permissions, dataset presence, and executes
a 5-batch dry run of the NetworkBytePatcher model.

Usage:
  # On Cluster (CUDA required):
  python setup_and_verify.py --dataset_path ./data/cic-ids2017/cic_ids.pcap
  
  # Locally (MPS/CPU bypass):
  python setup_and_verify.py --dataset_path local_test.pcap --bypass_cuda_check
"""

import os
import sys
import argparse
import torch
from src.dataloader import get_pcap_dataloader
from src.model import NetworkBytePatcher

def parse_args():
    parser = argparse.ArgumentParser(description="AI Kosh Diagnostic & Setup Verification Script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="local_test.pcap",
        help="Path to the PCAP dataset file (default: local_test.pcap)"
    )
    parser.add_argument(
        "--bypass_cuda_check",
        action="store_true",
        default=False,
        help="Bypass the CUDA GPU requirement (used for local CPU/MPS debugging)"
    )
    return parser.parse_args()

def verify_system(args):
    print("==================================================")
    print("          AI KOSH SYSTEM DIAGNOSTICS REPORT       ")
    print("==================================================")
    
    # 1. Directory Pre-flight (Strict Writable check)
    current_dir = os.getcwd()
    if not os.access(current_dir, os.W_OK):
        print(f"ERROR: Current directory '{current_dir}' is NOT writable.", file=sys.stderr)
        sys.exit(1)
    print(f"Directory Status: Current working directory '{current_dir}' is WRITABLE. [OK]")

    # 2. Hardware Enforcement Checks
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available and not args.bypass_cuda_check:
        print("ERROR: CUDA GPU is not available on this machine.", file=sys.stderr)
        print("To run local debug mode on macOS/CPU, please pass the '--bypass_cuda_check' flag.", file=sys.stderr)
        raise RuntimeError("Production training requires an NVIDIA GPU with CUDA.")
        
    if cuda_available:
        # Check GPU compute capability to avoid CUDA error: no kernel image is available
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"ERROR: Detected GPU '{gpu_name}' with CUDA capability {major}.{minor}.", file=sys.stderr)
            print("The installed PyTorch version requires CUDA capability >= 7.0 (sm_70+).", file=sys.stderr)
            print("Tesla P100 (sm_60) is NOT compatible. Please use a newer GPU (like T4, A100, V100).", file=sys.stderr)
            raise RuntimeError("Unsupported GPU architecture (compute capability < 7.0).")
        
    # 3. Print System Report
    print(f"CUDA Available:   {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model:        {gpu_name}")
        print(f"CUDA Version:     {cuda_version}")
        print(f"Total VRAM:       {total_mem_gb:.2f} GB")
    else:
        print("WARNING: Running in Local Debug Mode. GPU training is NOT enabled.")
        
    # Set the target hardware device
    if cuda_available:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Selected Device:  {device}")
    print("==================================================\n")
    return device

def run_dry_run(device, dataset_path):
    print(f"Checking for dataset at: '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at: '{dataset_path}'", file=sys.stderr)
        print("Please download/place the target PCAP file first.", file=sys.stderr)
        sys.exit(1)
    print("Dataset Status: Found target PCAP file. [OK]")
    
    print("\nInitializing model dry-run execution...")
    # Instantiate lightweight defaults of the model
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2).to(device)
    model.eval()
    print("Model Status: NetworkBytePatcher initialized and loaded to device. [OK]")
    
    # Load Dataloader
    print("Dataloader Status: Creating streaming PCAP reader...")
    dataloader = get_pcap_dataloader(
        pcap_path=dataset_path,
        batch_size=32,
        num_workers=0,
        max_sequence_length=1024  # short length for fast validation pass
    )
    
    # Run exactly 5 dry run batches
    print("Executing 5-batch forward pass dry run...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= 5:
                break
                
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
            
            print(f"  Batch {step+1}/5 | Input Shape: {list(inputs.shape)} | Loss: {loss.item():.4f}")
            
    print("\n==================================================")
    print(" DRY RUN COMPLETED SUCCESSFULLY! SYSTEM READY.   ")
    print("==================================================")

if __name__ == "__main__":
    args = parse_args()
    device = verify_system(args)
    run_dry_run(device, args.dataset_path)
