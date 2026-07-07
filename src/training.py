import os
import sys
import json
import time
import datetime
import subprocess
import resource
import torch
import torch.nn as nn
from contextlib import nullcontext

def get_gpu_metrics():
    """
    Queries nvidia-smi for active GPU telemetry.
    Returns None if nvidia-smi is unavailable.
    """
    try:
        res = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu", 
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        line = res.stdout.strip().split("\n")[0]
        gpu_util, mem_util, total_mem, used_mem, temp = [float(x.strip()) for x in line.split(",")]
        return {
            "gpu_utilization_percent": gpu_util,
            "gpu_memory_utilization_percent": mem_util,
            "gpu_total_memory_mb": total_mem,
            "gpu_used_memory_mb": used_mem,
            "gpu_temperature_c": temp
        }
    except Exception:
        return None

def get_system_metrics():
    """
    Standard library CPU load and RAM RSS monitor fallback for local debug environments.
    """
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Max RSS is in bytes on macOS, and kilobytes on Linux
    if sys.platform == 'darwin':
        rss_mb = max_rss / (1024 * 1024)
    else:
        rss_mb = max_rss / 1024
        
    try:
        load1, _, _ = os.getloadavg()
    except Exception:
        load1 = 0.0
        
    return {
        "cpu_load_1min": load1,
        "ram_max_rss_mb": rss_mb
    }

def log_telemetry_atomic(step, epoch):
    """
    Logs hardware telemetry stats atomically to logs/hardware_metrics.json.
    """
    metrics_dir = "logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "hardware_metrics.json")
    
    gpu_metrics = get_gpu_metrics()
    sys_metrics = get_system_metrics()
    
    metrics_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "epoch": epoch,
        "metrics": gpu_metrics if gpu_metrics is not None else sys_metrics,
        "is_gpu": gpu_metrics is not None
    }
    
    existing_metrics = []
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except Exception:
            pass
            
    existing_metrics.append(metrics_entry)
    
    # Atomic write to prevent file corruption
    temp_path = metrics_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=4)
    os.replace(temp_path, metrics_path)

def train_patcher_on_kosh(model, dataloader, epochs=5, checkpoint_dir="./checkpoints", lr=1e-4):
    """
    Resilient training loop designed for the AI Kosh cluster (SLURM).
    Handles automatic checkpoint saving, restoration, PyTorch 2.x AMP mixed precision,
    GPU/CPU hardware telemetry logging, and metadata-enriched checkpoint audits.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize GradScaler strictly only for CUDA execution
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None
    
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "latest_patcher.pt")
    
    # Auto-resume logic if pre-empted
    if os.path.exists(checkpoint_path):
        print(f"Found active checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        if scaler is not None and 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        print(f"Resumed successfully from epoch {start_epoch}.")
    else:
        print("No active checkpoint found. Starting training from scratch.")

    model.train()
    
    # Identify dataset provenance hash
    dataset_hash = getattr(dataloader.dataset, "file_hash", "unknown")
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            steps = 0
            
            for step, byte_sequence in enumerate(dataloader):
                byte_sequence = byte_sequence.to(device)
                
                # Predict the next byte: slice inputs and targets
                inputs = byte_sequence[:, :-1]
                targets = byte_sequence[:, 1:]
                
                optimizer.zero_grad()
                
                try:
                    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
                except (RuntimeError, ValueError):
                    autocast_ctx = nullcontext()
                    
                with autocast_ctx:
                    logits = model(inputs)
                    loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
                
                # Backward pass and step using GradScaler for CUDA, or standard backward for CPU/MPS
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item()
                steps += 1
                
                # Log telemetry metrics every 100 iterations
                if step % 100 == 0:
                    log_telemetry_atomic(step, epoch)
                    print(f"Epoch [{epoch}/{epochs}] | Step {step} | Loss: {loss.item():.4f}")
                    
            # Calculate and print epoch average loss
            avg_loss = epoch_loss / steps if steps > 0 else 0.0
            print(f"Epoch [{epoch}/{epochs}] Complete | Average Loss: {avg_loss:.4f}")
            
            # Metadata-enriched checkpoint dictionary
            checkpoint_metadata = {
                "model_version": "1.0.0",
                "dataset_hash": dataset_hash,
                "epoch": epoch + 1
            }
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metadata': checkpoint_metadata
            }
            if scaler is not None:
                checkpoint_state['scaler_state'] = scaler.state_dict()
                
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint successfully secured at {checkpoint_path} for Epoch {epoch + 1} with metadata hash: {dataset_hash}.")
            
    except (KeyboardInterrupt, SystemExit):
        print("Training execution interrupted! Securing final checkpoint state before termination...")
        checkpoint_metadata = {
            "model_version": "1.0.0",
            "dataset_hash": dataset_hash,
            "epoch": start_epoch  # current epoch state
        }
        checkpoint_state = {
            'epoch': start_epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metadata': checkpoint_metadata
        }
        if scaler is not None:
            checkpoint_state['scaler_state'] = scaler.state_dict()
            
        torch.save(checkpoint_state, checkpoint_path)
        print("Termination checkpoint saved successfully.")
        
    print("Training job complete!")
    return model
