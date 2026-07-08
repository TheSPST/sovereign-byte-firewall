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

def train_patcher_on_kosh(model, dataloader, epochs=5, checkpoint_dir="./checkpoints", lr=1e-4, use_focal_loss=True, focal_gamma=2.0):
    """
    Resilient training loop designed for the AI Kosh cluster (SLURM).
    Handles automatic checkpoint saving, restoration, PyTorch 2.x AMP mixed precision,
    GPU/CPU hardware telemetry logging, and metadata-enriched checkpoint audits.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Calculate total steps for OneCycleLR scheduler
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    if use_focal_loss:
        from src.losses import FocalLoss
        # FIX W5: ignore_index=-1 matches the sentinel written by the dataloader for
        # trailing-window padding (torch.full(..., -1)). This ensures padded positions
        # are excluded from every gradient update across the entire training run.
        criterion = FocalLoss(gamma=focal_gamma, ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Disable GradScaler for bfloat16 training (safest A100 pipeline)
    scaler = None
    
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "latest_patcher.pt")
    
    # Auto-resume logic if pre-empted
    if os.path.exists(checkpoint_path):
        print(f"Found active checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint['model_state']
        # Dynamically adjust for DataParallel module. prefix mismatch
        is_dp = isinstance(model, torch.nn.DataParallel)
        has_prefix = any(k.startswith('module.') for k in state_dict.keys())
        
        if not is_dp and has_prefix:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif is_dp and not has_prefix:
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
            
        # Dynamically handle size mismatches for pos_embedding.weight (e.g. 2048 -> 512)
        pos_key = 'module.pos_embedding.weight' if (is_dp or has_prefix) else 'pos_embedding.weight'
        if pos_key in state_dict:
            checkpoint_pos_weight = state_dict[pos_key]
            model_pos_embedding = model.module.pos_embedding if isinstance(model, torch.nn.DataParallel) else model.pos_embedding
            target_pos_shape = model_pos_embedding.weight.shape
            if checkpoint_pos_weight.shape != target_pos_shape:
                print(f"Adapting pos_embedding.weight shape from {checkpoint_pos_weight.shape} to {target_pos_shape}...")
                new_weight = model_pos_embedding.weight.clone().detach()
                min_len = min(checkpoint_pos_weight.shape[0], target_pos_shape[0])
                new_weight[:min_len, :] = checkpoint_pos_weight[:min_len, :]
                state_dict[pos_key] = new_weight

        # Align optimizer state shapes with current model parameters to avoid size mismatch on resume
        model_params = list(model.parameters())
        param_ids = checkpoint['optimizer_state']['param_groups'][0]['params']
        for idx, param_id in enumerate(param_ids):
            if param_id in checkpoint['optimizer_state']['state']:
                state_entry = checkpoint['optimizer_state']['state'][param_id]
                target_shape = model_params[idx].shape
                for state_key in ['exp_avg', 'exp_avg_sq']:
                    if state_key in state_entry:
                        old_tensor = state_entry[state_key]
                        if old_tensor.shape != target_shape:
                            print(f"Resizing optimizer state '{state_key}' for param {idx} from {old_tensor.shape} to {target_shape}...")
                            new_tensor = torch.zeros(target_shape, dtype=old_tensor.dtype, device=old_tensor.device)
                            min_dim0 = min(old_tensor.shape[0], target_shape[0])
                            new_tensor[:min_dim0, :] = old_tensor[:min_dim0, :]
                            state_entry[state_key] = new_tensor

        # Preserve scheduler-related keys in optimizer param_groups to prevent KeyError: 'max_lr'
        saved_keys = []
        for group in optimizer.param_groups:
            saved_keys.append({k: v for k, v in group.items() if k not in ['params']})

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Restore preserved keys
        for idx, group in enumerate(optimizer.param_groups):
            for k, v in saved_keys[idx].items():
                group[k] = v
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
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
                    if use_focal_loss:
                        loss = criterion(logits, targets)
                    else:
                        loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
                
                # Backward pass and step using GradScaler for CUDA, or standard backward for CPU/MPS
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()
                    
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
            
            # Unpack model state dict cleanly to remove DataParallel prefix
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
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
        # Unpack model state dict cleanly to remove DataParallel prefix
        model_state_interrupted = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        
        checkpoint_state = {
            'epoch': start_epoch,
            'model_state': model_state_interrupted,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'metadata': checkpoint_metadata
        }
        if scaler is not None:
            checkpoint_state['scaler_state'] = scaler.state_dict()
            
        torch.save(checkpoint_state, checkpoint_path)
        print("Termination checkpoint saved successfully.")
        
    print("Training job complete!")
    return model
