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
from src.cloud_backup import push_checkpoint

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

from accelerate import Accelerator

def train_patcher_on_kosh(model, dataloader, epochs=5, checkpoint_dir="./checkpoints", lr=1e-4, use_focal_loss=True, focal_gamma=2.0, checkpoint_interval_steps=5000):
    """
    Unified training loop for Kaggle (T4/P100) and AI Kosh (A100).
    Powered by Hugging Face Accelerate for maximum multi-GPU throughput.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Auto-detect environment for precision mapping
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    precision = "fp16" if is_kaggle else "bf16"
    print(f"[ENV] Accelerate initialized with mixed_precision='{precision}'")
    
    # Initialize Accelerator (Replaces DataParallel, GradScaler, and Autocast)
    accelerator = Accelerator(mixed_precision=precision)
    device = accelerator.device
    
    # Optional: Compile model for a free 20% speedup on PyTorch 2.0+
    # model = torch.compile(model) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
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
        criterion = FocalLoss(gamma=focal_gamma, ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # --- ACCELERATE PREPARE ---
    # This automatically distributes the model across GPUs and prepares the dataloader
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    start_epoch = 0
    global_step = 0 
    checkpoint_path = os.path.join(checkpoint_dir, "latest_patcher.pt")
    dataset_hash = getattr(dataloader.dataset, "file_hash", "unknown")
    
    # Auto-resume logic if pre-empted by SLURM
    if os.path.exists(checkpoint_path):
        print(f"Found active checkpoint. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # When restoring weights under Accelerate, we load into the unwrapped model
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = checkpoint['model_state']
        
        # Clean 'module.' prefix if present
        has_prefix = any(k.startswith('module.') for k in state_dict.keys())
        if has_prefix:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        # Dynamically handle size mismatches for pos_embedding.weight
        pos_key = 'pos_embedding.weight'
        if pos_key in state_dict:
            checkpoint_pos_weight = state_dict[pos_key]
            model_pos_embedding = getattr(unwrapped_model, 'pos_embedding', None)
            if model_pos_embedding is not None:
                target_pos_shape = model_pos_embedding.weight.shape
                if checkpoint_pos_weight.shape != target_pos_shape:
                    print(f"Adapting pos_embedding.weight shape from {checkpoint_pos_weight.shape} to {target_pos_shape}...")
                    new_weight = model_pos_embedding.weight.clone().detach()
                    min_len = min(checkpoint_pos_weight.shape[0], target_pos_shape[0])
                    new_weight[:min_len, :] = checkpoint_pos_weight[:min_len, :]
                    state_dict[pos_key] = new_weight

        unwrapped_model.load_state_dict(state_dict)
        
        if 'optimizer_state' in checkpoint:
            # Align optimizer state shapes with current model parameters
            try:
                model_params = list(unwrapped_model.parameters())
                if 'param_groups' in checkpoint['optimizer_state']:
                    param_ids = checkpoint['optimizer_state']['param_groups'][0]['params']
                    for idx, param_id in enumerate(param_ids):
                        if idx < len(model_params) and param_id in checkpoint['optimizer_state']['state']:
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
            except Exception as e:
                print(f"Warning: could not adapt optimizer state shapes: {e}")
            
            # Preserve scheduler-related keys in optimizer param_groups to prevent KeyError: 'max_lr'
            saved_keys = []
            for group in optimizer.param_groups:
                saved_keys.append({k: v for k, v in group.items() if k not in ['params']})
                
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            for idx, group in enumerate(optimizer.param_groups):
                for k, v in saved_keys[idx].items():
                    group[k] = v
                    
        if 'scheduler_state' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            
        print(f"Successfully resumed from epoch {start_epoch}, step {global_step}")
    else:
        print("No active checkpoint found. Starting training from scratch.")

    model.train()
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            steps = 0
            
            for step, byte_sequence in enumerate(dataloader):
                # No need to call .to(device), Accelerate's dataloader handles it!
                
                inputs = byte_sequence[:, :-1]
                targets = byte_sequence[:, 1:]
                
                optimizer.zero_grad()
                
                # No autocast context needed, Accelerate handles precision automatically
                logits = model(inputs)
                
                if use_focal_loss:
                    loss = criterion(logits, targets)
                else:
                    loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
                
                # Use accelerator.backward instead of loss.backward or scaler.scale
                accelerator.backward(loss)
                optimizer.step()
                
                # Unmask the native PyTorch scheduler to check its internal step counter
                underlying_scheduler = getattr(scheduler, "scheduler", scheduler)
                if underlying_scheduler.last_epoch < underlying_scheduler.total_steps:
                    scheduler.step()
                    
                epoch_loss += loss.item()
                steps += 1
                global_step += 1

                # Log telemetry
                if step % 100 == 0 and accelerator.is_main_process:
                    log_telemetry_atomic(step, epoch)
                    print(f"Epoch [{epoch}/{epochs}] | Step {step} | Global {global_step} | Loss: {loss.item():.4f}")

                # Mid-epoch checkpoint
                if global_step % checkpoint_interval_steps == 0 and accelerator.is_main_process:
                    # Unwrap the model to save the pure state_dict cleanly
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    _step_ckpt = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state': unwrapped_model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'metadata': {
                            'model_version': '1.0.0',
                            'dataset_hash': dataset_hash,
                            'checkpoint_type': 'mid_epoch_step'
                        }
                    }
                    torch.save(_step_ckpt, checkpoint_path)
                    print(f"[Step Checkpoint] → {checkpoint_path}")
                    push_checkpoint(checkpoint_path, epoch=epoch, global_step=global_step, checkpoint_type="mid_epoch")
                    
            if accelerator.is_main_process:
                avg_loss = epoch_loss / steps if steps > 0 else 0.0
                print(f"Epoch [{epoch}/{epochs}] Complete | Average Loss: {avg_loss:.4f}")
                
                # Unwrap model for epoch save
                unwrapped_model = accelerator.unwrap_model(model)
                
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state': unwrapped_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'metadata': {
                        'model_version': '1.0.0',
                        'dataset_hash': dataset_hash,
                        'epoch': epoch + 1
                    }
                }
                torch.save(checkpoint_state, checkpoint_path)
                push_checkpoint(checkpoint_path, epoch=epoch + 1, global_step=global_step, checkpoint_type="epoch")
            
    except (KeyboardInterrupt, SystemExit):
        if accelerator.is_main_process:
            _interrupted_epoch = locals().get('epoch', start_epoch)
            print(f"Training interrupted at epoch={_interrupted_epoch}. Securing checkpoint...")
            unwrapped_model = accelerator.unwrap_model(model)
            
            checkpoint_state = {
                'epoch': _interrupted_epoch,
                'global_step': global_step,
                'model_state': unwrapped_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'metadata': {
                    'model_version': '1.0.0',
                    'dataset_hash': dataset_hash,
                    'epoch': _interrupted_epoch,
                    'checkpoint_type': 'interrupt'
                }
            }
            torch.save(checkpoint_state, checkpoint_path)
            push_checkpoint(checkpoint_path, epoch=_interrupted_epoch, global_step=global_step, checkpoint_type="interrupt")
        
    print("Training job complete!")
    return model
