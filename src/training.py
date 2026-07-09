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
    
    # [Keep your existing Auto-resume logic here...]
    # When loading/saving weights, remember to use accelerator.unwrap_model(model)
    # instead of checking for isinstance(model, torch.nn.DataParallel)

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
                
                if scheduler.last_epoch < scheduler.total_steps:
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
                    'metadata': {"dataset_hash": dataset_hash, "epoch": epoch + 1}
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
                'metadata': {"epoch": _interrupted_epoch, "checkpoint_type": "interrupt"}
            }
            torch.save(checkpoint_state, checkpoint_path)
            push_checkpoint(checkpoint_path, epoch=_interrupted_epoch, global_step=global_step, checkpoint_type="interrupt")
        
    print("Training job complete!")
    return model
