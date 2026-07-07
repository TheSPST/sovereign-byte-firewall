import os
import torch
import torch.nn as nn
from contextlib import nullcontext

def train_patcher_on_kosh(model, dataloader, epochs=5, checkpoint_dir="./checkpoints", lr=1e-4):
    """
    Resilient training loop designed for the AI Kosh cluster (SLURM).
    Handles automatic checkpoint saving, restoration, and PyTorch 2.x AMP mixed precision.
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
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        steps = 0
        
        for step, byte_sequence in enumerate(dataloader):
            byte_sequence = byte_sequence.to(device)
            
            # Predict the next byte: slice inputs and targets
            inputs = byte_sequence[:, :-1]
            targets = byte_sequence[:, 1:]
            
            optimizer.zero_grad()
            
            # PyTorch 2.x Device-Agnostic AMP Autocast Context
            # We catch potential exceptions for 'mps' autocast if the local environment PyTorch build does not fully support it yet.
            try:
                autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
            except (RuntimeError, ValueError):
                # Fallback to no autocast for local compatibility on unsupported device types (like MPS on certain PyTorch builds)
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
            
            if step % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Step {step} | Loss: {loss.item():.4f}")
                
        # Calculate and print epoch average loss
        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        print(f"Epoch [{epoch}/{epochs}] Complete | Average Loss: {avg_loss:.4f}")
        
        # Mandatory end-of-epoch state save for cluster durability
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_state['scaler_state'] = scaler.state_dict()
            
        torch.save(checkpoint_state, checkpoint_path)
        print(f"Checkpoint successfully secured at {checkpoint_path} for Epoch {epoch + 1}.")
        
    print("Training job complete!")
    return model
