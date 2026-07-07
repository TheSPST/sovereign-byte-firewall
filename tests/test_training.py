import os
import shutil
import torch
from src.model import NetworkBytePatcher
from src.dataloader import get_pcap_dataloader
from src.training import train_patcher_on_kosh

def test_checkpoint_and_resume():
    print("=== Testing Training & Resilient Checkpointing ===")
    
    # 1. Use relative path to local_test.pcap and check presence
    pcap_path = "local_test.pcap"
    assert os.path.exists(pcap_path), f"Error: Required test PCAP file '{pcap_path}' not found in the project root."
    
    # Configure directories and parameters
    checkpoint_dir = "./test_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running training test on device: {device}")
    
    # 2. Setup Dataloader and Lightweight Model
    dataloader = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=2,
        num_workers=0,
        max_sequence_length=1024  # shorter sequences for faster test run
    )
    
    model = NetworkBytePatcher(d_model=64, nhead=2, num_layers=1).to(device)
    
    # 3. Simulate Phase 1: Train for 1 Epoch
    print("\n--- Phase 1: Training for 1 epoch (creating initial checkpoint) ---")
    train_patcher_on_kosh(
        model=model,
        dataloader=dataloader,
        epochs=1,
        checkpoint_dir=checkpoint_dir,
        lr=1e-3
    )
    
    # Verify checkpoint creation
    checkpoint_path = os.path.join(checkpoint_dir, "latest_patcher.pt")
    assert os.path.exists(checkpoint_path), "Error: Checkpoint file was not saved!"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    assert checkpoint['epoch'] == 1, f"Expected checkpoint epoch to be 1, got {checkpoint['epoch']}"
    print("Phase 1 complete. Initial checkpoint successfully verified!")
    
    # 4. Simulate Phase 2: Resume training to run Epoch 2 (total epochs = 2)
    print("\n--- Phase 2: Resuming training for a total of 2 epochs ---")
    
    # We instantiate a fresh model and optimizer shell to guarantee weight restoration is verified
    resumed_model = NetworkBytePatcher(d_model=64, nhead=2, num_layers=1).to(device)
    
    train_patcher_on_kosh(
        model=resumed_model,
        dataloader=dataloader,
        epochs=2,
        checkpoint_dir=checkpoint_dir,
        lr=1e-3
    )
    
    # Verify final checkpoint
    checkpoint_final = torch.load(checkpoint_path, map_location=device)
    assert checkpoint_final['epoch'] == 2, f"Expected final checkpoint epoch to be 2, got {checkpoint_final['epoch']}"
    print("Phase 2 complete. Resume training and final checkpoint verified!")
    
    # Clean up test checkpoints directory
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("Test checkpoints cleaned up.")
        
    print("\nTraining and Checkpointing Wrapper verification successfully complete!")

if __name__ == "__main__":
    test_checkpoint_and_resume()
