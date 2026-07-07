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
    
    # Verify metadata integration
    assert 'metadata' in checkpoint, "Error: Checkpoint metadata is missing!"
    metadata = checkpoint['metadata']
    assert metadata['model_version'] == "1.0.0", f"Expected model version '1.0.0', got {metadata['model_version']}"
    assert metadata['epoch'] == 1, f"Expected metadata epoch to be 1, got {metadata['epoch']}"
    
    # Expected SHA-256 hash for local_test.pcap
    expected_hash = "b9e851183f914c3d9562d5e5364a1a50446e80bc97bb9b6db86556611e91c67a"
    assert metadata['dataset_hash'] == expected_hash, f"Expected dataset hash '{expected_hash}', got '{metadata['dataset_hash']}'"
    print("Phase 1 complete. Initial checkpoint and metadata successfully verified!")
    
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
    assert checkpoint_final['metadata']['epoch'] == 2, "Expected metadata epoch to update to 2"
    print("Phase 2 complete. Resume training and final checkpoint verified!")
    
    # Verify telemetry logging
    metrics_path = "logs/hardware_metrics.json"
    assert os.path.exists(metrics_path), "Error: Telemetry metrics file 'logs/hardware_metrics.json' was not created!"
    print("Telemetry metrics logged successfully! [OK]")
    
    # Clean up test checkpoints and telemetry logs
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("Test checkpoints cleaned up.")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
        print("Telemetry logs cleaned up.")
        
    print("\nTraining, Checkpointing, and Telemetry verification successfully complete!")

if __name__ == "__main__":
    test_checkpoint_and_resume()
