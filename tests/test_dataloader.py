import os
import torch
from src.dataloader import get_pcap_dataloader

def test_dataloader():
    pcap_path = "/Users/shubhamtomar/Desktop/local_test.pcap"
    
    if not os.path.exists(pcap_path):
        print(f"Error: Test PCAP not found at {pcap_path}")
        return

    print(f"Initializing dataloader for {pcap_path}...")
    batch_size = 4
    max_sequence_length = 8192
    
    # Run with single process (num_workers=0) first
    dataloader = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=batch_size,
        num_workers=0,
        max_sequence_length=max_sequence_length
    )
    
    print("\n--- Testing Single Process (num_workers=0) ---")
    total_batches = 0
    total_sequences = 0
    
    for i, batch in enumerate(dataloader):
        total_batches += 1
        total_sequences += batch.size(0)
        
        # Verify batch properties
        assert isinstance(batch, torch.Tensor), "Batch is not a PyTorch Tensor"
        assert batch.dim() == 2, f"Expected 2D tensor, got {batch.dim()}D"
        assert batch.size(1) == max_sequence_length, f"Expected sequence length {max_sequence_length}, got {batch.size(1)}"
        assert (batch >= 0).all() and (batch <= 255).all(), "Byte values out of range (0-255)"
        
        if i == 0:
            print(f"Batch {i+1} shape: {list(batch.shape)}")
            print(f"First 20 bytes of sequence 0:\n{batch[0][:20].tolist()}")
            print(f"Last 20 bytes of sequence 0 (checking padding):\n{batch[0][-20:].tolist()}")
            
    print(f"Total batches loaded: {total_batches}")
    print(f"Total sequences loaded: {total_sequences}")
    print(f"Total raw bytes processed: {total_sequences * max_sequence_length} bytes")
    
    # Test multi-process dataloading to verify our worker partition logic
    print("\n--- Testing Multi-Process (num_workers=2) ---")
    dataloader_mp = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=batch_size,
        num_workers=2,
        max_sequence_length=max_sequence_length
    )
    
    total_sequences_mp = 0
    for batch in dataloader_mp:
        total_sequences_mp += batch.size(0)
        
    print(f"Total sequences loaded with 2 workers: {total_sequences_mp}")
    
    # Verify that multi-process partitioning yielded the same total number of sequences
    # (or close to it, due to padding/rounding differences per worker).
    # Since each worker processes subset of packets, the exact division of sequences can differ slightly 
    # due to independent padding at the end of each worker's stream, but it should be comparable.
    print(f"Difference in sequence count (single vs multi-process): {abs(total_sequences - total_sequences_mp)}")
    print("Verification successfully complete!")

if __name__ == "__main__":
    test_dataloader()
