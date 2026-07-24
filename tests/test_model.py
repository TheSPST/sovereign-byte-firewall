import os
import torch
import time
import pytest
from src.model import NetworkBytePatcher

def test_model_properties():
    print("=== Testing Model Properties ===")
    
    # 1. Verify default configurations and scale constraints
    model = NetworkBytePatcher()
    print(f"Model vocab size: {model.vocab_size}")
    print(f"Model max patch size: {model.max_patch_size}")
    print(f"Model max sequence length: {model.max_sequence_length}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    
    # 2. Check device compatibility
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running tests on device: {device}")
    model = model.to(device)
    
    # 3. Verify forward pass shape
    batch_size = 4
    seq_len = 512
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)
    
    model.eval()
    with torch.no_grad():
        out = model(x)
        
    print(f"Input shape: {list(x.shape)}")
    print(f"Output logits shape: {list(out.shape)}")
    assert out.shape == (batch_size, seq_len, 256), f"Expected shape {(batch_size, seq_len, 256)}, got {out.shape}"
    
    # 4. Verify Causal masking (no leakage of future information)
    x1 = torch.randint(0, 256, (1, 100), device=device)
    x2 = x1.clone()
    # Change value at index 50
    x2[0, 50] = (x1[0, 50] + 1) % 256
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        
    # Outputs up to index 49 (which predict index 50) must be identical
    diff_before = torch.abs(out1[0, :50] - out2[0, :50]).max().item()
    diff_after = torch.abs(out1[0, 50:] - out2[0, 50:]).max().item()
    
    print(f"Max logit diff before modification index (0-49): {diff_before}")
    print(f"Max logit diff at/after modification index (50-99): {diff_after}")
    assert diff_before < 1e-4, "Causal leakage detected! Modification in future affected past predictions."
    assert diff_after > 1e-4, "Expected modification to affect future predictions, but it didn't."
    print("Causal masking verified successfully!")

def test_entropy_computation():
    print("\n=== Testing Entropy Computation ===")
    model = NetworkBytePatcher()
    
    # Uniform logits (equally probable next bytes) should yield maximum Shannon Entropy: log2(256) = 8.0 bits
    uniform_logits = torch.ones(1, 1, 256)
    entropy_uniform = model.compute_entropy(uniform_logits).item()
    print(f"Entropy of uniform distribution: {entropy_uniform:.4f} bits (Expected: ~8.0000)")
    assert abs(entropy_uniform - 8.0) < 1e-4, "Uniform entropy calculation incorrect"
    
    # Peaked logits (one highly probable next byte) should yield near-zero entropy
    peaked_logits = torch.zeros(1, 1, 256)
    peaked_logits[0, 0, 42] = 100.0  # Extremely high logit for token 42
    entropy_peaked = model.compute_entropy(peaked_logits).item()
    print(f"Entropy of peaked distribution: {entropy_peaked:.4f} bits (Expected: ~0.0000)")
    assert entropy_peaked < 1e-3, "Peaked entropy calculation incorrect"

def test_patch_ceiling_and_boundaries():
    print("\n=== Testing Patch Boundary Ceiling & Triggers ===")
    model = NetworkBytePatcher(max_patch_size=64)
    
    # Create input sequence of length 150
    seq_len = 150
    x = torch.zeros(1, seq_len, dtype=torch.long)
    
    # 1. With low threshold, it will trigger splits at every token because random weights yield ~8.0 entropy
    lengths_low = model.generate_patch_lengths(x, entropy_threshold=2.0)
    print(f"Generated patch lengths (threshold 2.0): {lengths_low[0][:10]}...")
    for l in lengths_low[0]:
        assert l <= 64, f"Patch length {l} exceeded max_patch_size 64!"
        
    # 2. With high threshold (9.0, unreachable), it should only split at the hard ceiling (64, 64, 22)
    lengths_high = model.generate_patch_lengths(x, entropy_threshold=9.0)
    print(f"Generated patch lengths (threshold 9.0): {lengths_high[0]}")
    assert lengths_high[0] == [64, 64, 22], f"Expected ceiling splits [64, 64, 22], got {lengths_high[0]}"
    
    print("Hard patch size ceiling and dynamic triggers verified successfully!")

@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason=(
        "Latency budget (8ms) is tuned for local Apple Silicon (M2 Pro) "
        "scheduling; shared/CPU-only CI runners are noisy and slower, so this "
        "would be flaky there. Runs normally outside CI."
    ),
)
def test_inference_latency():
    print("\n=== Testing Inference Latency ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NetworkBytePatcher().to(device)
    model.eval()
    
    # standard packet sequence chunk of 64 bytes (typical max patch size/header chunk)
    x = torch.randint(0, 256, (1, 64), device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            
    # Measure latency
    start_time = time.perf_counter()
    iters = 500
    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
            
    avg_latency_ms = ((time.perf_counter() - start_time) / iters) * 1000
    print(f"Average inference latency for 64-byte chunk on {device}: {avg_latency_ms:.4f} ms")
    assert avg_latency_ms < 8.0, f"Inference latency {avg_latency_ms:.2f}ms is too high"
    print("Latency benchmark verified successfully!")

if __name__ == "__main__":
    test_model_properties()
    test_entropy_computation()
    test_patch_ceiling_and_boundaries()
    test_inference_latency()
