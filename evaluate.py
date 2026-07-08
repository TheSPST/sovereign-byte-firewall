#!/usr/bin/env python3
# Headless matplotlib configuration must run before importing pyplot
import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataloader import get_pcap_dataloader
from src.model import NetworkBytePatcher

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NetworkBytePatcher and Visualize running packet entropy profile")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest_patcher.pt",
        help="Path to the training checkpoint (default: checkpoints/latest_patcher.pt)"
    )
    parser.add_argument(
        "--pcap_path",
        type=str,
        default="local_test.pcap",
        help="Path to the PCAP file for evaluation (default: local_test.pcap)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/entropy_profile.png",
        help="Path to save the generated entropy plot (default: results/entropy_profile.png)"
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=4.5,
        help="Shannon entropy threshold to trigger dynamic patch cuts (default: 4.5)"
    )
    parser.add_argument(
        "--max_patch_size",
        type=int,
        default=64,
        help="Hard ceiling limit for dynamic patch size (default: 64)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup results directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            print(f"ERROR: Output directory '{output_dir}' is not writable.", file=sys.stderr)
            sys.exit(1)

    # 2. Check path existences
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint file '{args.checkpoint_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.pcap_path):
        print(f"ERROR: Evaluation PCAP file '{args.pcap_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 3. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Evaluating using device: {device}")

    # 4. Load checkpoint and initialize model with matching parameters
    print(f"Loading checkpoint state from '{args.checkpoint_path}'...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state']
    
    # Dynamically extract max_sequence_length from checkpoint's pos_embedding shape
    pos_weight = state_dict.get('pos_embedding.weight', None)
    if pos_weight is None:
        pos_weight = state_dict.get('module.pos_embedding.weight', None)
        
    if pos_weight is not None:
        max_seq_len = pos_weight.shape[0]
    else:
        max_seq_len = 8192
        
    print(f"Dynamic Shape Detection: max_sequence_length={max_seq_len}")
    model = NetworkBytePatcher(max_patch_size=args.max_patch_size, max_sequence_length=max_seq_len).to(device)
    
    # Resilient loading mapping DataParallel module wrapper
    is_dp = isinstance(model, torch.nn.DataParallel)
    has_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if not is_dp and has_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif is_dp and not has_prefix:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Extract a single sequence batch from the dataloader
    dataloader = get_pcap_dataloader(
        pcap_path=args.pcap_path,
        batch_size=1,
        num_workers=0,
        max_sequence_length=1024  # Evaluate a 1024-byte window for visualization clarity
    )
    
    batch = next(iter(dataloader))
    batch = batch.to(device)
    
    # 6. Run model forward pass and compute vectorized Shannon entropy
    with torch.no_grad():
        logits = model(batch)
        # Vectorized calculation over logits tensor (no loops)
        entropies = model.compute_entropy(logits).squeeze(0).cpu().numpy()
        
    # Get dynamic boundaries from the model
    lengths = model.generate_patch_lengths(batch, entropy_threshold=args.entropy_threshold)[0]
    
    # 7. Map boundary markers
    # We calculate the cumulative sums of the patch lengths to locate slice points along the byte offsets
    boundary_indices = np.cumsum(lengths)
    
    print("\nEvaluation Statistics:")
    print(f"  Total processed bytes: {len(entropies)}")
    print(f"  Total patches created: {len(lengths)}")
    print(f"  Average patch length:  {np.mean(lengths):.2f} bytes")
    print(f"  Max patch length:      {np.max(lengths)} bytes")
    print(f"  Min patch length:      {np.min(lengths)} bytes")

    # 8. Plot results
    print(f"\nGenerating visualization plot...")
    plt.figure(figsize=(15, 6))
    
    # Plot the running entropy profile
    plt.plot(entropies, label="Running Shannon Entropy (bits)", color="#1f77b4", linewidth=1.5)
    
    # Draw a line for the threshold
    plt.axhline(y=args.entropy_threshold, color="#ff7f0e", linestyle="--", linewidth=1.5, 
                label=f"Entropy Threshold ({args.entropy_threshold} bits)")
    
    # Plot the patch boundaries as vertical lines
    first_boundary = True
    for idx in boundary_indices:
        if idx < len(entropies):
            plt.axvline(x=idx, color="#d62728", linestyle=":", alpha=0.6, 
                        label="Dynamic Patch Boundary" if first_boundary else "")
            first_boundary = False

    # Formatting
    plt.title("Sovereign Firewall: Causal Next-Byte Entropy Profile & Dynamic Patches", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Byte Offset in Sequence Window", fontsize=12)
    plt.ylabel("Shannon Entropy (bits)", fontsize=12)
    plt.xlim(0, len(entropies))
    plt.ylim(-0.2, 8.5)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    # Highlight typical header region vs payload region
    # Protocol headers usually have low entropy in the first ~60 bytes
    plt.axvspan(0, min(64, len(entropies)), color="#2ca02c", alpha=0.1)
    plt.text(min(32, len(entropies)/2), 1.5, "Protocol Headers\n(Low Entropy)", 
             color="#2ca02c", fontsize=10, ha="center", style="italic")
    
    # Save the output visualization plot
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved successfully to: '{args.output_path}'")

if __name__ == "__main__":
    main()
