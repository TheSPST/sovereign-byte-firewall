#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from src.model import NetworkBytePatcher

def parse_args():
    parser = argparse.ArgumentParser(description="Export Trained NetworkBytePatcher Model to ONNX Format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest_patcher.pt",
        help="Path to the trained PyTorch checkpoint (default: checkpoints/latest_patcher.pt)"
    )
    parser.add_argument(
        "--output_onnx",
        type=str,
        default="checkpoints/latest_patcher.onnx",
        help="Target path to save the exported ONNX model (default: checkpoints/latest_patcher.onnx)"
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
    
    # 1. Setup paths
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint file '{args.checkpoint_path}' not found. Cannot export.")
        return

    # 2. Setup device (cpu is safest for ONNX export initialization)
    device = torch.device("cpu")
    print(f"Initializing export process on device: {device}")

    # 3. Load model using evaluate_zero_day load_model
    print(f"Loading checkpoint from '{args.checkpoint_path}'...")
    from evaluate_zero_day import load_model
    model, max_seq_len = load_model(args.checkpoint_path, device)
    model.eval()
    print("Model state loaded successfully.")

    # 5. Create dummy inputs representing a single packet sequence
    # Shape: (batch_size=1, sequence_length=2048)
    dummy_seq_len = min(2048, max_seq_len)
    dummy_input = torch.randint(0, 256, (1, dummy_seq_len), dtype=torch.long, device=device)
    print(f"Generating dummy trace input with shape: {dummy_input.shape}")

    # 6. Execute ONNX export
    print(f"Initiating ONNX export compile to '{args.output_onnx}'...")
    os.makedirs(os.path.dirname(args.output_onnx), exist_ok=True)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output_onnx,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_bytes'],
            output_names=['predicted_logits'],
            dynamo=False
        )
        print(f"SUCCESS: Model successfully compiled and saved to: '{args.output_onnx}' [OK]")
    except Exception as e:
        print(f"ERROR: ONNX compilation failed: {e}")

if __name__ == "__main__":
    main()
