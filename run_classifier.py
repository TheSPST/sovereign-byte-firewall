#!/usr/bin/env python3
"""
run_classifier.py
=================
Stage 2 Downstream Classifier.
Loads a pre-trained unsupervised NetworkBytePatcher checkpoint, extracts 
mean-pooled contextual features from raw PCAPs, and trains a fast, lightweight
Random Forest classifier to categorize known attacks.

Usage:
  python run_classifier.py \
    --checkpoint_path checkpoints/latest_patcher.pt \
    --dataset_dir scratch/archive_upload/
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.model import NetworkBytePatcher
from src.dataloader import get_pcap_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Downstream Classifier")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest_patcher.pt",
        help="Path to the pre-trained Stage 1 transformer checkpoint (default: checkpoints/latest_patcher.pt)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="scratch/archive_upload/",
        help="Directory containing benign and attack PCAP files"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=12,
        help="Maximum depth of the trees in the Random Forest (default: 12)"
    )
    return parser.parse_args()

def extract_packet_features(model, dataloader, device):
    """
    Extracts mean-pooled contextual embeddings from the final layer 
    of the Transformer for each packet sequence in the dataloader.
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Dataloader outputs sequence tokens of shape: [Batch_Size, Seq_Len]
            batch = batch.to(device)
            B, T = batch.size()
            
            # Mask-safe input clamping to prevent IndexError on -1 sentinels
            x_clamped = torch.clamp(batch, min=0)
            
            # Contextual embedding projection
            positions = torch.arange(T, device=device).unsqueeze(0)
            h = model.byte_embedding(x_clamped) + model.pos_embedding(positions)
            h = model.dropout(h)
            
            for block in model.blocks:
                h = block(h)
            
            h = model.ln_f(h)
            
            # Mean pool across sequence dimension to get a single vector per window
            pooled = torch.mean(h, dim=1)
            features.append(pooled.cpu().numpy())
            
    if not features:
        return np.empty((0, model.byte_embedding.embedding_dim))
        
    return np.concatenate(features, axis=0)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"==================================================")
    # Human-readable model information report
    print(f"      STAGE 2 HYBRID THREAT CLASSIFIER")
    print(f"==================================================")
    print(f"Active Device:  {device}")
    print(f"Checkpoint:     {args.checkpoint_path}")
    print(f"Dataset Dir:    {args.dataset_dir}")
    print(f"==================================================\n")
    
    # 1. Verify Checkpoint Existence
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Pre-trained Stage 1 checkpoint not found at: '{args.checkpoint_path}'", file=sys.stderr)
        print("Please train your baseline model first using run_kaggle.py or run_training.py.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Initialize and Restore Model
    print("Initializing NetworkBytePatcher...")
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Clean 'module.' prefix if present
    state_dict = checkpoint['model_state']
    has_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Pre-trained model weights restored successfully.\n")
    
    # 3. Scan dataset files
    # Define dataset target file mappings
    datasets = {
        "benign": "normal.pcap",
        "brute_force": "hydra_ftp.pcap",
        "dos": "vsftpd.pcap",
        "botnet": "zeus.pcap"
    }
    
    X_list, y_list = [], []
    valid_classes = []
    
    for label_id, (label_name, pcap_file) in enumerate(datasets.items()):
        pcap_path = os.path.join(args.dataset_dir, pcap_file)
        if not os.path.exists(pcap_path):
            # Try workspace root fallback as well
            pcap_path_fallback = pcap_file
            if os.path.exists(pcap_path_fallback):
                pcap_path = pcap_path_fallback
            else:
                print(f"Warning: PCAP file '{pcap_file}' not found in '{args.dataset_dir}'. Skipping class '{label_name}'.")
                continue
                
        print(f"Extracting features for class '{label_name}' from: '{pcap_path}'...")
        # Lightweight sequence length configurations for fast mapping
        dl = get_pcap_dataloader(
            pcap_path=pcap_path,
            batch_size=32,
            num_workers=0,
            max_sequence_length=512
        )
        
        feats = extract_packet_features(model, dl, device)
        if len(feats) > 0:
            X_list.append(feats)
            y_list.append(np.full(len(feats), len(valid_classes)))
            valid_classes.append(label_name)
            print(f"  -> Extracted {len(feats)} sequence feature vectors. [OK]")
        else:
            print(f"  -> No sequence features extracted from: '{pcap_path}'")
            
    if not X_list:
        print("ERROR: No valid features extracted. Verify that your dataset PCAPs are present.", file=sys.stderr)
        sys.exit(1)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\nTotal Dataset Footprint: {X.shape[0]} samples, {X.shape[1]}-dim feature vectors.")
    
    # Check class representation
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        print("WARNING: Classification requires at least 2 represented classes. Downstream training skipped.")
        sys.exit(0)
        
    # 4. Partition Train/Test Splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=y
    )
    
    # 5. Train Downstream Classifier
    print(f"\nTraining Downstream Random Forest Classifier (trees={args.n_estimators}, max_depth={args.max_depth})...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("✓ Downstream classifier training complete.")
    
    # 6. Evaluation Reporting
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n================ CLASSIFIER EVALUATION REPORT ================")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("-" * 62)
    print(classification_report(y_test, preds, target_names=valid_classes))
    print(f"==============================================================\n")
    
if __name__ == "__main__":
    main()
