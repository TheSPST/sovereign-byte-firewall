#!/usr/bin/env python3
"""
run_classifier.py
=================
Stage 2 Downstream Classifier.
Loads a pre-trained unsupervised NetworkBytePatcher checkpoint, extracts 
mean-pooled contextual features from raw PCAPs using high-speed dpkt parsing,
and trains a fast, lightweight Random Forest classifier to categorize known attacks.

Usage:
  python run_classifier.py \
    --checkpoint_path checkpoints/latest_patcher.pt \
    --dataset_dir scratch/archive_upload/
"""

import os
import sys
import argparse
import dpkt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.model import NetworkBytePatcher
from src.dataloader import RawPcapIterableDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Downstream Classifier")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt",
        help="Path to the pre-trained Stage 1 checkpoint"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for transformer feature extraction (default: 64)"
    )
    return parser.parse_args()

def extract_packet_features(model, pcap_path, device, batch_size=64, seq_len=512):
    """
    Extracts mean-pooled contextual embeddings from the final layer 
    of the Transformer for each packet sequence using fast dpkt parsing.
    """
    model.eval()
    features = []
    
    fobj = open(pcap_path, "rb")
    try:
        reader = dpkt.pcap.Reader(fobj)
    except ValueError:
        fobj.seek(0)
        reader = dpkt.pcapng.Reader(fobj)
        
    masker = RawPcapIterableDataset(pcap_path)
    tls_state = {}
    buffer = bytearray()
    pending_windows = []
    
    def process_batch(windows):
        with torch.no_grad():
            batch = torch.tensor(windows, dtype=torch.long, device=device)
            B, T = batch.size()
            x_clamped = torch.clamp(batch, min=0)
            positions = torch.arange(T, device=device).unsqueeze(0)
            
            h = model.byte_embedding(x_clamped) + model.pos_embedding(positions)
            h = model.dropout(h)
            
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
            
            pooled = torch.mean(h, dim=1)
            features.append(pooled.cpu().numpy())
    
    for ts, packet_data in reader:
        masked = masker._mask_packet_addresses(packet_data, stream_tls_state=tls_state)
        buffer.extend(masked)
        
        while len(buffer) >= seq_len:
            w = list(buffer[:seq_len])
            del buffer[:seq_len]
            pending_windows.append(w)
            
            if len(pending_windows) >= batch_size:
                process_batch(pending_windows)
                pending_windows = []
                
    if pending_windows:
        process_batch(pending_windows)
        
    fobj.close()
    
    if not features:
        return np.empty((0, model.byte_embedding.embedding_dim))
        
    return np.concatenate(features, axis=0)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"==================================================")
    print(f"      STAGE 2 HYBRID THREAT CLASSIFIER")
    print(f"==================================================")
    print(f"Active Device:  {device}")
    print(f"Checkpoint:     {args.checkpoint_path}")
    print(f"Dataset Dir:    {args.dataset_dir}")
    print(f"==================================================\n")
    
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Pre-trained Stage 1 checkpoint not found at: '{args.checkpoint_path}'", file=sys.stderr)
        sys.exit(1)
        
    print("Initializing NetworkBytePatcher...")
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2, max_sequence_length=512)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Pre-trained model weights restored successfully.\n")
    
    datasets = {
        "benign": "normal.pcap",
        "brute_force": "hydra_ftp.pcap",
        "dos": "tomcat.pcap", # swapped vsftpd for tomcat as vsftpd was missing in prev runs
        "botnet": "zeus.pcap"
    }
    
    X_list, y_list = [], []
    valid_classes = []
    
    for label_id, (label_name, pcap_file) in enumerate(datasets.items()):
        pcap_path = os.path.join(args.dataset_dir, pcap_file)
        if not os.path.exists(pcap_path):
            pcap_path_fallback = pcap_file
            if os.path.exists(pcap_path_fallback):
                pcap_path = pcap_path_fallback
            else:
                print(f"Warning: PCAP file '{pcap_file}' not found. Skipping class '{label_name}'.")
                continue
                
        print(f"Extracting features for class '{label_name}' from: '{pcap_path}'...")
        feats = extract_packet_features(model, pcap_path, device, batch_size=args.batch_size)
        if len(feats) > 0:
            X_list.append(feats)
            y_list.append(np.full(len(feats), len(valid_classes)))
            valid_classes.append(label_name)
            print(f"  -> Extracted {len(feats)} sequence feature vectors. [OK]")
        else:
            print(f"  -> No sequence features extracted from: '{pcap_path}'")
            
    if not X_list:
        print("ERROR: No valid features extracted.", file=sys.stderr)
        sys.exit(1)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\nTotal Dataset Footprint: {X.shape[0]} samples, {X.shape[1]}-dim feature vectors.")
    
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        print("WARNING: Classification requires at least 2 represented classes. Downstream training skipped.")
        sys.exit(0)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nTraining Downstream Random Forest Classifier (trees={args.n_estimators}, max_depth={args.max_depth})...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("✓ Downstream classifier training complete.")
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n================ CLASSIFIER EVALUATION REPORT ================")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("-" * 62)
    print(classification_report(y_test, preds, target_names=valid_classes))
    print(f"==============================================================\n")
    
if __name__ == "__main__":
    main()
