import os
import tempfile
import torch
import numpy as np
import pytest
from scapy.all import IP, TCP, wrpcap
from src.model import NetworkBytePatcher
from run_classifier import extract_packet_features

def test_feature_extraction():
    print("=== Testing Context Feature Extraction ===")
    
    device = torch.device("cpu")
    # Instantiate lightweight model
    model = NetworkBytePatcher(d_model=64, nhead=2, num_layers=1).to(device)
    model.eval()
    
    # Create 4 mock TCP packets (size 128 bytes each: 20 IP + 20 TCP + 88 payload)
    packets = [IP(dst="192.168.1.1")/TCP()/("A"*88) for _ in range(4)]
    
    fd, temp_pcap_path = tempfile.mkstemp(suffix=".pcap")
    os.close(fd)
    try:
        wrpcap(temp_pcap_path, packets)
        
        # Run feature extraction with seq_len=128
        features = extract_packet_features(model, temp_pcap_path, device, batch_size=2, seq_len=128)
        
        print(f"Extracted features shape: {features.shape}")
        assert features.shape == (4, 64), f"Expected shape (4, 64), got {features.shape}"
        assert np.isnan(features).sum() == 0, "Extracted features contain NaN values!"
        print("Feature extraction verified successfully!")
    finally:
        if os.path.exists(temp_pcap_path):
            os.remove(temp_pcap_path)

def test_classifier_flow():
    print("=== Testing Downstream Classifier Training Flow ===")
    
    # Generate mock features (2 classes, 10 samples each, 64 dimensions)
    np.random.seed(42)
    benign_features = np.random.normal(loc=0.0, scale=1.0, size=(10, 64))
    attack_features = np.random.normal(loc=2.0, scale=1.0, size=(10, 64))
    
    X = np.concatenate([benign_features, attack_features], axis=0)
    y = np.concatenate([np.zeros(10), np.ones(10)], axis=0)
    
    # Partition Train/Test splits
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    assert len(preds) == 4, f"Expected 4 test predictions, got {len(preds)}"
    
    # Verify predictions contain valid binary labels (0 or 1)
    for p in preds:
        assert p in [0, 1], f"Invalid class predicted: {p}"
        
    print("Downstream classifier training and validation flow verified successfully!")

if __name__ == "__main__":
    test_feature_extraction()
    test_classifier_flow()
