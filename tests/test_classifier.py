import os
import torch
import numpy as np
import pytest
from torch.utils.data import TensorDataset, DataLoader
from src.model import NetworkBytePatcher
from run_classifier import extract_packet_features

def test_feature_extraction():
    print("=== Testing Context Feature Extraction ===")
    
    device = torch.device("cpu")
    # Instantiate lightweight model
    model = NetworkBytePatcher(d_model=64, nhead=2, num_layers=1).to(device)
    model.eval()
    
    # Create mock dataloader containing 4 sample sequences of length 128
    dummy_input = torch.randint(0, 256, (4, 128))
    dataset = TensorDataset(dummy_input)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # We must mock the dataloader iteration because get_pcap_dataloader yields tensors directly,
    # whereas standard TensorDataset returns tuples. We override the dataloader yield format:
    class MockDataloaderWrapper:
        def __init__(self, dl):
            self.dl = dl
        def __iter__(self):
            for batch in self.dl:
                yield batch[0]
                
    wrapped_dl = MockDataloaderWrapper(dataloader)
    
    # Run feature extraction
    features = extract_packet_features(model, wrapped_dl, device)
    
    print(f"Extracted features shape: {features.shape}")
    assert features.shape == (4, 64), f"Expected shape (4, 64), got {features.shape}"
    assert np.isnan(features).sum() == 0, "Extracted features contain NaN values!"
    print("Feature extraction verified successfully!")

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
