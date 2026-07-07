import os
import shutil
import subprocess
import torch
from src.model import NetworkBytePatcher

def test_evaluation_flow():
    print("=== Testing Evaluation & Visualization Script ===")
    
    # 1. Create a temporary mock environment in a subdirectory
    mock_dir = "./test_mock_eval_env"
    checkpoint_dir = os.path.join(mock_dir, "checkpoints")
    results_dir = os.path.join(mock_dir, "results")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    mock_checkpoint_path = os.path.join(checkpoint_dir, "latest_patcher.pt")
    mock_plot_path = os.path.join(results_dir, "entropy_profile.png")
    
    print(f"Creating mock checkpoint at: {mock_checkpoint_path}")
    
    # 2. Instantiate and save a mock model checkpoint (with default configurations)
    model = NetworkBytePatcher()
    torch.save({
        'epoch': 1,
        'model_state': model.state_dict(),
        'optimizer_state': {}
    }, mock_checkpoint_path)
    
    pcap_path = "local_test.pcap"
    assert os.path.exists(pcap_path), f"Error: '{pcap_path}' not found. Run download_data.sh first."
    
    # 3. Call evaluate.py using subprocess to verify full execution loop
    print("Running evaluate.py script on mock environment...")
    eval_command = [
        ".venv/bin/python", "evaluate.py",
        "--checkpoint_path", mock_checkpoint_path,
        "--pcap_path", pcap_path,
        "--output_path", mock_plot_path,
        "--entropy_threshold", "4.5"
    ]
    
    result = subprocess.run(eval_command, capture_output=True, text=True)
    
    print("\nScript Output:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nScript Errors:")
        print(result.stderr)
        raise RuntimeError(f"evaluate.py failed with exit code {result.returncode}")
        
    # 4. Verify visual plot output is generated
    assert os.path.exists(mock_plot_path), f"Error: Visual plot '{mock_plot_path}' was not generated."
    file_size = os.path.getsize(mock_plot_path)
    print(f"Visual plot generated successfully. Size: {file_size} bytes.")
    assert file_size > 0, "Error: Generated plot file is empty (0 bytes)."
    
    # 5. Clean up the temporary mock environment
    print(f"Cleaning up mock environment directory '{mock_dir}'...")
    shutil.rmtree(mock_dir)
    
    print("\nVisualizer and Evaluation Script validation successfully complete!")

if __name__ == "__main__":
    test_evaluation_flow()
