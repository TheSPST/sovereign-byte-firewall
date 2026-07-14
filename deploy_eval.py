import os
import json
import subprocess

KERNEL_SLUG = "spsttomar/eval-gs865000-fused"
NOTEBOOK_FILE = "eval_gs865000.ipynb"
DATASET = "dummy4321/cybera"

# The evaluation code provided by the user, updated for gs865000
code_cell = """
import os, sys, glob, subprocess
from huggingface_hub import login, hf_hub_download
from kaggle_secrets import UserSecretsClient

# 1. Authenticate with Hugging Face
login(token=UserSecretsClient().get_secret('HF_TOKEN'))
REPO = '/kaggle/working/sovereign-byte-firewall'

# 2. Clone or Update GitHub Repository
if os.path.exists(REPO):
    subprocess.run(['git', '-C', REPO, 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', 'https://github.com/TheSPST/sovereign-byte-firewall.git', REPO], check=True)

print("HEAD:", subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-1'], capture_output=True, text=True).stdout.strip())

# 3. Install Dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', 'scapy', 'huggingface_hub', 'dpkt', '-q'], check=True)

# 4. Download Checkpoint & Locate PCAP
# We update this to gs865000
ckpt = hf_hub_download(
    repo_id='spst01/sovereign-byte-firewall-v2-masking-v2',
    filename='checkpoints/latest_patcher_ep0_gs865000_mid_epoch.pt', 
    repo_type='model', 
    local_dir='/kaggle/working/ckpt_best'
)
wed = glob.glob('/kaggle/input/**/Wednesday-workingHours.pcap', recursive=True)[0]

print('Using Checkpoint:', ckpt)
print('Using PCAP:', wed)

# 5. Patch the DeprecationWarning
patch_cmd = r"sed -i 's/datetime.datetime.utcfromtimestamp(ts + off \* 3600)/datetime.datetime.fromtimestamp(ts + off \* 3600, datetime.timezone.utc)/g' evaluate_cic_days_fused.py"
subprocess.run(patch_cmd, shell=True, cwd=REPO)
subprocess.run("sed -i '1i import datetime' evaluate_cic_days_fused.py", shell=True, cwd=REPO)

# 6. Create the DPKT Speed-Up Wrapper (FIXED FILE HANDLING)
wrapper_code = f\"\"\"
import sys
import dpkt
import scapy.utils
import scapy.all

# Mimic Scapy packet behavior for raw bytes
class FastDPKTPacket:
    def __init__(self, buf, ts):
        self.buf = buf
        self.time = ts
    def __bytes__(self): return self.buf
    def __len__(self): return len(self.buf)

# Mimic Scapy PcapReader with support for both strings and file objects
class FastPcapReader:
    def __init__(self, filename_or_fileobj):
        if hasattr(filename_or_fileobj, 'read'):
            self.f = filename_or_fileobj
            self.we_opened_it = False
        else:
            self.f = open(filename_or_fileobj, 'rb')
            self.we_opened_it = True
            
        try:
            self.reader = dpkt.pcap.Reader(self.f)
        except ValueError:
            self.f.seek(0)
            self.reader = dpkt.pcapng.Reader(self.f)
            
    def __iter__(self):
        for ts, buf in self.reader:
            yield FastDPKTPacket(buf, ts)
            
    def __enter__(self): return self
    
    def __exit__(self, *args): 
        if self.we_opened_it:
            self.f.close()

# Intercept and overwrite Scapy's reader with our fast one
scapy.utils.PcapReader = FastPcapReader
scapy.all.PcapReader = FastPcapReader
scapy.utils.RawPcapReader = FastPcapReader
scapy.all.RawPcapReader = FastPcapReader

# Inject arguments and run the original script
sys.argv = [
    'evaluate_cic_days_fused.py',
    '--checkpoint_path', '{ckpt}', 
    '--pcap', '{wed}',
    '--day', 'wednesday', 
    '--max_sequence_length', '512',
    '--window_subsample', '20',
    '--target_alarms_per_10k', '0.03',
    '--output_dir', './results/wed_gs865000_fused'
]
import evaluate_cic_days_fused
evaluate_cic_days_fused.main()
\"\"\"

# Save the wrapper to the repository
with open(os.path.join(REPO, 'fast_eval_wrapper.py'), 'w') as f:
    f.write(wrapper_code)

# 7. Run Evaluation using the wrapper
print("Starting High-Speed Evaluation on T4...")
subprocess.run([sys.executable, '-u', 'fast_eval_wrapper.py'], cwd=REPO)
"""

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_cell.split("\n")]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

metadata = {
  "id": KERNEL_SLUG,
  "title": "eval-gs865000-fused",
  "code_file": NOTEBOOK_FILE,
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_internet": True,
  "dataset_sources": [DATASET],
  "kernel_sources": [],
  "competition_sources": [],
  "machine_shape": "Gpu"
}

os.makedirs("kaggle_eval", exist_ok=True)
with open(os.path.join("kaggle_eval", NOTEBOOK_FILE), "w") as f:
    json.dump(notebook, f, indent=2)
with open(os.path.join("kaggle_eval", "kernel-metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Created Kaggle payload in kaggle_eval/")
