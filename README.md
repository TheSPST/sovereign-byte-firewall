# Sovereign Byte-Level Anomaly Detection Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HPC A100 Optimized](https://img.shields.io/badge/HPC-A100%20Optimized-NVIDIA?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![SLURM Scheduler](https://img.shields.io/badge/Scheduler-SLURM-blue)](https://slurm.schedmd.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An encoder-free, sovereign byte-level network anomaly detection engine optimized for high-throughput execution on multi-GPU A100 HPC clusters (e.g., CDAC AI Kosh) and local Apple Silicon environments.

---

## 1. Executive Summary

Traditional network intrusion detection architectures introduce severe bottlenecks by translating raw packets into human-readable text logs (SIEM, Syslogs, NetFlow) or utilizing CPU-bound flow extractors (e.g., CICFlowMeter). This process inflates data volume by up to 500% and delays latency critical responses. 

The **Sovereign Byte-Level Anomaly Detection Engine** bypasses text-based translation entirely. It processes raw network traffic (`.pcap` format) as a continuous 1D stream of integers representing raw byte values ($0 \text{ to } 255$). By modeling network bytes as a sequence prediction task (similar to character-level language modeling), the engine natively learns protocol grammar at the packet level. Zero-day payloads, malware heartbeats, and fuzzing attacks are flagged instantly as anomalous structural entropy spikes.

---

## 2. Core Architecture & Mathematical Optimizations

To prevent CPU memory thrashing and GPU starvation when reading massive 12GB+ PCAP captures, the engine implements three core mathematical and memory-safety paradigms:

### A. $O(1)$ Zero-Copy Tensor Striding
Manual slicing and python loops for sequence generation are eliminated. The dataloader reads the raw packet stream into a flat 1D memory array and creates overlapping sequence windows of size $W = 8192$ with a stride step $S$ using PyTorch's native `torch.as_strided()`. 
* **Strict Trailing Remainder Math:** To prevent memory out-of-bounds segfaults, the tensor is sliced to exactly:
  $$\text{Limit} = (\text{num\_windows} \times S) + (W - S)$$
  where $\text{num\_windows} = \max(0, (N - W) // S + 1)$ for a flat buffer of size $N$.
* **Storage Detachment:** Yielded sequence tensors are decoupled from the large underlying 10MB chunk buffers using `.clone()`, allowing Python's garbage collector to instantly release memory pointers and prevent Out-Of-Memory (OOM) crashes on GPU nodes.

### B. Vectorized Shannon Entropy Matrix Math
We replace Python loops and `Counter` statistics with PyTorch's native `torch.bincount()` to bin byte frequency histograms. Probabilities $P(x_i)$ are computed across the batch, and Shannon Entropy is calculated in a single SIMD execution pass:
$$H_t = - \sum_{i=0}^{255} P(x_i) \log_2 P(x_i)$$
This Hadamard product and summation executes natively on GPU accelerators, allowing the engine to calculate and write packet-level anomalies (such as `TCP_SYN_Flood` or `Abnormal_Entropy` spikes) to a line-buffered CSV database (`anomaly_labels.csv`) without blocking training throughput.

### C. Native Sub-Quadratic Attention Math
The `NetworkBytePatcher` model defaults to an ultra-lightweight scale (`num_layers=2`, `d_model=128`, `nhead=4`) to achieve sub-millisecond per-packet inference.
The self-attention block utilizes PyTorch 2.x's native `torch.nn.functional.scaled_dot_product_attention` with `attn_mask=None` and `is_causal=True`. This bypasses custom attention masking matrix multiplications, triggering optimal FlashAttention kernels and utilizing fast SRAM tiling to avoid $O(N^2)$ memory scaling on long 8192-byte sequences.

---

## 3. Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── dataloader.py      # Strided PCAP dataset, SHA-256 caching, and anomaly labeler
│   ├── model.py           # Pre-LN causal transformer patcher with FlashAttention
│   └── training.py        # Telemetry-logged training loop with SLURM pre-emption checkpoints
├── tests/
│   ├── test_dataloader.py # Dataloader and cache validation
│   ├── test_model.py      # Causality, scale, and latency tests
│   ├── test_training.py   # Resilient load/resume validation
│   └── test_evaluation.py # Visualizer integration test sandbox
├── scripts/
│   ├── download_data.sh   # Initializes directories & downloads public PCAP datasets
│   └── prepare_artifacts.sh # Packages source files into a lightweight deployment zip
├── setup_and_verify.py    # Hardware diagnostics & dry-run validation gatekeeper
├── run_training.py        # Orchestration entrypoint script
├── evaluate.py            # Headless runner to visualize running entropy profiles
└── sbatch_train.sh        # SLURM script with pre-emption SIGTERM trap handlers
```

---

## 4. Cluster Deployment & Execution Guide

Follow this step-by-step workflow on your HPC cluster (e.g., CDAC AI Kosh):

### Step 1: Bootstrap the Environment
Clone the repository and install core dependencies.
```bash
# Clone the repository
git clone https://github.com/TheSPST/sovereign-byte-firewall.git
cd sovereign-byte-firewall

# Setup a clean environment (e.g., Conda)
conda create -n sovereign python=3.11 -y
conda activate sovereign

# Install PyTorch and tooling
pip install torch scapy matplotlib
```

### Step 2: Prepare Datasets
Fetch training baselines directly to the cluster's high-speed storage.
```bash
# Create folders and symlink fallbacks
bash scripts/download_data.sh

# Download Wednesday working hours PCAP (CIC-IDS2017)
wget -O data/cic-ids2017/dataset.zip http://205.174.165.80/CICDataset/CIC-IDS2017/Dataset/PCAPs/Wednesday-workingHours.pcap.zip
unzip data/cic-ids2017/dataset.zip -d data/cic-ids2017/
```

### Step 3: Run Diagnostic Validation
Run the diagnostic gatekeeper to verify CUDA visibility, VRAM allocations, and model dry-run passes:
```bash
python setup_and_verify.py --dataset_path ./data/cic-ids2017/Wednesday-workingHours.pcap
```

### Step 4: Submit Training Job
Submit the training script to the SLURM scheduler.
```bash
sbatch sbatch_train.sh
```

---

## 5. Resilient Checkpointing & Telemetry
During training, the engine maintains:
- **Resiliency Checkpoints:** Model parameters, optimizer states, epoch markers, and CUDA gradient scalers are saved periodically to `checkpoints/latest_patcher.pt`.
- **Pre-emption Handling:** `sbatch_train.sh` traps SIGTERM (shutdown warnings) and sends SIGINT to Python to force a final checkpoint save.
- **Hardware Telemetry logs:** Logs GPU temp, GPU utilization, power draw, and VRAM footprints to `logs/hardware_metrics.json` every 100 steps.
