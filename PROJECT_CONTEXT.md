# Project Context - Sovereign Byte-Level Anomaly Detection Engine

## Metadata
- **Last Updated:** 2026-07-13T00:19:16+05:30
- **Current Phase:** Component 4: Deployment Wrapper & System Verification
- **Last Successful Test:** 2026-07-07T15:37:05Z

---

## 1. Project Mission
To build an encoder-free, sovereign byte-level network anomaly detection engine that bypasses traditional Deep Packet Inspection (DPI) and text log translation entirely. By treating raw packet captures (`.pcap`) as continuous 1D streams of raw integers ($0 \text{ to } 255$), the system learns the structural syntax of network protocols and identifies anomalies/zero-day attacks as dynamic entropy spikes.

---

## 2. Architecture Rules
- **Pre-LayerNorm GPT-style Causal Architecture:** Standardizes on Pre-LN residuals to ensure stable gradients during training compared to post-LN configurations.
- **Strict Causal Masking:** Enforces causal attention boundaries (`is_causal=True`) to ensure predictions at time $t$ do not leak future information.
- **Hardware-Optimized Self-Attention:** Employs PyTorch's native `F.scaled_dot_product_attention` to ensure fast FlashAttention/MemoryEfficientAttention execution across local Apple Silicon (`mps`) and cluster GPUs (`cuda`).
- **Dynamic Sequence Cuts & Hard Ceiling:** Dynamic patch generation bounds segments dynamically on entropy thresholds or a strict hard ceiling of `max_patch_size = 64` bytes.

---

## 3. Hardware / Target Environment
- **Local Debugging:** macOS Apple Silicon (M2 Pro) utilizing local MPS backend (MPS compatibility layer via `torch.backends.mps`).
- **Cluster Production:** CDAC AI Kosh (Airawat) multi-GPU A100 cluster nodes utilizing CUDA.
- **Execution Constraint:** Production training strictly requires CUDA and will exit immediately on CPU execution to prevent waste of cluster compute resource.

---

## 4. Checkpoints & Key Artifacts
- **Training Checkpoints:** Saved to `./checkpoints/latest_patcher.pt` (automatically checkpoints and restores weights, optimizer states, epoch count, and gradient scalers).
- **Core Package Modules:**
  - [src/dataloader.py](file:///Users/shubhamtomar/Documents/sovereign-byte-firewall/src/dataloader.py): Raw PCAP streaming with worker partitioning.
  - [src/model.py](file:///Users/shubhamtomar/Documents/sovereign-byte-firewall/src/model.py): Causal transformer patcher.
  - [src/training.py](file:///Users/shubhamtomar/Documents/sovereign-byte-firewall/src/training.py): Resilient training loop.
- **Verification Scripts:**
  - [setup_and_verify.py](file:///Users/shubhamtomar/Documents/sovereign-byte-firewall/setup_and_verify.py): Hardware diagnostics and 5-batch model dry-run.
  - [run_training.py](file:///Users/shubhamtomar/Documents/sovereign-byte-firewall/run_training.py): CLI wrapper to launch cluster training session.

---

## 5. Next 3 Critical Steps
1. **Model Evaluation & Visualizer:** Implement the running entropy visualizer (Component 4) to trace dynamic token lengths and graph entropy profiles across network packets.
2. **Cluster Multi-GPU Tuning:** Benchmark Dataloader worker parsing speeds on CDAC nodes and scale training epochs.
3. **Anomaly Classifier Calibration:** Build downstream classifiers utilizing trained patch representations to label specific attack categories (DDoS, Botnets, Brute-Force).

---

## 6. Project Codebase Statistics
- **Total Source Files:** 7
- **Total Test Files:** 9
- **File Catalog:**
  - `src/__init__.py`
  - `src/cloud_backup.py`
  - `src/dataloader.py`
  - `src/losses.py`
  - `src/model.py`
  - `src/sniffer.py`
  - `src/training.py`
  - `tests/__init__.py`
  - `tests/test_classifier.py`
  - `tests/test_dataloader.py`
  - `tests/test_evaluation.py`
  - `tests/test_losses.py`
  - `tests/test_model.py`
  - `tests/test_p0_fixes.py`
  - `tests/test_tls_masking.py`
  - `tests/test_training.py`