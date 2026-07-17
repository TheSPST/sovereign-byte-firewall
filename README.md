# Sovereign Byte-Level Firewall

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A byte-level language model that detects zero-day network attacks — no signatures, no cloud, no log pipeline.**

A compact causal transformer (~1.6M parameters) is trained to predict raw network
traffic byte by byte. Having learned the structural grammar of benign traffic, it
scores live packets by *surprise* — how improbable each byte is given everything
before it. Attacks it has never seen produce byte sequences the model finds
improbable; that is the entire detection signal. Runs fully offline on anything
from an Apple Silicon laptop to an A100 node.

## Results (honest numbers)

Zero-day protocol on CIC-IDS2017: trained on **benign Monday traffic only**, all
attacks unseen. Fused system = byte model OR-fused with a lightweight
connection-rate detector.

| Measurement | Result |
|---|---|
| Full-day eval, fused (Wed: Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) | **5/5 attack campaigns detected** |
| False alarms that day | **0.9–1.4/hour** (byte detector alone: 0–0.5/hour) |
| Held-out zero-day window detection (byte model, strictest metric) | 32.6% @ **0.23% FPR** |
| Live deployment (real traffic, MacBook, 1h self-calibration) | measurement in progress |
| Second dataset (UNSW-NB15) | in progress |

Window-level detection and campaign-level detection differ because an attack
campaign generates thousands of windows — flagging a fraction catches the
campaign. We report both. Known blind spot: slow low-rate brute-force (see
`TECHNICAL_BRIEF.md` for what this tool is honestly for and not for).

## Quickstart — live demo on your own machine

```bash
git clone <repo-url> && cd sovereign-byte-firewall
conda create -n sovereign python=3.11 -y && conda activate sovereign
pip install torch scapy websockets

# 1 hour of silent self-calibration on your network, then live monitoring
# with a real-time dashboard (opens in your browser):
./start_dashboard.sh --learning_time 3600
```

Day one is silent: the engine learns *your* network's baseline and sets its own
threshold (saved to `calibration_<iface>.json`). Alerts are deduplicated into
incidents and logged to `incidents_<iface>.csv`. Default checkpoint:
`ckpt_best/checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt` (the validated peak).

## Reproduce the evaluation

```bash
python evaluate_zero_day.py \
  --checkpoint_path ckpt_best/checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt \
  --score_agg topk --topk_frac 0.1
```

`--score_agg topk` (mean surprise of the top 10% most surprising bytes per
window) is the validated recipe; every deployed component mirrors it.
`scripts/unsw_eval_kaggle.py` runs the same protocol on UNSW-NB15.

## How it works

1. **Bytes in, no translation.** Traffic (live or `.pcap`) is a 1D stream of
   values 0–255. No flow extraction, no DPI parsing, no log conversion.
2. **Masking.** Deterministic fields (addresses, checksums) and always-encrypted
   payloads (TLS app data with cross-packet record tracking, QUIC, WireGuard,
   SSH transport) are masked so the model learns protocol structure, not
   ciphertext noise (`src/dataloader.py`).
3. **Model.** Pre-LN causal transformer (2 layers, d_model=128, nhead=4) with
   native `scaled_dot_product_attention` (FlashAttention path). Sub-millisecond
   per-packet inference (`src/model.py`).
4. **Scoring.** Per-window surprise = mean of top-10% per-byte
   −log₂ P(byte | context). Threshold set per environment by self-calibration
   (99.9th percentile of benign baseline).
5. **Fusion.** OR-fused with a SYN-rate detector: byte model catches payload
   anomalies, rate detector catches volumetric floods (`fuse_detectors.py`).

## Repository map

```
src/                    model, strided pcap dataloader + masking, training loop
firewall_daemon.py      live sniffer -> fused detectors -> WebSocket alerts
dashboard/              real-time browser dashboard
evaluate_zero_day.py    the validated zero-day eval harness (topk-surprise)
evaluate_cic_days*.py   full-day CIC evaluation (single + fused)
scripts/unsw_eval_kaggle.py  second-dataset validation runner
tests/                  dataloader, model causality, masking, training tests
TECHNICAL_BRIEF.md      2-page brief: what it is, honest metrics, pilot offer
PRODUCT_ROADMAP.md      near-term feature roadmap (pcap-on-alert, heatmap, triage UI)
FUTURE_ROADMAP.md       pre-filter positioning + open-core strategy
```

## Training (HPC / Kaggle)

Training runs on CIC-IDS2017 Monday (benign only). Two supported paths:

- **Kaggle GPU:** `kaggle_train.ipynb` / `run_kaggle.py` — auto-resume from
  HF Hub checkpoints, background eval watcher uploads metrics per checkpoint.
- **SLURM (e.g. CDAC AI Kosh):** `sbatch sbatch_train.sh` — SIGTERM-trapped
  pre-emption checkpointing, hardware telemetry to `logs/`.
  `python setup_and_verify.py --dataset_path <pcap>` gatekeeps the environment.

Checkpoint selection is by **held-out zero-day detection**, not training loss —
detection peaks early and sharply (see `PROJECT_STATUS_2026-07-12.md`); training
past the peak overfits. The eval watcher exists precisely to catch the peak.

## License

MIT — see [LICENSE](LICENSE).
