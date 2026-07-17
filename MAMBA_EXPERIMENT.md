# Mamba backbone experiment (branch: experiment/mamba-backbone)

**Goal:** decide whether a Mamba / state-space backbone should replace the
transformer, by running a clean A/B — same data, same held-out protocol —
comparing **both accuracy and inference throughput**. This branch adds the model
only; `main` is untouched until the result justifies a merge.

## Why Mamba
Mamba processes a sequence in linear time O(N) vs the transformer's O(N²).
Literature (NetMamba arXiv:2405.11449 reports 1–60× faster inference;
MambaNetBurst arXiv:2605.11034 does byte-level with no pretraining) suggests it
can match accuracy at materially better speed — which is exactly what the edge /
pre-filter throughput story needs. It also encodes sequence order through its
recurrence, so there is **no positional embedding** (a genuine simplification).

## What's here
- `src/model_mamba.py` — `MambaBytePatcher`, a drop-in for `NetworkBytePatcher`
  (same `forward(x): (B,T) int -> (B,T,256)` contract). Two backends:
  - `mamba_ssm.Mamba` (fused CUDA kernel) when installed — for real training.
  - a pure-PyTorch selective-SSM fallback — correct + linear-time but the scan
    is Python-loop slow; for tests and CPU/MPS smoke runs.
  - `build_backbone(name, **kw)` factory to select transformer vs mamba.
- `tests/test_model_mamba.py` — shape, -1-padding safety, **causality**, no-pos-
  embedding, factory, param-count. (Causality also verified independently via a
  NumPy mock of the exact scan: future input produced 0.00e+00 change to past
  outputs.)

## How to run the A/B

**1. Environment (GPU — do the real run on CUDA/Kaggle/A100):**
```bash
pip install mamba-ssm causal-conv1d   # fused kernel; without it, falls back to the slow scan
pytest tests/test_model_mamba.py -v   # sanity (runs on CPU via force_torch_scan)
```

**2. Train Mamba on the same corpus as the transformer.** Swap the model class
in the training entrypoint for `MambaBytePatcher` (or wire `build_backbone`
behind a `--backbone mamba` flag in `run_kaggle.py` / `src/training.py`). Keep
everything else identical: Monday benign corpus, seq_len 512, same optimizer/LR
schedule, checkpoint by held-out zero-day eval (not train loss). Mamba trains
from scratch — checkpoints are NOT cross-compatible with the transformer.

**3. Evaluate with the same harness.** Point `evaluate_zero_day.py` at the Mamba
checkpoint (it must construct `MambaBytePatcher` instead of `NetworkBytePatcher`
— add a `--backbone` switch there too). Report on the SAME split as the
transformer (CIC held-out, and UNSW via `scripts/unsw_eval_kaggle.py`):
calibration AUC, held-out benign FPR, held-out detection @ ~1% FPR.

**4. Benchmark throughput** — the whole point. Measure windows/sec and sustained
Mbps on one A100 (and M2 Pro MPS if relevant) for both backbones at seq_len 512,
batch 512. This is the "cost at the edge" number the pre-filter positioning
depends on.

## Decision rule (write the verdict when done)
- Mamba matches transformer accuracy (AUC within ~1–2 pts, held-out FPR
  comparable) at **materially better throughput** → adopt for v2; it strengthens
  the pre-filter economics.
- Mamba is clearly less accurate → keep the transformer; note that at 512-byte
  context the transformer's quadratic cost is already small.
- Mixed → test a longer context (where Mamba's linear scaling wins more) before
  deciding.

Same discipline as the n-gram ablation: an experiment with a written decision
rule, not an assumption. Do NOT merge to `main` until the A/B is run and the
verdict favors it.
