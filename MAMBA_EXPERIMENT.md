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

## Mamba-1 vs Mamba-2 (SSD)
The first throughput run used **Mamba-1** (`mamba_ssm.Mamba`). **Mamba-2**
(`mamba_ssm.Mamba2`, Structured State Space Duality) recasts the scan as
tensor-core matmuls — typically 30–60% faster forward/backward and runs a larger
`d_state` well; its scalar-`A` structure *may* also regularize (a hypothesis for
the accuracy A/B, not a given). The code now supports both: `MambaBytePatcher(...,
variant="mamba2")` (default), `build_backbone("mamba2"|"mamba1")`, `train_ab.py
--backbone mamba2`, `evaluate_zero_day.py --backbone mamba2`. `bench_backbones.py`
now benchmarks transformer vs mamba1 vs mamba2 side by side (whichever kernels are
installed). Re-run the benchmark to get the Mamba-2 speedup vs the 1.85x below.

## Throughput A/B result (2026-07-18, Kaggle GPU, torch 2.10+cu128, fused mamba_ssm — Mamba-1)
Forward-pass throughput, d_model=128 / 2 layers / batch 64:

| seq_len | transformer | mamba | speedup |
|---|---|---|---|
| 512  | 3,353 win/s | 4,557 win/s | **1.36x** |
| 2048 | 526 win/s   | 970 win/s   | **1.85x** |

Params: transformer 724,736 vs mamba 299,520 (**0.41x** — 2.4x smaller).
Learning check: both converge (loss ~5.6 -> ~0.008), so the Mamba impl trains
end-to-end. **Verdict so far: Mamba is faster (speedup grows with seq len),
smaller, and trains.** The remaining gate is ACCURACY (below).

## Next-session runbook — the accuracy A/B (wiring done, UNTESTED until smoke test)
Wiring added on this branch: `evaluate_zero_day.py --backbone {transformer,mamba}`
and a self-contained `train_ab.py` that trains either backbone the same way and
saves an eval-compatible checkpoint. **These were written without a local torch
to test against — run the smoke test FIRST.**

1. **Smoke test (~2 min, proves wiring end-to-end before spending hours):**
   ```
   pip install "causal-conv1d>=1.2.0" mamba-ssm --no-build-isolation
   python train_ab.py --backbone mamba --train_pcap <any.pcap> --steps 20 --seq_len 128 --out /tmp/smoke.pt
   python evaluate_zero_day.py --backbone mamba --checkpoint_path /tmp/smoke.pt \
     --benign_calibration_pcap <a> --benign_holdout_pcap <b> \
     --attack_dir <d> --holdout_attack_pcap <c> --score_agg topk --topk_frac 0.1
   ```
   If both run without error, the wiring is good.
2. **Real A/B — train both to the SAME steps on the SAME corpus, then eval both
   on the SAME held-out split** (commands in `train_ab.py` header). Compare
   calibration AUC + held-out detection @ ~1% FPR to the transformer.
3. Fair-capacity option: raise `--d_model`/`--num_layers` on the Mamba side to
   match ~725k params, or report accuracy-per-param.

## Decision rule (write the verdict when done)
- Mamba matches transformer accuracy (AUC within ~1–2 pts, held-out FPR
  comparable) at **materially better throughput** → adopt for v2; it strengthens
  the pre-filter economics.
- Mamba is clearly less accurate → keep the transformer; note that at 512-byte
  context the transformer's quadratic cost is already small.
- Mixed → test a longer context (where Mamba's linear scaling wins more) before
  deciding.

## Note on capacity (fairness)
At d_model=128 / 2 layers the Mamba comes out **~300k params — roughly 5× SMALLER
than the transformer (~1.6M)** (no attention QKV/MLP, no positional embedding).
For a fair *accuracy* A/B, either match capacity (raise the Mamba's d_model or
num_layers to ~1.6M params) or report the accuracy-per-parameter and
accuracy-per-FLOP honestly. If the smaller Mamba already matches the transformer,
that is itself a strong result for the edge story.

Same discipline as the n-gram ablation: an experiment with a written decision
rule, not an assumption. Do NOT merge to `main` until the A/B is run and the
verdict favors it.
