#!/usr/bin/env python3
"""
evaluate_zero_day.py
=====================
Proof-of-Work evaluation harness for the Sovereign Byte-Level Anomaly
Detection Engine.

This is the missing piece between "we built an entropy-based anomaly
scorer" and "we proved it detects zero-days": it scores a trained Stage 1
checkpoint against KNOWN attack traffic and a genuinely held-out file
(default: 0day.pcap), then reports ROC-AUC, a calibrated detection
threshold, and the resulting detection rate / false-positive rate.

Scoring methodology (identical to src/sniffer.py's live inference path,
so the numbers reported here are a faithful proxy for live deployment):
  For each byte window, run the causal model and compute the mean
  "surprise" (bits) the model assigns to the ACTUAL observed next byte:
      surprise_t = -log2 P(byte_t+1 | byte_<=t)
  This is next-byte predictive cross-entropy, NOT the distribution's own
  entropy (which is what generate_patch_lengths() uses for patch cuts —
  a different, unconditional quantity). High surprise = the model has
  learned "normal" protocol grammar poorly predicts this byte sequence.

Dataset split (avoids circular validation):
  - benign_calibration_pcap + attack_dir files  -> used ONLY to fit the
    ROC curve and pick a threshold (Youden's J and a target-FPR point).
  - benign_holdout_pcap + holdout_attack_pcap    -> NEVER used for
    threshold selection. Scored once, at the end, to report a genuine
    generalization number (this is where 0day.pcap belongs).

Usage (matches project pcap layout under scratch/archive_upload/):
  python evaluate_zero_day.py \
    --checkpoint_path checkpoints/latest_patcher.pt \
    --benign_calibration_pcap scratch/archive_upload/normal.pcap \
    --benign_holdout_pcap scratch/archive_upload/normal2.pcap \
    --attack_dir scratch/archive_upload \
    --holdout_attack_pcap scratch/archive_upload/0day.pcap \
    --output_dir results/zero_day_eval
"""

import os
import sys
import json
import glob
import math
import zlib
import bz2
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import genpareto

from src.model import NetworkBytePatcher
from src.dataloader import get_pcap_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-day proof-of-work evaluation harness")
    parser.add_argument("--checkpoint_path", "--ckpt", type=str, default="checkpoints/latest_patcher.pt",
                         help="Path to trained PyTorch model checkpoint")
    parser.add_argument("--benign_calibration_pcap", "--pcap", type=str, default="local_test.pcap",
                         help="Benign traffic used to fit the threshold (label=0)")
    parser.add_argument("--benign_holdout_pcap", type=str, default="dummy.pcap",
                         help="Benign traffic NEVER used for threshold fitting (final FPR check)")
    parser.add_argument("--attack_dir", type=str, default="data",
                         help="Directory of labeled attack pcaps used to fit the threshold (label=1)")
    parser.add_argument("--holdout_attack_pcap", type=str, default="data/Mirai_pcap.pcap",
                         help="Attack file NEVER used for threshold fitting (final recall check)")
    parser.add_argument("--max_sequence_length", type=int, default=None,
                         help="Override sequence length; auto-detected from checkpoint by default")
    parser.add_argument("--batch_size", type=int, default=None,
                         help="Inference batch size (default: auto — 512 for CUDA, 32 otherwise)")
    parser.add_argument("--target_fpr", type=float, default=0.01,
                         help="Additionally report the threshold nearest this false-positive rate")
    parser.add_argument("--evt_tail_quantile", type=float, default=0.98,
                         help="Quantile of the calibration benign scores used as the EVT/POT "
                              "initial high threshold before fitting the Generalized Pareto tail "
                              "(default: 0.98, i.e. use the top 2%% of benign scores as the tail sample)")
    parser.add_argument("--output_dir", type=str, default="results/zero_day_eval")
    parser.add_argument("--score_agg", type=str, default="mean", choices=["mean", "max", "topk"],
                         help="How to collapse per-byte surprise into one per-window score. "
                              "'mean' is the original methodology (can dilute a small anomalous "
                              "trigger inside a mostly-ordinary window). 'max' uses the single most "
                              "surprising byte in the window. 'topk' averages the top --topk_frac "
                              "fraction of bytes by surprise (a smoothed version of 'max').")
    parser.add_argument("--topk_frac", type=float, default=0.1,
                         help="Fraction of bytes (by surprise, highest first) to average when "
                              "--score_agg=topk (default: 0.1 = top 10%% of the window)")
    parser.add_argument("--complexity_correction", type=str, default="none",
                         choices=["none", "zlib", "bz2"],
                         help="Input-complexity corrected score (Serra et al., ICLR 2020, "
                              "arXiv:1909.11480). Likelihood-based anomaly scores are known to be "
                              "confounded by raw input COMPLEXITY: a window of ciphertext gets high "
                              "surprise simply because it is incompressible, not because it violates "
                              "learned protocol grammar — the classic failure mode of NLL-based OOD "
                              "detection (Nalisnick et al., arXiv:1810.09136). The fix is a "
                              "likelihood-ratio-style score S(x) = NLL(x) - L(x), where L(x) is a "
                              "universal-compressor estimate of the window's Kolmogorov complexity "
                              "(bits/byte via zlib or bz2). Structured-but-unusual attack bytes keep "
                              "a high corrected score; mere randomness is cancelled out.")
    parser.add_argument("--typicality", action="store_true", default=False,
                         help="Two-sided typicality score |s - mean(benign calibration s)| "
                              "(Nalisnick et al., arXiv:1906.02994): flags windows that are "
                              "TOO-predictable as well as too-surprising. Catches low-entropy "
                              "attacks (padding floods, repeated probes, C2 heartbeats) that "
                              "one-sided surprise misses by construction.")
    parser.add_argument("--num_workers", type=int, default=None,
                         help="Number of background workers for data loading (default: auto — 4 for CUDA, 0 otherwise)")
    parser.add_argument("--fp16", action="store_true", default=False,
                         help="Use FP16 mixed-precision inference (faster on CUDA, auto-enabled on CUDA unless --no_fp16 is set)")
    parser.add_argument("--no_fp16", action="store_true", default=False,
                         help="Disable automatic FP16 inference even on CUDA")
    parser.add_argument("--max_pcap_size_mb", type=float, default=None,
                         help="Skip files in the calibration attack directory larger than this MB size (default: None)")
    return parser.parse_args()


def load_model(checkpoint_path, device, override_seq_len=None):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at '{checkpoint_path}'", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]

    has_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if has_prefix:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    pos_weight = state_dict.get("pos_embedding.weight")
    checkpoint_max_seq_len = pos_weight.shape[0] if pos_weight is not None else 8192

    # IMPORTANT: the model architecture (and its pos_embedding table) must always be
    # built at the checkpoint's native size, or load_state_dict will fail on a shape
    # mismatch. A shorter --max_sequence_length only limits how long a window we feed
    # through the model during *scoring* (see main()) — the model just uses the first
    # N rows of the same embedding table, which is always valid since N <= native size.
    is_mamba = any("A_log" in k or "conv1d" in k for k in state_dict.keys())
    if is_mamba:
        from src.model_mamba import MambaBytePatcher
        model = MambaBytePatcher(max_sequence_length=checkpoint_max_seq_len).to(device)
        print(f"Detected Mamba-2 Backbone Checkpoint")
    else:
        model = NetworkBytePatcher(max_sequence_length=checkpoint_max_seq_len).to(device)
        print(f"Detected Transformer Backbone Checkpoint")
    model.load_state_dict(state_dict)
    model.eval()

    eval_window_len = override_seq_len or checkpoint_max_seq_len
    if eval_window_len > checkpoint_max_seq_len:
        print(f"WARNING: requested --max_sequence_length {eval_window_len} exceeds the "
              f"checkpoint's native {checkpoint_max_seq_len}; capping to native size.")
        eval_window_len = checkpoint_max_seq_len

    print(f"Loaded checkpoint '{checkpoint_path}' | native max_sequence_length={checkpoint_max_seq_len} "
          f"| scoring window length={eval_window_len}")
    return model, eval_window_len


def _window_complexity_bits_per_byte(rows, method):
    """
    Universal-compressor estimate L(x) of each window's complexity, in bits/byte.
    rows: (B, T) numpy int array with -1 padding sentinels.
    """
    out = np.full(rows.shape[0], np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        valid = row[row >= 0]
        if valid.size == 0:
            continue
        raw = valid.astype(np.uint8).tobytes()
        comp = zlib.compress(raw, 6) if method == "zlib" else bz2.compress(raw, 9)
        out[i] = 8.0 * len(comp) / len(raw)
    return out


@torch.no_grad()
def score_pcap(model, pcap_path, device, batch_size, max_sequence_length,
               num_workers=0, agg="mean", topk_frac=0.1,
               complexity_correction="none", use_fp16=False):
    """
    Returns a list of per-window "surprise" scores (bits) — negative log2
    probability the model assigned to the true next byte in each window,
    collapsed to one score per window via `agg`. NaN-safe against -1
    padding sentinels.

    agg="mean"  : average surprise across the whole window (original method;
                  a small anomalous trigger inside a mostly-ordinary window
                  gets diluted by all the ordinary bytes around it).
    agg="max"   : the single most surprising byte in the window. Sensitive
                  to a lone outlier, but can be noisy (one weird-but-benign
                  byte can trigger it).
    agg="topk"  : average of the top `topk_frac` fraction of bytes by
                  surprise — a smoothed middle ground between mean and max.
    """
    if not os.path.exists(pcap_path):
        print(f"  WARNING: '{pcap_path}' not found, skipping.")
        return []

    dataloader = get_pcap_dataloader(
        pcap_path=pcap_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_sequence_length=max_sequence_length,
        pin_memory=(device.type == "cuda"),
        # Eval must not pollute data/anomaly_labels.csv with side-channel rows,
        # and skipping the per-packet scapy parse roughly halves scoring time.
        label_anomalies=False,
    )

    scores = []
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16 else torch.amp.autocast("cpu", enabled=False)
    with torch.no_grad(), amp_ctx:
        for batch in dataloader:
            comp_bits = None
            if complexity_correction != "none":
                # Complexity of the TARGET bytes (batch[:, 1:]), matching what the
                # surprise score is computed over. Done on the CPU copy pre-transfer.
                comp_bits = _window_complexity_bits_per_byte(batch[:, 1:].numpy(), complexity_correction)
            batch = batch.to(device, non_blocking=True)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            valid_mask = targets != -1

            logits = model(inputs)  # model handles -1 clamping internally
            # Cast back to float32 for numerical stability of log_softmax
            logits = logits.float()
            log_probs = F.log_softmax(logits, dim=-1)

            gather_idx = torch.clamp(targets, min=0).unsqueeze(-1)
            token_logprob = log_probs.gather(-1, gather_idx).squeeze(-1)
            surprise_bits = -token_logprob / math.log(2)

            valid_counts = valid_mask.sum(dim=1)

            if agg == "mean":
                surprise_masked = surprise_bits.masked_fill(~valid_mask, float("nan"))
                per_window = torch.nanmean(surprise_masked, dim=1)
            elif agg == "max":
                filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
                per_window = filled.max(dim=1).values
                per_window = torch.where(valid_counts > 0, per_window, torch.full_like(per_window, float("nan")))
            elif agg == "topk":
                filled = torch.where(valid_mask, surprise_bits, torch.full_like(surprise_bits, float("-inf")))
                k = max(1, min(filled.shape[1], int(round(topk_frac * filled.shape[1]))))
                topk_vals, _ = torch.topk(filled, k=k, dim=1)
                topk_valid = torch.isfinite(topk_vals)
                topk_vals = torch.where(topk_valid, topk_vals, torch.zeros_like(topk_vals))
                denom = topk_valid.sum(dim=1).clamp(min=1)
                per_window = topk_vals.sum(dim=1) / denom
                per_window = torch.where(valid_counts > 0, per_window, torch.full_like(per_window, float("nan")))
            else:
                raise ValueError(f"Unknown --score_agg: {agg}")

            per_window = per_window.cpu().numpy()
            if comp_bits is not None:
                # Likelihood-ratio-style correction: S = NLL - L (both bits/byte).
                # For agg="mean" this is exactly Serra et al.'s parameter-free OOD
                # score restricted to the window; for max/topk it is the same
                # correction applied to the aggregated statistic (heuristic but
                # consistently applied to benign and attack windows alike).
                per_window = per_window - comp_bits
            scores.extend([s for s in per_window.tolist() if not math.isnan(s)])

    return scores



def split_calibration_holdout(scores):
    """Split one file's per-window scores into a (calibration, holdout) pair --
    first half / second half. Used as a same-file smoke-test fallback in main()
    when a dedicated distinct calibration or holdout file isn't available (e.g.
    a minimal 'out-of-the-box' sample dataset that ships only one benign pcap
    and/or one attack pcap). NOT a substitute for genuinely distinct captures --
    callers must surface this as a caveat, not a real generalization number."""
    half = len(scores) // 2
    return scores[:half], scores[half:]


def discover_attack_files(attack_dir, exclude_basenames, max_size_mb=None):
    files = sorted(glob.glob(os.path.join(attack_dir, "*.pcap")))
    filtered = []
    for f in files:
        if os.path.basename(f) in exclude_basenames:
            continue
        if max_size_mb is not None:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            if size_mb > max_size_mb:
                continue
        filtered.append(f)
    return filtered


def pick_thresholds(y_true, y_scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    # Youden's J statistic: maximize (tpr - fpr)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    youden_threshold = float(thresholds[best_idx])
    youden_tpr, youden_fpr = float(tpr[best_idx]), float(fpr[best_idx])

    # Nearest point to the requested target FPR
    fpr_idx = int(np.argmin(np.abs(fpr - target_fpr)))
    target_threshold = float(thresholds[fpr_idx])
    target_tpr, target_fpr_actual = float(tpr[fpr_idx]), float(fpr[fpr_idx])

    return {
        "auc": float(auc),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
        "youden": {"threshold": youden_threshold, "tpr": youden_tpr, "fpr": youden_fpr},
        "target_fpr": {"requested": target_fpr, "threshold": target_threshold,
                        "tpr": target_tpr, "fpr": target_fpr_actual},
    }


def fit_evt_threshold(benign_scores, target_fpr, tail_quantile=0.98):
    """
    Extreme Value Theory (Peaks-Over-Threshold) threshold selection.

    Youden's J picks a threshold that maximizes (TPR - FPR) over the WHOLE
    ROC curve — the right objective for overall separation, but the wrong
    one when what actually matters operationally is the extreme tail (a
    specific low FPR like 0.1-1%). Empirically across this project's
    checkpoints, Youden's threshold has proven noisy: similar calibration
    AUC has produced held-out detection rates swinging from ~3% to ~48%
    depending on exactly where the ROC curve happened to bend on a finite
    calibration sample.

    POT instead: pick a high quantile `tail_quantile` of the calibration
    BENIGN scores as an initial threshold t0, fit a Generalized Pareto
    Distribution to the excesses above t0, then solve analytically for the
    score threshold that should produce exactly `target_fpr` false
    positives on benign data, under the fitted tail model. This targets the
    tail directly instead of relying on wherever the calibration sample's
    ROC curve happens to bend. Method follows Siffer et al., "Anomaly
    Detection in Streams with Extreme Value Theory" (KDD 2017).
    """
    arr = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(arr)
    if n < 30:
        return None  # not enough calibration data for a meaningful tail fit

    t0 = float(np.quantile(arr, tail_quantile))
    excesses = arr[arr > t0] - t0
    Nt = len(excesses)
    if Nt < 10:
        return None  # tail sample too thin to fit a GPD reliably

    # Fit GPD to the excesses. loc is fixed at 0 since excesses are >= 0 by
    # construction (they're defined as (score - t0) for scores above t0).
    shape, _, scale = genpareto.fit(excesses, floc=0)

    q = target_fpr
    if abs(shape) < 1e-6:
        # Degenerate GPD (shape -> 0) reduces to an exponential tail.
        threshold = t0 - scale * math.log(q * n / Nt)
    else:
        threshold = t0 + (scale / shape) * (((n / Nt) * q) ** (-shape) - 1)

    return {
        "threshold": float(threshold),
        "tail_quantile": tail_quantile,
        "t0": t0,
        "gpd_shape": float(shape),
        "gpd_scale": float(scale),
        "num_tail_points": Nt,
        "target_fpr": target_fpr,
    }


def apply_threshold(scores, threshold):
    if not scores:
        return None
    arr = np.array(scores)
    return float(np.mean(arr > threshold))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device)
        except Exception as e:
            print(f"WARNING: CUDA device acceleration failed ({e}); falling back to CPU.")
            device = torch.device("cpu")

    # Auto-scale batch_size and num_workers based on device
    if args.batch_size is None:
        args.batch_size = 512 if device.type == "cuda" else 32
    if args.num_workers is None:
        args.num_workers = 4 if device.type == "cuda" else 0

    # Auto-enable FP16 on CUDA unless explicitly disabled
    use_fp16 = (device.type == "cuda") and (not args.no_fp16)
    if args.fp16:
        use_fp16 = True  # explicit override
    if use_fp16:
        print(f"FP16 AMP inference: ENABLED (use --no_fp16 to disable)")
    print("==================================================")
    print("   ZERO-DAY PROOF-OF-WORK EVALUATION HARNESS")
    print("==================================================")
    print(f"Device: {device}")
    agg_label = f"{args.score_agg}" + (f" (top {args.topk_frac:.0%})" if args.score_agg == "topk" else "")
    if args.complexity_correction != "none":
        agg_label += f" - {args.complexity_correction} complexity"
    if args.typicality:
        agg_label = f"|{agg_label} - benign mean|"
    print(f"Score aggregation: {agg_label}")

    model, max_seq_len = load_model(args.checkpoint_path, device, args.max_sequence_length)

    # Wrap with DataParallel if multiple GPUs are available
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Multi-GPU: wrapping model with DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    print(f"Batch size: {args.batch_size} | DataLoader workers: {args.num_workers}")

    exclude = {
        os.path.basename(args.benign_calibration_pcap),
        os.path.basename(args.benign_holdout_pcap),
        os.path.basename(args.holdout_attack_pcap),
    }
    attack_files = discover_attack_files(args.attack_dir, exclude, max_size_mb=args.max_pcap_size_mb)
    print(f"\nCalibration attack files ({len(attack_files)}): "
          f"{[os.path.basename(f) for f in attack_files]}")


    caveats = []

    # --- Calibration set: benign ---
    # If the calibration and holdout benign files are literally the same path (common
    # with a minimal "out-of-the-box" sample dataset that only ships one benign pcap),
    # scoring it twice would silently report a meaningless "held-out" FPR -- it's the
    # exact same windows, not a genuine generalization check. Score once and split the
    # windows in half instead, with a loud warning.
    benign_same_file = os.path.abspath(args.benign_calibration_pcap) == os.path.abspath(args.benign_holdout_pcap)
    holdout_benign_scores_presplit = None
    if benign_same_file:
        msg = (f"--benign_calibration_pcap and --benign_holdout_pcap are the same file "
               f"('{args.benign_calibration_pcap}'). Falling back to a same-file split "
               f"(first half of windows = calibration, second half = holdout) instead of "
               f"double-scoring identical data as if it were held out. This is a SMOKE-TEST "
               f"split, not a genuine cross-capture generalization check -- pass a distinct "
               f"--benign_holdout_pcap for a real number.")
        print(f"\nWARNING: {msg}")
        caveats.append(msg)
        print(f"Scoring benign file (calibration + holdout, same-file split): {args.benign_calibration_pcap}")
        all_benign_scores = score_pcap(model, args.benign_calibration_pcap, device, args.batch_size, max_seq_len,
                                        num_workers=args.num_workers,
                                        agg=args.score_agg, topk_frac=args.topk_frac,
                                        complexity_correction=args.complexity_correction,
                                        use_fp16=use_fp16)
        benign_scores, holdout_benign_scores_presplit = split_calibration_holdout(all_benign_scores)
        print(f"  -> {len(all_benign_scores)} windows scored total "
              f"({len(benign_scores)} calibration / {len(holdout_benign_scores_presplit)} holdout)")
    else:
        print(f"\nScoring benign calibration file: {args.benign_calibration_pcap}")
        benign_scores = score_pcap(model, args.benign_calibration_pcap, device, args.batch_size, max_seq_len,
                                    num_workers=args.num_workers,
                                    agg=args.score_agg, topk_frac=args.topk_frac,
                                    complexity_correction=args.complexity_correction,
                                    use_fp16=use_fp16)
        print(f"  -> {len(benign_scores)} windows scored")

    # --- Calibration set: attacks ---
    # Same problem, attack-side: a minimal sample dataset may ship exactly ONE attack
    # pcap, which discover_attack_files() correctly excludes from calibration because
    # it's also --holdout_attack_pcap (avoiding leakage) -- but that leaves zero
    # calibration attack files and the run has nothing to fit a threshold against.
    # Fall back to scoring that one file once and splitting it, same as above, rather
    # than hard-erroring on what is otherwise a working pipeline.
    per_file_attack_scores = {}
    holdout_attack_scores_presplit = None
    if not attack_files:
        msg = (f"no calibration attack files found in '{args.attack_dir}' after excluding "
               f"the benign/holdout files ({sorted(exclude)}). Falling back to a same-file "
               f"split of the holdout attack file '{args.holdout_attack_pcap}' (first half of "
               f"windows = calibration, second half = holdout) instead of erroring out. This "
               f"is a SMOKE-TEST split, not a genuine cross-capture generalization check -- add "
               f"a second, distinct attack pcap to --attack_dir for a real zero-day proof.")
        print(f"\nWARNING: {msg}")
        caveats.append(msg)
        print(f"Scoring attack file (calibration + holdout, same-file split): {args.holdout_attack_pcap}")
        all_attack_scores = score_pcap(model, args.holdout_attack_pcap, device, args.batch_size, max_seq_len,
                                        num_workers=args.num_workers,
                                        agg=args.score_agg, topk_frac=args.topk_frac,
                                        complexity_correction=args.complexity_correction,
                                        use_fp16=use_fp16)
        attack_scores, holdout_attack_scores_presplit = split_calibration_holdout(all_attack_scores)
        per_file_attack_scores[os.path.basename(args.holdout_attack_pcap) + " (calibration half)"] = attack_scores
        print(f"  -> {len(all_attack_scores)} windows scored total "
              f"({len(attack_scores)} calibration / {len(holdout_attack_scores_presplit)} holdout)")
    else:
        attack_scores = []
        for f in attack_files:
            print(f"Scoring attack file: {f}")
            s = score_pcap(model, f, device, args.batch_size, max_seq_len,
                            num_workers=args.num_workers,
                            agg=args.score_agg, topk_frac=args.topk_frac,
                            complexity_correction=args.complexity_correction,
                            use_fp16=use_fp16)
            print(f"  -> {len(s)} windows scored")
            per_file_attack_scores[os.path.basename(f)] = s
            attack_scores.extend(s)

    if not benign_scores or not attack_scores:
        print("ERROR: Insufficient calibration data (need both benign and attack windows, even "
              "after the same-file split fallback -- the source file(s) are too small to yield "
              "even 2 windows each). Provide a larger sample pcap or a second distinct file.",
              file=sys.stderr)
        sys.exit(1)

    MIN_RECOMMENDED_WINDOWS = 30
    if len(benign_scores) < MIN_RECOMMENDED_WINDOWS or len(attack_scores) < MIN_RECOMMENDED_WINDOWS:
        msg = (f"calibration set is small (benign={len(benign_scores)}, attack={len(attack_scores)} "
               f"windows). Below ~{MIN_RECOMMENDED_WINDOWS} windows per class, the ROC/threshold "
               f"numbers below are a pipeline smoke test, not a statistically meaningful benchmark.")
        print(f"\nWARNING: {msg}")
        caveats.append(msg)

    # --- Optional two-sided typicality transform (arXiv:1906.02994) ---
    # High likelihood is NOT the same as typical: benign traffic concentrates in
    # a typical set of near-average surprise, and anomalies can fall on EITHER
    # side (ciphertext-like: too surprising; padding floods / repeated probes:
    # too predictable). The transform |s - mu_benign| makes both tails score high.
    typicality_mu = None
    if args.typicality:
        typicality_mu = float(np.mean(benign_scores))
        print(f"Typicality transform enabled: mu_benign = {typicality_mu:.3f} bits")
        benign_scores = [abs(s - typicality_mu) for s in benign_scores]
        attack_scores = [abs(s - typicality_mu) for s in attack_scores]
        per_file_attack_scores = {k: [abs(s - typicality_mu) for s in v]
                                  for k, v in per_file_attack_scores.items()}

    y_true = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
    y_scores = np.array(benign_scores + attack_scores)
    calib_metrics = pick_thresholds(y_true, y_scores, args.target_fpr)

    print(f"\nCalibration AUC: {calib_metrics['auc']:.4f}")
    print(f"Youden threshold: {calib_metrics['youden']['threshold']:.3f} bits "
          f"(TPR={calib_metrics['youden']['tpr']:.3f}, FPR={calib_metrics['youden']['fpr']:.3f})")
    print(f"Target-FPR threshold (~{args.target_fpr:.1%}): "
          f"{calib_metrics['target_fpr']['threshold']:.3f} bits "
          f"(TPR={calib_metrics['target_fpr']['tpr']:.3f}, FPR={calib_metrics['target_fpr']['fpr']:.3f})")

    evt_result = fit_evt_threshold(benign_scores, args.target_fpr, tail_quantile=args.evt_tail_quantile)
    if evt_result is not None:
        print(f"EVT/POT threshold (targeting {args.target_fpr:.1%} FPR, GPD shape={evt_result['gpd_shape']:.3f}, "
              f"{evt_result['num_tail_points']} tail points above the {args.evt_tail_quantile:.0%} quantile): "
              f"{evt_result['threshold']:.3f} bits")
    else:
        print("EVT/POT threshold: skipped (not enough calibration benign windows for a reliable tail fit)")

    # --- True held-out generalization check ---
    # (skipped/reused above when a same-file split fallback already produced these halves)
    print(f"\nScoring HELD-OUT benign file (never used for calibration): {args.benign_holdout_pcap}")
    if holdout_benign_scores_presplit is not None:
        holdout_benign_scores = holdout_benign_scores_presplit
        print(f"  -> {len(holdout_benign_scores)} windows scored (reused from the same-file split above)")
    else:
        holdout_benign_scores = score_pcap(model, args.benign_holdout_pcap, device, args.batch_size, max_seq_len,
                                            num_workers=args.num_workers,
                                            agg=args.score_agg, topk_frac=args.topk_frac,
                                            complexity_correction=args.complexity_correction,
                                            use_fp16=use_fp16)
        print(f"  -> {len(holdout_benign_scores)} windows scored")

    print(f"Scoring HELD-OUT attack file (never used for calibration): {args.holdout_attack_pcap}")
    if holdout_attack_scores_presplit is not None:
        holdout_attack_scores = holdout_attack_scores_presplit
        print(f"  -> {len(holdout_attack_scores)} windows scored (reused from the same-file split above)")
    else:
        holdout_attack_scores = score_pcap(model, args.holdout_attack_pcap, device, args.batch_size, max_seq_len,
                                            num_workers=args.num_workers,
                                            agg=args.score_agg, topk_frac=args.topk_frac,
                                            complexity_correction=args.complexity_correction,
                                            use_fp16=use_fp16)
        print(f"  -> {len(holdout_attack_scores)} windows scored")

    if typicality_mu is not None:
        # Same transform, same mu (fit on calibration benign only — no leakage).
        holdout_benign_scores = [abs(s - typicality_mu) for s in holdout_benign_scores]
        holdout_attack_scores = [abs(s - typicality_mu) for s in holdout_attack_scores]

    threshold_candidates = [("youden", calib_metrics["youden"]["threshold"]),
                            ("target_fpr", calib_metrics["target_fpr"]["threshold"])]
    if evt_result is not None:
        threshold_candidates.append(("evt", evt_result["threshold"]))

    holdout_results = {}
    for label, thresh in threshold_candidates:
        holdout_results[label] = {
            "threshold": thresh,
            "holdout_benign_false_positive_rate": apply_threshold(holdout_benign_scores, thresh),
            "holdout_attack_detection_rate": apply_threshold(holdout_attack_scores, thresh),
        }

    print("\n================ HELD-OUT GENERALIZATION RESULTS ================")
    for label, r in holdout_results.items():
        fpr_str = f"{r['holdout_benign_false_positive_rate']:.3f}" if r["holdout_benign_false_positive_rate"] is not None else "N/A"
        det_str = f"{r['holdout_attack_detection_rate']:.3f}" if r["holdout_attack_detection_rate"] is not None else "N/A"
        print(f"  [{label}] threshold={r['threshold']:.3f} bits | "
              f"held-out benign FPR={fpr_str} | held-out 0day detection rate={det_str}")
    print("===================================================================\n")

    if caveats:
        print("\n================ CAVEATS (read before trusting these numbers) ================")
        for c in caveats:
            print(f"  - {c}")
        print("===================================================================\n")

    # --- Save artifacts ---
    metrics = {
        "caveats": caveats,
        "checkpoint_path": args.checkpoint_path,
        "max_sequence_length": max_seq_len,
        "score_agg": args.score_agg,
        "topk_frac": args.topk_frac if args.score_agg == "topk" else None,
        "complexity_correction": args.complexity_correction,
        "typicality": args.typicality,
        "typicality_mu": typicality_mu,
        "calibration": {
            "benign_file": args.benign_calibration_pcap,
            "attack_files": list(per_file_attack_scores.keys()),
            "num_benign_windows": len(benign_scores),
            "num_attack_windows": len(attack_scores),
            "auc": calib_metrics["auc"],
            "youden": calib_metrics["youden"],
            "target_fpr": calib_metrics["target_fpr"],
            "evt": evt_result,
            "per_file_mean_score": {
                "benign_calibration": float(np.mean(benign_scores)),
                **{k: float(np.mean(v)) for k, v in per_file_attack_scores.items() if v},
            },
        },
        "holdout": {
            "benign_file": args.benign_holdout_pcap,
            "attack_file": args.holdout_attack_pcap,
            "num_benign_windows": len(holdout_benign_scores),
            "num_attack_windows": len(holdout_attack_scores),
            "results": holdout_results,
        },
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # ROC curve plot
    plt.figure(figsize=(7, 6))
    plt.plot(calib_metrics["roc_curve"]["fpr"], calib_metrics["roc_curve"]["tpr"],
              label=f"ROC (AUC={calib_metrics['auc']:.3f})", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    plt.scatter([calib_metrics["youden"]["fpr"]], [calib_metrics["youden"]["tpr"]],
                color="#d62728", zorder=5, label="Youden's J threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Zero-Day Proof-of-Work: Calibration ROC Curve (agg={agg_label})")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.5)
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {roc_path}")

    # Score distribution plot (calibration + holdout overlay)
    plt.figure(figsize=(10, 6))
    plt.hist(benign_scores, bins=50, alpha=0.5, label="Benign (calibration)", color="#2ca02c", density=True)
    plt.hist(attack_scores, bins=50, alpha=0.5, label="Attacks (calibration)", color="#d62728", density=True)
    if holdout_benign_scores:
        plt.hist(holdout_benign_scores, bins=50, alpha=0.5, label="Benign (held-out)", color="#98df8a",
                  density=True, histtype="step", linewidth=2)
    if holdout_attack_scores:
        plt.hist(holdout_attack_scores, bins=50, alpha=0.5, label="0day (held-out)", color="#ff9896",
                  density=True, histtype="step", linewidth=2)
    plt.axvline(calib_metrics["youden"]["threshold"], color="black", linestyle="--",
                label=f"Youden threshold ({calib_metrics['youden']['threshold']:.2f} bits)")
    if evt_result is not None:
        plt.axvline(evt_result["threshold"], color="#9467bd", linestyle=":",
                    linewidth=2,
                    label=f"EVT/POT threshold ({evt_result['threshold']:.2f} bits, target {args.target_fpr:.1%} FPR)")
    plt.xlabel(f"Next-Byte Surprise (bits), agg={agg_label}")
    plt.ylabel("Density")
    plt.title(f"Zero-Day Proof-of-Work: Score Distributions (agg={agg_label})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    dist_path = os.path.join(args.output_dir, "score_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {dist_path}")

    print("\nProof-of-work evaluation complete.")


if __name__ == "__main__":
    main()
