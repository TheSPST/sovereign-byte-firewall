# Sovereign Byte-Level Firewall — Status Report
**As of: 2026-07-12**

## 1. Goal
Detect zero-day network attacks from raw `.pcap` bytes using a causal transformer (`NetworkBytePatcher`, d_model=128, nhead=4, 2 layers) trained on next-byte prediction. Anomaly score = "surprise" (−log2 P(true next byte)). Target: 80-90% detection at sub-1% false positive rate.

## 2. Best result so far
**Checkpoint gs865000, topk-10% score aggregation, Youden threshold (12.025 bits):**
- Held-out benign FPR: **1.0%**
- Held-out zero-day detection: **32.0%**

**Fused Same-Environment Results (Wednesday CIC-IDS2017 Day Evaluation):**
- **gs75000:** 100% Detection (5/5 attacks). FPR/hour: byte=0.00, rate=0.89, fused=0.89. (Zero byte-level false alarms).
- **gs865000:** 100% Detection (5/5 attacks). FPR/hour: byte=0.54, rate=0.89, fused=1.43. (Highly sensitive, caught Slowhttptest natively on byte detector).

This strongly validates the OR-fusion strategy: the byte model captures payload exploits (Heartbleed) and standard floods (Hulk, GoldenEye), while the rate detector covers the blindspot of slow volumetric attacks (Slowhttptest).

## 3. What we confirmed this session

**gs865000 is a genuine, narrow peak — not a plateau.**
Swept neighboring checkpoints:
| Checkpoint | Calibration AUC | Held-out detection |
|---|---|---|
| gs750000 | 0.702 | 3.2% |
| gs800000 | 0.783 | 2.3% |
| **gs865000** | **0.779** | **32.0%** |

gs800000 and gs865000 have nearly identical calibration AUC but wildly different held-out detection — calibration AUC does **not** reliably predict real zero-day generalization. The 865k point needs a checkpoint_path sanity check (not yet done) given past terminal-paste mix-ups, but the shape of the result is consistent with a real, sharp optimum.

**Training past gs865000 causes real regression, not noise.**
Extended sweep to gs1.76M–gs2.065M (repeated epochs over the same Monday file) showed detection dropping to ~6-7% and staying there — clear evidence of overfitting from re-seeing identical training data, not random variance.

**topk-aggregation breakthrough is robust, not a fluke of exactly 10%.**
Stable AUC (0.75-0.77) across `topk_frac` 5-20%. Does not rescue undertrained checkpoints (gs50000 stays near-random) or fix contaminated ones (gs770000 has high raw AUC but the worst true generalization — a contamination echo, not signal).

**Why 80-90%/sub-1% is a hard target, not a project failure.**
Grounded in literature: the base-rate fallacy (Axelsson 2000) makes low FPR the dominant constraint on any IDS; CIC-IDS2017 has documented data-quality issues (Engelen et al. 2021, >20% of traces needed relabeling) independent of our own earlier contamination bug; TLS/encryption entropy is known to swamp payload-based anomaly detection.

## 4. What we built/fixed this session

| Item | Status |
|---|---|
| EVT/POT thresholding (`evaluate_zero_day.py`) | Implemented. On gs865000 it converged to ~the same threshold as the already-broken naive target-FPR method (0% detection) — **no improvement over Youden here**. Inconclusive; not yet tested on a checkpoint where Youden's natural FPR is far from 1% (where EVT should have more room to help). |
| TLS continuation-masking fix (`src/dataloader.py`) | Found and fixed a real gap — TLS record bytes spanning multiple packets weren't being masked correctly. Unit-tested. **Needs a fresh retrain to take effect** — no existing checkpoint benefits from it yet. |
| Rate-based companion detector (`evaluate_rate_based.py`, new) | Built and validated. AUC 0.908. Catches `hydra_ssh` (SYN-rate spike) which the byte-level model is structurally blind to. |
| OR-fusion (`fuse_detectors.py`, new) | Built and validated. 18/24 calibration attack files caught by at least one detector. `hydra_ftp`/`hydra_ftp2` still slip through both — its rate (11.6 SYNs/window) is well under the 34.0 threshold; lowering the threshold to catch it would flag benign noise elsewhere. Real fix needs a longer aggregation window or session-based features, not a threshold tweak. |
| AI Kosh SLURM path audit (`sbatch_train.sh`) | Found and fixed a real bug: FocalLoss was hardcoded on despite being proven worse earlier in the project. Also fixed: `EPOCHS` default 10→1 (overfitting evidence above), `HF_REPO_ID` changed to `spst01/sovereign-byte-firewall-aikosh` (avoids checkpoint-filename collision with two other repos already in use). Flagged the default dataset path as a known 0-byte placeholder that must be overridden explicitly. |

## 5. Operational context
- Kaggle GPU quota resets ~2026-07-18 (was 23h/30h left as of 2026-07-11).
- **Full-day A100 access via AI Kosh on 2026-07-15** — the precious, one-shot window this week's SLURM audit was done to protect.
- Separate Kaggle training session continuing independently past gs2M steps (repo `spst01/sovereign-byte-firewall-monday`).
- AI Kosh run decisions already made: keep current model size (128-dim/2-layer), train on the Monday file capped at ~1 epoch.

## 6. Open / pending
1. **Not yet done:** commit and push this session's code changes to GitHub (`evaluate_zero_day.py`, `src/dataloader.py`, `evaluate_rate_based.py`, `fuse_detectors.py`, `sbatch_train.sh`) — needed before cloning fresh for AI Kosh.
2. **Not yet built:** an active-monitoring checklist/script for the AI Kosh day — pulling checkpoints periodically from the new HF repo during the run and killing the job early if detection regresses, using the gs865000 peak as the benchmark to beat.
3. **Unresolved:** `hydra_ftp` gap in the rate-based detector.
4. **Inconclusive:** whether EVT thresholding is genuinely more useful than Youden anywhere — only tested on one checkpoint so far.
5. **Deprioritized until after AI Kosh results:** bigger-lift R&D (Deep SVDD one-class objective, temporal graph neural networks).

## 7. Recommended order for the next 3 days
1. Git push (fast, protects all work done this session).
2. Build the AI Kosh monitoring checklist (needed before the 15th regardless of what else happens).
3. If time remains: decide between finishing the checkpoint sweep (800k-865k) or testing EVT on a checkpoint with a higher natural Youden FPR (e.g. gs590000) to get a real verdict on EVT's value.
