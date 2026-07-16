# Sovereign Byte-Level Firewall — Project Report & Go-to-Market Plan
**As of: 2026-07-16**

## 1. One-line status
Research-stage IDS with a working live demo. Core science is validated but detection is at ~33% zero-day @ <1% FPR standalone (100% on fused same-environment eval) — strong enough to sell a **paid pilot**, not yet a product.

## 2. Where the project stands

### Results (verified against eval data)
| Metric | Value | Source |
|---|---|---|
| Best held-out zero-day detection (v2-masking, gs75000) | **32.6% @ 0.23% FPR** (calib AUC 0.726) | `eval_watcher_results.csv` |
| Previous peak (old masking, gs865000) | 32.0% @ 1.0% FPR | `PROJECT_STATUS_2026-07-12.md` |
| Fused (byte + rate) on CIC-IDS2017 Wednesday | **100% detection (5/5 attacks)**, 0.89–1.43 false alarms/hour | same-environment eval |
| Rate detector standalone | AUC 0.908, catches SYN-rate attacks byte model misses | `evaluate_rate_based.py` |

Key insight now proven: **OR-fusion works** — the byte model catches payload exploits (Heartbleed) and floods (Hulk, GoldenEye); the rate detector covers slow attacks (Slowhttptest). The v2-masking retrain matches the old peak at 4× lower FPR and much earlier in training (75k vs 865k steps).

### What's built
- Training pipeline: Kaggle-based (A100/AI Kosh path also scripted), HF Hub checkpointing (`spst01/sovereign-byte-firewall*`), auto-resume, parallel eval watcher.
- Evaluation harness: `evaluate_zero_day.py` (topk-10% aggregation + Youden threshold — the proven scoring recipe), CIC full-day eval, fusion eval.
- **Live deployment (new)**: `firewall_daemon.py` — scapy sniffer + fused detectors + WebSocket alerts, and a real-time web `dashboard/`. This is your demo asset.
- 9 test files incl. TLS-masking and P0-fix regression tests.

### Known gaps / risks
1. **32% standalone is far from the 80–90% target.** Literature-grounded reality (base-rate fallacy, TLS entropy) — set expectations accordingly with any buyer.
2. `hydra_ftp` slips through both detectors; needs session-level features, not threshold tweaks.
3. Detection peaks are **narrow and non-monotonic** (gs105k–365k rows are degenerate threshold-collapse; training past the peak regresses to 3–7%). Checkpoint selection must be locked and documented.
4. Uncommitted work: `firewall_daemon.py` (modified), `dashboard/` (untracked). Push it.
5. Single benchmark dataset (CIC-IDS2017, known label-quality issues). No eval yet on a second dataset or real customer traffic — this is the #1 credibility gap for sales.

## 3. Cracking the first customer

### Honest positioning
Do not sell "an IDS" (you'd be compared to Darktrace/Vectra/Suricata and lose). Sell what's uniquely true today:
- **"Zero-day anomaly layer on top of your existing stack"** — signature tools (Suricata/Snort) miss novel payloads; you flag structural anomalies they can't.
- **Sovereign/offline** — no cloud, no vendor telemetry, runs on-prem on a single GPU. This is a real differentiator in Indian gov/defense/BFSI.
- **Zero integration cost** — feed it a span-port pcap; no agents, no log pipeline.

### Who to target (ranked for a solo founder)
1. **Paid PoC via MSSPs / boutique SOCs (fastest cash).** They already have customer pcap and a pain: alert fatigue + zero-day blind spots. Offer: "Give me one week of pcap, I return an anomaly report + FP analysis. ₹X fixed." Consulting-shaped, fits freelancer workflow, converts to a license later.
2. **Indian defense/gov grants (best fit, slower).** The sovereign angle is tailor-made for **iDEX (DISC challenges)**, MeitY GENESIS/TIDE, DSCI, and a C-DAC collaboration (you already have the AI Kosh relationship). Grants are your "first customer" in disguise — non-dilutive and they buy credibility.
3. **University/research paid pilots.** Low friction, publishable validation, but low revenue.

Skip direct enterprise sales for now — 6–12 month cycles and they'll demand certifications (CERT-In empanelment) you don't have yet.

### 30-day action plan
1. **Week 1 — lock the artifact.** Commit dashboard + daemon; pin the winning checkpoint (v2 gs75000) as `release/v0.1`; one-command demo: replay a CIC attack pcap through the daemon → dashboard lights up. Record a 2-min screen capture.
2. **Week 2 — proof pack.** 2-page technical brief with the *honest* numbers (100% fused detection on CIC Wednesday @ ~1 FP/hour; 33% standalone zero-day @ 0.23% FPR; what it misses and why). Honest metrics beat inflated ones in security — buyers test everything.
3. **Week 3 — second dataset.** Run the harness on one more public dataset (e.g. UNSW-NB15 pcaps or CIC-IDS2018). Even a modest result kills the "works on one dataset" objection.
4. **Week 4 — outreach.** (a) Apply to the current iDEX/MeitY open challenges; (b) pitch 5 Indian MSSPs the paid-PoC offer with the video + brief; (c) publish the brief on GitHub/LinkedIn — inbound in this niche is real.

### The one metric to improve before outreach
False-positives-per-hour on *benign real-world traffic* (not CIC). Run the daemon on your own network for 48h and report the number. An SOC's first question is never "what do you catch?" — it's "how often will you wake me at 3am?" 0.89 FP/hour ≈ 21/day is currently too chatty for production; document it and frame the PoC as threshold-tuning per environment.

## 4. Immediate next steps (this week)
1. `git add dashboard/ firewall_daemon.py start_dashboard.sh && commit && push` — demo assets are unprotected.
2. Verify AI Kosh (July 15) run outcome and fold results into this report.
3. Tag `v0.1` with pinned checkpoint + reproducible eval command.
4. Start the 48h benign-traffic FP measurement.
