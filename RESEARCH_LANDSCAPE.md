# Research & Tech-Stack Landscape (survey July 2026)
**Purpose:** keep architecture, positioning, and roadmap decisions grounded in current literature. Sources are arXiv IDs / links inline.

---

## Where our approach sits

We run a small (~1.6M param) **byte-level causal transformer** doing next-byte prediction on raw pcap, scoring by *surprise* (−log₂P), unsupervised, calibrated per-environment, fused with a rate detector. That places us at the **lightweight / sovereign / anomaly-first** corner of a field currently dominated by large pre-trained *classification* models. Understanding the neighbours tells us what to adopt, what to defend against, and where our niche is genuinely defensible.

---

## 1. Foundation models for network traffic (the "big" neighbours)

The dominant research thread is *train-once, adapt-anywhere* foundation models pre-trained on unlabelled traffic, then fine-tuned:

- **netFound** (arXiv:2310.17025) — self-supervised, hierarchical/multi-modal, beats SOTA on traffic classification, NIDS, and APT detection; explicitly designed to fine-tune on low-quality/limited labels.
- **ET-BERT, TrafficFormer, YaTC, TrafficMAE** — masked-byte / transformer pre-training for encrypted traffic classification.
- Systematic review: *Network traffic foundation models* (ScienceDirect S1389128626000101), 51 studies 2017–2025.

**Implication for us:** do **not** try to out-scale netFound — it needs GPU-cluster pre-training and a labelled fine-tune set. Our differentiation is the opposite end: tiny, on-prem, no pre-training corpus required, anomaly-first (no attack labels needed), runs on a laptop. Position as **the sovereign-edge alternative**, not a competitor on the classification benchmark leaderboard.

---

## 2. Mamba / state-space models — the architecture upgrade to watch

Linear-time SSMs are displacing quadratic attention for packet models, which matters directly for our edge/pre-filter throughput goals:

- **NetMamba** (arXiv:2405.11449) — unidirectional Mamba, **1.22–60× faster inference** than transformer baselines, low GPU memory, >90–99% accuracy, strong few-shot.
- **NetMamba+** (arXiv:2601.21792, Jan 2026) — SSM + Flash-Attention hybrid.
- **MambaNetBurst** (arXiv:2605.11034, 2026) — **direct byte-level classification, no tokenization, no pre-training** — architecturally the closest to us.

**Implication:** our CLAUDE.md already prizes "sub-quadratic attention"; Mamba is the concrete next step. A NetMamba-style backbone could cut per-packet inference cost 1–60× and lower the "what does it cost at the edge" number that the pre-filter pitch depends on. **Action: prototype a Mamba backbone as a v2 architecture experiment after first-customer.** MambaNetBurst proves byte-level Mamba works without the pre-training we don't do.

---

## 3. The byte-flattening critique — a direct challenge to our thesis (address head-on)

Several 2025–2026 papers argue byte-level modelling is fundamentally flawed:

- *Where Do Flow Semantics Reside?* (arXiv:2603.10051) and *Debunking Representation Learning for Encrypted Traffic* (ACM 3718958.3750498): under **frozen-encoder** evaluation, accuracy collapses **0.9 → <0.47**; previously reported high numbers were **data leakage**, not learned representation. Named failure modes: (1) **field unpredictability** — random fields like `ip.id` are unlearnable yet used as reconstruction targets; (2) **embedding confusion** — semantically distinct fields collapse together; (3) **metadata loss**.

**Why this is a manageable objection for us — and partly a validation:**
1. **We already mask the exact fields they name.** `src/dataloader.py` masks `ip.id`, checksums, addresses, TCP options, and encrypted payloads — precisely the "unpredictable / random" fields the critique says pollute byte modelling. Our design pre-empts issue (1).
2. **The leakage critique targets *supervised classification* benchmarks.** We do **unsupervised anomaly detection** with a strict zero-day protocol (train benign-only, held-out unseen attack class). That protocol is structurally resistant to the label-leakage they debunk.
3. **Our cross-dataset transfer is direct counter-evidence.** A CIC-trained model scoring **AUC 0.75 on UNSW-NB15** (a different capture) cannot be memorising one dataset — the exact failure the frozen-encoder critique exposes in others.

**Action:** add a short "Why byte-level, and why the leakage critique doesn't apply here" note to the technical brief — turning a known objection into a credibility moment. This is the single most important thing to have an answer ready for with a technical buyer.

Also note: *Convolutions are Competitive with Transformers for Encrypted Traffic* (arXiv:2508.02001) — a reminder that architecture is not the moat; the masking + anomaly framing + sovereign deployment is.

---

## 4. Zero-day / unsupervised peers (our direct cohort)

- **AEGIS** (arXiv:2604.02149, 2026) — *adversarial entropy-guided* state-space models for zero-day evasion detection. Closest in spirit to our entropy/surprise signal; worth reading for how they combine entropy + SSM.
- **NetVAD** (arXiv:2606.01452, 2026) — foundation-model reps for **identifier-free unsupervised** intrusion detection; philosophically aligned (no reliance on IPs/ports).
- **Contrastive self-supervised zero-day** (Koukoulis et al., 2025) — augment flow sequences, maximise agreement; adapts to concept drift.
- **GraphIDS** (Guerra et al., 2025) — graph reps + transformer autoencoder for local+global structure.
- **In-context learning for zero-day** (arXiv:2501.16453) — detect novel attacks with few examples, no retraining.

**Implication:** our surprise+fusion approach is in good, current company. Two concrete upgrades already on our radar are corroborated: **CUSUM/sequential scoring** (untested per our memory) and **contrastive self-supervised pre-training** as a way to sharpen benign representation without labels. Entropy-guided (AEGIS) validates the entropy signal but also shows the frontier is entropy **+ sequence model**, which is what we already do.

---

## 5. LLM-assisted SOC triage — validates our Phase 3, with guardrails

The SOC-automation literature exploded in 2025–2026:

- 80% of SOC alerts are false positives; alert fatigue is the named crisis (survey arXiv:2509.10858; MDPI 2624-800X/5/4/95).
- **Local** models automate ~78% of triage (Llama 3.1); **CORTEX** (arXiv:2510.00311) collaborative LLM agents for high-stakes triage; agentic tool-augmented investigation without training data.
- Metrics that matter: MTTD, MTTR, FP rate, analyst productivity.

**Implication:** our PRODUCT_ROADMAP Phase 3 (local LLM triage summaries) is squarely on-trend, and the **local/sovereign** angle is exactly what this literature is moving toward (data can't leave the network). Keep our earlier guardrail: the LLM summarises *grounded facts* (talkers, ports, surprise heatmap), it does not speculate. The 78%-automation figure is a credible north star for a paid enterprise feature.

---

## 6. Deployment: eBPF/XDP is the line-rate pre-filter path

Our FUTURE_ROADMAP pre-filter vision has a concrete 2025–2026 deployment pattern:

- **In-kernel CNN inference via eBPF** (Springer 978-981-95-8417-8_34); **SmartX Intelligent Sec** (arXiv:2410.20244) XDP capture + BiLSTM, 99.3% acc; the emerging pattern of "**intelligent in-kernel XDP data paths that pre-filter and steer traffic to ML services**."

**Implication:** the architecture for our pre-filter is: **XDP does cheap in-kernel steering; our byte-model runs as the userspace ML service it steers the *suspicious* fraction to.** Our transformer is too heavy for in-kernel, but that's fine — it's the second-stage brain, not the first-stage filter. This grounds the "triage filter in front of the heavy stack" story in a real, current deployment idiom and is worth citing in the FUTURE_ROADMAP.

---

## Decision summary (ranked)

1. **Have the byte-flattening rebuttal ready** (Section 3) — highest priority, it's the sharpest technical objection and we have a strong, honest answer. Add to the brief.
2. **Prototype a Mamba backbone (v2, post-first-customer)** — 1–60× inference speedup directly serves edge/pre-filter economics; MambaNetBurst shows byte-level Mamba needs no pre-training.
3. **Frame pre-filter deployment as XDP-steer + userspace byte-model** — real, citable, line-rate story for FUTURE_ROADMAP.
4. **Keep LLM-triage as a grounded, local Phase-3 feature** — strong market validation, sovereign angle aligned.
5. **Own the sovereign-edge niche vs foundation models** — don't chase netFound's benchmark; our moat is tiny + on-prem + anomaly-first + no labels.
6. **Next scoring experiments corroborated:** CUSUM/sequential + contrastive SSL pre-training (both already flagged internally).

## Watch-list (re-check quarterly)
netFound · NetMamba/NetMamba+/MambaNetBurst · AEGIS · NetVAD · "Where Do Flow Semantics Reside?" · CORTEX / local-LLM SOC triage · eBPF/XDP inline-ML frameworks · Awesome-NTA (github.com/wangtz19/Awesome-NTA) as a living index.
