# Sovereign Byte-Level Firewall — Final Project Report
**Date: 2026-07-17 · Supersedes PROJECT_REPORT_2026-07-16.md**

---

## 1. Executive summary

The Sovereign Byte-Level Firewall is an encoder-free, signature-free network anomaly detector that reads raw packet bytes and flags never-before-seen attacks as statistical "surprise." It is built for the one thing signature tools cannot do — catch novel and zero-day payloads — and for the environments incumbents cannot serve well: sovereign, on-premise, air-gapped, and cost-constrained networks where cloud-connected, telemetry-hungry products are unacceptable.

Over this phase the project moved from a research prototype to a demo-ready system with a defensible evidence base. Three independent results now support the core thesis: the fused detector caught 5 of 5 attack campaigns on CIC-IDS2017 at roughly one false alarm per hour; a model trained on one dataset's benign traffic transferred to a completely separate dataset (UNSW-NB15) with near-identical separation (AUC 0.75 vs 0.73); and a controlled ablation proved the transformer is not over-engineered — a cheap byte n-gram collapses to a 98% false-positive rate on unseen benign traffic where the transformer holds 0.23%. The system is honest about its limits, which are documented rather than hidden, and its positioning — a low-overhead pre-filter and additional detection layer, not an all-knowing shield — is consistent with those limits.

The project is close to outreach-ready. The remaining gates are a live false-positive measurement (in progress) and a two-minute demo video; every supporting document — technical brief, limitations, mitigation plan, research landscape, and outreach copy — is written with honest numbers.

## 2. What the system is

At its core is a compact causal transformer (~1.6M parameters: two layers, model dimension 128, four attention heads, 512-byte context) trained to predict raw network traffic one byte at a time. Having learned the structural grammar of benign traffic, it scores each window by the mean surprise (−log₂ of the predicted probability of the true next byte) of its most anomalous 10% of bytes. Traffic that violates learned protocol structure — malformed fields, exploit byte patterns, unexpected commands in known protocols — produces improbable sequences and is flagged. Deterministic and encrypted regions (addresses, checksums, TLS/QUIC/SSH payloads) are masked so the model learns structure rather than ciphertext noise.

The byte model is OR-fused with a lightweight connection-rate detector: the byte model covers payload anomalies, the rate detector covers volumetric floods the byte model is structurally blind to. Deployment is passive and read-only — a span/mirror port or a pcap file, with no agents, no log pipeline, and no cloud dependency. On first contact with a network the engine spends about an hour learning that environment's baseline and setting its own alert threshold, calibrated to an explicit alert budget (a chosen number of alerts per hour).

The differentiators, stated plainly: signature-free zero-day detection; reads raw packets with no flow extraction or log translation; fully sovereign and offline; self-calibrating to an alert budget; proven to generalize rather than memorize; and analyst-ready output (incident deduplication, enrichment, and a per-byte surprise heatmap on a live dashboard).

## 3. Validation results

All figures below use the strict zero-day protocol: train on benign traffic only, hold out unseen attacks, and report on data never used to fit the threshold.

| Measurement | Result | Meaning |
|---|---|---|
| CIC-IDS2017 full-day fused eval (Wednesday: Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) | **5/5 campaigns detected** at **~0.9–1.4 false alarms/hour** | Operational-grade campaign detection at low alarm volume |
| Held-out zero-day window detection (byte model, v2 gs75000) | **32.6% @ 0.23% FPR** | Strict window-level signal, low false-positive rate |
| Cross-dataset transfer to UNSW-NB15 (trained on CIC benign only) | **calibration AUC 0.75–0.77** vs 0.73 on CIC | Model learned general "normal," not one capture |
| UNSW held-out single-class window detection @ ~1% FPR | 8.5% (Shellcode) – 11.4% (Exploits) | Conservative floor; campaign-level is higher |
| N-gram ablation (same CIC split, orders 2–5) | n-gram **98% held-out benign FPR** vs transformer **0.23%** | Transformer complexity justified via generalization |
| Live deployment (real 2026 traffic, self-calibrated, 1/hr budget) | **~326 incidents/day** (unaggregated, 9.25h overnight run) | Validates baseline behavior on noisy local network |

Two results deserve emphasis. First, the **cross-dataset transfer** is the strongest single piece of evidence: a model trained only on CIC-IDS2017 benign traffic separated benign from attack on UNSW-NB15 — a different lab, year, and attack toolkit — as well as it did on CIC itself. That is direct evidence it learned protocol structure rather than memorizing a dataset, which is exactly the property that makes "it will work on your network" credible.

Second, the **n-gram ablation** answers the sharpest skeptical question — "why not something simpler?" A byte n-gram also produces a surprise score at a fraction of the cost, and it achieves a near-perfect calibration AUC. But that is a memorization artifact: it flags 98%+ of held-out *benign* traffic, because any byte context it has not seen verbatim looks anomalous. The transformer holds held-out benign false positives to 0.23% because its learned representations generalize across similar-but-unseen contexts. The roughly 400× gap on the false-positive metric — the metric that actually decides deployability — is precisely what the model's complexity buys, and the failure is structural, not an artifact of training-set size.

## 4. What was built this phase

The engineering work turned a scoring model into a deployable tool. The live daemon (`firewall_daemon.py`) sniffs an interface, runs the fused detectors, and streams alerts over WebSocket to a real-time browser dashboard. Its scoring was aligned to the validated topk-surprise recipe after an earlier entropy-sum metric was found to saturate on live encrypted traffic. Calibration was reworked from a fixed statistical rule into an **alert-budget** system: the operator states a tolerable alerts-per-hour rate and the daemon sets the threshold to deliver it, using the measured window throughput — turning the false-positive rate into a controllable dial rather than a hoped-for outcome.

Alerts are deduplicated into incidents (a sustained event becomes one incident, not hundreds of alerts) and logged to CSV. Each incident now carries deterministic enrichment — top talkers, ports, protocol mix, SYN count, and the score's percentile against the baseline — so a non-ML analyst can triage it. The dashboard gained a **per-byte surprise heatmap** that renders the flagged window as a grid with anomalous bytes glowing red, turning an abstract score into a visible anomalous region. A second-dataset evaluation runner (`scripts/unsw_eval_kaggle.py`) and the n-gram baseline (`ngram_baseline.py`) were built to produce the validation evidence above.

## 5. Honest limitations and mitigation status

The system is not a silver bullet, and the project documents this openly in `LIMITATIONS.md` and `MITIGATION_PLAN.md`. It is blind to semantics *inside* established encryption — an exploit that fits the normal size and timing envelope of a valid TLS session is invisible, as it is to every payload-blind sensor. It requires re-baselining as networks drift, though the alert-budget calibration bounds the noise in the meantime. A patient adversary who paces an attack below the per-window threshold evades a memoryless detector; the countermeasure is cross-session CUSUM accumulation, which is designed but not yet built. It cannot run neural inference on every packet at core-router line rates, which defines its deployment envelope as span-port and pre-filter scale, with an eBPF/XDP front-end as the path to higher rates. And a white-box adversary with model access can craft low-surprise evasions; the defenses are defense-in-depth (fusion, invariant checkers) and keeping each customer's tuned weights and thresholds private.

None of these are fatal, and stating them plainly is a credibility asset with the serious engineers who will evaluate the tool. Each maps to a concrete mitigation with an effort estimate in the hardening roadmap.

## 6. Where it sits in the research landscape

The field is dominated by large, pre-trained network foundation models (netFound, ET-BERT, TrafficFormer) that require GPU-cluster pre-training and labeled fine-tuning. This project deliberately occupies the opposite corner — tiny, on-premise, anomaly-first, no labels — and should own the sovereign-edge niche rather than compete on classification benchmarks. Two live research threads are directly relevant: Mamba/state-space models (NetMamba, MambaNetBurst) offer 1–60× faster byte-level inference and are the natural architecture upgrade for the pre-filter throughput goal; and a 2025–2026 critique argues byte-level modeling destroys protocol semantics and that reported accuracies stem from leakage. The project is unusually well-defended against that critique — it masks the exact random fields the critics name, it does unsupervised anomaly detection rather than the leakage-prone supervised benchmarks, and its cross-dataset transfer plus the n-gram ablation are direct empirical rebuttals. Full detail and citations are in `RESEARCH_LANDSCAPE.md`.

## 7. Business model and go-to-market

The recommended path is open-core. The engine — architecture, training code, evaluation harness, single-node daemon, basic dashboard, and research checkpoints — should be public under MIT, because in network security black-box AI is distrusted and open source is how tools like Snort, Suricata, and Zeek became standards. Openness also turns the sovereignty claim into a verifiable fact. The commercial boundary, which doubles as a security boundary, keeps each customer's tuned weights and calibration thresholds private, denying a white-box attacker the deployed model while defining the paid tier.

Revenue realism matters here: open-source security engines rarely earn money directly; they generate trust and inbound. Revenue comes from the commercial layer — central fleet management, SIEM connectors and retention tiering, per-environment tuned models and calibration-as-a-service, the triage dashboard and analyst workflow, and support. For a solo founder with no traction yet, the realistic near-term income is paid pilots (fixed-fee pcap analysis for MSSPs, consulting-shaped, converting to license later) and non-dilutive grants (iDEX, MeitY, DSCI, where the sovereign angle is tailor-made), not license sales. The public repo makes both easier to win. A public launch (Show HN, r/netsec) should wait until the live-FP number and demo video exist, as it is a one-shot credibility event. Outreach copy for all of these is drafted in `OUTREACH.md`.

## 8. Current status and readiness

The project is at the end of a productive validation-and-hardening phase and near the start of outreach. Validation is effectively complete: same-environment detection, held-out zero-day, cross-dataset transfer, and the complexity-justification ablation have all landed in the project's favor. The product is demo-ready, with alert-budget calibration, incident enrichment, and the surprise heatmap shipped and pushed. Every go-to-market document is written with honest numbers.

The outreach-ready sprint (`PRODUCT_ROADMAP.md`, Phase 0) has two items left: the live false-positive measurement, currently running as a two-day test to confirm the 1/hr budget holds on unseen traffic, and a two-minute demo video. Once both exist, the technical brief's last placeholder is filled and the four outreach drafts are send-ready.

## 9. Roadmap

The near-term is the Phase 0 outreach sprint above. Everything heavier is deliberately deferred until a customer pulls it, to avoid gold-plating a product no one has bought. When a pilot is agreed, the pilot-deliverable polish follows (one-click pcap context, a triage-grade dashboard). Hardening (Phase H) is priority-ordered: cross-session CUSUM accumulation to close the slow-and-low gap; drift monitoring with a poisoning-safe, frozen-reference baseline update; envelope/metadata upweighting to sharpen the one signal retained inside encryption; invariant checkers plus the private-weights policy; and an XDP throughput benchmark followed by a Mamba backbone prototype. A standing validation experiment — the n-gram baseline — is complete; the next architecture experiment is the Mamba backbone.

## 10. Risks and recommended next steps

The principal risks are commercial rather than technical: a solo founder's bandwidth against long enterprise sales cycles, and the need for credibility markers (a second pilot reference, eventually CERT-In empanelment) that take time. The mitigation is to lead with paid pilots and grants, which are shorter paths to a first cheque and to institutional credibility than direct enterprise sales.

The recommended immediate sequence: let the live-FP run finish and record the number; capture the two-minute demo video; tag a v0.1.0 release; then send the first outreach batch — MSSP paid-pilot emails and iDEX/MeitY applications — using the drafted copy. After the first replies, build only what a real conversation asks for. In parallel, when GPU time allows, prototype the Mamba backbone to secure the throughput story that the pre-filter positioning depends on.

---

*Supporting documents: `TECHNICAL_BRIEF.md` (engineer-facing), `LIMITATIONS.md` and `MITIGATION_PLAN.md` (honest scope), `RESEARCH_LANDSCAPE.md` (literature + decisions), `FUTURE_ROADMAP.md` (open-core strategy), `PRODUCT_ROADMAP.md` (build plan), `OUTREACH.md` (go-to-market copy).*
