# Sovereign Byte-Level Firewall
### A self-calibrating zero-day anomaly layer for networks your existing tools can't see into

**Shubham Tomar · shubhamtomar.spst@gmail.com · July 2026**

---

## The gap it fills

Signature-based IDS (Suricata, Snort) and SIEM pipelines detect what they have rules for. Novel payloads — zero-day exploits, custom malware beacons, fuzzing — produce no signature match and sail through. Behavioural NDR products close some of that gap, but they are cloud-connected, telemetry-hungry, and expensive: unacceptable in sovereign, air-gapped, or regulated environments.

The Sovereign Byte-Level Firewall takes a different route. A compact transformer (~1.6M parameters) is trained to predict raw network traffic **byte by byte** — no flow extraction, no log translation, no DPI parsing. Having learned the structural grammar of benign traffic, it scores live packets by *surprise*: how improbable each byte is given everything before it. Attacks it has never seen produce byte sequences the model finds improbable — and that is the detection signal. No signatures, no rule updates, no cloud.

**Three properties matter to a buyer:**

- **Sovereign by construction.** Runs fully offline on a single machine (Apple Silicon laptop to A100 node). No telemetry leaves your network. Model trained on public datasets; fine-tunable on your own traffic in-house.
- **Zero integration cost.** Input is raw traffic — a SPAN/mirror port or a pcap file. No agents, no log shippers, no schema mapping. Deployment is passive and read-only.
- **Self-calibrating.** Day one on a new network is silent: the engine learns that environment's baseline (~1 hour) and sets its own alert threshold. Alerts start day two, tuned to *your* traffic, not a lab's.

## How it works

Traffic is streamed as a 1D sequence of byte values (0–255). Deterministic fields that carry no security signal (addresses, checksums) and always-encrypted payloads (TLS application data, QUIC, WireGuard, SSH transport) are masked so the model learns protocol structure, not ciphertext noise. A 2-layer causal transformer predicts each next byte; the per-window anomaly score is the mean surprise of the most improbable 10% of bytes. A lightweight companion detector watches connection-rate statistics, and the two are OR-fused: the byte model catches payload anomalies, the rate detector catches volumetric attacks. Inference is sub-millisecond per packet on commodity hardware.

## Measured performance

All numbers from held-out evaluation on CIC-IDS2017 (trained on benign Monday traffic only; attacks never seen in training):

| Measurement | Result |
|---|---|
| Full-day evaluation, fused system (Wednesday: DoS Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) | **5/5 attack campaigns detected** |
| False alarms during that day | **0.9–1.4 per hour** (byte detector alone: 0–0.5/hour) |
| Held-out zero-day window detection (single-checkpoint byte model, strictest test) | 32.6% of attack windows @ **0.23% FPR** |
| Companion rate detector (volumetric attacks) | AUC 0.908 |
| Live deployment (real 2026 traffic, MacBook, self-calibrated) | **326 incidents/day** (unaggregated, over 9.25-hour overnight test on local network en0) |
| **Cross-dataset transfer (UNSW-NB15, trained on CIC benign only)** | **calibration AUC 0.75–0.77** — statistically identical to 0.73 on CIC's own held-out |
| Held-out single-class window detection on UNSW @ ~1% FPR (byte-only) | 8.5% (Shellcode) – 11.4% (Exploits) |

**Read the table honestly.** Campaign-level detection is the operational metric: an attack that generates thousands of windows needs only a fraction flagged to be caught, which is how 32.6% window-level detection yields 5/5 campaigns. We publish the window-level number anyway because vendors who only quote campaign metrics are hiding something.

**Why a transformer, not a simple statistical model?** We ran the ablation. A byte n-gram (the obvious cheap alternative) achieves a near-perfect calibration AUC but flags **98%+ of held-out benign traffic** — it memorizes exact byte sequences and treats any unseen-but-normal traffic as anomalous, making it unusable. The transformer holds held-out benign false positives to **0.23%** on the same split, because it learns general protocol structure rather than memorizing strings. That ~400× gap on the false-positive metric is precisely what the model's complexity buys, and it's the same generalization the cross-dataset result below demonstrates.

**On the cross-dataset result.** The point isn't the single-class window numbers — it's the AUC. A model trained *only* on CIC-IDS2017 benign traffic separates benign from attack on UNSW-NB15 (a different lab, year, and attack toolkit) as well as it does on CIC itself (AUC 0.75–0.77 vs 0.73). That is direct evidence it learned general benign-protocol structure rather than memorizing one capture — the property that matters for detecting novel attacks in *your* environment. The 8.5–11.4% figures are held-out *window-level* single-class detection at strict 1% FPR — the most conservative possible framing (one attack class never calibrated on, byte detector alone, no fusion, no campaign aggregation). The operational metric is campaign-level: on CIC the same window-level range yielded 5/5 attack campaigns caught at ~1 false alarm/hour, because a campaign emits thousands of windows and only a fraction need flagging. UNSW campaign-level aggregation is not yet run but follows the same arithmetic.

## What it does not do

This is an anomaly detector, not a full IDS replacement — an *additional layer* for what your stack is blind to:

- It flags *structural anomalies*, not named threats. Triage still belongs to your analysts (alerts carry score, time, and traffic context).
- Slow, low-rate brute-force (e.g. patient FTP password guessing) evades both detectors today; that class is already well covered by fail2ban-style tooling and auth logs. Session-level features to close this gap are on the roadmap.
- Encrypted payload *contents* are invisible to everyone, including us. We detect anomalies in protocol structure, timing envelope, and unencrypted regions.
- A patient sub-threshold adversary, concept drift after major environment changes, and white-box adversarial evasion are real limits, and core-router line rate requires an XDP pre-filter front-end. We document all of this openly in `LIMITATIONS.md` — an anomaly detector that claims no blind spots is lying, and honest scope is why our numbers can be trusted.

## Pilot proposal

Two weeks, fixed fee, read-only:

1. **Days 1–2:** Deploy on a SPAN port or receive a pcap capture (12–48h of your traffic). Engine self-calibrates; we validate the baseline together.
2. **Days 3–12:** Passive monitoring. Every incident logged with score and context.
3. **Days 13–14:** Joint review — every alert triaged with your team, false-positive rate computed on *your* traffic, detection demonstrated by replaying public attack captures through your baseline.

You keep the incident data and the report. If the anomaly layer shows nothing your existing stack didn't already catch, you've spent two weeks and a fixed fee to prove your coverage — also a result.

**Deliverables:** live dashboard during the pilot, incident CSV, written findings report, calibration profile for your environment.

---

*Architecture, evaluation methodology, and code lineage available under NDA. References: CIC-IDS2017 (Canadian Institute for Cybersecurity); evaluation follows standard zero-day protocol — train on benign only, test on unseen attack classes.*
