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
| Live deployment (real 2026 traffic, MacBook, self-calibrated) | *[measurement in progress — N incidents/day over M days]* |
| Second-dataset validation (UNSW-NB15) | *[in progress]* |

**Read the table honestly.** Campaign-level detection is the operational metric: an attack that generates thousands of windows needs only a fraction flagged to be caught, which is how 32.6% window-level detection yields 5/5 campaigns. We publish the window-level number anyway because vendors who only quote campaign metrics are hiding something.

## What it does not do

This is an anomaly detector, not a full IDS replacement — an *additional layer* for what your stack is blind to:

- It flags *structural anomalies*, not named threats. Triage still belongs to your analysts (alerts carry score, time, and traffic context).
- Slow, low-rate brute-force (e.g. patient FTP password guessing) evades both detectors today; that class is already well covered by fail2ban-style tooling and auth logs. Session-level features to close this gap are on the roadmap.
- Encrypted payload *contents* are invisible to everyone, including us. We detect anomalies in protocol structure, timing envelope, and unencrypted regions.

## Pilot proposal

Two weeks, fixed fee, read-only:

1. **Days 1–2:** Deploy on a SPAN port or receive a pcap capture (12–48h of your traffic). Engine self-calibrates; we validate the baseline together.
2. **Days 3–12:** Passive monitoring. Every incident logged with score and context.
3. **Days 13–14:** Joint review — every alert triaged with your team, false-positive rate computed on *your* traffic, detection demonstrated by replaying public attack captures through your baseline.

You keep the incident data and the report. If the anomaly layer shows nothing your existing stack didn't already catch, you've spent two weeks and a fixed fee to prove your coverage — also a result.

**Deliverables:** live dashboard during the pilot, incident CSV, written findings report, calibration profile for your environment.

---

*Architecture, evaluation methodology, and code lineage available under NDA. References: CIC-IDS2017 (Canadian Institute for Cybersecurity); evaluation follows standard zero-day protocol — train on benign only, test on unseen attack classes.*
